import pandas as pd
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from utils import normalize_disease_name, get_logger
from llm_utils import get_report_year_llm

class GroundTruthGenerator:
    """
    Responsável por determinar o 'Ground Truth' usando um LLM local para verificar
    se um composto é reportado como tratamento para a doença.
    """
    
    def __init__(self, disease_name: str, logger: Optional[logging.Logger] = None):
        self.disease_name = normalize_disease_name(disease_name)
        self.logger = logger or get_logger(self.__class__.__name__)
        self.base_path = Path(f"data/{self.disease_name}")
        self.corpus_path = self.base_path / "corpus/clean_abstracts/clean_abstracts.csv"
        self.whitelist_path = Path("data/compound_whitelist.txt")
        self.cache_dir = self.base_path / "ground_truth_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM initialization
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_report_year_llm()
        return self._llm

    def _verify_with_llm(self, compound: str, abstract: str) -> bool:
        """
        Usa o LLM para verificar se o abstract indica que o composto é um tratamento.
        """
        prompt = f"""<|im_start|>system
You are a strict clinical data scientist. Your task is to extract therapeutic relationships from medical abstracts.
Analyze the provided abstract to determine if "{compound}" is being used or investigated as a DIRECT therapeutic agent for "{self.disease_name}".

CRITERIA FOR "YES":
- The abstract describes "{compound}" as a treatment, drug, or therapeutic agent specifically targeting "{self.disease_name}".
- It results in cancer cell death, tumor reduction, or clinical improvement in "{self.disease_name}".
- Includes experimental, preclinical, or clinical investigations of the compound for this specific disease.

CRITERIA FOR "NO" (STRICT EXCLUSIONS):
- SUPPORTIVE CARE: Mentioned only for side effects, secondary infections, or general comfort (e.g., antiemetics, antibiotics, painkillers).
- CAUSATION: Mentioned as a cause of the disease or a risk factor.
- NEGATIVE STUDY: The abstract explicitly concludes that the compound is ineffective or purely toxic without benefit.
- CONTEXT ONLY: Mentioned as a chemical tool or unrelated drug in the background (e.g., "The patient was previously on {compound} for an unrelated condition").

Respond ONLY with "YES" or "NO". Do not provide reasoning.
<|im_end|>
<|im_start|>user
Does this abstract indicate that {compound} is a DIRECT treatment or therapeutic candidate for {self.disease_name}?

Abstract: {abstract}
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=10, 
                temperature=0.01,
            )
            response = output['choices'][0]['text'].strip().upper()
            return "YES" in response
        except Exception as e:
            self.logger.error(f"LLM inference error: {e}")
            return False

    def _save_checkpoint(self, checkpoint_file: Path, processed_compounds: set, year_reported: dict):
        """Salva o progresso atual em um arquivo de checkpoint."""
        try:
            records = []
            for comp in processed_compounds:
                if comp in year_reported:
                    yr, abs_text = year_reported[comp]
                    records.append({'compound': comp, 'year': yr, 'abstract_evidence': abs_text})
                else:
                    records.append({'compound': comp, 'year': None, 'abstract_evidence': None})
            
            pd.DataFrame(records).to_csv(checkpoint_file, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _load_corpus(self) -> pd.DataFrame:
        if not self.corpus_path.exists():
            # Check for spark directory
            if self.corpus_path.parent.exists() and self.corpus_path.parent.is_dir():
                 csv_files = list(self.corpus_path.parent.glob('*.csv'))
                 if not csv_files:
                     self.logger.error(f"Corpus not found at {self.corpus_path}")
                     return pd.DataFrame()
            else:
                self.logger.error(f"Corpus not found at {self.corpus_path}")
                return pd.DataFrame()
        else:
            csv_files = [self.corpus_path]

        all_dfs = []
        for path in csv_files:
            try:
                all_dfs.append(pd.read_csv(path))
            except Exception:
                self.logger.warning(f"Standard read failed for {path.name}. Using rsplit fallback.")
                raw_data = []
                with open(path, 'r', encoding='utf-8') as f:
                    f.readline() # Skip header
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.rsplit(',', 1)
                        if len(parts) == 2:
                            raw_data.append({
                                'summary': parts[0].strip().strip('"'), 
                                'year_extracted': parts[1].strip()
                            })
                if raw_data:
                    df_fallback = pd.DataFrame(raw_data)
                    df_fallback['year_extracted'] = pd.to_numeric(df_fallback['year_extracted'], errors='coerce')
                    all_dfs.append(df_fallback)
        
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _load_whitelist(self) -> List[str]:
        if not self.whitelist_path.exists():
            self.logger.warning("Whitelist file not found.")
            return []
        with open(self.whitelist_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def generate_ground_truth(self, threshold: int = 1, force_regenerate: bool = False) -> Dict[str, int]:
        """
        Retorna dicionário {composto: ano_primeiro_reporte}.
        Usa LLM para validar abstracts que mencionam a doença e o composto.
        """
        cache_file = self.cache_dir / f"ground_truth_llm.csv"
        checkpoint_file = self.cache_dir / "ground_truth_llm_checkpoint.csv"

        # 1. Tentar carregar do cache
        if cache_file.exists() and not force_regenerate:
            self.logger.info("Loading Ground Truth from cache.")
            try:
                df_cache = pd.read_csv(cache_file)
                return dict(zip(df_cache['compound'], df_cache['year']))
            except Exception as e:
                self.logger.warning(f"Failed to load cache ({e}). Regenerating...")

        year_reported = {}
        processed_compounds = set()

        # 2. Tentar carregar do checkpoint (se não estiver forçando regeneração)
        if checkpoint_file.exists() and not force_regenerate:
            self.logger.info("Loading progress from checkpoint.")
            try:
                df_cp = pd.read_csv(checkpoint_file)
                for _, row in df_cp.iterrows():
                    comp = row['compound']
                    processed_compounds.add(comp)
                    if not pd.isna(row['year']):
                        year_reported[comp] = (int(row['year']), row['abstract_evidence'])
                self.logger.info(f"Resuming from checkpoint: {len(processed_compounds)} compounds already processed.")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint ({e}).")

        # 3. Gerar do zero ou continuar
        self.logger.info("Generating Ground Truth using LLM...")
        df = self._load_corpus()
        compounds = self._load_whitelist()
        
        if df.empty or not compounds:
            return {}

        df['summary'] = df['summary'].astype(str)
        
        # Pre-filter papers that mention the disease
        disease_mask = df['summary'].str.contains(self.disease_name)
        disease_df = df[disease_mask].sort_values('year_extracted').copy()
        
        if disease_df.empty:
            self.logger.warning("No abstracts found for the disease.")
            return {}

        year_reported = year_reported # Mantém o que foi carregado do checkpoint
        
        for compound in tqdm(compounds, desc="Verifying compounds with LLM"):
            if compound in processed_compounds:
                continue
                
            try:
                compound_pat = r'\b' + re.escape(compound) + r'\b'
                # Filter abstracts mentioning the compound
                compound_mask = disease_df['summary'].str.contains(compound_pat, flags=re.IGNORECASE)
                eligible_abstracts = disease_df[compound_mask]
                
                if eligible_abstracts.empty:
                    continue

                valid_count = 0
                first_year = None
                first_abstract = None
                
                # Check abstracts in chronological order
                for _, row in eligible_abstracts.iterrows():
                    if self._verify_with_llm(compound, row['summary']):
                        valid_count += 1
                        if first_year is None:
                            first_year = int(row['year_extracted'])
                            first_abstract = row['summary']
                        
                        if valid_count >= threshold:
                            break
                
                if first_year is not None:
                    year_reported[compound] = (first_year, first_abstract)
                    
            except Exception as e:
                self.logger.error(f"Error processing {compound}: {e}")
                # Mesmo com erro, marcamos como processado para não travar o loop infinitamente se for erro de dado
                processed_compounds.add(compound)
                continue
            
            # Salvar checkpoint após cada composto processado
            processed_compounds.add(compound)
            self._save_checkpoint(checkpoint_file, processed_compounds, year_reported)
        
        self.logger.info(f"Ground Truth generated: {len(year_reported)} compounds found.")

        # 3. Salvar no cache
        try:
            # year_reported agora é {composto: (ano, abstract)}
            data_out = []
            for comp, (yr, abs_text) in year_reported.items():
                data_out.append({'compound': comp, 'year': yr, 'abstract_evidence': abs_text})
            
            df_out = pd.DataFrame(data_out)
            df_out.to_csv(cache_file, index=False)
            self.logger.info(f"Ground Truth saved to cache: {cache_file}")
            
            # Se terminou com sucesso, podemos remover o checkpoint ou mantê-lo
            # Vou mantê-lo por segurança, mas o cache_file agora é o mestre.
        except Exception as e:
            self.logger.error(f"Could not save cache: {e}")

        # Retorna o formato original {composto: ano} para compatibilidade
        return {comp: yr for comp, (yr, _) in year_reported.items()}
