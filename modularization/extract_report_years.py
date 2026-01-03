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
        prompt = f"""[INST] <<SYS>>
You are a medical research assistant. Answer only YES or NO.
<</SYS>>

Does the following abstract indicate that {compound} is officialized as a treatment for {self.disease_name}?
Abstract: {abstract}
[/INST]"""
        
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

    def _load_corpus(self) -> pd.DataFrame:
        if not self.corpus_path.exists():
            # Check for spark directory
            if self.corpus_path.parent.exists() and self.corpus_path.parent.is_dir():
                 csv_files = list(self.corpus_path.parent.glob('*.csv'))
                 if csv_files:
                     return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            
            self.logger.error(f"Corpus not found at {self.corpus_path}")
            return pd.DataFrame()
        return pd.read_csv(self.corpus_path)

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

        # 1. Tentar carregar do cache
        if cache_file.exists() and not force_regenerate:
            self.logger.info("Loading Ground Truth from cache.")
            try:
                df_cache = pd.read_csv(cache_file)
                return dict(zip(df_cache['compound'], df_cache['year']))
            except Exception as e:
                self.logger.warning(f"Failed to load cache ({e}). Regenerating...")

        # 2. Gerar do zero
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

        year_reported = {}
        
        for compound in tqdm(compounds, desc="Verifying compounds with LLM"):
            try:
                compound_pat = r'\b' + re.escape(compound) + r'\b'
                # Filter abstracts mentioning the compound
                compound_mask = disease_df['summary'].str.contains(compound_pat, flags=re.IGNORECASE)
                eligible_abstracts = disease_df[compound_mask]
                
                if eligible_abstracts.empty:
                    continue

                valid_count = 0
                first_year = None
                
                # Check abstracts in chronological order
                for _, row in eligible_abstracts.iterrows():
                    if self._verify_with_llm(compound, row['summary']):
                        valid_count += 1
                        if first_year is None:
                            first_year = int(row['year_extracted'])
                        
                        if valid_count >= threshold:
                            break
                
                if first_year is not None:
                    year_reported[compound] = first_year
                    
            except Exception as e:
                self.logger.error(f"Error processing {compound}: {e}")
                continue
        
        self.logger.info(f"Ground Truth generated: {len(year_reported)} compounds found.")

        # 3. Salvar no cache
        try:
            df_out = pd.DataFrame(list(year_reported.items()), columns=['compound', 'year'])
            df_out.to_csv(cache_file, index=False)
            self.logger.info(f"Ground Truth saved to cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"Could not save cache: {e}")

        return year_reported
