import pandas as pd
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from utils import normalize_disease_name, get_logger

class GroundTruthGenerator:
    """
    Responsável por determinar o 'Ground Truth': o ano em que cada composto
    foi de fato reportado na literatura associado à doença em contexto terapêutico.
    Inclui sistema de cache para evitar reprocessamento do corpus.
    """
    
    THERAPEUTIC_KEYWORDS = {
        'treat', 'treatment', 'therapy', 'therapeutic', 'efficacy', 'effective',
        'clinical trial', 'patients', 'remission', 'response', 'inhibit', 
        'antiproliferative', 'antitumor', 'antineoplastic', 'chemotherapy', 'regimen'
    }

    NON_THERAPEUTIC_KEYWORDS = {
        'toxic', 'toxicity', 'carcinogen', 'carcinogenic', 'mutagen', 'mutagenic',
        'side effect', 'adverse', 'poison', 'environmental', 'exposure', 'risk factor'
    }

    def __init__(self, disease_name: str, logger: Optional[logging.Logger] = None):
        self.disease_name = normalize_disease_name(disease_name)
        self.logger = logger or get_logger(self.__class__.__name__)
        self.base_path = Path(f"data/{self.disease_name}")
        self.corpus_path = self.base_path / "corpus/clean_abstracts/clean_abstracts.csv"
        self.whitelist_path = Path("data/compound_whitelist.txt")
        self.cache_dir = self.base_path / "ground_truth_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Regex compilation
        self.positive_regex = re.compile(r'\b(?:' + '|'.join(self.THERAPEUTIC_KEYWORDS) + r')\b', re.IGNORECASE)
        self.negative_regex = re.compile(r'\b(?:' + '|'.join(self.NON_THERAPEUTIC_KEYWORDS) + r')\b', re.IGNORECASE)
        self.disease_regex = re.compile(r'\b' + re.escape(self.disease_name) + r'\b', re.IGNORECASE)

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

    def generate_ground_truth(self, threshold: int = 3, force_regenerate: bool = False) -> Dict[str, int]:
        """
        Retorna dicionário {composto: ano_primeiro_reporte}.
        Usa cache baseado no threshold para evitar reprocessamento.
        """
        cache_file = self.cache_dir / f"ground_truth_t{threshold}.csv"

        # 1. Tentar carregar do cache
        if cache_file.exists() and not force_regenerate:
            self.logger.info("Loading Ground Truth from cache.")
            try:
                df_cache = pd.read_csv(cache_file)
                # Converter para dicionário: compound -> year
                return dict(zip(df_cache['compound'], df_cache['year']))
            except Exception as e:
                self.logger.warning(f"Failed to load cache ({e}). Regenerating...")

        # 2. Gerar do zero (Lógica pesada)
        self.logger.info("Generating Ground Truth (Year Reported) from corpus...")
        df = self._load_corpus()
        compounds = self._load_whitelist()
        
        if df.empty or not compounds:
            return {}

        # Ensure text columns are strings
        df['summary'] = df['summary'].astype(str)
        
        # Pre-filter: Keyword Context
        mask_positive = df['summary'].str.contains(self.positive_regex)
        mask_negative = df['summary'].str.contains(self.negative_regex)
        
        context_df = df[mask_positive & ~mask_negative].copy()
        context_df = context_df[context_df['summary'].str.contains(self.disease_regex)]
        
        if context_df.empty:
            self.logger.warning("No therapeutic abstracts found for the disease.")
            return {}

        year_reported = {}
        
        for compound in tqdm(compounds, desc="Scanning compounds in corpus"):
            try:
                compound_pat = r'\b' + re.escape(compound) + r'\b'
                counts = context_df['summary'].str.count(compound_pat, flags=re.IGNORECASE)
                eligible = context_df[counts >= threshold]
                
                if not eligible.empty:
                    first_year = eligible['year_extracted'].min()
                    year_reported[compound] = int(first_year)
            except Exception:
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
