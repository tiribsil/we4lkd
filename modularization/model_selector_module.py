import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import logging

from candidate_model_training_module import CandidateModelTraining
from dotproduct_generation_module import ValidationModule
from utils import LoggerFactory, normalize_disease_name

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

    def __init__(self, disease_name: str, logger: logging.Logger):
        self.disease_name = normalize_disease_name(disease_name)
        self.logger = logger
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
            self.logger.info(f"Loading Ground Truth from cache: {cache_file}")
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


class ModelEvaluator:
    """
    Calcula a score de um modelo comparando suas recomendações (top_n files)
    com o Ground Truth.
    """
    def __init__(self, model_subfolder: str, ground_truth: Dict[str, int], base_path: Path, logger: logging.Logger, start_year: int, end_year: int):
        self.model_subfolder = model_subfolder
        self.ground_truth = ground_truth
        self.logger = logger
        self.start_year = start_year
        self.end_year = end_year
        self.top_n_base_path = base_path / "validation" / model_subfolder / "top_n_compounds"

    def _get_first_recommendations(self) -> Dict[str, int]:
        """
        Varre todas as pastas de ano dentro de top_n_compounds para encontrar
        o primeiro ano em que um composto apareceu no ranking.
        """
        first_recommendation = {}
        if not self.top_n_base_path.exists():
            self.logger.warning(f"Top N path not found: {self.top_n_base_path}")
            return {}

        # Listar pastas de anos (ordenadas)
        year_dirs = sorted([
            d for d in self.top_n_base_path.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x.name))

        for year_path in year_dirs:
            year = int(year_path.name)
            
            if not (self.start_year <= year <= self.end_year):
                continue

            # Procurar arquivo CSV de score (ex: top_20_score.csv ou top_50_score.csv)
            csv_files = list(year_path.glob("top_*_score.csv"))
            if not csv_files:
                continue
            
            # Ler o primeiro arquivo encontrado
            try:
                df = pd.read_csv(csv_files[0])
                # Coluna de nome
                col = 'chemical_name' if 'chemical_name' in df.columns else 'compound_name'
                
                if col in df.columns:
                    for compound in df[col].values:
                        if compound not in first_recommendation:
                            first_recommendation[compound] = year
            except Exception as e:
                self.logger.error(f"Error reading {csv_files[0]}: {e}")
                continue
        
        return first_recommendation

    def compute_metrics(self) -> float:
        """
        Calcula a métrica 'Mean How Early' e exibe estatísticas detalhadas.
        """
        recommendations = self._get_first_recommendations()
        
        data_for_stats = []
        
        for compound, rec_year in recommendations.items():
            if compound in self.ground_truth:
                report_year = self.ground_truth[compound]
                # Ignora se já foi reportado no passado
                if report_year < self.start_year:
                    continue
                how_early = report_year - rec_year
                
                # Armazena tupla para o DataFrame
                data_for_stats.append({
                    'compound_name': compound,
                    'how_early': how_early
                })
        
        if not data_for_stats:
            return 0.0

        # Criar DataFrame para facilitar cálculos estatísticos
        df = pd.DataFrame(data_for_stats)
        
        mean_early = df['how_early'].mean()
        median_early = df['how_early'].median()
        std_dev = df['how_early'].std() if len(df) > 1 else 0.0
        mode = df['how_early'].mode().tolist()
        
        # Logar estatísticas gerais
        self.logger.info(f"Model '{self.model_subfolder}':")
        self.logger.info(f"  Mean: {mean_early:.2f} years")
        self.logger.info(f"  Median: {median_early} years")
        self.logger.info(f"  Std Dev: {std_dev:.2f} years")
        self.logger.info(f"  Mode: {mode} years")

        # Top 5 Maiores (Antecipações)
        top_5_biggest = df.nlargest(5, 'how_early')
        self.logger.info("  Top 5 Biggest Anticipations (Years):")
        for _, row in top_5_biggest.iterrows():
            self.logger.info(f"    {row['how_early']} years - {row['compound_name']}")

        # Top 5 Menores (Atrasos)
        top_5_smallest = df.nsmallest(5, 'how_early')
        self.logger.info("  Top 5 Smallest (Delays/Reactive):")
        for _, row in top_5_smallest.iterrows():
            self.logger.info(f"    {row['how_early']} years - {row['compound_name']}")
        
        return mean_early


class ModelSelector:
    """
    Maestro do processo:
    1. Gera Ground Truth (uma vez).
    2. Para cada candidato:
       - Treina (Iterativo)
       - Valida (Gera Rankings)
       - Avalia (Compara Rankings vs Ground Truth)
    3. Escolhe o melhor.
    """

    def __init__(self, disease_name: str, models: Dict[str, List], corpus_start_year: int, start_year: int, end_year: int):
        self.disease_name = disease_name
        self.models = models 
        self.corpus_start_year = corpus_start_year
        self.start_year = start_year
        self.end_year = end_year
        self.logger = LoggerFactory.setup_logger(self.__class__.__name__, log_to_file=True, log_file="model_selection.log")
        self.base_path = Path(f"data/{normalize_disease_name(disease_name)}")
        
        self.trainer = CandidateModelTraining(
            disease_name=disease_name,
            start_year=corpus_start_year,
            end_year=end_year
        )

    def select_best_model(self) -> str:
        self.logger.info("="*60)
        self.logger.info("PHASE: MODEL SELECTION")
        self.logger.info("="*60)

        # 1. Gerar Ground Truth (uma vez para todos os modelos)
        gt_gen = GroundTruthGenerator(self.disease_name, self.logger)
        ground_truth = gt_gen.generate_ground_truth(threshold=3)
        
        if not ground_truth:
            self.logger.error("Could not generate Ground Truth from corpus. Selecting first model as fallback.")
            return list(self.models.keys())[0]

        model_scores = {}

        # 2. Loop de Candidatos
        for model_name, params in self.models.items():
            self.logger.info(f"--- Evaluating Candidate: {model_name} ---")
            
            # A) Treinamento Iterativo
            training_success = True
            for year in range(self.start_year, self.end_year + 1):
                if not self.trainer.train_specific_model(model_name, params, year):
                    training_success = False
                    break
            
            if not training_success:
                self.logger.error(f"Training failed for {model_name}. Skipping.")
                continue

            # B) Geração de Métricas e Rankings (ValidationModule)
            try:
                validator = ValidationModule(
                    disease_name=self.disease_name,
                    model_subfolder=model_name,
                    start_year=self.start_year,
                    end_year=self.end_year,
                    use_chembl=True,
                    top_n_to_save=20 
                )
                if not validator.run():
                    self.logger.warning(f"Validation failed for {model_name}")
                    continue
            except Exception as e:
                self.logger.error(f"Validation exception for {model_name}: {e}")
                continue

            evaluator = ModelEvaluator(
                model_subfolder=model_name,
                ground_truth=ground_truth,
                base_path=self.base_path,
                logger=self.logger,
                start_year=self.start_year,
                end_year=self.end_year
            )
            score = evaluator.compute_metrics()
            model_scores[model_name] = score
            
            self.logger.info(f"Candidate {model_name} Score: {score:.2f}")

        # 3. Seleção
        if not model_scores:
            self.logger.error("No models scored successfully.")
            return list(self.models.keys())[0]

        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]

        self.logger.info("="*60)
        self.logger.info(f"SELECTION COMPLETE")
        self.logger.info(f"Best Model: {best_model}")
        self.logger.info(f"Best Score (Mean Years Early): {best_score:.2f}")
        self.logger.info("="*60)

        return best_model
