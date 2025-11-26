import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from candidate_model_training_module import CandidateModelTraining
from dotproduct_generation_module import ValidationModule
from utils import LoggerFactory, normalize_disease_name

class LatentKnowledgeScorer:
    """
    Calcula a pontuação de conhecimento latente comparando as predições do modelo
    (Top N compostos) com um 'Ground Truth' (ex: dados do FDA).
    """

    def __init__(self, disease_name: str, year: int, model_subfolder: str, logger: logging.Logger):
        self.disease_name = normalize_disease_name(disease_name)
        self.year = year
        self.model_subfolder = model_subfolder
        self.logger = logger
        self.base_path = Path(f"data/{self.disease_name}")
        
        # Caminho dos rankings gerados pelo ValidationModule
        self.top_n_path = self.base_path / "validation" / self.model_subfolder / "top_n_compounds" / str(self.year)
        
        # Caminho do Ground Truth (FDA Data)
        self.fda_path = self.base_path / "fda_data" / "agg_reports.csv"
        self.fda_df = self._load_fda_data()

    def _load_fda_data(self) -> pd.DataFrame:
        if not self.fda_path.exists():
            self.logger.warning(f"FDA data not found at {self.fda_path}. Scoring will be 0.")
            return pd.DataFrame(columns=['compounds', 'year'])
        try:
            return pd.read_csv(self.fda_path)
        except Exception as e:
            self.logger.error(f"Error loading FDA data: {e}")
            return pd.DataFrame(columns=['compounds', 'year'])

    def _get_ranking_file(self, metric: str = 'score') -> Optional[Path]:
        if not self.top_n_path.exists():
            return None
        # Tenta encontrar qualquer arquivo que combine com o padrão (top_20, top_50, etc)
        candidates = list(self.top_n_path.glob(f"top_*_{metric}.csv"))
        return candidates[0] if candidates else None

    def get_score(self) -> float:
        """
        Retorna a pontuação média 'how_early' (antecipação).
        """
        try:
            ranking_file = self._get_ranking_file('score')
            if not ranking_file:
                return 0.0

            df = pd.read_csv(ranking_file)
            top_compounds = df.head(10) # Avalia apenas top 10

            hits = []
            for _, row in top_compounds.iterrows():
                compound_name = row['chemical_name']
                first_mention = self._find_first_mention(compound_name)

                if first_mention:
                    # Só conta pontos se a primeira menção no FDA for no futuro ou no mesmo ano
                    if first_mention >= self.year:
                        years_early = first_mention - self.year
                        hits.append(years_early)

            if not hits:
                return 0.0

            # Score é a média de anos de antecipação
            return sum(hits) / len(hits)

        except Exception as e:
            self.logger.error(f"Error scoring {self.model_subfolder}/{self.year}: {e}")
            return 0.0

    def _find_first_mention(self, compound_name: str) -> Optional[int]:
        if self.fda_df.empty: return None
        import re
        safe_name = re.escape(compound_name)
        matches = self.fda_df[self.fda_df['compounds'].str.contains(safe_name, case=False, na=False, regex=True)]
        if not matches.empty:
            return int(matches['year'].min())
        return None


class ModelSelector:
    """
    Responsável por:
    1. Treinar modelos candidatos incrementalmente nos anos de seleção (usando CandidateModelTraining).
    2. Gerar métricas (chamando ValidationModule).
    3. Calcular Latent Knowledge Score.
    4. Selecionar o melhor modelo.
    """

    def __init__(self, disease_name: str, models: Dict[str, List], corpus_start_year: int, start_year: int, end_year: int):
        self.disease_name = disease_name
        self.models = models # Dict[model_name, list_of_params]
        self.corpus_start_year = corpus_start_year
        self.start_year = start_year
        self.end_year = end_year
        self.logger = LoggerFactory.setup_logger(self.__class__.__name__, log_to_file=True, log_file="model_selection.log")
        
        # Instancia o treinador (note que o end_year aqui no init é o teto máximo, 
        # mas o método train_specific_model usará o ano alvo específico)
        self.trainer = CandidateModelTraining(
            disease_name=disease_name,
            start_year=corpus_start_year,
            end_year=end_year
        )

    def select_best_model(self) -> str:
        model_scores = {name: 0.0 for name in self.models.keys()}
        self.logger.info("="*60)
        self.logger.info(f"Starting Model Selection Phase ({self.start_year}-{self.end_year})")
        self.logger.info(f"Candidates: {list(self.models.keys())}")
        self.logger.info("="*60)

        for model_name, params in self.models.items():
            self.logger.info(f"Processing candidate: {model_name}")
            
            # --- PASSO 1: Treinamento Iterativo ---
            self.logger.info(f"1. Training models iteratively ({self.start_year} to {self.end_year})...")
            training_success = True
            
            for year in range(self.start_year, self.end_year + 1):
                # Chama o treinador existente para criar o modelo específico desse ano
                if not self.trainer.train_specific_model(model_name, params, year):
                    training_success = False
                    break
            
            if not training_success:
                self.logger.error(f"Training failed for {model_name}. Skipping this candidate.")
                continue

            # --- PASSO 2: Geração de Métricas (Validation Module) ---
            self.logger.info(f"2. Generating validation metrics for {model_name}...")
            try:
                # ValidationModule varre start->end e gera os CSVs para cada ano
                validator = ValidationModule(
                    disease_name=self.disease_name,
                    model_subfolder=model_name,
                    start_year=self.start_year,
                    end_year=self.end_year,
                    use_chembl=True,
                    top_n_to_save=20
                )
                if not validator.run():
                    self.logger.warning(f"Validation module returned False for {model_name}")
            except Exception as e:
                self.logger.error(f"Validation failed for {model_name}: {e}")
                continue

            # --- PASSO 3: Cálculo de Score (Latent Knowledge) ---
            self.logger.info(f"3. Calculating Latent Knowledge Score...")
            total_score = 0.0
            
            for year in range(self.start_year, self.end_year + 1):
                scorer = LatentKnowledgeScorer(
                    disease_name=self.disease_name,
                    year=year,
                    model_subfolder=model_name,
                    logger=self.logger
                )
                score = scorer.get_score()
                total_score += score
            
            model_scores[model_name] = total_score
            self.logger.info(f"CANDIDATE SCORE [{model_name}]: {total_score:.4f}")
            self.logger.info("-" * 40)

        # Seleção Final
        if not model_scores:
            self.logger.error("No models were successfully evaluated.")
            return list(self.models.keys())[0] if self.models else ""

        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]

        self.logger.info("="*60)
        self.logger.info(f"SELECTION COMPLETE")
        self.logger.info(f"Winner: {best_model}")
        self.logger.info(f"Score: {best_score:.4f}")
        self.logger.info("="*60)

        return best_model
