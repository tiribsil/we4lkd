import pandas as pd
import re

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import logging

from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from model_evaluation import ModelEvaluator
from extract_report_years import GroundTruthGenerator
from utils import get_logger, normalize_disease_name


# ModelEvaluator importado de model_evaluation_module


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
        self.logger = get_logger(self.__class__.__name__)
        self.base_path = Path(f"data/{normalize_disease_name(disease_name)}")
        
        self.trainer = CandidateModelTraining(
            disease_name=disease_name,
            start_year=corpus_start_year,
            end_year=end_year
        )

    def select_best_model(self) -> str:
        self.logger.info("Starting model selection")

        # 1. Gerar Ground Truth (uma vez para todos os modelos)
        gt_gen = GroundTruthGenerator(self.disease_name, self.logger)
        ground_truth = gt_gen.generate_ground_truth(threshold=3)
        
        if not ground_truth:
            self.logger.error("Could not generate Ground Truth from corpus. Selecting first model as fallback.")
            return list(self.models.keys())[0]

        model_scores = {}

        # 2. Loop de Candidatos
        for model_name, params in self.models.items():
            self.logger.info(f"Eval candidate: {model_name}")
            
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
                disease_name=self.disease_name,
                model_name=model_name,
                corpus_start_year=self.corpus_start_year,
                test_start_year=self.start_year,
                test_end_year=self.end_year,
                ground_truth=ground_truth
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

        self.logger.info(f"Best: {best_model} ({best_score:.2f})")

        return best_model
