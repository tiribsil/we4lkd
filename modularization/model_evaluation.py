import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from reporting import LatentKnowledgeReportGenerator
from extract_report_years import GroundTruthGenerator
from utils import get_logger, normalize_disease_name

class ModelEvaluator:
    """
    Módulo responsável pela Fase 4 (Final Test / Report).
    Executa o treinamento incremental do MELHOR modelo selecionado no período de teste,
    avalia sua performance final e gera o relatório visual.
    """

    def __init__(
        self, 
        disease_name: str, 
        model_name: str, 
        corpus_start_year: int, 
        test_start_year: int, 
        test_end_year: int,
        ground_truth: Optional[Dict[str, int]] = None
    ):
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.model_name = model_name
        
        self.corpus_start_year = corpus_start_year
        self.test_start_year = test_start_year
        self.test_end_year = test_end_year
        self.ground_truth = ground_truth
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.base_path = Path(f"data/{self.normalized_disease_name}")
        self.top_n_base_path = self.base_path / "validation" / self.model_name / "top_n_compounds"

        # Instancia o treinador para ter acesso aos hiperparâmetros e métodos de treino
        self.trainer = CandidateModelTraining(
            disease_name=disease_name,
            start_year=corpus_start_year,
            end_year=test_end_year
        )

    def _get_model_params(self) -> List[any]:
        """Recupera os hiperparâmetros do modelo selecionado (ex: w2v_comb2)."""
        if self.model_name in self.trainer.model_combinations:
            return self.trainer.model_combinations[self.model_name]
        else:
            raise ValueError(f"Model '{self.model_name}' not found in defined combinations.")

    def compute_metrics(self) -> Tuple[float, Dict[int, float]]:
        """Alias para _calculate_final_performance, usado pelo ModelSelector."""
        return self._calculate_final_performance()

    def _calculate_final_performance(self) -> Tuple[float, Dict[int, float]]:
        """
        Calcula a métrica 'Mean Years Early' e gera estatísticas detalhadas.
        """
        if self.ground_truth:
            ground_truth = self.ground_truth
        else:
            self.logger.info("Generating Ground Truth for evaluation...")
            gt_gen = GroundTruthGenerator(self.disease_name, self.logger)
            ground_truth = gt_gen.generate_ground_truth(threshold=3)
            self.ground_truth = ground_truth
        
        if not ground_truth:
            self.logger.warning("No ground truth generated. Score will be 0.")
            return 0.0, {}

        # Encontrar primeira recomendação no período de teste
        first_recommendation = {}
        
        for year in range(self.test_start_year, self.test_end_year + 1):
            year_path = self.top_n_base_path / str(year)
            if not year_path.exists(): continue
                
            csv_files = list(year_path.glob("top_*_score.csv"))
            if not csv_files: continue
            
            try:
                df = pd.read_csv(csv_files[0])
                col = 'chemical_name' if 'chemical_name' in df.columns else 'compound_name'
                
                if col in df.columns:
                    for compound in df[col].values:
                        if compound not in first_recommendation and compound in ground_truth:
                            first_recommendation[compound] = year
            except Exception as e:
                self.logger.error(f"Error reading ranking file for year {year}: {e}")

        # Preparar dados
        details = []
        for compound, rec_year in first_recommendation.items():
            report_year = ground_truth[compound]
            if report_year < self.test_start_year:
                continue
            how_early = report_year - rec_year
            details.append((compound, rec_year, report_year, how_early))

        if not details:
            self.logger.warning("No intersection between recommendations and ground truth in test period.")
            return 0.0

        # Criar DataFrame
        details_df = pd.DataFrame(details, columns=['compound', 'recommendation_year', 'literature_report_year', 'years_early'])
        
        # Cálculos Estatísticos
        mean_early = details_df['years_early'].mean()
        median_early = details_df['years_early'].median()
        std_dev = details_df['years_early'].std() if len(details_df) > 1 else 0.0
        mode = details_df['years_early'].mode().tolist()
        
        # Logar Detalhes (Formato similar ao script antigo)
        self.logger.info(f"{'='*40}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"  Test Period: {self.test_start_year}-{self.test_end_year}")
        self.logger.info(f"  Anticipation Mean: {mean_early:.2f}")
        self.logger.info(f"  Anticipation Median: {median_early}")
        self.logger.info(f"  Anticipation Std Dev: {std_dev:.2f}")
        self.logger.info(f"  Anticipation Mode: {mode}")
        
        self.logger.info("Top 5 Biggest Anticipations:")
        top_5_biggest = details_df.nlargest(5, 'years_early')
        for _, row in top_5_biggest.iterrows():
            self.logger.info(f"  {row['years_early']} years - {row['compound']}")

        self.logger.info("Top 5 Smallest (Delays):")
        top_5_smallest = details_df.nsmallest(5, 'years_early')
        for _, row in top_5_smallest.iterrows():
            self.logger.info(f"  {row['years_early']} years - {row['compound']}")
        self.logger.info(f"{'='*40}")
        
        # Salvar CSV
        output_csv = self.base_path / "reports" / "final_model_validation.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(output_csv, index=False)
        self.logger.info(f"Full validation details saved to {output_csv}")

        # Cálculos Anuais
        annual_scores = {}
        for year in range(self.test_start_year, self.test_end_year + 1):
            year_data = details_df[details_df['recommendation_year'] == year]
            if not year_data.empty:
                annual_scores[year] = year_data['years_early'].mean()
            else:
                annual_scores[year] = 0.0

        return mean_early, annual_scores

    def run(self) -> bool:
        """
        Executa o pipeline de avaliação final.
        """
        self.logger.info(f"=== Starting Final Evaluation for '{self.model_name}' ===")
        
        try:
            # 1. Recuperar Parâmetros
            params = self._get_model_params()
            self.logger.info(f"Hyperparameters: {params}")

            # 2. Treinamento Incremental (Ano a Ano no período de teste)
            self.logger.info("--- Step 1: Incremental Training ---")
            for year in range(self.test_start_year, self.test_end_year + 1):
                # O método train_specific_model verifica se o modelo já existe antes de treinar
                success = self.trainer.train_specific_model(self.model_name, params, year)
                if not success:
                    self.logger.error(f"Failed to train model for year {year}. Aborting.")
                    return False

            # 3. Geração de Métricas e Rankings
            self.logger.info("--- Step 2: Generating Metrics & Rankings ---")
            # ValidationModule é inteligente o suficiente para processar o range de uma vez
            validator = ValidationModule(
                disease_name=self.disease_name,
                model_subfolder=self.model_name,
                start_year=self.test_start_year,
                end_year=self.test_end_year,
                use_chembl=True,
                top_n_to_save=20
            )
            if not validator.run():
                self.logger.error("Validation module failed.")
                return False

            # 4. Cálculo de Performance (Score Final)
            self.logger.info("--- Step 3: Calculating Final Performance ---")
            # Atualiza self.ground_truth se for gerado internamente
            if not self.ground_truth:
                self._calculate_final_performance()
            else:
                self._calculate_final_performance()

            # 5. Geração do Relatório Visual
            self.logger.info("--- Step 4: Generating Visual Report ---")
            # O ReportGenerator vai pegar os dados gerados pelo ValidationModule
            # e criar os gráficos (incluindo a trajetória PCA).
            reporter = LatentKnowledgeReportGenerator(
                disease_name=self.disease_name,
                model_subfolder=self.model_name,
                start_year=self.test_start_year,
                target_year=self.test_end_year, # Foca o relatório no último ano
                top_n_to_plot=15,
                ground_truth=self.ground_truth
            )
            reporter.run()

            self.logger.info("=== Final Evaluation Completed Successfully ===")
            return True

        except Exception as e:
            self.logger.exception(f"Fatal error in ModelEvaluator: {e}")
            return False

if __name__ == "__main__":
    # Exemplo de uso manual
    evaluator = ModelEvaluator(
        disease_name="acute myeloid leukemia",
        model_name="w2v_comb1",
        corpus_start_year=1990,
        test_start_year=2021,
        test_end_year=2024
    )
    evaluator.run()
