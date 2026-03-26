import datetime
from pathlib import Path
import pandas as pd

from topic_expansion import IterativeTopicExpansion
from data_collection import DataCollection
from preprocessing import Preprocessing
from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from model_selection import ModelSelector
from model_evaluation import ModelEvaluator
from contextualization import ContextualizationModule
from utils import LoggerFactory, get_logger, _load_checkpoint, _save_checkpoint, normalize_disease_name

class PreliminaryPipeline:
    def __init__(
        self, 
        disease_name: str, 
        max_topics: int, 
        max_new_topics: int, 
        train_val_test_split: list[float]
    ):
        if len(train_val_test_split) != 3:
            raise ValueError("The train_val_test_split list must have 3 values!")
            
        self.disease_name = disease_name
        self.max_topics = max_topics
        self.max_new_topics = max_new_topics
        self.train_val_test_split = train_val_test_split
        self.use_lhs = True
        self.num_combinations = 30
        
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.today_year = datetime.datetime.now().year
        
        # Initialize logging
        LoggerFactory.setup(log_to_file=True, log_file='logs/preliminary_pipeline.log')
        self.logger = get_logger(__name__)
        
        # State variables
        self.checkpoint_data = _load_checkpoint(self.normalized_disease_name)
        self.last_expansion_year = self.checkpoint_data.get("last_expansion_year")
        self.model_dev_end_year = self.checkpoint_data.get("model_dev_end_year")
        self.model_selection_end_year = self.checkpoint_data.get("model_selection_end_year")
        self.best_model = self.checkpoint_data.get("best_model_name")
        self.trained_models_info = self.checkpoint_data.get("trained_models_info")

    def run(self):
        """Runs the complete preliminary pipeline."""
        self.logger.info(f"Starting Preliminary Pipeline: {self.disease_name}")
        
        self._phase_1_topic_expansion()
        self._phase_2_dev_and_training()
        self._phase_3_selection()
        self._phase_4_evaluation()
        self._phase_5_summary()
        
        self.logger.info("PRELIMINARY PIPELINE FINISHED")

    def _phase_1_topic_expansion(self):
        self.logger.info(">>> Phase 1 (Expansion)")
        
        if not self.checkpoint_data.get("phase_1_topic_expansion_completed"):
            topic_expander = IterativeTopicExpansion(
                disease_name=self.disease_name,
                max_topics=self.max_topics,
                max_new_topics=self.max_new_topics
            )
            self.last_expansion_year = topic_expander.run()
            self.logger.info(f"Expansion done: {self.last_expansion_year}")
            
            self.checkpoint_data["last_expansion_year"] = self.last_expansion_year
            self.checkpoint_data["phase_1_topic_expansion_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info(f"Phase 1 skipped ({self.last_expansion_year})")

    def _phase_2_dev_and_training(self):
        self.logger.info(">>> Phase 2 (Dev & Training)")
        
        model_dev_start_year = self.last_expansion_year + 1
        year_range = self.today_year - model_dev_start_year
        
        if self.model_dev_end_year is None:
            self.model_dev_end_year = int(model_dev_start_year + self.train_val_test_split[0] * year_range)
            self.checkpoint_data["model_dev_end_year"] = self.model_dev_end_year
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        
        self.logger.info(f"Model dev period: {model_dev_start_year}-{int(self.model_dev_end_year)}")

        # Data Collection
        if not self.checkpoint_data.get("phase_2_data_collection_completed"):
            self.logger.info("Collecting data...")
            for year in range(model_dev_start_year, self.today_year + 1):
                self.logger.info(f"Year {year}...")
                dc = DataCollection(disease_name=self.disease_name, target_year=year)
                dc.run()
            self.checkpoint_data["phase_2_data_collection_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info("Data collection skipped")

        # Preprocessing
        if not self.checkpoint_data.get("phase_3_preprocessing_completed"):
            self.logger.info("Preprocessing data...")
            preprocessor = Preprocessing(disease_name=self.disease_name, target_year=self.today_year, incremental=False)
            preprocessor.run(force_full=True)
            self.checkpoint_data["phase_3_preprocessing_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info("Preprocessing skipped")

        # Model Training
        if not self.checkpoint_data.get("phase_4_automl_training_completed"):
            self.logger.info("Training candidate models...")
            cmt = CandidateModelTraining(
                disease_name=self.disease_name,
                start_year=model_dev_start_year,
                end_year=self.model_dev_end_year,
                use_lhs=self.use_lhs,
                num_combinations=self.num_combinations
            )
            self.trained_models_info = cmt.run()
            self.checkpoint_data["trained_models_info"] = self.trained_models_info
            self.checkpoint_data["phase_4_automl_training_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info("Model training skipped")
            self.trained_models_info = self.checkpoint_data.get("trained_models_info")

        # Metric Generation
        if not self.checkpoint_data.get("phase_5_metric_generation_completed"):
            if not self.trained_models_info:
                self.logger.error("No models found for metric generation.")
                return
                
            self.logger.info("Generating metrics for candidate models...")
            for model_name, _ in self.trained_models_info.items():
                self.logger.info(f"Model: {model_name}")
                validator = ValidationModule(
                    disease_name=self.disease_name,
                    model_subfolder=model_name,
                    start_year=self.model_dev_end_year,
                    end_year=self.model_dev_end_year
                )
                validator.run()
            self.checkpoint_data["phase_5_metric_generation_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info("Metric generation skipped")

    def _phase_3_selection(self):
        self.logger.info(">>> Phase 3 (Selection)")
        
        model_selection_start_year = self.model_dev_end_year + 1
        model_dev_start_year = self.last_expansion_year + 1
        year_range = self.today_year - model_dev_start_year

        if self.model_selection_end_year is None:
            self.model_selection_end_year = int(model_selection_start_year + self.train_val_test_split[1] * year_range)
            self.checkpoint_data["model_selection_end_year"] = self.model_selection_end_year
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)

        if not self.checkpoint_data.get("phase_6_model_selection_completed"):
            self.logger.info("Selecting best model...")
            selector = ModelSelector(
                disease_name=self.disease_name,
                models=self.trained_models_info,
                corpus_start_year=model_dev_start_year,
                start_year=model_selection_start_year,
                end_year=self.model_selection_end_year
            )
            self.best_model = selector.select_best_model()
            self.logger.info(f"Best model selected: {self.best_model}")
            
            self.checkpoint_data["best_model_name"] = self.best_model
            self.checkpoint_data["phase_6_model_selection_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info(f"Model selection skipped (Best: {self.best_model})")

    def _phase_4_evaluation(self):
        self.logger.info(">>> Phase 4 (Evaluation)")
        
        test_start_year = self.model_selection_end_year + 1
        test_end_year = self.today_year
        model_dev_start_year = self.last_expansion_year + 1

        if not self.checkpoint_data.get("phase_7_final_report_completed"):
            self.logger.info(f"Generating evaluation report for: {self.best_model}")
            me = ModelEvaluator(
                self.disease_name, 
                self.best_model, 
                model_dev_start_year, 
                test_start_year, 
                test_end_year
            )
            me.run()
            self.checkpoint_data["phase_7_final_report_completed"] = True
            _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
        else:
            self.logger.info("Evaluation report skipped")

    def _phase_5_summary(self):
        self.logger.info(">>> Phase 5 (Summary)")
        
        test_end_year = self.today_year
        
        if not self.checkpoint_data.get("phase_8_contextualization_completed"):
            n = 20
            top_compounds_csv = Path(f'./data/{self.normalized_disease_name}/validation/{self.best_model}/top_n_compounds/{test_end_year}/top_{n}_score.csv')
            
            if top_compounds_csv.exists():
                try:
                    self.logger.info(f"Contextualizing results for {test_end_year}...")
                    df = pd.read_csv(top_compounds_csv)
                    if 'chemical_name' in df.columns:
                        compounds = df['chemical_name'].tolist()
                        context_module = ContextualizationModule(disease=self.disease_name)
                        results = context_module.analyze_batch(compounds)
                        
                        output_file = Path(f'./data/{self.normalized_disease_name}/contextualization_results_{test_end_year}.json')
                        context_module.export_json(results, str(output_file))
                        
                        self.checkpoint_data["phase_8_contextualization_completed"] = True
                        _save_checkpoint(self.normalized_disease_name, self.checkpoint_data)
                    else:
                        self.logger.error(f"Missing 'chemical_name' in {top_compounds_csv}")
                except Exception as e:
                    self.logger.error(f"Contextualization error: {e}")
            else:
                self.logger.warning(f"Top compounds file not found: {top_compounds_csv}")
        else:
            self.logger.info("Contextualization skipped")

if __name__ == '__main__':
    pipeline = PreliminaryPipeline(
        disease_name='acute myeloid leukemia',
        max_topics=9,
        max_new_topics=9,
        train_val_test_split=[0.6, 0.2, 0.2]
    )
    pipeline.run()
