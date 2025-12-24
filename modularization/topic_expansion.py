from pathlib import Path
import time
import datetime

from data_collection import DataCollection
from preprocessing import Preprocessing
from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from reporting import LatentKnowledgeReportGenerator
from utils import get_logger, normalize_disease_name


class IterativeTopicExpansion:
    def __init__(self, disease_name: str, max_topics: int, max_new_topics: int):
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.max_topics = max_topics
        self.max_new_topics = max_new_topics
        self.logger = get_logger(self.__class__.__name__)

        self.base_path = Path('./data') / self.normalized_disease_name
        self.topics_file = self.base_path / 'topics_of_interest.txt'


    def _get_current_topics_count(self) -> int:
        if not self.topics_file.exists():
            return 0
        with open(self.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
        return len(topics)

    def run(self):
        self.logger.info(">>> Phase 1: Expansion")

        # Ensure topics_of_interest.txt is initialized with disease name if empty
        if not self.topics_file.exists() or self._get_current_topics_count() == 0:
            self.topics_file.parent.mkdir(parents=True, exist_ok=True)
            self.topics_file.write_text(self.disease_name + '\n', encoding='utf-8')
            self.logger.info(f"Init topics: '{self.disease_name}'")



        self.logger.info("Finding start year...")
        # We need a temporary DC instance just to get the start year
        temp_dc_for_year_finding = DataCollection(self.disease_name, target_year=datetime.datetime.now().year) # Year is irrelevant here
        start_year = temp_dc_for_year_finding.get_first_publication_year()

        if start_year is None:
            start_year = 1950 # Default fallback year
            self.logger.warning(f"Could not determine start year dynamically. Defaulting to {start_year}.")
        else:
            self.logger.info(f"Start year: {start_year}")


        self.logger.info(f"Params: start={start_year}, target={self.max_topics}")

        iteration = 1
        current_year = start_year
        while self._get_current_topics_count() < self.max_topics:
            self.logger.info(f"--- Iter {iteration} ({current_year}) ---")
            self.logger.info(f"Topics: {self._get_current_topics_count()}")

            # 1. Data Collection
            self.logger.info(f"Collecting {current_year}...")
            data_collection = DataCollection(
                disease_name=self.disease_name,
                target_year=current_year,
                expand_synonyms=True,
                filter_synonyms=True
            )
            data_collection.run()
            self.logger.info("Data collected.")

            # 2. Preprocessing
            self.logger.info("Preprocessing...")
            preprocessing = Preprocessing(
                target_year=current_year,
                disease_name=self.disease_name,
                incremental=True
            )
            if not preprocessing.run(force_full=False):
                self.logger.error("Preprocessing failed. Skipping.")
                iteration += 1
                current_year += 1
                continue
            self.logger.info("Preprocessing done.")

            # 3. Fixed Embedding Training (Cumulative)
            # 3. Fixed Embedding Training (Cumulative)
            self.logger.info("Training...")
            # [vector_size, window, min_count, sg, negative, alpha, epochs, workers, ns_exponent, sample]
            fixed_params = [200, 5, 2, 1, 15, 0.025, 15, 4, 0.75, 0.001]
            
            trainer = CandidateModelTraining(
                disease_name=self.disease_name,
                start_year=start_year,
                end_year=current_year
            )
            # Use 'w2v_fixed' as model name so it maps to 'w2v' architecture and saves in 'w2v_fixed' folder
            if not trainer.train_specific_model("w2v_fixed", fixed_params, current_year):
                self.logger.error("Embedding training failed. Skipping.")
                iteration += 1
                current_year += 1
                continue
            self.logger.info("Embedding training complete.")

            # 4. Dot Product Generation (Cumulative)
            self.logger.info("Running Dot Product Generation module...")
            validator = ValidationModule(
                disease_name=self.disease_name,
                model_subfolder='w2v_fixed', 
                start_year=start_year,
                end_year=current_year,
                use_chembl=True
            )
            validator.run()
            self.logger.info("Dot Product Generation complete.")

            # 5. Latent Knowledge Report and Feedback
            self.logger.info("Running Latent Knowledge Report module for feedback...")
            
            report_generator = LatentKnowledgeReportGenerator(
                disease_name=self.disease_name,
                model_subfolder='w2v_fixed',
                target_year=current_year,
                top_n_to_plot=20
            )
            report_generator.run(max_total_topics=self.max_topics, max_new_topics=self.max_new_topics)

            self.logger.info("Latent Knowledge Report and feedback complete.")
            iteration += 1
            current_year += 1
            time.sleep(1) # Small delay between iterations
        
        self.logger.info("Iterative topic expansion process finished.")
        final_expansion_year = current_year - 1
        self.logger.info(f"Iterative topic expansion process finished. Final expansion year: {final_expansion_year}.")
        return final_expansion_year

if __name__ == '__main__':
    expander = IterativeTopicExpansion(
        disease_name='diabetes',
        max_topics=9,
        max_new_topics=9
    )
    expander.run()
