import os
from pathlib import Path
import time, datetime

from data_collection_module import DataCollection
from preprocessing_module import Preprocessing
from fixed_embedding_training import FixedEmbeddingTraining
from dotproduct_generation_module import ValidationModule
from latent_knowledge_report_module import LatentKnowledgeReportGenerator
from utils import LoggerFactory, normalize_disease_name


class IterativeTopicExpansion:
    def __init__(self, disease_name: str, max_topics: int, max_new_topics: int):
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.max_topics = max_topics
        self.max_new_topics = max_new_topics
        self.logger = LoggerFactory.setup_logger("IterativeTopicExpansion", log_to_file=True, log_file='iterative_topic_expansion.log')

        self.base_path = Path('./data') / self.normalized_disease_name
        self.topics_file = self.base_path / 'topics_of_interest.txt'


    def _get_current_topics_count(self) -> int:
        if not self.topics_file.exists():
            return 0
        with open(self.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
        return len(topics)

    def run(self):
        self.logger.info("Starting iterative topic expansion process.")

        # Ensure topics_of_interest.txt is initialized with disease name if empty
        if not self.topics_file.exists() or self._get_current_topics_count() == 0:
            self.topics_file.parent.mkdir(parents=True, exist_ok=True)
            self.topics_file.write_text(self.disease_name + '\n', encoding='utf-8')
            self.logger.info(f"Initialized topics_of_interest.txt with '{self.disease_name}'.")



        self.logger.info("Determining start year based on topics...")
        # We need a temporary DC instance just to get the start year
        temp_dc_for_year_finding = DataCollection(self.disease_name, target_year=datetime.datetime.now().year) # Year is irrelevant here
        start_year = temp_dc_for_year_finding.get_first_publication_year()

        if start_year is None:
            start_year = 1950 # Default fallback year
            self.logger.warning(f"Could not determine start year dynamically. Defaulting to {start_year}.")
        else:
            self.logger.info(f"Dynamically determined start year: {start_year}")


        self.logger.info("Starting iterative topic expansion process.")
        self.logger.info(f"Start year: {start_year}")
        self.logger.info(f"Max topics to reach: {self.max_topics}")
        self.logger.info(f"Max new topics per iteration: {self.max_new_topics}")

        iteration = 1
        current_year = start_year
        while self._get_current_topics_count() < self.max_topics:
            self.logger.info(f"{'='*20} Iteration: {iteration} | Year: {current_year} {'='*20}")
            current_topics = self._get_current_topics_count()
            self.logger.info(f"Current number of topics: {current_topics}")

            # 1. Data Collection
            self.logger.info(f"Running Data Collection module for year {current_year}...")
            data_collection = DataCollection(
                disease_name=self.disease_name,
                target_year=current_year,
                expand_synonyms=True,
                filter_synonyms=True
            )
            data_collection.run()
            self.logger.info("Data Collection complete.")

            # 2. Preprocessing
            self.logger.info("Running Preprocessing module...")
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
            self.logger.info("Preprocessing complete.")

            # 3. Fixed Embedding Training (Cumulative)
            self.logger.info("Running Fixed Embedding Training module...")
            embedding_trainer = FixedEmbeddingTraining(
                disease_name=self.disease_name,
                start_year=start_year,
                end_year=current_year
            )
            if not embedding_trainer.run():
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
