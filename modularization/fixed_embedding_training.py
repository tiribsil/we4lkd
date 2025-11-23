import os
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from utils import LoggerFactory, normalize_disease_name

class FixedEmbeddingTraining:
    def __init__(self, disease_name: str, start_year: int, end_year: int, model_type: str = 'w2v'):
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        self.model_type = 'w2v_fixed'
        self.logger = LoggerFactory.setup_logger("FixedEmbeddingTraining", f"{start_year}-{end_year}", log_to_file=True, log_file=f'logs/{end_year}.log')

        self.base_path = Path('./data') / self.normalized_disease_name
        self.corpus_path = self.base_path / 'corpus' / 'clean_abstracts'
        self.models_path = self.base_path / 'models' / self.model_type
        self.models_path.mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """
        Trains a fixed Word2Vec model based on the provided year range.
        Returns:
            bool: True if training was successful, False otherwise.
        """
        self.logger.info(f"Starting fixed Word2Vec training for disease '{self.disease_name}' for years {self.start_year}-{self.end_year}.")

        vector_size = 200
        alpha = 0.025
        negative = 15
        min_count = 2
        sg = 1
        hs = 0
        epochs = 15
        min_corpus_size = 10

        clean_abstracts_file = self.corpus_path / 'clean_abstracts.csv'
        if not clean_abstracts_file.exists():
            self.logger.error(f"Clean abstracts file not found at: {clean_abstracts_file}")
            return False

        self.logger.info("Reading clean abstracts...")
        try:
            df = pd.read_csv(clean_abstracts_file, header=0)
        except Exception as e:
            self.logger.error(f"Failed to read clean abstracts file: {e}")
            return False

        self.logger.info(f"Filtering abstracts from {self.start_year} to {self.end_year}.")
        abstracts_in_range = df[df['year_extracted'] <= self.end_year]['summary'].dropna().to_list()
        self.logger.info(f'{len(abstracts_in_range)} abstracts found in range.')

        if len(abstracts_in_range) < min_corpus_size:
            self.logger.warning(f"Corpus for year {self.end_year} is too small ({len(abstracts_in_range)} abstracts). Skipping training.")
            return False

        self.logger.info("Preparing data for training...")
        sentences = [abstract.split() for abstract in abstracts_in_range]

        self.logger.info("Training Word2Vec model...")
        try:
            model = Word2Vec(
                sentences=sentences,
                vector_size=vector_size,
                alpha=alpha,
                negative=negative,
                min_count=min_count,
                sg=sg,
                hs=hs,
                epochs=epochs,
                sorted_vocab=True
            )
            model_filename = self.models_path / f'{self.model_type}_{self.start_year}_{self.end_year}.model'
            model.save(str(model_filename))
            self.logger.info(f"Model saved successfully to {model_filename}")
        except Exception as e:
            self.logger.error(f"An error occurred during model training or saving: {e}")
            return False

        self.logger.info("Fixed Word2Vec training complete.")
        return True

if __name__ == '__main__':
    # Example usage:
    trainer = FixedEmbeddingTraining(
        disease_name='acute myeloid leukemia',
        start_year=1970,
        end_year=1980
    )
    trainer.run()
