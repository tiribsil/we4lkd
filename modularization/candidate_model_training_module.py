import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import shutil

from utils import LoggerFactory, normalize_disease_name
from embeddings_training_automl import ModelType, EmbeddingConfig, ModelFactory, GloVeModel as MittensGloVeModel

class CandidateModelTraining:
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
    ):
        self.logger = LoggerFactory.setup_logger(
            "candidate_model_training", target_year=str(start_year), log_to_file=False
        )
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        # This dictionary should be easily modifiable and extendable by the user
        # Hyperparameters order: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, (min_n, max_n for FastText)]
        self.model_combinations: Dict[str, List[Any]] = {
            "w2v_comb1": [200, 5, 2, 1, 15, 0.025, 15, 4], # Word2Vec
            "w2v_comb2": [200, 10, 3, 0, 5, 0.03, 20, 4],  # Word2Vec (CBOW)
            "ft_comb1": [100, 5, 2, 1, 10, 0.025, 15, 4, 3, 6],   # FastText
            "glove_comb1": [300, 8, 30, 0.05, 5, 100] # GloVe: [vector_size, window, max_iter, learning_rate, min_count, x_max]
        }

        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_base_path = Path(f'{self.base_path}/models')
        self.models_base_path.mkdir(parents=True, exist_ok=True)

        self._corpus_df = None
        self.logger.info(f"CandidateModelTraining initialized for {self.disease_name}, years {start_year}-{end_year}")

    @property
    def corpus_df(self) -> pd.DataFrame:
        """Lazy loading do corpus."""
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df

    def _load_corpus(self) -> Optional[pd.DataFrame]:
        """Carrega corpus de abstracts limpos, suportando diretórios de saída do Spark."""
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus path not found at {self.corpus_path}")
            self.logger.error("Have you run the preprocessing module?")
            return None
        
        try:
            self.logger.info(f"Loading corpus from {self.corpus_path}")
            
            if self.corpus_path.is_dir():
                # If it's a directory (Spark output), read all part-xxxx.csv files
                csv_files = list(self.corpus_path.glob('*.csv'))
                if not csv_files:
                    self.logger.error(f"No CSV files found in Spark output directory: {self.corpus_path}")
                    return None
                
                list_df = []
                for f in csv_files:
                    list_df.append(pd.read_csv(f))
                df = pd.concat(list_df, ignore_index=True)
            else:
                # If it's a single file (older output format or non-Spark)
                df = pd.read_csv(self.corpus_path)
            
            if 'summary' not in df.columns:
                self.logger.error("Column 'summary' not found in corpus")
                return None
            
            if 'year_extracted' not in df.columns and 'year' not in df.columns:
                self.logger.warning("No year column found. Using all data without year filtering.")
                df['year_extracted'] = self.end_year
            elif 'year' in df.columns:
                df['year_extracted'] = df['year']
            
            self.logger.info(f"Loaded {len(df)} abstracts")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            return None

    def _prepare_sentences(self, start_year: int, end_year: int) -> List[List[str]]:
        """
        Prepara sentenças para treinamento, filtrando por ano.
        """
        df = self.corpus_df
        
        if df is None or df.empty:
            self.logger.warning("No corpus data available.")
            return []
        
        # Filter by year_extracted within the specified range
        if 'year_extracted' in df.columns:
            df = df[(df['year_extracted'] >= start_year) & (df['year_extracted'] <= end_year)]
        
        abstracts = df['summary'].dropna().tolist()
        
        self.logger.info(f"Preparing {len(abstracts)} abstracts for training from {start_year} to {end_year}")
        
        sentences = [abstract.split() for abstract in abstracts if abstract]
        sentences = [s for s in sentences if len(s) > 0]
        
        return sentences
    
    def _train_single_candidate_model(
        self,
        sentences: List[List[str]],
        model_key: str,
        architecture: str,
        hyperparameters: List[Any],
    ) -> Optional[Any]:
        """
        Trains a single embedding model with the specified architecture and hyperparameters.
        """
        self.logger.info(f"Training model {model_key} ({architecture}) with hyperparameters: {hyperparameters}")

        model_type = None
        if architecture == 'w2v':
            model_type = ModelType.WORD2VEC
            # Expected hyperparameters: [vector_size, window, min_count, sg, negative, alpha, epochs, workers]
            custom_params = {
                'vector_size': hyperparameters[0],
                'window': hyperparameters[1],
                'min_count': hyperparameters[2],
                'sg': hyperparameters[3],
                'negative': hyperparameters[4],
                'alpha': hyperparameters[5],
                'epochs': hyperparameters[6],
                'workers': hyperparameters[7]
            }
        elif architecture == 'ft':
            model_type = ModelType.FASTTEXT
            # Expected hyperparameters: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, min_n, max_n]
            custom_params = {
                'vector_size': hyperparameters[0],
                'window': hyperparameters[1],
                'min_count': hyperparameters[2],
                'sg': hyperparameters[3],
                'negative': hyperparameters[4],
                'alpha': hyperparameters[5],
                'epochs': hyperparameters[6],
                'workers': hyperparameters[7],
                'min_n': hyperparameters[8],
                'max_n': hyperparameters[9]
            }
        elif architecture == 'glove':
            model_type = ModelType.GLOVE
            # Expected hyperparameters for GloVe: [vector_size, window, max_iter, learning_rate, min_count, x_max]
            custom_params = {
                'vector_size': hyperparameters[0],
                'window': hyperparameters[1],
                'max_iter': hyperparameters[2],
                'learning_rate': hyperparameters[3],
                'min_count': hyperparameters[4],
                'x_max': hyperparameters[5],
            }
        else:
            self.logger.error(f"Unsupported architecture: {architecture}")
            return None

        config = EmbeddingConfig(
            model_type=model_type,
            use_pca=False, # Not using PCA for candidate training by default
            vector_size=hyperparameters[0], # vector_size is typically the first hyperparameter
            custom_params=custom_params,
            end_year=self.end_year # Pass end_year for logging in BaseEmbeddingModel
        )

        model_instance = ModelFactory.create_model(config)
        model_instance.train(sentences)

        return model_instance
    
    def _save_trained_model(
        self,
        model_instance: Any, # BaseEmbeddingModel instance
        model_key: str,
        start_year: int,
        end_year: int,
    ):
        """
        Saves the trained model to the specified path.
        """
        model_output_dir = self.models_base_path / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_key}_{start_year}_{end_year}.model"
        model_path = model_output_dir / model_filename

        if hasattr(model_instance.model, 'save'): # For Word2Vec and FastText
            model_instance.model.save(str(model_path))
        elif isinstance(model_instance, MittensGloVeModel): # For GloVe trained with Mittens
            if model_instance.word_vectors_keyed_vectors:
                model_instance.word_vectors_keyed_vectors.save_word2vec_format(str(model_path), binary=True)
            else:
                self.logger.warning(f"Mittens GloVe model {model_key} has no word vectors to save.")
        elif isinstance(model_instance.model, KeyedVectors): # Fallback for other Gensim KeyedVectors
            model_instance.model.save_word2vec_format(str(model_path), binary=True)
        else:
            # Generic fallback for any other model type
            import pickle
            self.logger.warning(f"Model {model_key} does not have a .save() or is not a known KeyedVectors type. Attempting to pickle the model.")
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance.model, f)
        
        self.logger.info(f"Model {model_key} saved to {model_path}")

    def run(self) -> bool:
        """
        Executes the candidate model training pipeline.
        """
        self.logger.info("=== Starting Candidate Model Training Pipeline ===")
        
        sentences = self._prepare_sentences(self.start_year, self.end_year)
        if not sentences:
            self.logger.error("No sentences available for training. Aborting.")
            return False
        
        for model_key, hyperparameters in self.model_combinations.items():
            try:
                architecture_prefix = model_key.split('_')[0]
                
                trained_model_instance = self._train_single_candidate_model(
                    sentences=sentences,
                    model_key=model_key,
                    architecture=architecture_prefix,
                    hyperparameters=hyperparameters
                )
                
                if trained_model_instance:
                    self._save_trained_model(
                        model_instance=trained_model_instance,
                        model_key=model_key,
                        start_year=self.start_year,
                        end_year=self.end_year
                    )
                else:
                    self.logger.error(f"Failed to train model for {model_key}. Skipping saving.")

            except Exception as e:
                self.logger.exception(f"Error training or saving model {model_key}: {e}")
                continue # Continue to the next model combination even if one fails
        
        self.logger.info("=== Candidate Model Training Pipeline Completed ===")
        return True

if __name__ == '__main__':
    # Example usage:
    # This dictionary should be easily modifiable and extendable by the user
    # Hyperparameters order: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, (min_n, max_n for FastText), (max_iter, learning_rate, min_count, x_max for GloVe)]
    model_configs = {
        "w2v_comb1": [100, 5, 2, 1, 10, 0.025, 15, 4], # Word2Vec
        "w2v_comb2": [200, 10, 3, 0, 5, 0.03, 20, 4],  # Word2Vec (CBOW)
        "ft_comb1": [100, 5, 2, 1, 10, 0.025, 15, 4, 3, 6],   # FastText
        "glove_comb1": [300, 8, 30, 0.05, 5, 100] # GloVe: [vector_size, window, max_iter, learning_rate, min_count, x_max]
    }

    trainer = CandidateModelTraining(
        disease_name="acute myeloid leukemia",
        start_year=2020,
        end_year=2023
    )

    success = trainer.run()
    exit(0 if success else 1)
