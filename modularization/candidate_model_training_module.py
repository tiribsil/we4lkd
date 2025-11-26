import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import shutil

from utils import LoggerFactory, normalize_disease_name
# Mantemos o import do EmbeddingConfig e ModelFactory pois são utilitários de construção de classes,
# não o processo de AutoML em si.
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
        
        # Dicionário padrão para a fase de geração de candidatos
        self.model_combinations: Dict[str, List[Any]] = {
            "w2v_comb1": [200, 5, 2, 1, 15, 0.025, 15, 4], 
            "w2v_comb2": [200, 10, 3, 0, 5, 0.03, 20, 4],  
            "ft_comb1": [100, 5, 2, 1, 10, 0.025, 15, 4, 3, 6],   
            "glove_comb1": [300, 8, 30, 0.05, 5, 100] 
        }

        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_base_path = Path(f'{self.base_path}/models')
        self.models_base_path.mkdir(parents=True, exist_ok=True)

        self._corpus_df = None

    @property
    def corpus_df(self) -> pd.DataFrame:
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df

    def _load_corpus(self) -> Optional[pd.DataFrame]:
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus path not found at {self.corpus_path}")
            return None
        
        try:
            if self.corpus_path.is_dir():
                csv_files = list(self.corpus_path.glob('*.csv'))
                if not csv_files: return None
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                df = pd.read_csv(self.corpus_path)
            
            if 'summary' not in df.columns:
                self.logger.error("Column 'summary' not found in corpus")
                return None
            
            if 'year_extracted' not in df.columns:
                if 'year' in df.columns:
                    df['year_extracted'] = df['year']
                else:
                    df['year_extracted'] = self.end_year # Fallback
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            return None

    def _prepare_sentences(self, start_year: int, target_end_year: int) -> List[List[str]]:
        """
        Prepara sentenças filtrando até o ano alvo especificado.
        """
        df = self.corpus_df
        if df is None or df.empty: return []
        
        # Filtro temporal
        df_filtered = df[(df['year_extracted'] >= start_year) & (df['year_extracted'] <= target_end_year)]
        
        abstracts = df_filtered['summary'].dropna().tolist()
        self.logger.info(f"Prepared {len(abstracts)} abstracts ({start_year}-{target_end_year})")
        
        sentences = [str(abstract).split() for abstract in abstracts if abstract]
        return [s for s in sentences if len(s) > 0]
    
    def _create_config_from_params(self, architecture: str, params: List[Any], end_year: int) -> Optional[EmbeddingConfig]:
        """Helper para criar config baseado na lista de parâmetros."""
        if architecture == 'w2v':
            model_type = ModelType.WORD2VEC
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'min_count': params[2],
                'sg': params[3], 'negative': params[4], 'alpha': params[5],
                'epochs': params[6], 'workers': params[7]
            }
        elif architecture == 'ft':
            model_type = ModelType.FASTTEXT
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'min_count': params[2],
                'sg': params[3], 'negative': params[4], 'alpha': params[5],
                'epochs': params[6], 'workers': params[7], 'min_n': params[8], 'max_n': params[9]
            }
        elif architecture == 'glove':
            model_type = ModelType.GLOVE
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'max_iter': params[2],
                'learning_rate': params[3], 'min_count': params[4], 'x_max': params[5],
            }
        else:
            return None

        return EmbeddingConfig(
            model_type=model_type,
            use_pca=False,
            vector_size=params[0],
            custom_params=custom_params,
            end_year=end_year
        )

    def train_specific_model(self, model_name: str, params: List[Any], target_year: int) -> bool:
        """
        Método público para treinar um modelo específico até um ano alvo.
        Usado pelo ModelSelector.
        """
        # Determinar arquitetura pelo nome (ex: w2v_comb1 -> w2v)
        architecture = model_name.split('_')[0]
        
        # Verificar se modelo já existe para economizar tempo
        model_output_dir = self.models_base_path / model_name
        model_filename = f"{model_name}_{self.start_year}_{target_year}.model"
        model_path = model_output_dir / model_filename
        
        if model_path.exists():
            self.logger.info(f"Model {model_filename} already exists. Skipping training.")
            return True

        # Preparar dados
        sentences = self._prepare_sentences(self.start_year, target_year)
        if not sentences:
            self.logger.error(f"No data found for {self.start_year}-{target_year}")
            return False

        # Configurar e Treinar
        config = self._create_config_from_params(architecture, params, target_year)
        if not config:
            self.logger.error(f"Unknown architecture for {model_name}")
            return False

        try:
            self.logger.info(f"Training {model_name} up to {target_year}...")
            model_instance = ModelFactory.create_model(config)
            model_instance.train(sentences)
            
            # Salvar
            self._save_trained_model(model_instance, model_name, self.start_year, target_year)
            return True
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {e}")
            return False

    def _save_trained_model(self, model_instance, model_key: str, start_year: int, end_year: int):
        model_output_dir = self.models_base_path / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_key}_{start_year}_{end_year}.model"
        model_path = model_output_dir / model_filename

        if hasattr(model_instance.model, 'save'):
            model_instance.model.save(str(model_path))
        elif isinstance(model_instance, MittensGloVeModel):
             if model_instance.word_vectors_keyed_vectors:
                model_instance.word_vectors_keyed_vectors.save_word2vec_format(str(model_path), binary=True)
        elif isinstance(model_instance.model, KeyedVectors):
            model_instance.model.save_word2vec_format(str(model_path), binary=True)
        else:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance.model, f)
        
        self.logger.info(f"Saved: {model_path}")

    def run(self) -> Dict[str, List[Any]]:
        """
        Executa o treinamento de TODOS os candidatos definidos em self.model_combinations
        (Fase 4 do pipeline principal).
        """
        self.logger.info("=== Starting Candidate Training (Bulk) ===")
        trained_models = {}
        
        sentences = self._prepare_sentences(self.start_year, self.end_year)
        if not sentences: return {}
        
        for model_key, params in self.model_combinations.items():
            arch = model_key.split('_')[0]
            config = self._create_config_from_params(arch, params, self.end_year)
            
            if config:
                try:
                    self.logger.info(f"Bulk training {model_key}...")
                    model = ModelFactory.create_model(config)
                    model.train(sentences)
                    self._save_trained_model(model, model_key, self.start_year, self.end_year)
                    trained_models[model_key] = params
                except Exception as e:
                    self.logger.error(f"Error training {model_key}: {e}")
        
        return trained_models

if __name__ == '__main__':
    # Teste
    t = CandidateModelTraining("acute myeloid leukemia", 1990, 2000)
    # Exemplo de treino específico
    t.train_specific_model("w2v_test", [100, 5, 2, 1, 5, 0.025, 5, 4], 1995)
