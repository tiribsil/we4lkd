import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Literal, Callable
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from utils import *

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)


class EmbeddingCallback(CallbackAny2Vec):
    """Callback para monitorar treinamento."""
    
    def __init__(self, logger, total_epochs: int, trial: Optional[Trial] = None):
        self.logger = logger
        self.total_epochs = total_epochs
        self.epoch = 0
        self.trial = trial

    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        
        if self.epoch % 5 == 0 or self.epoch == self.total_epochs:
            self.logger.info(f"Epoch {self.epoch}/{self.total_epochs} - Loss: {loss:.4f}")
        
        # Report intermediate value para pruning
        if self.trial is not None:
            self.trial.report(loss, self.epoch)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                raise optuna.TrialPruned()


class EmbeddingTraining:
    """
    Module for training Word2Vec and FastText embedding models with Optuna optimization.
    Supports year-over-year training on cleaned abstracts.
    """
    
    # Parâmetros padrão otimizados
    DEFAULT_W2V_PARAMS = [
        {'vector_size': 100, 'alpha': 0.0025, 'negative': 10},
        {'vector_size': 200, 'alpha': 0.025, 'negative': 15}
    ]
    
    DEFAULT_FT_PARAMS = [
        {'vector_size': 300, 'alpha': 0.0025, 'negative': 5}
    ]
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        model_type: Literal['w2v', 'ft'] = 'w2v',
        parameters: Optional[List[Dict[str, float]]] = None,
        use_optuna: bool = False,
        optuna_trials: int = 50,
        optuna_timeout: Optional[int] = None,
    ):
        """
        Initialize embedding training module.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year for training data
            end_year: Ending year for training data
            model_type: Type of model ('w2v' for Word2Vec, 'ft' for FastText)
            parameters: List of parameter combinations for the model
            use_optuna: Whether to use Optuna for hyperparameter optimization
            optuna_trials: Number of Optuna trials
            optuna_timeout: Timeout in seconds for Optuna optimization
        """
        self.logger = LoggerFactory.setup_logger("embedding_training", target_year=str(start_year), log_to_file=False)
        
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        
        # Validar model_type
        if model_type not in ['w2v', 'ft']:
            raise ValueError(f"model_type must be 'w2v' or 'ft', got '{model_type}'")
        self.model_type = model_type
        
        # Optuna settings
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        
        # Configurar parâmetros
        if parameters is None and not use_optuna:
            self.parameters = (
                self.DEFAULT_W2V_PARAMS if self.model_type == 'w2v' 
                else self.DEFAULT_FT_PARAMS
            )
        elif parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = []  # Será definido pelo Optuna
        
        # Parâmetros de treinamento
        self.min_count = 2
        self.sg = 1
        self.hs = 0
        self.epochs = 15
        self.min_corpus_size = 5
        self.workers = 4
        
        # Paths
        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_path = Path(f'{self.base_path}/models')
        self.w2v_path = Path(f'{self.models_path}/w2v_models')
        self.ft_path = Path(f'{self.models_path}/ft_models')
        self.optuna_path = Path(f'{self.models_path}/optuna_studies')
        
        # Criar diretórios
        self.w2v_path.mkdir(parents=True, exist_ok=True)
        self.ft_path.mkdir(parents=True, exist_ok=True)
        self.optuna_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._corpus_df = None
        self._train_sentences = None
        self._val_sentences = None
        
        self.logger.info(f"EmbeddingTraining initialized for {self.disease_name}")
        self.logger.info(f"Model type: {self.model_type.upper()}, Years: {start_year}-{end_year}")
        if use_optuna:
            self.logger.info(f"Optuna optimization enabled: {optuna_trials} trials")

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

    def _prepare_sentences(self, year_filter: Optional[int] = None, 
                          validation_split: float = 0.0) -> Tuple[List[List[str]], Optional[List[List[str]]]]:
        """
        Prepara sentenças para treinamento e validação.
        
        Args:
            year_filter: Se fornecido, filtra abstracts até este ano
            validation_split: Proporção para validação (0.0 a 1.0)
            
        Returns:
            Tupla (train_sentences, val_sentences)
        """
        df = self.corpus_df
        
        if df is None or df.empty:
            return [], None
        
        if year_filter and 'year_extracted' in df.columns:
            df = df[df['year_extracted'] <= year_filter]
        
        abstracts = df['summary'].dropna().tolist()
        
        self.logger.info(f"Preparing {len(abstracts)} abstracts for training")
        
        sentences = [abstract.split() for abstract in abstracts if abstract]
        sentences = [s for s in sentences if len(s) > 0]
        
        # Split para validação se necessário
        if validation_split > 0 and len(sentences) > 1:
            np.random.seed(42)
            np.random.shuffle(sentences)
            
            split_idx = int(len(sentences) * (1 - validation_split))
            train_sentences = sentences[:split_idx]
            val_sentences = sentences[split_idx:]
            
            self.logger.info(f"Split: {len(train_sentences)} train, {len(val_sentences)} validation")
            return train_sentences, val_sentences
        
        return sentences, None

    def _evaluate_model(self, model, val_sentences: List[List[str]]) -> float:
        """
        Avalia modelo usando perplexidade aproximada no conjunto de validação.
        
        Args:
            model: Modelo treinado
            val_sentences: Sentenças de validação
            
        Returns:
            Score de avaliação (menor é melhor)
        """
        if not val_sentences:
            return float('inf')
        
        total_score = 0
        valid_words = 0
        
        for sentence in val_sentences:
            for i, word in enumerate(sentence):
                if word in model.wv:
                    # Context words
                    context = [w for j, w in enumerate(sentence) 
                             if j != i and w in model.wv]
                    
                    if context:
                        try:
                            # Similaridade média com contexto
                            similarities = [model.wv.similarity(word, ctx) 
                                          for ctx in context[:5]]  # Limitar contexto
                            total_score += np.mean(similarities)
                            valid_words += 1
                        except:
                            continue
        
        if valid_words == 0:
            return float('inf')
        
        # Retornar score negativo (queremos maximizar similaridade)
        return -total_score / valid_words

    def _get_model_path(self, param_idx: int = 0) -> Path:
        """Retorna o caminho para salvar o modelo."""
        base_path = self.w2v_path if self.model_type == 'w2v' else self.ft_path
        
        model_name = f"model_{self.start_year}_{self.end_year}"
        
        """if len(self.parameters) > 1:
            model_name += f"_params{param_idx}"""
        
        return Path(f'{base_path}/{model_name}.model')

    def _objective(self, trial: Trial, sentences: List[List[str]], 
                   val_sentences: Optional[List[List[str]]]) -> float:
        """
        Função objetivo para Optuna.
        
        Args:
            trial: Trial do Optuna
            sentences: Sentenças de treinamento
            val_sentences: Sentenças de validação
            
        Returns:
            Score a minimizar
        """
        # Sugerir hiperparâmetros
        params = {
            'vector_size': trial.suggest_categorical('vector_size', [50, 100, 150, 200, 300]),
            'alpha': trial.suggest_float('alpha', 0.001, 0.05, log=True),
            'negative': trial.suggest_int('negative', 5, 20),
            'window': trial.suggest_int('window', 3, 10),
        }
        
        # Parâmetros específicos do FastText
        if self.model_type == 'ft':
            params['min_n'] = trial.suggest_int('min_n', 2, 4)
            params['max_n'] = trial.suggest_int('max_n', 4, 6)
        
        try:
            callback = EmbeddingCallback(self.logger, self.epochs, trial)
            
            common_params = {
                'sentences': sentences,
                'vector_size': params['vector_size'],
                'alpha': params['alpha'],
                'negative': params['negative'],
                'window': params['window'],
                'min_count': self.min_count,
                'sg': self.sg,
                'hs': self.hs,
                'epochs': self.epochs,
                'workers': self.workers,
                'sorted_vocab': True,
                'compute_loss': True,
                'callbacks': [callback]
            }
            
            if self.model_type == 'w2v':
                model = Word2Vec(**common_params)
            else:
                common_params['min_n'] = params['min_n']
                common_params['max_n'] = params['max_n']
                model = FastText(**common_params)
            
            # Avaliar no conjunto de validação
            if val_sentences:
                score = self._evaluate_model(model, val_sentences)
            else:
                # Usar loss de treinamento se não houver validação
                score = model.get_latest_training_loss()
            
            # Salvar melhores parâmetros
            trial.set_user_attr('params', params)
            
            return score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Error in trial: {e}")
            return float('inf')

    def optimize_hyperparameters(self, year_filter: Optional[int] = None,
                                validation_split: float = 0.2) -> Dict:
        """
        Otimiza hiperparâmetros usando Optuna.
        
        Args:
            year_filter: Se fornecido, usa apenas abstracts até este ano
            validation_split: Proporção para validação
            
        Returns:
            Dicionário com melhores parâmetros
        """
        self.logger.info("=== Starting Optuna Hyperparameter Optimization ===")
        
        # Preparar dados
        train_sentences, val_sentences = self._prepare_sentences(
            year_filter or self.end_year, 
            validation_split
        )
        
        if not train_sentences:
            self.logger.error("No sentences available for optimization")
            return {}
        
        if len(train_sentences) < self.min_corpus_size:
            self.logger.warning(f"Corpus too small: {len(train_sentences)} abstracts")
            return {}
        
        # Criar estudo Optuna
        study_name = f"{self.model_type}_{self.disease_name}_{self.start_year}_{self.end_year}"
        storage_path = Path(f'{self.optuna_path}/{study_name}.db')
        
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        # Otimizar
        self.logger.info(f"Running {self.optuna_trials} trials...")
        study.optimize(
            lambda trial: self._objective(trial, train_sentences, val_sentences),
            n_trials=self.optuna_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=True
        )
        
        # Resultados
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Optimization Complete!")
        self.logger.info(f"Best value: {best_value:.4f}")
        self.logger.info(f"Best parameters:")
        for param, value in best_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info(f"{'='*60}\n")
        
        # Salvar resumo
        summary_path = Path(f'{self.optuna_path}/{study_name}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Optuna Study Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Study name: {study_name}\n")
            f.write(f"Best value: {best_value:.4f}\n")
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"\nBest parameters:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write(f"\nTotal trials: {len(study.trials)}\n")
            f.write(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
        
        self.logger.info(f"Study summary saved to {summary_path}")
        
        # Atualizar parâmetros para usar os melhores
        self.parameters = [best_params]
        
        return best_params

    def _train_single_model(
        self,
        sentences: List[List[str]],
        params: Dict[str, float],
        param_idx: int = 0,
        suffix: str = ""
    ) -> Optional[object]:
        """
        Treina um único modelo com os parâmetros especificados.
        
        Args:
            sentences: Sentenças tokenizadas
            params: Dicionário com parâmetros do modelo
            param_idx: Índice da combinação de parâmetros
            suffix: Sufixo para nome do arquivo
            
        Returns:
            Modelo treinado ou None se falhar
        """
        try:
            callback = EmbeddingCallback(self.logger, self.epochs)
            
            # Parâmetros comuns
            common_params = {
                'sentences': sentences,
                'vector_size': params.get('vector_size', 100),
                'alpha': params.get('alpha', 0.025),
                'negative': params.get('negative', 10),
                'window': params.get('window', 5),
                'min_count': params.get('min_count', self.min_count),
                'sg': self.sg,
                'hs': self.hs,
                'epochs': self.epochs,
                'workers': self.workers,
                'sorted_vocab': True,
                'compute_loss': True,
                'callbacks': [callback]
            }
            
            self.logger.info(f"Training {self.model_type.upper()} model with params: {params}")
            
            # Treinar modelo
            if self.model_type == 'w2v':
                model = Word2Vec(**common_params)
            else:
                common_params['min_n'] = params.get('min_n', 3)
                common_params['max_n'] = params.get('max_n', 5)
                model = FastText(**common_params)
            
            # Salvar modelo
            model_path = self._get_model_path(param_idx)
            self.logger.info(f'MODEL PATH: {model_path}')
            model.save(str(model_path))
            
            # Informações do modelo
            vocab_size = len(model.wv)
            total_loss = model.get_latest_training_loss()
            
            self.logger.info(f"Model saved to {model_path}")
            self.logger.info(f"Vocabulary size: {vocab_size}")
            self.logger.info(f"Final training loss: {total_loss:.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None

    def train(self, year_filter: Optional[int] = None) -> Dict[int, object]:
        """
        Treina modelos de embedding.
        
        Args:
            year_filter: Se fornecido, usa apenas abstracts até este ano
            
        Returns:
            Dicionário com índice de parâmetros -> modelo treinado
        """
        # Otimizar hiperparâmetros se necessário
        if self.use_optuna and not self.parameters:
            self.optimize_hyperparameters(year_filter)
        
        # Preparar sentenças
        sentences, _ = self._prepare_sentences(year_filter or self.end_year)
        
        if not sentences:
            self.logger.error("No sentences available for training")
            return {}
        
        if len(sentences) < self.min_corpus_size:
            self.logger.warning(
                f"Corpus too small: {len(sentences)} abstracts "
                f"(minimum: {self.min_corpus_size}). Skipping."
            )
            return {}
        
        self.logger.info(f"Training on {len(sentences)} sentences")
        
        # Treinar modelos para cada combinação de parâmetros
        models = {}
        for idx, params in enumerate(self.parameters):
            self.logger.info(f"=== Training model {idx + 1}/{len(self.parameters)} ===")
            
            suffix = "optuna_best" if self.use_optuna and idx == 0 else ""
            model = self._train_single_model(sentences, params, idx, suffix)
            if model:
                models[idx] = model
        
        self.logger.info(f"Training complete. {len(models)} models trained successfully.")
        return models

    def train_year_over_year(self, step: int = 1) -> Dict[int, Dict[int, object]]:
        """
        Treina modelos incrementalmente, ano a ano.
        
        Args:
            step: Incremento de anos entre cada modelo
            
        Returns:
            Dicionário aninhado: ano -> {param_idx -> modelo}
        """
        all_models = {}
        
        for year in range(self.start_year, self.end_year + 1, step):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training models for year range {self.start_year}-{year}")
            self.logger.info(f"{'='*60}\n")
            
            # Atualizar ano final temporariamente
            original_end = self.end_year
            self.end_year = year
            
            # Treinar modelos
            year_models = self.train(year_filter=year)
            
            if year_models:
                all_models[year] = year_models
            
            # Restaurar ano final
            self.end_year = original_end
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Year-over-year training complete")
        self.logger.info(f"Trained models for {len(all_models)} year ranges")
        self.logger.info(f"{'='*60}\n")
        
        return all_models

    def get_model_info(self, model_path: Path) -> Optional[Dict[str, any]]:
        """Carrega informações sobre um modelo salvo."""
        try:
            if self.model_type == 'w2v':
                model = Word2Vec.load(str(model_path))
            else:
                model = FastText.load(str(model_path))
            
            return {
                'vocab_size': len(model.wv),
                'vector_size': model.wv.vector_size,
                'total_words': model.corpus_total_words,
                'epochs_trained': model.epochs
            }
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def list_trained_models(self) -> List[Path]:
        """Lista todos os modelos treinados."""
        model_dir = self.w2v_path if self.model_type == 'w2v' else self.ft_path
        models = sorted(model_dir.glob('*.model'))
        
        self.logger.info(f"Found {len(models)} trained models in {model_dir}")
        for model_path in models:
            self.logger.info(f"  - {model_path.name}")
        
        return models

    def run(self, year_over_year: bool = False, step: int = 1) -> bool:
        """
        Executa pipeline de treinamento.
        
        Args:
            year_over_year: Se True, treina incrementalmente ano a ano
            step: Incremento de anos (apenas para year_over_year)
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            self.logger.info("=== Starting Embedding Training Pipeline ===")
            self.logger.info(f"Disease: {self.disease_name}")
            self.logger.info(f"Model type: {self.model_type.upper()}")
            
            if self.use_optuna:
                self.logger.info(f"Optuna optimization: {self.optuna_trials} trials")
            else:
                self.logger.info(f"Parameters combinations: {len(self.parameters)}")
            
            if year_over_year:
                models = self.train_year_over_year(step=step)
                success = len(models) > 0
            else:
                models = self.train()
                success = len(models) > 0
            
            if success:
                self.logger.info("=== Training completed successfully ===")
                self.list_trained_models()
            else:
                self.logger.error("=== Training failed ===")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error in training pipeline: {e}")
            return False


if __name__ == '__main__':
    # Exemplo 1: Treinar com Optuna
    trainer_optuna = EmbeddingTraining(
        disease_name="acute myeloid leukemia",
        start_year=2024,
        end_year=2025,
        model_type='w2v',
        use_optuna=True,
        optuna_trials=2,
        optuna_timeout=3600
    )
    
    success = trainer_optuna.run(year_over_year=False)
    
    exit(0 if success else 1)
