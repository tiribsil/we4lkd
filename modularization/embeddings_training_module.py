import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Literal
import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from utils import *


class EmbeddingCallback(CallbackAny2Vec):
    """Callback para monitorar treinamento."""
    
    def __init__(self, logger, total_epochs: int):
        self.logger = logger
        self.total_epochs = total_epochs
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch % 5 == 0 or self.epoch == self.total_epochs:
            self.logger.info(f"Epoch {self.epoch}/{self.total_epochs} completed")


class EmbeddingTraining:
    """
    Module for training Word2Vec and FastText embedding models.
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
        parameters: Optional[List[Dict[str, float]]] = None,
    ):
        """
        Initialize embedding training module.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year for training data
            end_year: Ending year for training data
            model_type: Type of model ('w2v' for Word2Vec, 'ft' for FastText)
            parameters: List of parameter combinations for the model
            min_count: Minimum word frequency threshold
            sg: Training algorithm (1=skip-gram, 0=CBOW)
            hs: Use hierarchical softmax (1=yes, 0=negative sampling)
            epochs: Number of training epochs
            min_corpus_size: Minimum number of abstracts required
            workers: Number of worker threads
        """
        self.logger = setup_logger("embedding_training", log_to_file=False)
        
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = 2024
        self.end_year = 2025
        
        # Validar model_type
        self.model_type = 'w2v' #['w2v', 'ft']
        
        # Configurar parâmetros
        if parameters is None:
            self.parameters = (
                self.DEFAULT_W2V_PARAMS if self.model_type == 'w2v' 
                else self.DEFAULT_FT_PARAMS
            )
        else:
            self.parameters = parameters
        
        # Parâmetros de treinamento
        self.min_count = 2
        self.sg = 1
        self.hs = 0
        self.epochs = 15
        self.min_corpus_size = 10
        self.workers = 4
        
        # Configurar caminhos
        self.base_path = Path('.') / 'data' / self.disease_name
        self.corpus_path = self.base_path / 'corpus' / 'clean_abstracts' / 'clean_abstracts.csv'
        
        # Caminhos para modelos
        self.models_path = self.base_path / 'models'
        self.w2v_path = self.models_path / 'w2v_combination15'
        self.ft_path = self.models_path / 'ft_combination16'
        
        # Criar diretórios
        self.w2v_path.mkdir(parents=True, exist_ok=True)
        self.ft_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._corpus_df = None
        
        self.logger.info(f"EmbeddingTraining initialized for {self.disease_name}")
        self.logger.info(f"Model type: {self.model_type.upper()}, Years: {start_year}-{end_year}")

    @property
    def corpus_df(self) -> pd.DataFrame:
        """Lazy loading do corpus."""
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df

    def _load_corpus(self) -> Optional[pd.DataFrame]:
        """Carrega corpus de abstracts limpos."""
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus file not found at {self.corpus_path}")
            self.logger.error("Have you run the preprocessing module?")
            return None
        
        try:
            self.logger.info(f"Loading corpus from {self.corpus_path}")
            
            # Ler CSV
            df = pd.read_csv(self.corpus_path)
            
            # Verificar colunas necessárias
            if 'summary' not in df.columns:
                self.logger.error("Column 'summary' not found in corpus")
                return None
            
            # Adicionar coluna de ano se não existir (extrair do índice se necessário)
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

    def _prepare_sentences(self, year_filter: Optional[int] = None) -> List[List[str]]:
        """
        Prepara sentenças para treinamento.
        
        Args:
            year_filter: Se fornecido, filtra abstracts até este ano
            
        Returns:
            Lista de sentenças tokenizadas
        """
        df = self.corpus_df
        
        if df is None or df.empty:
            return []
        
        # Filtrar por ano se especificado
        if year_filter and 'year_extracted' in df.columns:
            df = df[df['year_extracted'] <= year_filter]
        
        # Remover NaN e converter para lista
        abstracts = df['summary'].dropna().tolist()
        
        self.logger.info(f"Preparing {len(abstracts)} abstracts for training")
        
        # Tokenizar (split por espaços)
        sentences = [abstract.split() for abstract in abstracts if abstract]
        
        # Filtrar sentenças vazias
        sentences = [s for s in sentences if len(s) > 0]
        
        return sentences

    def _get_model_path(self, param_idx: int = 0) -> Path:
        """Retorna o caminho para salvar o modelo."""
        if self.model_type == 'w2v':
            base_path = self.w2v_path
        else:
            base_path = self.ft_path
        
        model_name = f"model_{self.start_year}_{self.end_year}"
        
        if len(self.parameters) > 1:
            model_name += f"_params{param_idx}"
        
        return base_path / f"{model_name}.model"

    def _train_single_model(
        self,
        sentences: List[List[str]],
        params: Dict[str, float],
        param_idx: int = 0
    ) -> Optional[object]:
        """
        Treina um único modelo com os parâmetros especificados.
        
        Args:
            sentences: Sentenças tokenizadas
            params: Dicionário com parâmetros do modelo
            param_idx: Índice da combinação de parâmetros
            
        Returns:
            Modelo treinado ou None se falhar
        """
        try:
            # Callback para monitoramento
            callback = EmbeddingCallback(self.logger, self.epochs)
            
            # Parâmetros comuns
            common_params = {
                'sentences': sentences,
                'vector_size': params['vector_size'],
                'alpha': params['alpha'],
                'negative': params['negative'],
                'min_count': self.min_count,
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
                model = FastText(**common_params)
            
            # Salvar modelo
            model_path = self._get_model_path(param_idx)
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
        # Preparar sentenças
        sentences = self._prepare_sentences(year_filter or self.end_year)
        
        if not sentences:
            self.logger.error("No sentences available for training")
            return {}
        
        # Verificar tamanho mínimo do corpus
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
            self.logger.info(f"\n=== Training model {idx + 1}/{len(self.parameters)} ===")
            
            model = self._train_single_model(sentences, params, idx)
            if model:
                models[idx] = model
        
        self.logger.info(f"\nTraining complete. {len(models)} models trained successfully.")
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
        """
        Carrega informações sobre um modelo salvo.
        
        Args:
            model_path: Caminho do modelo
            
        Returns:
            Dicionário com informações do modelo
        """
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

    def run(self, year_over_year: bool = True, step: int = 1) -> bool:
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
    trainer = EmbeddingTraining(disease_name="acute myeloid leukemia",
                                start_year=2024,
                                end_year=2025)
    
    success = trainer.run(year_over_year=True)
    
    exit(0 if success else 1)