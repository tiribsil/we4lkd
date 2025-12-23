import os
from gensim.models import KeyedVectors
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import wget
from zipfile import ZipFile
import numpy as np
from gensim.models import KeyedVectors, Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import optuna
from optuna.trial import Trial
import logging
import time
import json
import gc

from utils import LoggerFactory, normalize_disease_name
    

class ModelType(Enum):
    """Supported embedding model types."""
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_type: ModelType
    use_pca: bool = False
    pca_components: Optional[int] = None
    vector_size: int = 300
    custom_params: Dict[str, Any] = field(default_factory=dict)
    end_year: int = None


@dataclass
class EvaluationMetrics:
    """Metrics for unsupervised embedding evaluation."""
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    vocabulary_coverage: float
    oov_handling: float
    cosine_consistency: float
    euclidean_consistency: float
    dot_product_consistency: float
    neighborhood_preservation: float
    rank_correlation: float
    similarity_score: float
    intrinsic_score: float
    domain_vocabulary_coverage: float = 0.0
    embedding_variance: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


class EmbeddingEvaluator:
    """Evaluates embedding quality for similarity-based tasks."""
    
    def __init__(
        self,
        end_year: Optional[int] = None,
        n_clusters: int = 10, 
        k_neighbors: int = 10,
        test_compounds: Optional[List[str]] = None,
        domain_vocabulary: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.test_compounds = test_compounds or []
        self.domain_vocabulary = domain_vocabulary or []
        self.random_state = random_state
        self.end_year = end_year
        self.logger = LoggerFactory.setup_logger(
            name="EmbeddingEvaluator",
            target_year=str(self.end_year),
            log_to_file=True,
            log_file=f'logs/{self.end_year}.log'
        )
    
    def _calculate_similarity_consistency(
        self, embeddings: np.ndarray, n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """Calculate consistency of different similarity metrics."""
        n_samples = min(n_samples, len(embeddings))
        np.random.seed(self.random_state)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = sample_embeddings / norms
        
        cosine_sim = np.dot(normalized, normalized.T)
        dot_product = np.dot(normalized, normalized.T) 
        
        from scipy.spatial.distance import pdist, squareform
        euclidean_dist = squareform(pdist(normalized, metric='euclidean'))
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        upper_tri = np.triu_indices_from(cosine_sim, k=1)
        cosine_std = np.std(cosine_sim[upper_tri])
        dot_std = np.std(dot_product[upper_tri])
        euclidean_std = np.std(euclidean_sim[upper_tri])
        
        return (1 / (1 + cosine_std), 1 / (1 + euclidean_std), 1 / (1 + dot_std))
    
    def _calculate_neighborhood_preservation(self, embeddings: np.ndarray) -> float:
        """Calculate how well local neighborhoods are preserved."""
        if len(embeddings) < self.k_neighbors + 1:
            return 0.0
        
        from sklearn.neighbors import NearestNeighbors
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        
        nbrs_cosine = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='cosine').fit(normalized_embeddings)
        _, indices_cosine = nbrs_cosine.kneighbors(normalized_embeddings)
        
        nbrs_euclidean = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean').fit(normalized_embeddings)
        _, indices_euclidean = nbrs_euclidean.kneighbors(normalized_embeddings)
        
        overlaps = [
            len(set(indices_cosine[i][1:]) & set(indices_euclidean[i][1:])) / self.k_neighbors
            for i in range(len(embeddings))
        ]
        return np.mean(overlaps)
    
    def _calculate_rank_correlation(self, embeddings: np.ndarray, n_samples: int = 50) -> float:
        """Calculate rank correlation between different distance metrics."""
        n_samples = min(n_samples, len(embeddings))
        from scipy.stats import spearmanr
        
        np.random.seed(self.random_state)
        reference_idx = np.random.choice(len(embeddings))
        reference = embeddings[reference_idx]
        
        sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        reference_norm = reference / (np.linalg.norm(reference) + 1e-8)
        sample_norms = sample_embeddings / (np.linalg.norm(sample_embeddings, axis=1, keepdims=True) + 1e-8)
        
        cosine_dists = 1 - np.dot(sample_norms, reference_norm)
        euclidean_dists = np.linalg.norm(sample_norms - reference_norm, axis=1)
        dot_dists = -np.dot(sample_norms, reference_norm)
        
        corr_ce, _ = spearmanr(cosine_dists, euclidean_dists)
        corr_cd, _ = spearmanr(cosine_dists, dot_dists)
        corr_ed, _ = spearmanr(euclidean_dists, dot_dists)
        
        return np.mean([abs(corr_ce), abs(corr_cd), abs(corr_ed)])
    
    def _calculate_oov_handling(
        self, 
        model,
        model_type: ModelType, 
        test_words: List[str]
    ) -> float:
        """Testa empiricamente a capacidade de lidar com palavras fora do vocabulário."""
        if not test_words:
            # Palavras de teste biomédicas incluindo variações
            test_words = [
                'leukemia', 'leukaemia', 'leucemia',  # Variações ortográficas
                'cd34', 'cd38', 'flt3',  # Marcadores biomédicos
                'chemotherapy', 'chemotherapeutic',  # Derivações
                'xyzunknown123',  # Palavra inexistente
            ]
        
        found = 0
        for word in test_words:
            try:
                if model_type == ModelType.FASTTEXT:
                    # FastText usa subword
                    _ = model.wv[word]
                    found += 1
                elif model_type == ModelType.GLOVE:
                    # GloVe precisa ter a palavra exata
                    if word in model.key_to_index:
                        found += 1
                elif model_type == ModelType.WORD2VEC:
                    if word in model.wv:
                        found += 1
            except:
                continue
        
        return found / len(test_words)
    
    def _calculate_domain_vocabulary_coverage(
        self, vocabulary: Optional[List[str]], model_type: ModelType
    ) -> float:
        """NOVO: Calcula cobertura do vocabulário específico do domínio."""
        if not vocabulary or not self.domain_vocabulary:
            return 0.0
        
        vocab_set = set(v.lower() for v in vocabulary)
        domain_set = set(v.lower() for v in self.domain_vocabulary)
        
        coverage = len(vocab_set & domain_set) / len(domain_set)
        return coverage
    
    def _calculate_embedding_variance(self, embeddings: np.ndarray) -> float:
        """NOVO: Calcula variância dos embeddings (diversidade)."""
        # Normalizar primeiro
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        variance = np.var(normalized, axis=0).mean()
        return float(variance)
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        vocabulary: Optional[List[str]] = None,
        total_words: Optional[int] = None,
        model_type: Optional[ModelType] = None,
        model: Optional[Any] = None
    ) -> EvaluationMetrics:
        """Comprehensive evaluation for similarity-based tasks."""
        min_samples = max(self.n_clusters, self.k_neighbors + 1)
        if len(embeddings) < min_samples:
            self.logger.warning(f"Not enough samples ({len(embeddings)}) for evaluation")
            return EvaluationMetrics(0.0, 0.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        
        # Clustering metrics
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_embeddings)
        
        silhouette = silhouette_score(normalized_embeddings, cluster_labels)
        calinski = calinski_harabasz_score(normalized_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(normalized_embeddings, cluster_labels)
        
        # Vocabulary coverage
        vocab_coverage = 1.0
        if vocabulary and total_words:
            vocab_coverage = len(vocabulary) / total_words
        

        domain_vocab_coverage = self._calculate_domain_vocabulary_coverage(vocabulary, model_type)
        embedding_variance = self._calculate_embedding_variance(embeddings)

        oov_handling = self._calculate_oov_handling(model, model_type, self.test_compounds) if model and model_type else 0.5
        
        # Similarity metrics
        cosine_cons, euclidean_cons, dot_cons = self._calculate_similarity_consistency(normalized_embeddings)
        neighborhood_pres = self._calculate_neighborhood_preservation(normalized_embeddings)
        rank_corr = self._calculate_rank_correlation(normalized_embeddings)
        
        # Normalize clustering metrics
        silhouette_norm = (silhouette + 1) / 2
        calinski_norm = min(calinski / 1000, 1.0)
        davies_bouldin_norm = 1 / (1 + davies_bouldin)
        
        # Combined scores - AJUSTADO: Dar mais peso à cobertura do domínio
        similarity_score = (
            0.25 * cosine_cons + 0.25 * euclidean_cons + 0.20 * dot_cons +
            0.15 * neighborhood_pres + 0.15 * rank_corr
        )
        
        intrinsic_score = (
            0.12 * silhouette_norm + 0.08 * calinski_norm + 0.08 * davies_bouldin_norm +
            0.08 * vocab_coverage + 0.12 * oov_handling + 0.32 * similarity_score +
            0.15 * domain_vocab_coverage + 0.05 * embedding_variance  # NOVO
        )
        
        return EvaluationMetrics(
            silhouette, calinski, davies_bouldin, vocab_coverage, oov_handling,
            cosine_cons, euclidean_cons, dot_cons, neighborhood_pres, rank_corr,
            similarity_score, intrinsic_score, domain_vocab_coverage, embedding_variance
        )


class BaseEmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.pca = None
        self.end_year = self.config.end_year
        self.logger = LoggerFactory.setup_logger(
            name="BaseEmbeddingModel",
            target_year=str(self.end_year),
            log_to_file=True,
            log_file=f'logs/{self.end_year}.log'
        )
    
    def train(self, sentences: List[List[str]]) -> None:
        raise NotImplementedError
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        raise NotImplementedError
    
    def apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA reduction if configured."""
        if not self.config.use_pca or embeddings.shape[0] == 0:
            return embeddings
        
        n_components = self.config.pca_components or min(50, embeddings.shape[1] // 2)
        n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            reduced = self.pca.fit_transform(embeddings)
        else:
            reduced = self.pca.transform(embeddings)
        
        self.logger.info(f"PCA: {embeddings.shape[1]} → {reduced.shape[1]} dims")
        return reduced


class Word2VecModel(BaseEmbeddingModel):
    """Word2Vec embedding model."""
    
    def train(self, sentences: List[List[str]]) -> None:
        params = {
            'vector_size': self.config.vector_size,
            'window': self.config.custom_params.get('window', 5),
            'min_count': self.config.custom_params.get('min_count', 2),
            'sg': self.config.custom_params.get('sg', 1),
            'negative': self.config.custom_params.get('negative', 10),
            'alpha': self.config.custom_params.get('alpha', 0.025),
            'epochs': self.config.custom_params.get('epochs', 15),
            'workers': self.config.custom_params.get('workers', 1),
            'ns_exponent': self.config.custom_params.get('ns_exponent', 0.75),
            'sample': self.config.custom_params.get('sample', 0.001),
        }
        self.model = Word2Vec(sentences=sentences, **params)
        self.logger.info(f"Word2Vec trained: {len(self.model.wv)} words")
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        if sentences:
            embeddings = []
            for sentence in sentences:
                words = sentence.lower().split()
                vectors = [self.model.wv[w] for w in words if w in self.model.wv]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.config.vector_size))
            embeddings = np.array(embeddings)
        else:
            embeddings = self.model.wv.vectors
        
        return self.apply_pca(embeddings)


class FastTextModel(BaseEmbeddingModel):
    """FastText embedding model."""
    
    def train(self, sentences: List[List[str]]) -> None:
        params = {
            'vector_size': self.config.vector_size,
            'window': self.config.custom_params.get('window', 5),
            'min_count': self.config.custom_params.get('min_count', 2),
            'sg': self.config.custom_params.get('sg', 1),
            'negative': self.config.custom_params.get('negative', 10),
            'alpha': self.config.custom_params.get('alpha', 0.025),
            'epochs': self.config.custom_params.get('epochs', 15),
            'workers': self.config.custom_params.get('workers', 1),
            'ns_exponent': self.config.custom_params.get('ns_exponent', 0.75),
            'sample': self.config.custom_params.get('sample', 0.001),
            'min_n': self.config.custom_params.get('min_n', 3),
            'max_n': self.config.custom_params.get('max_n', 5),
        }
        self.model = FastText(sentences=sentences, **params)
        self.logger.info(f"FastText trained: {len(self.model.wv)} words")
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        if sentences:
            embeddings = []
            for sentence in sentences:
                words = sentence.lower().split()
                vectors = [self.model.wv[w] for w in words if w in self.model.wv]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.config.vector_size))
            embeddings = np.array(embeddings)
        else:
            embeddings = self.model.wv.vectors
        
        return self.apply_pca(embeddings)


import logging
import itertools
import gc # Adicionado para gerenciamento de memória

from mittens import Mittens, GloVe # Alterado para Mittens


class GloVeModel(BaseEmbeddingModel):
    """GloVe embedding model trained from scratch using Mittens."""

    def train(self, sentences: List[List[str]]) -> None:
        # Expected hyperparameters for GloVe training:
        # [vector_size, window, max_iter, learning_rate, min_count, x_max]
        params = {
            'vector_size': self.config.vector_size,
            'window': self.config.custom_params.get('window', 5),
            'min_count': self.config.custom_params.get('min_count', 2),
            'max_iter': self.config.custom_params.get('max_iter', 15),
            'alpha': self.config.custom_params.get('alpha', 0.05),
            'learning_rate': self.config.custom_params.get('learning_rate', 0.05),
            'x_max': self.config.custom_params.get('x_max', 100),
        }

        self.logger.info(f"Training GloVe model with Mittens using params: {params}")

        # 1. Create Corpus for GloVe and build co-occurrence matrix
        # Flatten list of lists for vocabulary extraction, then build co-occurrence
        all_words = list(itertools.chain.from_iterable(sentences))
        vocab = {word: i for i, word in enumerate(sorted(set(all_words)))} # Ensure consistent vocabulary order

        # Filter vocabulary by min_count
        word_counts = pd.Series(all_words).value_counts()
        filtered_vocab_list = word_counts[word_counts >= params['min_count']].index.tolist()
        filtered_vocab = {word: i for i, word in enumerate(sorted(filtered_vocab_list))}

        if not filtered_vocab:
            self.logger.warning("No words left after min_count filtering for GloVe. Skipping training.")
            self.model = None
            self.word_vectors_keyed_vectors = KeyedVectors(vector_size=params['vector_size'])
            return

        # Mittens expects raw sentences (list of lists of words)
        # It uses gensim's build_cooccurrence_matrix internally
        cooc_model = GloVe(params['window'])
        cooc_matrix = cooc_model.build_cooccurrence_matrix(sentences, vocabulary=filtered_vocab, min_count=params['min_count'])
        
        # 2. Initialize and train Mittens model
        mittens_model = Mittens(
            n=params['vector_size'],
            max_iter=params['max_iter'],
            eta=params['learning_rate'],
            alpha=params['alpha'], # Although GloVe itself doesn't use alpha directly in its objective, mittens includes it.
            max_count=params['x_max'],
            # Note: GloVe from mittens does not directly expose 'workers' for multi-threading like gensim
        )

        # Train the model
        mittens_model.fit(cooc_matrix)

        # 3. Store word vectors in Gensim KeyedVectors format for compatibility
        self.model = mittens_model
        self.vocabulary = list(filtered_vocab.keys())
        self.word_vectors_keyed_vectors = KeyedVectors(vector_size=params['vector_size'])

        # Add vectors to KeyedVectors object
        for i, word in enumerate(self.vocabulary):
            self.word_vectors_keyed_vectors.add_vector(word, mittens_model.get_embedding(word))
        
        self.logger.info(f"GloVe trained: {len(self.vocabulary)} words, {len(mittens_model.get_vectors())} embeddings")

        # Clean up large objects
        del cooc_matrix
        del all_words
        del vocab
        del word_counts
        del filtered_vocab
        gc.collect()

    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        """
        Get sentence embeddings by averaging word vectors or all word vectors if no sentences provided.
        Returns a numpy array of embeddings.
        """
        if self.model is None or self.word_vectors_keyed_vectors is None:
            return np.array([])

        if sentences:
            embeddings = []
            for sentence in sentences:
                words = sentence.lower().split()
                vectors = [self.word_vectors_keyed_vectors[w] 
                           for w in words if w in self.word_vectors_keyed_vectors.key_to_index]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.config.vector_size))
            embeddings = np.array(embeddings)
        else:
            # Return all word embeddings if no specific sentences are provided
            embeddings = self.word_vectors_keyed_vectors.vectors
        
        return self.apply_pca(embeddings)


class ModelFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create_model(config: EmbeddingConfig) -> BaseEmbeddingModel:
        model_map = {
            ModelType.WORD2VEC: Word2VecModel,
            ModelType.FASTTEXT: FastTextModel,
            ModelType.GLOVE: GloVeModel,
        }
        
        model_class = model_map.get(config.model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        return model_class(config)


def train_and_evaluate_single(
    config: EmbeddingConfig,
    sentences: List[List[str]],
    evaluator_params: Dict[str, Any],
    end_year: Optional[int]
) -> Tuple[EmbeddingConfig, EvaluationMetrics, float]:
    """Sequential training and evaluation (NO multiprocessing)."""
    logger = LoggerFactory.setup_logger(
        name="ModelTrainer",
        target_year=str(end_year),
        log_to_file=True,
        log_file=f'logs/{end_year}.log'
    )
    start_time = time.time()
    
    try:
        logger.info(f"Training {config.model_type.value}...")
        
        model = ModelFactory.create_model(config)
        model.train(sentences)
        embeddings = model.get_embeddings()
        
        vocabulary = None
        if hasattr(model.model, 'wv'):
            vocabulary = list(model.model.wv.key_to_index.keys())
        elif hasattr(model.model, 'key_to_index'):
            vocabulary = list(model.model.key_to_index.keys())
        elif hasattr(model.model, 'vocabulary'):
            vocabulary = list(model.model['vocabulary'].keys())
        
        total_words = len(set(word for sent in sentences for word in sent))
        
        evaluator = EmbeddingEvaluator(**evaluator_params, end_year=end_year)
        metrics = evaluator.evaluate(
            embeddings, vocabulary, total_words, config.model_type, model.model  # NOVO: passar modelo
        )
        
        training_time = time.time() - start_time
        logger.info(
            f"✓ {config.model_type.value}: Score={metrics.intrinsic_score:.4f}, "
            f"DomainCov={metrics.domain_vocabulary_coverage:.3f}, "
            f"OOV={metrics.oov_handling:.3f}, Time={training_time:.2f}s"
        )
        
        # Clean up
        del model
        del embeddings
        gc.collect()
        
        return config, metrics, training_time
        
    except Exception as e:
        logger.error(f"Error training {config.model_type.value}: {e}", exc_info=True)
        return (
            config, 
            EvaluationMetrics(0.0, 0.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            time.time() - start_time
        )


class SequentialModelSelector:
    """Sequential model selection (avoids multiprocessing issues)."""
    
    def __init__(
        self,
        candidate_models: List[ModelType],
        use_pca_variants: bool = True,
        evaluator: Optional[EmbeddingEvaluator] = None,
        domain_vocabulary: Optional[List[str]] = None,
        end_year: Optional[int] = None
    ):
        self.candidate_models = candidate_models
        self.use_pca_variants = use_pca_variants
        self.end_year = end_year

        self.evaluator = evaluator or EmbeddingEvaluator(
            domain_vocabulary=domain_vocabulary, 
            end_year=self.end_year
        )
        self.logger = LoggerFactory.setup_logger(
            name="SequentialModelSelector",
            target_year=str(end_year) if end_year else None,
            log_to_file=True,
            log_file=f'logs/{end_year}.log' if end_year else 'logs/model_selector.log'
        )
        self.results = []
    
    def select_best_model(
        self, sentences: List[List[str]], vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """Select best model sequentially."""
        self.logger.info("="*80)
        self.logger.info("SEQUENTIAL MODEL SELECTION")
        self.logger.info(f"Models: {[m.value for m in self.candidate_models]}")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        evaluator_params = {
            'n_clusters': self.evaluator.n_clusters,
            'k_neighbors': self.evaluator.k_neighbors,
            'test_compounds': self.evaluator.test_compounds,
            'domain_vocabulary': self.evaluator.domain_vocabulary,
            'random_state': self.evaluator.random_state,
        }
        
        # Phase 1: Quick screening
        screening_results = []
        for model_type in self.candidate_models:
            config = EmbeddingConfig(
                model_type=model_type, 
                use_pca=False, 
                vector_size=vector_size,
                use_pretrained=(model_type == ModelType.GLOVE),
                end_year=self.end_year
            )
            config, metrics, train_time = train_and_evaluate_single(
                config, sentences, evaluator_params, self.end_year
            )
            screening_results.append({
                'config': config,
                'metrics': metrics,
                'score': metrics.intrinsic_score,
                'training_time': train_time
            })
        
        screening_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Phase 2: Try PCA on top models
        detailed_results = []
        top_k = min(3, len(screening_results))
        
        for result in screening_results[:top_k]:
            mt = result['config'].model_type
            
            if self.use_pca_variants:
                for use_pca in [False, True]:
                    config = EmbeddingConfig(
                        model_type=mt,
                        use_pca=use_pca,
                        pca_components=50 if use_pca else None,
                        vector_size=vector_size,
                        use_pretrained=(mt == ModelType.GLOVE),
                        end_year=self.end_year
                    )
                    config, metrics, train_time = train_and_evaluate_single(
                        config, sentences, evaluator_params, self.end_year
                    )
                    detailed_results.append({
                        'config': config,
                        'metrics': metrics,
                        'score': metrics.intrinsic_score,
                        'training_time': train_time
                    })
        
        # Combine and find best
        self.results = screening_results + detailed_results
        best_result = max(self.results, key=lambda x: x['score'])
        
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Best: {best_result['config'].model_type.value}")
        self.logger.info(f"Score: {best_result['score']:.4f}")
        self.logger.info(f"Domain Coverage: {best_result['metrics'].domain_vocabulary_coverage:.3f}")
        self.logger.info(f"OOV Handling: {best_result['metrics'].oov_handling:.3f}")
        self.logger.info(f"Time: {total_time:.2f}s")
        self.logger.info(f"{'='*80}")
        
        return best_result['config'], best_result['metrics']
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        data = []
        for result in self.results:
            config = result['config']
            metrics = result['metrics']
            row = {
                'model_type': config.model_type.value,
                'use_pca': config.use_pca,
                'pca_components': config.pca_components,
                'training_time': result.get('training_time', 0.0),
                **metrics.to_dict()
            }
            data.append(row)
        return pd.DataFrame(data).sort_values('intrinsic_score', ascending=False)


class SequentialHyperparameterOptimizer:
    """Sequential hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        model_config: EmbeddingConfig,
        evaluator: Optional[EmbeddingEvaluator] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        end_year: Optional[int] = None
    ):
        self.model_config = model_config
        self.end_year = end_year
        self.evaluator = evaluator or EmbeddingEvaluator(end_year=self.end_year)
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optim_{model_config.model_type.value}"
        
        if storage_path is None:
            storage_path = f"./{self.study_name}.db"
        self.storage = f"sqlite:///{storage_path}"
        
        self.logger = LoggerFactory.setup_logger(
            name="SequentialHyperparameterOptimizer",
            target_year=str(self.end_year) if self.end_year else None,
            log_to_file=True,
            log_file=f'logs/{self.end_year}.log' if self.end_year else 'logs/hyperopt.log'
        )
        self.study = None
    
    def _get_search_space(self, trial: Trial) -> Dict[str, Any]:
        """Define espaço de busca de hiperparâmetros com Optuna."""
        params: Dict[str, Any] = {}
        
        if self.model_config.model_type in [ModelType.WORD2VEC, ModelType.FASTTEXT]:
            params.update({
                'vector_size': trial.suggest_categorical('vector_size', [100, 200, 300, 400]),
                'window': trial.suggest_int('window', 3, 10),
                'min_count': trial.suggest_int('min_count', 1, 5),
                'sg': trial.suggest_categorical('sg', [0, 1]),  # skip-gram vs CBOW
                'hs': trial.suggest_categorical('hs', [0, 1]),  # hierarchical softmax
                'negative': trial.suggest_int('negative', 5, 20),
                'alpha': trial.suggest_float('alpha', 0.001, 0.05, log=True),
                'min_alpha': trial.suggest_float('min_alpha', 1e-5, 0.01, log=True),
                'sample': trial.suggest_float('sample', 1e-6, 1e-2, log=True),
                'ns_exponent': trial.suggest_float('ns_exponent', 0.5, 1.0),
                'cbow_mean': trial.suggest_categorical('cbow_mean', [0, 1]),
                'epochs': trial.suggest_int('epochs', 10, 30),
            })
            if self.model_config.model_type == ModelType.FASTTEXT:
                params['min_n'] = trial.suggest_int('min_n', 2, 4)
                params['max_n'] = trial.suggest_int('max_n', 5, 8)
                # 'bucket' e 'word_ngrams' podem ser fixos (usualmente bucket grande, word_ngrams=1)
        
        elif self.model_config.model_type == ModelType.GLOVE:
            params.update({
                'vector_size': trial.suggest_categorical('vector_size', [100, 200, 300]),
                'window_size': trial.suggest_int('window_size', 5, 15),
                'max_iter': trial.suggest_int('max_iter', 10, 50),
                'x_max': trial.suggest_int('x_max', 50, 200),
                'alpha': trial.suggest_float('alpha', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'min_count': trial.suggest_int('min_count', 1, 5),
            })
        
        elif self.model_config.model_type == ModelType.LSA:
            params.update({
                'max_features': trial.suggest_categorical('max_features', [5000, 10000, 20000]),
                'min_df': trial.suggest_int('min_df', 1, 5),
                'max_df': trial.suggest_float('max_df', 0.85, 0.95),
                'use_tfidf': trial.suggest_categorical('use_tfidf', [True, False]),
                'n_components': trial.suggest_int('n_components', 50, 300),
                'n_iter': trial.suggest_int('n_iter', 5, 15),
            })
        
        if self.model_config.use_pca:
            params['pca_components'] = trial.suggest_int('pca_components', 30, 200)
        
        return params
    
    def _objective(self, trial: Trial, sentences: List[List[str]]) -> float:
        """Optuna objective function."""
        try:
            params = self._get_search_space(trial)
            
            config = EmbeddingConfig(
                model_type=self.model_config.model_type,
                use_pca=self.model_config.use_pca,
                pca_components=params.get('pca_components'),
                vector_size=params.get('vector_size', self.model_config.vector_size),
                use_pretrained=self.model_config.use_pretrained,
                custom_params=params,
                end_year=self.end_year
            )
            
            model = ModelFactory.create_model(config)
            model.train(sentences)
            embeddings = model.get_embeddings()
            
            metrics = self.evaluator.evaluate(
                embeddings, 
                model_type=config.model_type,
                model=model.model
            )
            trial.report(metrics.intrinsic_score, step=0)
            
            # Clean up
            del model
            del embeddings
            gc.collect()
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metrics.intrinsic_score
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial error: {e}")
            return 0.0
    
    def optimize(self, sentences: List[List[str]]) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters sequentially."""
        self.logger.info("="*80)
        self.logger.info("SEQUENTIAL HYPERPARAMETER OPTIMIZATION")
        self.logger.info(f"Model: {self.model_config.model_type.value}")
        self.logger.info(f"Trials: {self.n_trials}")
        self.logger.info("="*80)
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, sentences),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=1,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        self.logger.info("\n" + "="*80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info(f"Best Score: {best_score:.4f}")
        self.logger.info("Best Parameters:")
        for param, value in best_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("="*80)
        
        return best_params, best_score
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history."""
        if not self.study:
            return pd.DataFrame()
        return self.study.trials_dataframe().sort_values('value', ascending=False)


class SequentialEmbeddingAutoML:
    """Complete sequential AutoML pipeline for embeddings (NO multiprocessing)."""
    
    def __init__(
        self,
        candidate_models: Optional[List[ModelType]] = None,
        use_pca_variants: bool = True,
        hyperopt_trials: int = 50,
        hyperopt_timeout: Optional[int] = None,
        output_dir: Optional[Path] = None,
        domain_vocabulary: Optional[List[str]] = None,
        end_year: Optional[int] = None
    ):
        self.candidate_models = candidate_models or [
            ModelType.WORD2VEC, ModelType.FASTTEXT, ModelType.GLOVE]
        self.use_pca_variants = use_pca_variants
        self.hyperopt_trials = hyperopt_trials
        self.hyperopt_timeout = hyperopt_timeout
        self.output_dir = output_dir or Path('./automl_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.domain_vocabulary = domain_vocabulary
        self.end_year = end_year
        
        self.logger = LoggerFactory.setup_logger(
            name="SequentialEmbeddingAutoML",
            target_year=str(self.end_year) if self.end_year else None,
            log_to_file=True,
            log_file=f'logs/{self.end_year}.log' if self.end_year else 'logs/automl.log'
        )
        
        self.selected_model_config = None
        self.optimized_params = None
        self.final_model = None
    
    def run(
        self,
        sentences: List[List[str]],
        skip_model_selection: bool = False,
        initial_model_config: Optional[EmbeddingConfig] = None
    ) -> Tuple[BaseEmbeddingModel, EvaluationMetrics, str]:
        """Run complete sequential AutoML pipeline."""
        self.logger.info("\n" + "="*80)
        self.logger.info("SEQUENTIAL EMBEDDING AUTOML PIPELINE")
        self.logger.info("="*80)
        
        overall_start = time.time()
        
        # Phase 1: Model Selection
        if not skip_model_selection:
            self.logger.info("\n📊 PHASE 1: MODEL SELECTION")
            selector = SequentialModelSelector(
                candidate_models=self.candidate_models,
                use_pca_variants=self.use_pca_variants,
                domain_vocabulary=self.domain_vocabulary,
                end_year=self.end_year
            )
            
            self.selected_model_config, _ = selector.select_best_model(sentences)
            
            results_df = selector.get_results_dataframe()
            results_path = self.output_dir / 'model_selection_results.csv'
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"Results saved: {results_path}")
        else:
            if not initial_model_config:
                raise ValueError("initial_model_config required when skipping selection")
            self.selected_model_config = initial_model_config
            self.logger.info(f"Using: {initial_model_config.model_type.value}")
        
        # Phase 2: Hyperparameter Optimization
        self.logger.info("\n🔧 PHASE 2: HYPERPARAMETER OPTIMIZATION")
        storage_path = str(self.output_dir / 'optuna_study.db')
        
        optimizer = SequentialHyperparameterOptimizer(
            model_config=self.selected_model_config,
            evaluator=EmbeddingEvaluator(
                domain_vocabulary=self.domain_vocabulary, 
                end_year=self.end_year
            ),
            n_trials=self.hyperopt_trials,
            timeout=self.hyperopt_timeout,
            study_name=f"hyperopt_{self.selected_model_config.model_type.value}",
            storage_path=storage_path,
            end_year=self.end_year
        )
        
        self.optimized_params, best_score = optimizer.optimize(sentences)
        
        history_df = optimizer.get_optimization_history()
        history_path = self.output_dir / 'hyperopt_history.csv'
        history_df.to_csv(history_path, index=False)
        
        # Phase 3: Train Final Model
        self.logger.info("\n🚀 PHASE 3: TRAINING FINAL MODEL")
        final_config = EmbeddingConfig(
            model_type=self.selected_model_config.model_type,
            use_pca=self.selected_model_config.use_pca,
            pca_components=self.optimized_params.get('pca_components'),
            vector_size=self.optimized_params.get('vector_size', 300),
            use_pretrained=self.selected_model_config.use_pretrained,
            custom_params=self.optimized_params,
            end_year=self.end_year
        )
        
        self.final_model = ModelFactory.create_model(final_config)
        self.final_model.train(sentences)
        
        embeddings = self.final_model.get_embeddings()
        evaluator = EmbeddingEvaluator(
            domain_vocabulary=self.domain_vocabulary, 
            end_year=self.end_year
        )
        final_metrics = evaluator.evaluate(
            embeddings, 
            model_type=final_config.model_type,
            model=self.final_model.model
        )
        
        total_time = time.time() - overall_start
        model_type = final_config.model_type.value
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("AUTOML PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Model: {model_type}")
        self.logger.info(f"PCA: {final_config.use_pca}")
        self.logger.info(f"Score: {final_metrics.intrinsic_score:.4f}")
        self.logger.info(f"Domain Coverage: {final_metrics.domain_vocabulary_coverage:.3f}")
        self.logger.info(f"OOV Handling: {final_metrics.oov_handling:.3f}")
        self.logger.info(f"Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        self.logger.info("="*80)
        
        # Save configuration
        config_dict = {
            'model_type': model_type,
            'use_pca': final_config.use_pca,
            'use_pretrained': final_config.use_pretrained,
            'pca_components': final_config.pca_components,
            'vector_size': final_config.vector_size,
            'optimized_params': self.optimized_params,
            'final_metrics': final_metrics.to_dict(),
            'total_time': total_time,
        }
        
        config_path = self.output_dir / 'final_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.logger.info(f"Config saved: {config_path}")
        
        return self.final_model, final_metrics, model_type
    
    def save_model(self, path: Path) -> None:
        """Save final trained model."""
        if not self.final_model:
            raise ValueError("No model trained yet")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.final_model.model, 'save'):
            self.final_model.model.save(str(path))
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.final_model, f)
        
        self.logger.info(f"Model saved: {path}")


class SequentialEmbeddingTrainingAutoML:
    """Enhanced embedding training with sequential AutoML (PROCESS-SAFE)."""
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        automl_config: Optional[Dict[str, Any]] = None,
    ):
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        
        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_path = Path(f'{self.base_path}/models')
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'candidate_models': [
                ModelType.WORD2VEC, ModelType.FASTTEXT, ModelType.GLOVE],
            'use_pca_variants': True,
            'hyperopt_trials': 30,
            'hyperopt_timeout': 1800,
        }
        self.automl_config = {**default_config, **(automl_config or {})}
        
        self.logger = LoggerFactory.setup_logger(
            "SequentialEmbeddingTrainingAutoML",
            target_year=str(end_year),
            log_to_file=True,
            log_file=f'logs/{self.end_year}.log'
        )
        self._corpus_df = None
        self._domain_vocabulary = None  # NOVO
        
        self.logger.info(f"Initialized for {self.disease_name}")
        self.logger.info(f"Years: {start_year}-{end_year}")
    
    def _load_corpus(self) -> Optional[pd.DataFrame]:
        """Load corpus from path."""
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus not found: {self.corpus_path}")
            return None
        
        try:
            if self.corpus_path.is_dir():
                csv_files = list(self.corpus_path.glob('*.csv'))
                if not csv_files:
                    return None
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                df = pd.read_csv(self.corpus_path)
            
            if 'summary' not in df.columns:
                self.logger.error("'summary' column not found")
                return None
            
            if 'year_extracted' not in df.columns:
                df['year_extracted'] = df.get('year', self.end_year)
            
            self.logger.info(f"Loaded {len(df)} abstracts")
            return df
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            return None
    
    @property
    def corpus_df(self) -> pd.DataFrame:
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df
    
    def _extract_domain_vocabulary(self, sentences: List[List[str]]) -> List[str]:
        """NOVO: Extrair vocabulário específico do domínio."""
        if self._domain_vocabulary is not None:
            return self._domain_vocabulary
        
        from collections import Counter
        
        # Contar frequência de todas as palavras
        word_freq = Counter(word.lower() for sent in sentences for word in sent)
        
        # Palavras que aparecem entre 10 e 1000 vezes (filtro heurístico)
        domain_words = [word for word, freq in word_freq.items() if 10 <= freq <= 1000]
        
        self.logger.info(f"Extracted {len(domain_words)} domain-specific words")
        self._domain_vocabulary = domain_words
        return domain_words
    
    def _prepare_sentences(self, year_filter: Optional[int] = None) -> List[List[str]]:
        """Prepare sentences for training."""
        df = self.corpus_df
        if df is None or df.empty:
            return []
        
        if year_filter and 'year_extracted' in df.columns:
            df = df[df['year_extracted'] <= year_filter]
        
        abstracts = df['summary'].dropna().tolist()
        sentences = [abstract.split() for abstract in abstracts if abstract]
        sentences = [s for s in sentences if len(s) > 0]
        
        self.logger.info(f"Prepared {len(sentences)} sentences")
        return sentences
    
    def run_automl(
        self,
        year_filter: Optional[int] = None,
        skip_model_selection: bool = False,
        force_model_type: Optional[ModelType] = None
    ) -> bool:
        """Run sequential AutoML pipeline."""
        try:
            sentences = self._prepare_sentences(year_filter or self.end_year)
            
            if not sentences or len(sentences) < 10:
                self.logger.error(f"Insufficient data: {len(sentences)} sentences")
                return False
            
            # NOVO: Extrair vocabulário do domínio
            domain_vocab = self._extract_domain_vocabulary(sentences)
            
            automl = SequentialEmbeddingAutoML(
                candidate_models=self.automl_config['candidate_models'],
                use_pca_variants=self.automl_config['use_pca_variants'],
                hyperopt_trials=self.automl_config['hyperopt_trials'],
                hyperopt_timeout=self.automl_config['hyperopt_timeout'],
                output_dir=self.models_path / f'{self.start_year}_{self.end_year}',
                domain_vocabulary=domain_vocab,
                end_year=self.end_year
            )
            
            initial_config = None
            if skip_model_selection and force_model_type:
                initial_config = EmbeddingConfig(
                    model_type=force_model_type, 
                    use_pca=False, 
                    vector_size=300,
                    use_pretrained=(force_model_type == ModelType.GLOVE),
                    end_year=self.end_year
                    
                )
            
            final_model, _, model_type = automl.run(sentences, skip_model_selection, initial_config)
            
            model_path = Path(f'{self.models_path}/{model_type}_{self.start_year}_{self.end_year}.model')
            automl.save_model(model_path)
            
            self.logger.info("AutoML pipeline completed successfully")
            return True
        except Exception as e:
            self.logger.exception(f"AutoML pipeline error: {e}")
            return False
    
    def run_year_over_year(self, step: int = 1) -> bool:
        """Run AutoML year-over-year."""
        for year in range(self.start_year, self.end_year + 1, step):
            self.logger.info(f"{'='*80}")
            self.logger.info(f"AutoML for {self.start_year}-{year}")
            self.logger.info(f"{'='*80}")
            
            if not self.run_automl(year_filter=year):
                self.logger.error(f"Failed for year {year}")
                return False
        return True


if __name__ == '__main__':
    # IMPORTANT: Use Sequential version to avoid multiprocessing issues
    automl_trainer = SequentialEmbeddingTrainingAutoML(
        disease_name="acute myeloid leukemia",
        start_year=1990,
        end_year=2000,
        automl_config={
            'candidate_models': [
                ModelType.WORD2VEC,
                ModelType.FASTTEXT,
                ModelType.GLOVE],
            'use_pca_variants': True,
            'hyperopt_trials': 20,
            'hyperopt_timeout': 900,
        },
    )
    
    success = automl_trainer.run_automl()
