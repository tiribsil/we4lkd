import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import optuna
from optuna.trial import Trial
from gensim.models import Word2Vec, FastText
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import pickle
from functools import partial

from utils import LoggerFactory, normalize_disease_name

# Note: FLAML is designed for supervised learning (classification/regression).
# For unsupervised embedding selection, we use a custom efficient search strategy
# inspired by FLAML's Cost-Frugal Optimization approach.

# For transformer models
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class ModelType(Enum):
    """Supported embedding model types."""
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    LSA = "lsa"
    DOC2VEC = "doc2vec"
    BIOBERT = "biobert"
    PUBMEDBERT = "pubmedbert"
    SCIBERT = "scibert"
    SBERT = "sbert"
    BIOCLINICALBERT = "bioclinicalbert"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_type: ModelType
    use_pca: bool = False
    pca_components: Optional[int] = None
    vector_size: int = 300
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Metrics for unsupervised embedding evaluation focused on similarity tasks."""
    # Clustering quality
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    
    # Vocabulary and coverage
    vocabulary_coverage: float
    oov_handling: float  # Out-of-vocabulary handling (important for compounds)
    
    # Similarity quality
    cosine_consistency: float  # Consistency of cosine similarities
    euclidean_consistency: float  # Consistency of euclidean distances
    dot_product_consistency: float  # Consistency of dot products
    
    # Neighborhood quality
    neighborhood_preservation: float  # K-NN preservation
    rank_correlation: float  # Spearman correlation of rankings
    
    # Combined metrics
    similarity_score: float  # For similarity-based tasks
    intrinsic_score: float  # Overall quality
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'silhouette': self.silhouette,
            'calinski_harabasz': self.calinski_harabasz,
            'davies_bouldin': self.davies_bouldin,
            'vocabulary_coverage': self.vocabulary_coverage,
            'oov_handling': self.oov_handling,
            'cosine_consistency': self.cosine_consistency,
            'euclidean_consistency': self.euclidean_consistency,
            'dot_product_consistency': self.dot_product_consistency,
            'neighborhood_preservation': self.neighborhood_preservation,
            'rank_correlation': self.rank_correlation,
            'similarity_score': self.similarity_score,
            'intrinsic_score': self.intrinsic_score
        }


class EmbeddingEvaluator:
    """
    Evaluates embedding quality for SIMILARITY-BASED tasks.
    
    Critical for compound similarity analysis where we need:
    - Dot product similarities
    - Normalized dot products (cosine similarity)
    - Euclidean distances
    - Consistent rankings across metrics
    """
    
    def __init__(
        self, 
        n_clusters: int = 10, 
        k_neighbors: int = 10,
        test_compounds: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.test_compounds = test_compounds or []
        self.random_state = random_state

        self.logger = LoggerFactory.setup_logger(
                "EmbeddingEvaluator",
                target_year=str(0000),
                log_to_file=True
            )

    
    def _calculate_similarity_consistency(
        self,
        embeddings: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """Calculate consistency of different similarity metrics."""
        if len(embeddings) < n_samples:
            n_samples = len(embeddings)
        
        np.random.seed(self.random_state)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = sample_embeddings / norms
        
        cosine_sim = np.dot(normalized_embeddings, normalized_embeddings.T)
        dot_product = np.dot(sample_embeddings, sample_embeddings.T)
        
        from scipy.spatial.distance import pdist, squareform
        euclidean_dist = squareform(pdist(sample_embeddings, metric='euclidean'))
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        cosine_std = np.std(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
        dot_std = np.std(dot_product[np.triu_indices_from(dot_product, k=1)])
        euclidean_std = np.std(euclidean_sim[np.triu_indices_from(euclidean_sim, k=1)])
        
        cosine_consistency = 1 / (1 + cosine_std)
        dot_consistency = 1 / (1 + dot_std)
        euclidean_consistency = 1 / (1 + euclidean_std)
        
        return cosine_consistency, euclidean_consistency, dot_consistency
    
    def _calculate_neighborhood_preservation(
        self,
        embeddings: np.ndarray
    ) -> float:
        """Calculate how well local neighborhoods are preserved."""
        if len(embeddings) < self.k_neighbors + 1:
            return 0.0
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs_cosine = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1, 
            metric='cosine'
        ).fit(embeddings)
        _, indices_cosine = nbrs_cosine.kneighbors(embeddings)
        
        nbrs_euclidean = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1, 
            metric='euclidean'
        ).fit(embeddings)
        _, indices_euclidean = nbrs_euclidean.kneighbors(embeddings)
        
        overlaps = []
        for i in range(len(embeddings)):
            neighbors_cosine = set(indices_cosine[i][1:])
            neighbors_euclidean = set(indices_euclidean[i][1:])
            overlap = len(neighbors_cosine & neighbors_euclidean) / self.k_neighbors
            overlaps.append(overlap)
        
        return np.mean(overlaps)
    
    def _calculate_rank_correlation(
        self,
        embeddings: np.ndarray,
        n_samples: int = 50
    ) -> float:
        """Calculate rank correlation between different distance metrics."""
        if len(embeddings) < n_samples:
            n_samples = len(embeddings)
        
        from scipy.stats import spearmanr
        
        np.random.seed(self.random_state)
        reference_idx = np.random.choice(len(embeddings))
        reference = embeddings[reference_idx]
        
        sample_indices = np.random.choice(
            len(embeddings), 
            min(n_samples, len(embeddings)), 
            replace=False
        )
        sample_embeddings = embeddings[sample_indices]
        
        reference_norm = reference / (np.linalg.norm(reference) + 1e-8)
        sample_norms = sample_embeddings / (np.linalg.norm(sample_embeddings, axis=1, keepdims=True) + 1e-8)
        cosine_dists = 1 - np.dot(sample_norms, reference_norm)
        
        euclidean_dists = np.linalg.norm(sample_embeddings - reference, axis=1)
        
        dot_products = np.dot(sample_embeddings, reference)
        dot_dists = -dot_products
        
        corr_cosine_euclidean, _ = spearmanr(cosine_dists, euclidean_dists)
        corr_cosine_dot, _ = spearmanr(cosine_dists, dot_dists)
        corr_euclidean_dot, _ = spearmanr(euclidean_dists, dot_dists)
        
        avg_correlation = np.mean([
            abs(corr_cosine_euclidean),
            abs(corr_cosine_dot),
            abs(corr_euclidean_dot)
        ])
        
        return avg_correlation
    
    def _calculate_oov_handling(
        self,
        model,
        vocabulary: Optional[List[str]] = None
    ) -> float:
        """Evaluate Out-of-Vocabulary handling capability."""
        if not self.test_compounds or not vocabulary:
            if hasattr(model, 'wv') and hasattr(model.wv, 'vectors_ngrams'):
                return 0.8
            else:
                return 0.3
        
        oov_compounds = [c for c in self.test_compounds if c not in vocabulary]
        if not oov_compounds:
            return 1.0
        
        handled = 0
        for compound in oov_compounds:
            try:
                if hasattr(model, '__getitem__'):
                    _ = model[compound]
                    handled += 1
                elif hasattr(model, 'wv'):
                    _ = model.wv[compound]
                    handled += 1
            except:
                pass
        
        return handled / len(oov_compounds) if oov_compounds else 0.0
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        vocabulary: Optional[List[str]] = None,
        total_words: Optional[int] = None,
        model: Optional[Any] = None
    ) -> EvaluationMetrics:
        """Comprehensive evaluation for similarity-based tasks."""
        if len(embeddings) < max(self.n_clusters, self.k_neighbors + 1):
            self.logger.warning(
                f"Not enough samples ({len(embeddings)}) for evaluation"
            )
            return EvaluationMetrics(
                0.0, 0.0, float('inf'), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        
        vocab_coverage = 1.0
        if vocabulary and total_words:
            vocab_coverage = len(vocabulary) / total_words
        
        oov_handling = self._calculate_oov_handling(model, vocabulary)
        
        cosine_cons, euclidean_cons, dot_cons = self._calculate_similarity_consistency(
            embeddings
        )
        
        neighborhood_pres = self._calculate_neighborhood_preservation(embeddings)
        rank_corr = self._calculate_rank_correlation(embeddings)
        
        silhouette_norm = (silhouette + 1) / 2
        calinski_norm = min(calinski / 1000, 1.0)
        davies_bouldin_norm = 1 / (1 + davies_bouldin)
        
        similarity_score = (
            0.25 * cosine_cons +
            0.25 * euclidean_cons +
            0.20 * dot_cons +
            0.15 * neighborhood_pres +
            0.15 * rank_corr
        )
        
        intrinsic_score = (
            0.15 * silhouette_norm +
            0.10 * calinski_norm +
            0.10 * davies_bouldin_norm +
            0.10 * vocab_coverage +
            0.15 * oov_handling +
            0.40 * similarity_score
        )
        
        return EvaluationMetrics(
            silhouette=silhouette,
            calinski_harabasz=calinski,
            davies_bouldin=davies_bouldin,
            vocabulary_coverage=vocab_coverage,
            oov_handling=oov_handling,
            cosine_consistency=cosine_cons,
            euclidean_consistency=euclidean_cons,
            dot_product_consistency=dot_cons,
            neighborhood_preservation=neighborhood_pres,
            rank_correlation=rank_corr,
            similarity_score=similarity_score,
            intrinsic_score=intrinsic_score
        )


class BaseEmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.pca = None
        self.logger = LoggerFactory.setup_logger(
                "BaseEmbeddingModel",
                target_year=str(0000),
                log_to_file=True
            )
    
    def train(self, sentences: List[List[str]]) -> None:
        """Train the embedding model."""
        raise NotImplementedError
    
    def get_embeddings(self, words: Optional[List[str]] = None) -> np.ndarray:
        """Get embeddings for words or entire vocabulary."""
        raise NotImplementedError
    
    def apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA reduction if configured."""
        if not self.config.use_pca:
            return embeddings
        
        n_components = self.config.pca_components or min(50, embeddings.shape[1] // 2)
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            reduced = self.pca.fit_transform(embeddings)
        else:
            reduced = self.pca.transform(embeddings)
        
        self.logger.info(f"PCA reduced dimensions from {embeddings.shape[1]} to {reduced.shape[1]}")
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
            'workers': self.config.custom_params.get('workers', 4),
        }
        
        self.model = Word2Vec(sentences=sentences, **params)
        self.logger.info(f"Trained Word2Vec with vocab size: {len(self.model.wv)}")
    
    def get_embeddings(self, words: Optional[List[str]] = None) -> np.ndarray:
        if words:
            embeddings = np.array([self.model.wv[w] for w in words if w in self.model.wv])
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
            'workers': self.config.custom_params.get('workers', 4),
            'min_n': self.config.custom_params.get('min_n', 3),
            'max_n': self.config.custom_params.get('max_n', 5),
        }
        
        self.model = FastText(sentences=sentences, **params)
        self.logger.info(f"Trained FastText with vocab size: {len(self.model.wv)}")
    
    def get_embeddings(self, words: Optional[List[str]] = None) -> np.ndarray:
        if words:
            embeddings = np.array([self.model.wv[w] for w in words if w in self.model.wv])
        else:
            embeddings = self.model.wv.vectors
        
        return self.apply_pca(embeddings)


class TransformerModel(BaseEmbeddingModel):
    """Transformer-based embedding models (BERT variants, SBERT, etc.)."""
    
    MODEL_NAMES = {
        ModelType.BIOBERT: "dmis-lab/biobert-v1.1",
        ModelType.PUBMEDBERT: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        ModelType.SCIBERT: "allenai/scibert_scivocab_uncased",
        ModelType.SBERT: "sentence-transformers/all-MiniLM-L6-v2",
        ModelType.BIOCLINICALBERT: "emilyalsentzer/Bio_ClinicalBERT",
    }
    
    def train(self, sentences: List[List[str]]) -> None:
        """Load pre-trained transformer model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for transformer models")
        
        model_name = self.MODEL_NAMES.get(self.config.model_type)
        if not model_name:
            raise ValueError(f"Unknown transformer model: {self.config.model_type}")
        
        self.logger.info(f"Loading pre-trained model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.sentences = [' '.join(sent) for sent in sentences]
    
    def get_embeddings(self, words: Optional[List[str]] = None) -> np.ndarray:
        """Get sentence embeddings."""
        if words:
            texts = words
        else:
            texts = self.sentences
        
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return self.apply_pca(embeddings)


class ModelFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create_model(config: EmbeddingConfig) -> BaseEmbeddingModel:
        """Create embedding model based on config."""
        model_map = {
            ModelType.WORD2VEC: Word2VecModel,
            ModelType.FASTTEXT: FastTextModel,
            ModelType.BIOBERT: TransformerModel,
            ModelType.PUBMEDBERT: TransformerModel,
            ModelType.SCIBERT: TransformerModel,
            ModelType.SBERT: TransformerModel,
            ModelType.BIOCLINICALBERT: TransformerModel,
        }
        
        model_class = model_map.get(config.model_type)
        if not model_class:
            raise ValueError(f"Model type not implemented: {config.model_type}")
        
        return model_class(config)


# ============================================================================
# PARALLEL EXECUTION FUNCTIONS
# ============================================================================

def _train_and_evaluate_config(
    config: EmbeddingConfig,
    sentences: List[List[str]],
    evaluator_params: Dict[str, Any],
    worker_id: int
) -> Tuple[EmbeddingConfig, EvaluationMetrics, float]:
    """
    Worker function to train and evaluate a single configuration.
    This runs in a separate process.
    """
    logger = LoggerFactory.setup_logger("_train_and_evaluate_config",
                                    target_year=str(start_year),
                                    log_to_file=True
                                )
    start_time = time.time()
    
    try:
        logger.info(f"[Worker {worker_id}] Evaluating {config.model_type.value} (PCA={config.use_pca})")
        
        # Create and train model
        model = ModelFactory.create_model(config)
        model.train(sentences)
        
        # Get embeddings
        embeddings = model.get_embeddings()
        
        # Get vocabulary info
        vocabulary = None
        if hasattr(model.model, 'wv'):
            vocabulary = list(model.model.wv.key_to_index.keys())
        
        total_words = len(set(word for sent in sentences for word in sent))
        
        # Create evaluator
        evaluator = EmbeddingEvaluator(**evaluator_params)
        
        # Evaluate
        metrics = evaluator.evaluate(
            embeddings,
            vocabulary=vocabulary,
            total_words=total_words,
            model=model.model
        )
        
        training_time = time.time() - start_time
        
        logger.info(
            f"[Worker {worker_id}] {config.model_type.value} - "
            f"Score: {metrics.intrinsic_score:.4f}, "
            f"Time: {training_time:.2f}s"
        )
        
        return config, metrics, training_time
        
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error: {e}")
        return config, EvaluationMetrics(
            0.0, 0.0, float('inf'), 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ), time.time() - start_time


class ParallelModelSelector:
    """
    Parallel AutoML-style Model Selection for Embeddings.
    
    Uses multiprocessing to train and evaluate multiple models in parallel.
    Significantly faster than sequential execution.
    """
    
    def __init__(
        self,
        candidate_models: List[ModelType],
        use_pca_variants: bool = True,
        evaluator: Optional[EmbeddingEvaluator] = None,
        time_budget: int = 3600,
        n_jobs: int = -1,
    ):
        """
        Initialize parallel model selector.
        
        Args:
            candidate_models: List of model types to evaluate
            use_pca_variants: If True, test each model with and without PCA
            evaluator: Custom evaluator (optional)
            time_budget: Total time budget in seconds
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.candidate_models = candidate_models
        self.use_pca_variants = use_pca_variants
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.time_budget = time_budget
        
        # Set number of workers
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        self.logger = LoggerFactory.setup_logger("ParallelModelSelector",
                                    target_year=str(0000),
                                    log_to_file=True
                                )
        
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')
        
        self.logger.info(f"Parallel ModelSelector initialized with {self.n_jobs} workers")
    
    def _parallel_screening(
        self,
        sentences: List[List[str]],
        vector_size: int
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Parallel screening of all models with default params.
        """
        self.logger.info("=== Phase 1: Parallel Quick Screening ===")
        self.logger.info(f"Testing {len(self.candidate_models)} models in parallel")
        
        # Create configurations for screening
        configs = []
        for model_type in self.candidate_models:
            config = EmbeddingConfig(
                model_type=model_type,
                use_pca=False,
                vector_size=vector_size
            )
            configs.append(config)
        
        # Prepare evaluator params for workers
        evaluator_params = {
            'n_clusters': self.evaluator.n_clusters,
            'k_neighbors': self.evaluator.k_neighbors,
            'test_compounds': self.evaluator.test_compounds,
            'random_state': self.evaluator.random_state,
        }
        
        # Execute in parallel
        screening_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _train_and_evaluate_config,
                    config,
                    sentences,
                    evaluator_params,
                    i
                ): (config, i)
                for i, config in enumerate(configs)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                config, worker_id = futures[future]
                try:
                    result_config, metrics, training_time = future.result()
                    
                    screening_results.append({
                        'config': result_config,
                        'metrics': metrics,
                        'score': metrics.intrinsic_score,
                        'training_time': training_time
                    })
                    
                    self.logger.info(
                        f"✓ {result_config.model_type.value}: "
                        f"Score={metrics.intrinsic_score:.3f}, "
                        f"Time={training_time:.2f}s"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to get result from worker {worker_id}: {e}")
        
        # Sort by score
        screening_results.sort(key=lambda x: x['score'], reverse=True)
        
        return screening_results
    
    def _parallel_detailed_evaluation(
        self,
        top_models: List[Dict[str, Any]],
        sentences: List[List[str]],
        vector_size: int
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Parallel detailed evaluation of top candidates with PCA variants.
        """
        self.logger.info(f"\n=== Phase 2: Parallel Detailed Evaluation ===")
        self.logger.info(f"Testing top {len(top_models)} models with PCA variants")
        
        # Create configurations for detailed evaluation
        configs = []
        for result in top_models:
            model_type = result['config'].model_type
            
            # Test with and without PCA
            for use_pca in [False, True] if self.use_pca_variants else [False]:
                config = EmbeddingConfig(
                    model_type=model_type,
                    use_pca=use_pca,
                    pca_components=50 if use_pca else None,
                    vector_size=vector_size
                )
                configs.append(config)
        
        # Prepare evaluator params
        evaluator_params = {
            'n_clusters': self.evaluator.n_clusters,
            'k_neighbors': self.evaluator.k_neighbors,
            'test_compounds': self.evaluator.test_compounds,
            'random_state': self.evaluator.random_state,
        }
        
        # Execute in parallel
        detailed_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(
                    _train_and_evaluate_config,
                    config,
                    sentences,
                    evaluator_params,
                    i
                ): (config, i)
                for i, config in enumerate(configs)
            }
            
            for future in as_completed(futures):
                config, worker_id = futures[future]
                try:
                    result_config, metrics, training_time = future.result()
                    
                    detailed_results.append({
                        'config': result_config,
                        'metrics': metrics,
                        'score': metrics.intrinsic_score,
                        'training_time': training_time
                    })
                    
                    pca_str = f"PCA={result_config.use_pca}"
                    self.logger.info(
                        f"✓ {result_config.model_type.value} ({pca_str}): "
                        f"Score={metrics.intrinsic_score:.3f}, "
                        f"Time={training_time:.2f}s"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to get result from worker {worker_id}: {e}")
        
        return detailed_results
    
    def select_best_model(
        self,
        sentences: List[List[str]],
        vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """
        Select best model from candidates using PARALLEL efficient search.
        
        Args:
            sentences: Training sentences
            vector_size: Embedding dimension
            
        Returns:
            Tuple of (best_config, best_metrics)
        """
        self.logger.info("="*80)
        self.logger.info("PARALLEL AUTOML: MODEL SELECTION")
        self.logger.info("="*80)
        self.logger.info(f"Strategy: Parallel Efficient Search")
        self.logger.info(f"Workers: {self.n_jobs}")
        self.logger.info(f"Candidates: {[m.value for m in self.candidate_models]}")
        self.logger.info(f"PCA variants: {self.use_pca_variants}")
        self.logger.info("="*80 + "\n")
        
        start_time = time.time()
        
        # Phase 1: Parallel screening
        screening_results = self._parallel_screening(sentences, vector_size)
        
        # Get top candidates
        top_k = min(3, len(screening_results))
        top_models = screening_results[:top_k]
        
        self.logger.info(f"\nTop {top_k} models from screening:")
        for i, result in enumerate(top_models, 1):
            self.logger.info(
                f"  {i}. {result['config'].model_type.value}: "
                f"Score={result['score']:.4f}, Time={result['training_time']:.2f}s"
            )
        
        # Phase 2: Parallel detailed evaluation
        detailed_results = self._parallel_detailed_evaluation(
            top_models, sentences, vector_size
        )
        
        # Combine all results
        self.results = screening_results + detailed_results
        
        # Find best
        best_result = max(self.results, key=lambda x: x['score'])
        self.best_config = best_result['config']
        self.best_score = best_result['score']
        
        total_time = time.time() - start_time
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Best Configuration Found:")
        self.logger.info(f"  Model: {self.best_config.model_type.value}")
        self.logger.info(f"  PCA: {self.best_config.use_pca}")
        self.logger.info(f"  Score: {self.best_score:.4f}")
        self.logger.info(f"  Total Time: {total_time:.2f}s")
        self.logger.info(f"  Speedup: ~{len(self.results) * best_result['training_time'] / total_time:.1f}x")
        self.logger.info(f"{'='*80}")
        
        return self.best_config, best_result['metrics']
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        data = []
        for result in self.results:
            config = result['config']
            metrics = result['metrics']
            
            row = {
                'model_type': config.model_type.value,
                'use_pca': config.use_pca,
                'pca_components': config.pca_components if config.use_pca else None,
                'training_time': result.get('training_time', 0.0),
                **metrics.to_dict()
            }
            data.append(row)
        
        return pd.DataFrame(data).sort_values('intrinsic_score', ascending=False)


class ParallelHyperparameterOptimizer:
    """
    Parallel hyperparameter optimization using Optuna with parallel trials.
    """
    
    def __init__(
        self,
        model_config: EmbeddingConfig,
        evaluator: Optional[EmbeddingEvaluator] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize parallel hyperparameter optimizer.
        
        Args:
            model_config: Base model configuration to optimize
            evaluator: Evaluation metrics calculator
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Optuna study name
            storage: Optuna storage URL (required for parallel execution)
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.model_config = model_config
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optim_{model_config.model_type.value}"
        self.storage = storage
        
        # Set number of workers
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        self.logger = LoggerFactory.setup_logger("ParallelHyperparameterOptimizer",
                                    target_year=str(0000),
                                    log_to_file=True
                                )
        
        self.best_params = None
        self.best_score = None
        self.study = None
        
        self.logger.info(f"Parallel HyperparameterOptimizer with {self.n_jobs} workers")
    
    def _get_search_space(self, trial: Trial) -> Dict[str, Any]:
        """Define search space based on model type."""
        model_type = self.model_config.model_type
        
        if model_type in [ModelType.WORD2VEC, ModelType.FASTTEXT]:
            params = {
                'vector_size': trial.suggest_categorical('vector_size', [100, 200, 300, 400]),
                'window': trial.suggest_int('window', 3, 10),
                'min_count': trial.suggest_int('min_count', 1, 5),
                'sg': trial.suggest_categorical('sg', [0, 1]),
                'negative': trial.suggest_int('negative', 5, 20),
                'alpha': trial.suggest_float('alpha', 0.001, 0.05, log=True),
                'epochs': trial.suggest_int('epochs', 10, 30),
            }
            
            if model_type == ModelType.FASTTEXT:
                params['min_n'] = trial.suggest_int('min_n', 2, 4)
                params['max_n'] = trial.suggest_int('max_n', 4, 7)
        
        elif model_type in [ModelType.BIOBERT, ModelType.PUBMEDBERT, 
                           ModelType.SCIBERT, ModelType.SBERT, ModelType.BIOCLINICALBERT]:
            params = {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            }
        
        else:
            params = {}
        
        if self.model_config.use_pca:
            params['pca_components'] = trial.suggest_int('pca_components', 30, 200)
        
        return params
    
    def _objective(
        self,
        trial: Trial,
        sentences: List[List[str]]
    ) -> float:
        """Optuna objective function."""
        try:
            params = self._get_search_space(trial)
            
            config = EmbeddingConfig(
                model_type=self.model_config.model_type,
                use_pca=self.model_config.use_pca,
                pca_components=params.get('pca_components'),
                vector_size=params.get('vector_size', self.model_config.vector_size),
                custom_params=params
            )
            
            model = ModelFactory.create_model(config)
            model.train(sentences)
            
            embeddings = model.get_embeddings()
            
            metrics = self.evaluator.evaluate(embeddings)
            
            trial.report(metrics.intrinsic_score, step=0)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return metrics.intrinsic_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Error in trial: {e}")
            return 0.0
    
    def optimize(
        self,
        sentences: List[List[str]]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters in PARALLEL.
        
        Args:
            sentences: Training sentences
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self.logger.info("="*80)
        self.logger.info("PARALLEL Hyperparameter Optimization")
        self.logger.info(f"Model: {self.model_config.model_type.value}")
        self.logger.info(f"Trials: {self.n_trials}")
        self.logger.info(f"Parallel Workers: {self.n_jobs}")
        self.logger.info("="*80)
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Optimize with parallel trials
        self.study.optimize(
            lambda trial: self._objective(trial, sentences),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,  # This enables parallel execution
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Optimization Complete")
        self.logger.info("="*80)
        self.logger.info(f"Best Score: {self.best_score:.4f}")
        self.logger.info("Best Parameters:")
        for param, value in self.best_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("="*80)
        
        return self.best_params, self.best_score
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not self.study:
            return pd.DataFrame()
        
        df = self.study.trials_dataframe()
        return df.sort_values('value', ascending=False)


class ParallelEmbeddingAutoML:
    """
    Complete PARALLEL AutoML pipeline for embeddings.
    1. Parallel Model Selection
    2. Parallel Hyperparameter Optimization
    """
    
    def __init__(
        self,
        candidate_models: Optional[List[ModelType]] = None,
        use_pca_variants: bool = True,
        model_selection_time_budget: int = 3600,
        hyperopt_trials: int = 50,
        hyperopt_timeout: Optional[int] = None,
        output_dir: Optional[Path] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize PARALLEL AutoML pipeline.
        
        Args:
            candidate_models: Models to consider (None = all available)
            use_pca_variants: Test PCA variants during selection
            model_selection_time_budget: Time budget for model selection
            hyperopt_trials: Number of hyperparameter optimization trials
            hyperopt_timeout: Timeout for hyperparameter optimization
            output_dir: Directory to save results
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.candidate_models = candidate_models or [
            ModelType.WORD2VEC,
            ModelType.FASTTEXT,
            ModelType.BIOBERT,
            ModelType.PUBMEDBERT,
            ModelType.SCIBERT,
        ]
        
        self.use_pca_variants = use_pca_variants
        self.model_selection_time_budget = model_selection_time_budget
        self.hyperopt_trials = hyperopt_trials
        self.hyperopt_timeout = hyperopt_timeout
        self.output_dir = output_dir or Path('./automl_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of workers
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        self.logger = LoggerFactory.setup_logger("ParallelEmbeddingAutoML",
                                    target_year=str(0000),
                                    log_to_file=True
                                )
        
        # Results
        self.selected_model_config = None
        self.optimized_params = None
        self.final_model = None
        
        self.logger.info(f"Parallel AutoML initialized with {self.n_jobs} workers")
    
    def run(
        self,
        sentences: List[List[str]],
        skip_model_selection: bool = False,
        initial_model_config: Optional[EmbeddingConfig] = None
    ) -> Tuple[BaseEmbeddingModel, EvaluationMetrics]:
        """
        Run complete PARALLEL AutoML pipeline.
        
        Args:
            sentences: Training sentences
            skip_model_selection: If True, skip model selection phase
            initial_model_config: Starting config (required if skipping selection)
            
        Returns:
            Tuple of (final_model, evaluation_metrics)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("PARALLEL EMBEDDING AUTOML PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"CPU Cores: {cpu_count()}")
        self.logger.info(f"Parallel Workers: {self.n_jobs}")
        self.logger.info("="*80)
        
        overall_start = time.time()
        
        # Phase 1: Parallel Model Selection
        if not skip_model_selection:
            self.logger.info("\n📊 PHASE 1: PARALLEL MODEL SELECTION")
            self.logger.info("-"*80)
            
            selector = ParallelModelSelector(
                candidate_models=self.candidate_models,
                use_pca_variants=self.use_pca_variants,
                time_budget=self.model_selection_time_budget,
                n_jobs=self.n_jobs
            )
            
            self.selected_model_config, selection_metrics = selector.select_best_model(
                sentences=sentences
            )
            
            # Save model selection results
            results_df = selector.get_results_dataframe()
            results_path = self.output_dir / 'model_selection_results.csv'
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"Model selection results saved to {results_path}")
        
        else:
            if initial_model_config is None:
                raise ValueError("initial_model_config required when skipping model selection")
            self.selected_model_config = initial_model_config
            self.logger.info(f"Skipping model selection. Using: {initial_model_config.model_type.value}")
        
        # Phase 2: Parallel Hyperparameter Optimization
        self.logger.info("\n🔧 PHASE 2: PARALLEL HYPERPARAMETER OPTIMIZATION")
        self.logger.info("-"*80)
        
        # Setup storage for parallel Optuna
        storage_path = self.output_dir / 'optuna_study.db'
        storage_url = f"sqlite:///{storage_path}"
        
        optimizer = ParallelHyperparameterOptimizer(
            model_config=self.selected_model_config,
            n_trials=self.hyperopt_trials,
            timeout=self.hyperopt_timeout,
            study_name=f"hyperopt_{self.selected_model_config.model_type.value}",
            storage=storage_url,
            n_jobs=self.n_jobs
        )
        
        self.optimized_params, best_score = optimizer.optimize(sentences)
        
        # Save optimization history
        history_df = optimizer.get_optimization_history()
        history_path = self.output_dir / 'hyperopt_history.csv'
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"Optimization history saved to {history_path}")
        
        # Phase 3: Train Final Model
        self.logger.info("\n🚀 PHASE 3: TRAINING FINAL MODEL")
        self.logger.info("-"*80)
        
        final_config = EmbeddingConfig(
            model_type=self.selected_model_config.model_type,
            use_pca=self.selected_model_config.use_pca,
            pca_components=self.optimized_params.get('pca_components'),
            vector_size=self.optimized_params.get('vector_size', 300),
            custom_params=self.optimized_params
        )
        
        self.final_model = ModelFactory.create_model(final_config)
        self.final_model.train(sentences)
        
        # Final evaluation
        embeddings = self.final_model.get_embeddings()
        evaluator = EmbeddingEvaluator()
        final_metrics = evaluator.evaluate(embeddings)
        
        total_time = time.time() - overall_start
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("PARALLEL AUTOML PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Selected Model: {final_config.model_type.value}")
        self.logger.info(f"Use PCA: {final_config.use_pca}")
        self.logger.info(f"Final Score: {final_metrics.intrinsic_score:.4f}")
        self.logger.info(f"Total Pipeline Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        self.logger.info("\nFinal Metrics:")
        for metric, value in final_metrics.to_dict().items():
            self.logger.info(f"  {metric}: {value:.4f}")
        self.logger.info("="*80 + "\n")
        
        # Save final configuration
        import json
        config_dict = {
            'model_type': final_config.model_type.value,
            'use_pca': final_config.use_pca,
            'pca_components': final_config.pca_components,
            'vector_size': final_config.vector_size,
            'optimized_params': self.optimized_params,
            'final_metrics': final_metrics.to_dict(),
            'total_time': total_time,
            'n_workers': self.n_jobs
        }
        
        config_path = self.output_dir / 'final_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.logger.info(f"Final configuration saved to {config_path}")
        
        return self.final_model, final_metrics
    
    def save_model(self, path: Path) -> None:
        """Save final trained model."""
        if self.final_model is None:
            raise ValueError("No model trained yet. Run the pipeline first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.final_model.model, 'save'):
            self.final_model.model.save(str(path))
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.final_model, f)
        
        self.logger.info(f"Model saved to {path}")


class ParallelEmbeddingTrainingAutoML:
    """
    Enhanced EmbeddingTraining with PARALLEL AutoML capabilities.
    Integrates with the existing pipeline.
    """
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        automl_config: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize with PARALLEL AutoML support.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year for training data
            end_year: Ending year for training data
            automl_config: AutoML configuration dictionary
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        # Import utils here to avoid circular imports
        try:
            
            
            self.logger = LoggerFactory.setup_logger(
                "ParallelEmbeddingTrainingAutoML",
                target_year=str(start_year),
                log_to_file=True
            )
            
            self.disease_name = normalize_disease_name(disease_name)
        except ImportError:
            # Fallback if utils not available
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.disease_name = disease_name.lower().replace(' ', '_')
        
        self.start_year = start_year
        self.end_year = end_year
        
        # Set number of workers
        if n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
        
        # Paths - using self.corpus_path as specified
        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_path = Path(f'{self.base_path}/models/automl')
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # AutoML configuration
        default_config = {
            'candidate_models': [
                ModelType.WORD2VEC,
                ModelType.FASTTEXT,
                ModelType.BIOBERT,
                ModelType.PUBMEDBERT,
            ],
            'use_pca_variants': True,
            'model_selection_time_budget': 3600,
            'hyperopt_trials': 30,
            'hyperopt_timeout': 1800,
        }
        
        self.automl_config = {**default_config, **(automl_config or {})}
        
        # Cache
        self._corpus_df = None
        
        self.logger.info(f"Parallel EmbeddingTrainingAutoML initialized for {self.disease_name}")
        self.logger.info(f"Years: {start_year}-{end_year}")
        self.logger.info(f"Corpus path: {self.corpus_path}")
        self.logger.info(f"Models path: {self.models_path}")
        self.logger.info(f"Parallel workers: {self.n_jobs}")
    
    def _load_corpus(self) -> Optional[pd.DataFrame]:
        """Load corpus from self.corpus_path."""
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus path not found at {self.corpus_path}")
            return None
        
        try:
            self.logger.info(f"Loading corpus from {self.corpus_path}")
            
            if self.corpus_path.is_dir():
                csv_files = list(self.corpus_path.glob('*.csv'))
                if not csv_files:
                    self.logger.error(f"No CSV files found in {self.corpus_path}")
                    return None
                
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                df = pd.read_csv(self.corpus_path)
            
            if 'summary' not in df.columns:
                self.logger.error("Column 'summary' not found in corpus")
                return None
            
            if 'year_extracted' not in df.columns and 'year' not in df.columns:
                df['year_extracted'] = self.end_year
            elif 'year' in df.columns:
                df['year_extracted'] = df['year']
            
            self.logger.info(f"Loaded {len(df)} abstracts from corpus")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            return None
    
    @property
    def corpus_df(self) -> pd.DataFrame:
        """Lazy loading of corpus."""
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df
    
    def _prepare_sentences(
        self,
        year_filter: Optional[int] = None
    ) -> List[List[str]]:
        """Prepare sentences for training from corpus."""
        df = self.corpus_df
        
        if df is None or df.empty:
            self.logger.error("Corpus is empty or None")
            return []
        
        if year_filter and 'year_extracted' in df.columns:
            df = df[df['year_extracted'] <= year_filter]
            self.logger.info(f"Filtered to {len(df)} abstracts up to year {year_filter}")
        
        abstracts = df['summary'].dropna().tolist()
        self.logger.info(f"Preparing {len(abstracts)} abstracts for training")
        
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
        """
        Run PARALLEL AutoML pipeline.
        
        Args:
            year_filter: Filter abstracts up to this year
            skip_model_selection: Skip model selection phase
            force_model_type: Force specific model type
            
        Returns:
            True if successful
        """
        try:
            # Prepare data from corpus
            sentences = self._prepare_sentences(year_filter or self.end_year)
            
            if not sentences:
                self.logger.error("No sentences available for training")
                return False
            
            if len(sentences) < 10:
                self.logger.warning(f"Corpus too small: {len(sentences)} sentences")
                return False
            
            # Initialize PARALLEL AutoML
            automl = ParallelEmbeddingAutoML(
                candidate_models=self.automl_config['candidate_models'],
                use_pca_variants=self.automl_config['use_pca_variants'],
                model_selection_time_budget=self.automl_config['model_selection_time_budget'],
                hyperopt_trials=self.automl_config['hyperopt_trials'],
                hyperopt_timeout=self.automl_config['hyperopt_timeout'],
                output_dir=self.models_path / f'{self.start_year}_{self.end_year}',
                n_jobs=self.n_jobs
            )
            
            # Prepare initial config if skipping selection
            initial_config = None
            if skip_model_selection and force_model_type:
                initial_config = EmbeddingConfig(
                    model_type=force_model_type,
                    use_pca=False,
                    vector_size=300
                )
            
            # Run pipeline
            final_model, final_metrics = automl.run(
                sentences=sentences,
                skip_model_selection=skip_model_selection,
                initial_model_config=initial_config
            )
            
            # Save final model
            model_path = self.models_path / f'model_{self.start_year}_{self.end_year}.model'
            automl.save_model(model_path)
            
            self.logger.info("PARALLEL AutoML pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error in PARALLEL AutoML pipeline: {e}")
            return False
    
    def run_year_over_year(self, step: int = 1) -> bool:
        """
        Run PARALLEL AutoML year-over-year.
        
        Args:
            step: Year increment
            
        Returns:
            True if successful
        """
        for year in range(self.start_year, self.end_year + 1, step):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"PARALLEL AutoML for year range {self.start_year}-{year}")
            self.logger.info(f"{'='*80}\n")
            
            success = self.run_automl(year_filter=year)
            
            if not success:
                self.logger.error(f"Failed for year {year}")
                return False
        
        return True


# Example usage and testing
if __name__ == '__main__':

    automl_trainer = ParallelEmbeddingTrainingAutoML(
        disease_name="acute myeloid leukemia",
        start_year=1990,
        end_year=2000,
        automl_config={
            'candidate_models': [
                ModelType.WORD2VEC,
                ModelType.FASTTEXT,
                ModelType.PUBMEDBERT,
            ],
            'use_pca_variants': True,
            'model_selection_time_budget': 1800,
            'hyperopt_trials': 20,
            'hyperopt_timeout': 900,
        },
        n_jobs=-1  # Use all available CPUs
    )
    
    success = automl_trainer.run_automl()
