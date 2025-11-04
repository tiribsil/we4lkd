import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import json

from utils import LoggerFactory, normalize_disease_name

# Transformer models
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")


class ModelType(Enum):
    """Supported embedding model types."""
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    LSA = "lsa"
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
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


class EmbeddingEvaluator:
    """Evaluates embedding quality for similarity-based tasks."""
    
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
        self.logger = LoggerFactory.setup_logger("EmbeddingEvaluator", log_to_file=True)
    
    def _calculate_similarity_consistency(
        self, embeddings: np.ndarray, n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """Calculate consistency of different similarity metrics."""
        n_samples = min(n_samples, len(embeddings))
        np.random.seed(self.random_state)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = sample_embeddings / norms
        
        # Calculate similarities
        cosine_sim = np.dot(normalized, normalized.T)
        dot_product = np.dot(sample_embeddings, sample_embeddings.T)
        
        from scipy.spatial.distance import pdist, squareform
        euclidean_dist = squareform(pdist(sample_embeddings, metric='euclidean'))
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Calculate consistency (lower std = higher consistency)
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
        
        nbrs_cosine = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='cosine').fit(embeddings)
        _, indices_cosine = nbrs_cosine.kneighbors(embeddings)
        
        nbrs_euclidean = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean').fit(embeddings)
        _, indices_euclidean = nbrs_euclidean.kneighbors(embeddings)
        
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
        
        # Calculate distances
        reference_norm = reference / (np.linalg.norm(reference) + 1e-8)
        sample_norms = sample_embeddings / (np.linalg.norm(sample_embeddings, axis=1, keepdims=True) + 1e-8)
        cosine_dists = 1 - np.dot(sample_norms, reference_norm)
        euclidean_dists = np.linalg.norm(sample_embeddings - reference, axis=1)
        dot_dists = -np.dot(sample_embeddings, reference)
        
        # Spearman correlations
        corr_ce, _ = spearmanr(cosine_dists, euclidean_dists)
        corr_cd, _ = spearmanr(cosine_dists, dot_dists)
        corr_ed, _ = spearmanr(euclidean_dists, dot_dists)
        
        return np.mean([abs(corr_ce), abs(corr_cd), abs(corr_ed)])
    
    def _calculate_oov_handling(self, model, vocabulary: Optional[List[str]] = None) -> float:
        """Evaluate Out-of-Vocabulary handling capability."""
        if not self.test_compounds or not vocabulary:
            # Estimate based on model type
            if hasattr(model, 'wv') and hasattr(model.wv, 'vectors_ngrams'):
                return 0.8
            return 0.3
        
        oov_compounds = [c for c in self.test_compounds if c not in vocabulary]
        if not oov_compounds:
            return 1.0
        
        handled = sum(1 for compound in oov_compounds if self._can_get_embedding(model, compound))
        return handled / len(oov_compounds)
    
    def _can_get_embedding(self, model, word: str) -> bool:
        """Check if model can get embedding for word."""
        try:
            if hasattr(model, '__getitem__'):
                _ = model[word]
            elif hasattr(model, 'wv'):
                _ = model.wv[word]
            else:
                return False
            return True
        except:
            return False
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        vocabulary: Optional[List[str]] = None,
        total_words: Optional[int] = None,
        model: Optional[Any] = None
    ) -> EvaluationMetrics:
        """Comprehensive evaluation for similarity-based tasks."""
        min_samples = max(self.n_clusters, self.k_neighbors + 1)
        if len(embeddings) < min_samples:
            self.logger.warning(f"Not enough samples ({len(embeddings)}) for evaluation")
            return EvaluationMetrics(0.0, 0.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Clustering metrics
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        
        # Vocabulary coverage
        vocab_coverage = 1.0
        if vocabulary and total_words:
            vocab_coverage = len(vocabulary) / total_words
        
        # OOV handling
        oov_handling = self._calculate_oov_handling(model, vocabulary)
        
        # Similarity metrics
        cosine_cons, euclidean_cons, dot_cons = self._calculate_similarity_consistency(embeddings)
        neighborhood_pres = self._calculate_neighborhood_preservation(embeddings)
        rank_corr = self._calculate_rank_correlation(embeddings)
        
        # Normalize clustering metrics
        silhouette_norm = (silhouette + 1) / 2
        calinski_norm = min(calinski / 1000, 1.0)
        davies_bouldin_norm = 1 / (1 + davies_bouldin)
        
        # Combined scores
        similarity_score = (
            0.25 * cosine_cons + 0.25 * euclidean_cons + 0.20 * dot_cons +
            0.15 * neighborhood_pres + 0.15 * rank_corr
        )
        
        intrinsic_score = (
            0.15 * silhouette_norm + 0.10 * calinski_norm + 0.10 * davies_bouldin_norm +
            0.10 * vocab_coverage + 0.15 * oov_handling + 0.40 * similarity_score
        )
        
        return EvaluationMetrics(
            silhouette, calinski, davies_bouldin, vocab_coverage, oov_handling,
            cosine_cons, euclidean_cons, dot_cons, neighborhood_pres, rank_corr,
            similarity_score, intrinsic_score
        )


class BaseEmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.pca = None
        self.logger = LoggerFactory.setup_logger("BaseEmbeddingModel", log_to_file=True)
    
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
            'workers': self.config.custom_params.get('workers', 4),
        }
        self.model = Word2Vec(sentences=sentences, **params)
        self.logger.info(f"Word2Vec trained: {len(self.model.wv)} words")
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        if sentences:
            # Average word vectors for sentences
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
            'workers': self.config.custom_params.get('workers', 4),
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


class GloVeModel(BaseEmbeddingModel):
    """GloVe embedding model with sentence support - FIXED."""
    
    GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
    GLOVE_FILENAME = "glove.6B.300d.txt"
    WORD2VEC_FILENAME = "glove.6B.300d.word2vec.txt"
    
    # Class-level cache for shared embeddings
    _cached_embeddings = {}
    _cache_lock = None
    
    @classmethod
    def _get_cache_lock(cls):
        """Get or create thread lock for cache."""
        if cls._cache_lock is None:
            import threading
            cls._cache_lock = threading.Lock()
        return cls._cache_lock
    
    @classmethod
    def get_or_load_glove(cls, glove_dir: str = "./glove") -> KeyedVectors:
        """Load GloVe once and cache it (thread-safe)."""
        glove_dir = str(glove_dir)
        
        # Check cache first (fast path, no lock needed)
        if glove_dir in cls._cached_embeddings:
            return cls._cached_embeddings[glove_dir]
        
        # Acquire lock for loading
        lock = cls._get_cache_lock()
        with lock:
            # Double-check after acquiring lock
            if glove_dir in cls._cached_embeddings:
                return cls._cached_embeddings[glove_dir]
            
            # Load GloVe
            logger = LoggerFactory.setup_logger("GloVeLoader", log_to_file=True)
            os.makedirs(glove_dir, exist_ok=True)
            
            glove_path = os.path.join(glove_dir, cls.GLOVE_FILENAME)
            w2v_path = os.path.join(glove_dir, cls.WORD2VEC_FILENAME)
            
            # Download if needed
            if not os.path.exists(glove_path):
                zip_path = os.path.join(glove_dir, "glove.6B.zip")
                logger.info(f"Downloading GloVe to {zip_path}...")
                wget.download(cls.GLOVE_URL, zip_path)
                logger.info("============ Extracting... ============")
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(glove_dir)
                logger.info("Extraction complete")
            
            # Convert to word2vec format if needed
            if not os.path.exists(w2v_path):
                logger.info("Converting GloVe to word2vec format...")
                glove2word2vec(glove_path, w2v_path)
                logger.info("Conversion complete")
            
            # Load embeddings
            logger.info(f"Loading GloVe from {w2v_path}...")
            embeddings = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
            logger.info(f"GloVe loaded: {len(embeddings.key_to_index)} words")
            
            # Cache it
            cls._cached_embeddings[glove_dir] = embeddings
            return embeddings
    
    def train(self, sentences: List[List[str]], glove_dir: str = "./glove") -> None:
        """Load GloVe embeddings using cache."""
        # Input validation
        if not isinstance(glove_dir, (str, os.PathLike)):
            raise TypeError(f"glove_dir must be string/path, got {type(glove_dir)}")
        
        # Use cached loader
        self.model = self.get_or_load_glove(glove_dir)
        self.logger.info(f"Using cached GloVe: {len(self.model.key_to_index)} words")
        
        # Store sentences for later use
        self.sentences = [' '.join(sent) for sent in sentences]
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        """Get sentence embeddings by averaging word vectors."""
        if sentences:
            texts = sentences
        else:
            texts = getattr(self, 'sentences', [])
            if not texts:
                # Fallback to all word embeddings
                return self.apply_pca(self.model.vectors)
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vectors = [self.model[w] for w in words if w in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.model.vector_size))
        
        embeddings = np.array(embeddings)
        return self.apply_pca(embeddings)


class LSAModel(BaseEmbeddingModel):
    """LSA embedding model."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.vectorizer = None
        self.svd = None
        self.vocabulary = {}
    
    def train(self, sentences: List[List[str]]) -> None:
        documents = [' '.join(sentence) for sentence in sentences]
        
        # Vectorization
        use_tfidf = self.config.custom_params.get('use_tfidf', True)
        VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
        
        self.vectorizer = VectorizerClass(
            max_features=self.config.custom_params.get('max_features', None),
            min_df=self.config.custom_params.get('min_df', 2),
            max_df=self.config.custom_params.get('max_df', 0.95),
            ngram_range=self.config.custom_params.get('ngram_range', (1, 1)),
            lowercase=True
        )
        
        self.logger.info(f"Vectorizing {len(documents)} documents...")
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.vocabulary = self.vectorizer.vocabulary_
        
        # SVD
        self.svd = TruncatedSVD(
            n_components=self.config.vector_size,
            random_state=42,
            n_iter=self.config.custom_params.get('n_iter', 10)
        )
        
        document_embeddings = self.svd.fit_transform(doc_term_matrix)
        term_embeddings = self.svd.components_.T
        
        self.model = {
            'term_embeddings': term_embeddings,
            'document_embeddings': document_embeddings,
            'vocabulary': self.vocabulary,
            'vectorizer': self.vectorizer,
            'svd': self.svd
        }
        
        explained_var = self.svd.explained_variance_ratio_.sum()
        self.logger.info(f"LSA: {len(self.vocabulary)} terms, var={explained_var:.2%}")
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        if sentences:
            doc_term_matrix = self.vectorizer.transform(sentences)
            embeddings = self.svd.transform(doc_term_matrix)
        else:
            embeddings = self.model['document_embeddings']
        
        return self.apply_pca(embeddings)


class TransformerModel(BaseEmbeddingModel):
    """Transformer-based embedding models."""
    
    MODEL_NAMES = {
        ModelType.BIOBERT: "dmis-lab/biobert-v1.1",
        ModelType.PUBMEDBERT: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        ModelType.SCIBERT: "allenai/scibert_scivocab_uncased",
        ModelType.SBERT: "sentence-transformers/all-MiniLM-L6-v2",
        ModelType.BIOCLINICALBERT: "emilyalsentzer/Bio_ClinicalBERT",
    }
    
    def train(self, sentences: List[List[str]]) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        model_name = self.MODEL_NAMES.get(self.config.model_type)
        if not model_name:
            raise ValueError(f"Unknown model: {self.config.model_type}")
        
        self.logger.info(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.sentences = [' '.join(sent) for sent in sentences]
    
    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        texts = sentences if sentences else self.sentences
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return self.apply_pca(embeddings)


class ModelFactory:
    """Factory for creating embedding models."""
    
    @staticmethod
    def create_model(config: EmbeddingConfig) -> BaseEmbeddingModel:
        model_map = {
            ModelType.WORD2VEC: Word2VecModel,
            ModelType.FASTTEXT: FastTextModel,
            ModelType.GLOVE: GloVeModel,
            ModelType.LSA: LSAModel,
            ModelType.BIOBERT: TransformerModel,
            ModelType.PUBMEDBERT: TransformerModel,
            ModelType.SCIBERT: TransformerModel,
            ModelType.SBERT: TransformerModel,
            ModelType.BIOCLINICALBERT: TransformerModel,
        }
        
        model_class = model_map.get(config.model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        return model_class(config)


def _train_and_evaluate_config(
    config: EmbeddingConfig,
    sentences: List[List[str]],
    evaluator_params: Dict[str, Any],
    worker_id: int
) -> Tuple[EmbeddingConfig, EvaluationMetrics, float]:
    """Worker function for parallel execution."""
    logger = LoggerFactory.setup_logger("Worker", log_to_file=True)
    start_time = time.time()
    
    try:
        logger.info(f"[Worker {worker_id}] Evaluating {config.model_type.value}")
        
        model = ModelFactory.create_model(config)
        model.train(sentences)
        embeddings = model.get_embeddings()
        
        vocabulary = None
        if hasattr(model.model, 'wv'):
            vocabulary = list(model.model.wv.key_to_index.keys())
        elif hasattr(model.model, 'key_to_index'):
            vocabulary = list(model.model.key_to_index.keys())
        
        total_words = len(set(word for sent in sentences for word in sent))
        
        evaluator = EmbeddingEvaluator(**evaluator_params)
        metrics = evaluator.evaluate(embeddings, vocabulary, total_words, model.model)
        
        training_time = time.time() - start_time
        logger.info(f"[Worker {worker_id}] Model: {config.model_type.value} Score: {metrics.intrinsic_score:.4f}, Time: {training_time:.2f}s")
        
        return config, metrics, training_time
        
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error: {e}")
        return config, EvaluationMetrics(0.0, 0.0, float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), time.time() - start_time


class ParallelModelSelector:
    """Parallel model selection with efficient search."""
    
    def __init__(
        self,
        candidate_models: List[ModelType],
        use_pca_variants: bool = True,
        evaluator: Optional[EmbeddingEvaluator] = None,
        n_jobs: int = -1,
    ):
        self.candidate_models = candidate_models
        self.use_pca_variants = use_pca_variants
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.n_jobs = max(1, cpu_count() - 1 if n_jobs == -1 else min(n_jobs, cpu_count()))
        self.logger = LoggerFactory.setup_logger("ParallelModelSelector", log_to_file=True)
        self.results = []
        
        self.logger.info(f"ParallelModelSelector: {self.n_jobs} workers")
    
    def select_best_model(
        self, sentences: List[List[str]], vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """Select best model using parallel search."""
        self.logger.info("="*80)
        self.logger.info("PARALLEL MODEL SELECTION")
        self.logger.info(f"Models: {[m.value for m in self.candidate_models]}")
        self.logger.info(f"Workers: {self.n_jobs}")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        # Phase 1: Quick screening
        configs = [EmbeddingConfig(model_type=mt, use_pca=False, vector_size=vector_size) 
                   for mt in self.candidate_models]
        
        evaluator_params = {
            'n_clusters': self.evaluator.n_clusters,
            'k_neighbors': self.evaluator.k_neighbors,
            'test_compounds': self.evaluator.test_compounds,
            'random_state': self.evaluator.random_state,
        }
        
        screening_results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(_train_and_evaluate_config, cfg, sentences, evaluator_params, i): cfg
                for i, cfg in enumerate(configs)
            }
            
            for future in as_completed(futures):
                try:
                    config, metrics, train_time = future.result()
                    screening_results.append({
                        'config': config, 'metrics': metrics, 
                        'score': metrics.intrinsic_score, 'training_time': train_time
                    })
                    self.logger.info(f"✓ {config.model_type.value}: {metrics.intrinsic_score:.3f}")
                except Exception as e:
                    self.logger.error(f"Failed: {e}")
        
        screening_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Phase 2: Detailed evaluation of top models
        top_k = min(3, len(screening_results))
        detailed_configs = []
        for result in screening_results[:top_k]:
            mt = result['config'].model_type
            for use_pca in ([False, True] if self.use_pca_variants else [False]):
                detailed_configs.append(EmbeddingConfig(
                    model_type=mt, use_pca=use_pca, 
                    pca_components=50 if use_pca else None, vector_size=vector_size
                ))
        
        detailed_results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(_train_and_evaluate_config, cfg, sentences, evaluator_params, i): cfg
                for i, cfg in enumerate(detailed_configs)
            }
            
            for future in as_completed(futures):
                try:
                    config, metrics, train_time = future.result()
                    detailed_results.append({
                        'config': config, 'metrics': metrics,
                        'score': metrics.intrinsic_score, 'training_time': train_time
                    })
                    pca_str = f"PCA={config.use_pca}"
                    self.logger.info(f"✓ {config.model_type.value} ({pca_str}): {metrics.intrinsic_score:.3f}")
                except Exception as e:
                    self.logger.error(f"Failed: {e}")
        
        # Combine and find best
        self.results = screening_results + detailed_results
        best_result = max(self.results, key=lambda x: x['score'])
        
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Best: {best_result['config'].model_type.value}")
        self.logger.info(f"Score: {best_result['score']:.4f}")
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


class ParallelHyperparameterOptimizer:
    """Parallel hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        model_config: EmbeddingConfig,
        evaluator: Optional[EmbeddingEvaluator] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage_path: Optional[str] = None,
        n_jobs: int = -1,
    ):
        self.model_config = model_config
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optim_{model_config.model_type.value}"
        
        if storage_path is None:
            storage_path = f"./{self.study_name}.db"
        self.storage = f"sqlite:///{storage_path}"
        
        self.n_jobs = max(1, cpu_count() - 1 if n_jobs == -1 else min(n_jobs, cpu_count()))
        self.logger = LoggerFactory.setup_logger("ParallelHyperparameterOptimizer", log_to_file=True)
        self.study = None
        self.logger.info(f"ParallelHyperparameterOptimizer: {self.n_jobs} workers")
    
    def _get_search_space(self, trial: Trial) -> Dict[str, Any]:
        """Define search space based on model type."""
        params = {}
        
        if self.model_config.model_type in [ModelType.WORD2VEC, ModelType.FASTTEXT]:
            params.update({
                'vector_size': trial.suggest_categorical('vector_size', [100, 200, 300, 400]),
                'window': trial.suggest_int('window', 3, 10),
                'min_count': trial.suggest_int('min_count', 1, 5),
                'sg': trial.suggest_categorical('sg', [0, 1]),
                'negative': trial.suggest_int('negative', 5, 20),
                'alpha': trial.suggest_float('alpha', 0.001, 0.05, log=True),
                'epochs': trial.suggest_int('epochs', 10, 30),
            })
            if self.model_config.model_type == ModelType.FASTTEXT:
                params['min_n'] = trial.suggest_int('min_n', 2, 4)
                params['max_n'] = trial.suggest_int('max_n', 4, 7)
        
        elif self.model_config.model_type == ModelType.LSA:
            params.update({
                'max_features': trial.suggest_categorical('max_features', [5000, 10000, 20000]),
                'min_df': trial.suggest_int('min_df', 1, 5),
                'max_df': trial.suggest_float('max_df', 0.85, 0.95),
                'use_tfidf': trial.suggest_categorical('use_tfidf', [True, False]),
            })
        
        elif self.model_config.model_type in [ModelType.BIOBERT, ModelType.PUBMEDBERT, 
                                                ModelType.SCIBERT, ModelType.SBERT, 
                                                ModelType.BIOCLINICALBERT]:
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        
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
            self.logger.error(f"Trial error: {e}")
            return 0.0
    
    def optimize(self, sentences: List[List[str]]) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters in parallel."""
        self.logger.info("="*80)
        self.logger.info("PARALLEL HYPERPARAMETER OPTIMIZATION")
        self.logger.info(f"Model: {self.model_config.model_type.value}")
        self.logger.info(f"Trials: {self.n_trials}, Workers: {self.n_jobs}")
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
            n_jobs=self.n_jobs,
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


class ParallelEmbeddingAutoML:
    """Complete parallel AutoML pipeline for embeddings."""
    
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
        self.candidate_models = candidate_models or [
            ModelType.WORD2VEC, ModelType.FASTTEXT, ModelType.GLOVE,
            ModelType.LSA, ModelType.BIOBERT, ModelType.PUBMEDBERT
        ]
        self.use_pca_variants = use_pca_variants
        self.model_selection_time_budget = model_selection_time_budget
        self.hyperopt_trials = hyperopt_trials
        self.hyperopt_timeout = hyperopt_timeout
        self.output_dir = output_dir or Path('./automl_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_jobs = max(1, cpu_count() - 1 if n_jobs == -1 else min(n_jobs, cpu_count()))
        self.logger = LoggerFactory.setup_logger("ParallelEmbeddingAutoML", log_to_file=True)
        
        self.selected_model_config = None
        self.optimized_params = None
        self.final_model = None
        
        self.logger.info(f"ParallelEmbeddingAutoML: {self.n_jobs} workers")
    
    def run(
        self,
        sentences: List[List[str]],
        skip_model_selection: bool = False,
        initial_model_config: Optional[EmbeddingConfig] = None
    ) -> Tuple[BaseEmbeddingModel, EvaluationMetrics]:
        """Run complete parallel AutoML pipeline."""
        self.logger.info("\n" + "="*80)
        self.logger.info("PARALLEL EMBEDDING AUTOML PIPELINE")
        self.logger.info(f"Workers: {self.n_jobs}")
        self.logger.info("="*80)
        
        overall_start = time.time()
        
        # Phase 1: Model Selection
        if not skip_model_selection:
            self.logger.info("\n📊 PHASE 1: MODEL SELECTION")
            selector = ParallelModelSelector(
                candidate_models=self.candidate_models,
                use_pca_variants=self.use_pca_variants,
                n_jobs=self.n_jobs
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
        
        optimizer = ParallelHyperparameterOptimizer(
            model_config=self.selected_model_config,
            n_trials=self.hyperopt_trials,
            timeout=self.hyperopt_timeout,
            study_name=f"hyperopt_{self.selected_model_config.model_type.value}",
            storage_path=storage_path,
            n_jobs=self.n_jobs
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
            custom_params=self.optimized_params
        )
        
        self.final_model = ModelFactory.create_model(final_config)
        self.final_model.train(sentences)
        
        embeddings = self.final_model.get_embeddings()
        evaluator = EmbeddingEvaluator()
        final_metrics = evaluator.evaluate(embeddings)
        
        total_time = time.time() - overall_start
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("AUTOML PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Model: {final_config.model_type.value}")
        self.logger.info(f"PCA: {final_config.use_pca}")
        self.logger.info(f"Score: {final_metrics.intrinsic_score:.4f}")
        self.logger.info(f"Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        self.logger.info("="*80)
        
        # Save configuration
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
        self.logger.info(f"Config saved: {config_path}")
        
        return self.final_model, final_metrics
    
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


class ParallelEmbeddingTrainingAutoML:
    """Enhanced embedding training with parallel AutoML."""
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        automl_config: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
    ):
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        self.n_jobs = max(1, cpu_count() - 1 if n_jobs == -1 else min(n_jobs, cpu_count()))
        
        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_path = Path(f'{self.base_path}/models/automl')
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'candidate_models': [
                ModelType.WORD2VEC, ModelType.FASTTEXT, ModelType.GLOVE,
                ModelType.BIOBERT, ModelType.PUBMEDBERT
            ],
            'use_pca_variants': True,
            'model_selection_time_budget': 3600,
            'hyperopt_trials': 30,
            'hyperopt_timeout': 1800,
        }
        self.automl_config = {**default_config, **(automl_config or {})}
        
        self.logger = LoggerFactory.setup_logger(
            "ParallelEmbeddingTrainingAutoML",
            target_year=str(start_year),
            log_to_file=True
        )
        self._corpus_df = None
        
        self.logger.info(f"Initialized for {self.disease_name}")
        self.logger.info(f"Years: {start_year}-{end_year}")
        self.logger.info(f"Workers: {self.n_jobs}")
    
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
        """Run parallel AutoML pipeline."""
        try:
            sentences = self._prepare_sentences(year_filter or self.end_year)
            
            if not sentences or len(sentences) < 10:
                self.logger.error(f"Insufficient data: {len(sentences)} sentences")
                return False
            
            automl = ParallelEmbeddingAutoML(
                candidate_models=self.automl_config['candidate_models'],
                use_pca_variants=self.automl_config['use_pca_variants'],
                model_selection_time_budget=self.automl_config['model_selection_time_budget'],
                hyperopt_trials=self.automl_config['hyperopt_trials'],
                hyperopt_timeout=self.automl_config['hyperopt_timeout'],
                output_dir=self.models_path / f'{self.start_year}_{self.end_year}',
                n_jobs=self.n_jobs
            )
            
            initial_config = None
            if skip_model_selection and force_model_type:
                initial_config = EmbeddingConfig(
                    model_type=force_model_type, use_pca=False, vector_size=300
                )
            
            final_model, _ = automl.run(sentences, skip_model_selection, initial_config)
            
            model_path = self.models_path / f'model_{self.start_year}_{self.end_year}.model'
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
    automl_trainer = ParallelEmbeddingTrainingAutoML(
        disease_name="acute myeloid leukemia",
        start_year=1990,
        end_year=2000,
        automl_config={
            'candidate_models': [
                ModelType.WORD2VEC,
                ModelType.FASTTEXT,
                ModelType.GLOVE,
                ModelType.LSA,
                ModelType.BIOBERT,
                ModelType.PUBMEDBERT,
                ModelType.SCIBERT,
                ModelType.SBERT,
                ModelType.BIOCLINICALBERT
            ],
            'use_pca_variants': True,
            'model_selection_time_budget': 1800,
            'hyperopt_trials': 20,
            'hyperopt_timeout': 900,
        },
        n_jobs=-1
    )
    
    success = automl_trainer.run_automl()