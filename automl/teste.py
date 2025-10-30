

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

# FLAML for potential future integration (currently not used for unsupervised embeddings)
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    logging.warning("FLAML not available. Install with: pip install flaml")

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
    
    Metrics:
    1. Clustering Quality - How well embeddings cluster similar concepts
    2. Vocabulary Coverage - Percentage of terms embedded
    3. OOV Handling - Ability to handle unseen compounds (important!)
    4. Similarity Consistency - Stability of similarity metrics
    5. Neighborhood Preservation - Local structure preservation
    6. Rank Correlation - Consistency between distance metrics
    """
    
    def __init__(
        self, 
        n_clusters: int = 10, 
        k_neighbors: int = 10,
        test_compounds: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize evaluator.
        
        Args:
            n_clusters: Number of clusters for clustering metrics
            k_neighbors: K for neighborhood preservation
            test_compounds: Known compounds to test OOV handling
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.test_compounds = test_compounds or []
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def _calculate_similarity_consistency(
        self,
        embeddings: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """
        Calculate consistency of different similarity metrics.
        
        For similarity tasks, we want metrics that produce stable,
        meaningful rankings regardless of the distance function used.
        
        Returns:
            (cosine_consistency, euclidean_consistency, dot_product_consistency)
        """
        if len(embeddings) < n_samples:
            n_samples = len(embeddings)
        
        # Sample random pairs
        np.random.seed(self.random_state)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = sample_embeddings / norms
        
        # Calculate different similarity matrices
        # 1. Cosine similarity (normalized dot product)
        cosine_sim = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # 2. Dot product
        dot_product = np.dot(sample_embeddings, sample_embeddings.T)
        
        # 3. Euclidean distance (converted to similarity)
        from scipy.spatial.distance import pdist, squareform
        euclidean_dist = squareform(pdist(sample_embeddings, metric='euclidean'))
        euclidean_sim = 1 / (1 + euclidean_dist)  # Convert to similarity
        
        # Calculate consistency (variance of similarities)
        # Lower variance = more consistent = better for downstream tasks
        cosine_std = np.std(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
        dot_std = np.std(dot_product[np.triu_indices_from(dot_product, k=1)])
        euclidean_std = np.std(euclidean_sim[np.triu_indices_from(euclidean_sim, k=1)])
        
        # Convert to scores (0-1 range, higher is better)
        # We want reasonable variance (not too high, not too low)
        cosine_consistency = 1 / (1 + cosine_std)
        dot_consistency = 1 / (1 + dot_std)
        euclidean_consistency = 1 / (1 + euclidean_std)
        
        return cosine_consistency, euclidean_consistency, dot_consistency
    
    def _calculate_neighborhood_preservation(
        self,
        embeddings: np.ndarray
    ) -> float:
        """
        Calculate how well local neighborhoods are preserved.
        
        Important for similarity tasks: similar compounds should have
        similar neighbors in embedding space.
        """
        if len(embeddings) < self.k_neighbors + 1:
            return 0.0
        
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-nearest neighbors using cosine similarity
        nbrs_cosine = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1, 
            metric='cosine'
        ).fit(embeddings)
        _, indices_cosine = nbrs_cosine.kneighbors(embeddings)
        
        # Find k-nearest neighbors using euclidean distance
        nbrs_euclidean = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1, 
            metric='euclidean'
        ).fit(embeddings)
        _, indices_euclidean = nbrs_euclidean.kneighbors(embeddings)
        
        # Calculate overlap between neighborhoods
        overlaps = []
        for i in range(len(embeddings)):
            neighbors_cosine = set(indices_cosine[i][1:])  # Exclude self
            neighbors_euclidean = set(indices_euclidean[i][1:])
            overlap = len(neighbors_cosine & neighbors_euclidean) / self.k_neighbors
            overlaps.append(overlap)
        
        # High overlap = consistent neighborhoods = good for similarity
        return np.mean(overlaps)
    
    def _calculate_rank_correlation(
        self,
        embeddings: np.ndarray,
        n_samples: int = 50
    ) -> float:
        """
        Calculate rank correlation between different distance metrics.
        
        For compound similarity, we want rankings to be consistent
        across dot product, cosine, and euclidean metrics.
        """
        if len(embeddings) < n_samples:
            n_samples = len(embeddings)
        
        from scipy.stats import spearmanr
        
        np.random.seed(self.random_state)
        reference_idx = np.random.choice(len(embeddings))
        reference = embeddings[reference_idx]
        
        # Sample points to compare
        sample_indices = np.random.choice(
            len(embeddings), 
            min(n_samples, len(embeddings)), 
            replace=False
        )
        sample_embeddings = embeddings[sample_indices]
        
        # Calculate distances using different metrics
        # 1. Cosine distance
        reference_norm = reference / (np.linalg.norm(reference) + 1e-8)
        sample_norms = sample_embeddings / (np.linalg.norm(sample_embeddings, axis=1, keepdims=True) + 1e-8)
        cosine_dists = 1 - np.dot(sample_norms, reference_norm)
        
        # 2. Euclidean distance
        euclidean_dists = np.linalg.norm(sample_embeddings - reference, axis=1)
        
        # 3. Dot product (as similarity, so negate for distance)
        dot_products = np.dot(sample_embeddings, reference)
        dot_dists = -dot_products  # Convert similarity to distance
        
        # Calculate rank correlations
        corr_cosine_euclidean, _ = spearmanr(cosine_dists, euclidean_dists)
        corr_cosine_dot, _ = spearmanr(cosine_dists, dot_dists)
        corr_euclidean_dot, _ = spearmanr(euclidean_dists, dot_dists)
        
        # Average correlation (higher = more consistent rankings)
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
        """
        Evaluate Out-of-Vocabulary handling capability.
        
        Critical for compounds: many compound names may not appear
        in training corpus. FastText handles this well, Word2Vec doesn't.
        """
        if not self.test_compounds or not vocabulary:
            # If no test compounds, check model type
            # FastText can handle OOV, Word2Vec cannot
            if hasattr(model, 'wv') and hasattr(model, 'wv.vectors_ngrams'):
                return 0.8  # FastText-like model
            else:
                return 0.3  # Word2Vec-like model
        
        # Test actual OOV compounds
        oov_compounds = [c for c in self.test_compounds if c not in vocabulary]
        if not oov_compounds:
            return 1.0  # All compounds in vocabulary
        
        handled = 0
        for compound in oov_compounds:
            try:
                # Try to get embedding for OOV word
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
        """
        Comprehensive evaluation for similarity-based tasks.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            vocabulary: List of words in vocabulary
            total_words: Total words in corpus
            model: Trained model (for OOV testing)
            
        Returns:
            EvaluationMetrics object with all scores
        """
        if len(embeddings) < max(self.n_clusters, self.k_neighbors + 1):
            self.logger.warning(
                f"Not enough samples ({len(embeddings)}) for evaluation"
            )
            return EvaluationMetrics(
                0.0, 0.0, float('inf'), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        
        # 1. Clustering quality metrics
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        
        # 2. Vocabulary coverage
        vocab_coverage = 1.0
        if vocabulary and total_words:
            vocab_coverage = len(vocabulary) / total_words
        
        # 3. OOV handling (critical for compounds!)
        oov_handling = self._calculate_oov_handling(model, vocabulary)
        
        # 4. Similarity consistency
        cosine_cons, euclidean_cons, dot_cons = self._calculate_similarity_consistency(
            embeddings
        )
        
        # 5. Neighborhood preservation
        neighborhood_pres = self._calculate_neighborhood_preservation(embeddings)
        
        # 6. Rank correlation
        rank_corr = self._calculate_rank_correlation(embeddings)
        
        # Normalize metrics to [0, 1]
        silhouette_norm = (silhouette + 1) / 2  # [-1, 1] -> [0, 1]
        calinski_norm = min(calinski / 1000, 1.0)  # Arbitrary scaling
        davies_bouldin_norm = 1 / (1 + davies_bouldin)  # Lower is better
        
        # Combined similarity score (optimized for similarity tasks)
        similarity_score = (
            0.25 * cosine_cons +
            0.25 * euclidean_cons +
            0.20 * dot_cons +
            0.15 * neighborhood_pres +
            0.15 * rank_corr
        )
        
        # Overall intrinsic score (balanced across all aspects)
        intrinsic_score = (
            0.15 * silhouette_norm +
            0.10 * calinski_norm +
            0.10 * davies_bouldin_norm +
            0.10 * vocab_coverage +
            0.15 * oov_handling +  # Important for compounds!
            0.40 * similarity_score  # Most important for your use case
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
        self.logger = logging.getLogger(__name__)
    
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
        
        # For transformers, we work with sentences, not individual words
        self.sentences = [' '.join(sent) for sent in sentences]
    
    def get_embeddings(self, words: Optional[List[str]] = None) -> np.ndarray:
        """Get sentence embeddings."""
        if words:
            # If specific words requested, treat each as a sentence
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


class ModelSelector:
    """
    AutoML-style Model Selection for Embeddings.
    
    Since FLAML/Auto-sklearn are designed for supervised learning,
    we implement a custom efficient search strategy for unsupervised
    embedding selection, inspired by FLAML's Cost-Frugal Optimization (CFO).
    
    Strategy:
    1. Quick screening: Test all models with default params
    2. Early pruning: Drop poor performers
    3. Detailed evaluation: Focus budget on top candidates
    4. PCA variants: Test dimensionality reduction on promising models
    
    This is TRUE AutoML (model selection + optimization), just adapted
    for unsupervised learning.
    """
    
    def __init__(
        self,
        candidate_models: List[ModelType],
        use_pca_variants: bool = True,
        evaluator: Optional[EmbeddingEvaluator] = None,
        time_budget: int = 3600,
    ):
        """
        Initialize model selector.
        
        Args:
            candidate_models: List of model types to evaluate
            use_pca_variants: If True, test each model with and without PCA
            evaluator: Custom evaluator (optional)
            time_budget: Total time budget in seconds (not enforced yet)
        """
        self.candidate_models = candidate_models
        self.use_pca_variants = use_pca_variants
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.time_budget = time_budget
        self.logger = logging.getLogger(__name__)
        
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')
        
        self.logger.info("Using custom efficient search for unsupervised embeddings")
        self.logger.info("(FLAML-style strategy adapted for embeddings without labels)")
    
    def _evaluate_config(
        self,
        config: EmbeddingConfig,
        sentences: List[List[str]]
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """Evaluate a single model configuration."""
        try:
            self.logger.info(f"Evaluating {config.model_type.value} (PCA={config.use_pca})")
            
            # Create and train model
            model = ModelFactory.create_model(config)
            model.train(sentences)
            
            # Get embeddings and vocabulary
            embeddings = model.get_embeddings()
            
            # Get vocabulary info
            vocabulary = None
            if hasattr(model.model, 'wv'):
                vocabulary = list(model.model.wv.key_to_index.keys())
            
            # Total words from sentences
            total_words = len(set(word for sent in sentences for word in sent))
            
            # Evaluate (pass model for OOV testing)
            metrics = self.evaluator.evaluate(
                embeddings, 
                vocabulary=vocabulary,
                total_words=total_words,
                model=model.model
            )
            
            self.logger.info(f"Scores - Similarity: {metrics.similarity_score:.4f}, "
                           f"Intrinsic: {metrics.intrinsic_score:.4f}, "
                           f"OOV: {metrics.oov_handling:.4f}")
            
            return config, metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {config.model_type.value}: {e}")
            # Return worst possible scores
            return config, EvaluationMetrics(
                0.0, 0.0, float('inf'), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
    
    def _efficient_search(
        self,
        sentences: List[List[str]],
        vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """
        Efficient search strategy inspired by FLAML's CFO (Cost-Frugal Optimization).
        
        Unlike grid search which tests all combinations equally,
        this strategy:
        1. Quickly identifies promising model types
        2. Focuses computational budget on best candidates
        3. Prunes poor performers early
        
        This is the core "AutoML" component - automatic model selection.
        
        Strategy:
        1. Quick screening of all models with default params
        2. Early stopping for poor performers
        3. Focus budget on promising models
        """
        self.logger.info("=== Phase 1: Quick Screening ===")
        screening_results = []
        
        for model_type in self.candidate_models:
            config = EmbeddingConfig(
                model_type=model_type,
                use_pca=False,
                vector_size=vector_size
            )
            
            _, metrics = self._evaluate_config(config, sentences)
            screening_results.append({
                'config': config,
                'metrics': metrics,
                'score': metrics.intrinsic_score
            })
            
            self.logger.info(
                f"{model_type.value}: "
                f"Score={metrics.intrinsic_score:.3f}, "
                f"Similarity={metrics.similarity_score:.3f}, "
                f"OOV={metrics.oov_handling:.3f}"
            )
        
        # Sort by score
        screening_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Phase 2: Detailed evaluation of top candidates
        top_k = min(3, len(screening_results))
        self.logger.info(f"\n=== Phase 2: Detailed Evaluation of Top {top_k} ===")
        
        detailed_results = []
        for i, result in enumerate(screening_results[:top_k], 1):
            model_type = result['config'].model_type
            self.logger.info(f"\n[{i}/{top_k}] Evaluating {model_type.value} variants")
            
            # Test with and without PCA
            for use_pca in [False, True] if self.use_pca_variants else [False]:
                config = EmbeddingConfig(
                    model_type=model_type,
                    use_pca=use_pca,
                    pca_components=50 if use_pca else None,
                    vector_size=vector_size
                )
                
                _, metrics = self._evaluate_config(config, sentences)
                detailed_results.append({
                    'config': config,
                    'metrics': metrics,
                    'score': metrics.intrinsic_score
                })
        
        # Combine all results
        self.results = screening_results + detailed_results
        
        # Find best
        best_result = max(self.results, key=lambda x: x['score'])
        self.best_config = best_result['config']
        self.best_score = best_result['score']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Best Configuration Found:")
        self.logger.info(f"  Model: {self.best_config.model_type.value}")
        self.logger.info(f"  PCA: {self.best_config.use_pca}")
        self.logger.info(f"  Score: {self.best_score:.4f}")
        self.logger.info(f"{'='*60}")
        
        return self.best_config, best_result['metrics']
    
    def _select_with_grid_search(
        self,
        sentences: List[List[str]],
        vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """
        Fallback: Exhaustive grid search (NOT AutoML, just systematic evaluation).
        """
        self.logger.info("Using grid search for model selection")
        
        configs_to_test = []
        
        # Generate all configurations
        for model_type in self.candidate_models:
            # Base config without PCA
            configs_to_test.append(
                EmbeddingConfig(
                    model_type=model_type,
                    use_pca=False,
                    vector_size=vector_size
                )
            )
            
            # Config with PCA if enabled
            if self.use_pca_variants:
                configs_to_test.append(
                    EmbeddingConfig(
                        model_type=model_type,
                        use_pca=True,
                        pca_components=50,
                        vector_size=vector_size
                    )
                )
        
        self.logger.info(f"Total configurations to test: {len(configs_to_test)}")
        
        # Evaluate all configurations
        for i, config in enumerate(configs_to_test, 1):
            self.logger.info(f"\n[{i}/{len(configs_to_test)}] Testing configuration...")
            
            config, metrics = self._evaluate_config(config, sentences)
            
            self.results.append({
                'config': config,
                'metrics': metrics,
                'score': metrics.intrinsic_score
            })
            
            # Update best
            if metrics.intrinsic_score > self.best_score:
                self.best_score = metrics.intrinsic_score
                self.best_config = config
                self.logger.info(f"🏆 New best model: {config.model_type.value} (PCA={config.use_pca})")
        
        best_result = next(r for r in self.results if r['config'] == self.best_config)
        return self.best_config, best_result['metrics']
    
    def select_best_model(
        self,
        sentences: List[List[str]],
        vector_size: int = 300
    ) -> Tuple[EmbeddingConfig, EvaluationMetrics]:
        """
        Select best model from candidates using efficient search.
        
        This is TRUE AutoML - automatically selects the best MODEL TYPE,
        not just hyperparameters.
        
        Args:
            sentences: Training sentences
            vector_size: Embedding dimension for models that support it
            
        Returns:
            Tuple of (best_config, best_metrics)
        """
        self.logger.info("="*60)
        self.logger.info("AUTOML: MODEL SELECTION")
        self.logger.info("="*60)
        self.logger.info(f"Strategy: Efficient Search (inspired by FLAML's CFO)")
        self.logger.info(f"Candidates: {[m.value for m in self.candidate_models]}")
        self.logger.info(f"PCA variants: {self.use_pca_variants}")
        self.logger.info(f"Evaluation: Optimized for similarity tasks")
        self.logger.info("="*60 + "\n")
        
        # Run efficient search
        best_config, best_metrics = self._efficient_search(sentences, vector_size)
        
        return best_config, best_metrics
    
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
                **metrics.to_dict()
            }
            data.append(row)
        
        return pd.DataFrame(data).sort_values('intrinsic_score', ascending=False)


class HyperparameterOptimizer:
    """
    AutoML for hyperparameter optimization using Optuna.
    Optimizes hyperparameters for a specific model type.
    """
    
    def __init__(
        self,
        model_config: EmbeddingConfig,
        evaluator: Optional[EmbeddingEvaluator] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            model_config: Base model configuration to optimize
            evaluator: Evaluation metrics calculator
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Optuna study name
            storage: Optuna storage URL
        """
        self.model_config = model_config
        self.evaluator = evaluator or EmbeddingEvaluator()
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optim_{model_config.model_type.value}"
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        self.best_params = None
        self.best_score = None
        self.study = None
    
    def _get_search_space(self, trial: Trial) -> Dict[str, Any]:
        """Define search space based on model type."""
        model_type = self.model_config.model_type
        
        # Common parameters for word-based models
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
            
            # FastText specific
            if model_type == ModelType.FASTTEXT:
                params['min_n'] = trial.suggest_int('min_n', 2, 4)
                params['max_n'] = trial.suggest_int('max_n', 4, 7)
        
        # Transformer models - less to tune (mostly pre-trained)
        elif model_type in [ModelType.BIOBERT, ModelType.PUBMEDBERT, 
                           ModelType.SCIBERT, ModelType.SBERT, ModelType.BIOCLINICALBERT]:
            params = {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            }
        
        else:
            params = {}
        
        # PCA optimization if enabled
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
            # Get hyperparameters for this trial
            params = self._get_search_space(trial)
            
            # Update config with trial params
            config = EmbeddingConfig(
                model_type=self.model_config.model_type,
                use_pca=self.model_config.use_pca,
                pca_components=params.get('pca_components'),
                vector_size=params.get('vector_size', self.model_config.vector_size),
                custom_params=params
            )
            
            # Create and train model
            model = ModelFactory.create_model(config)
            model.train(sentences)
            
            # Get embeddings
            embeddings = model.get_embeddings()
            
            # Evaluate
            metrics = self.evaluator.evaluate(embeddings)
            
            # Report intermediate value for pruning
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
        Optimize hyperparameters.
        
        Args:
            sentences: Training sentences
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self.logger.info("="*60)
        self.logger.info("Starting Hyperparameter Optimization")
        self.logger.info(f"Model: {self.model_config.model_type.value}")
        self.logger.info(f"Trials: {self.n_trials}")
        self.logger.info("="*60)
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, sentences),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Optimization Complete")
        self.logger.info("="*60)
        self.logger.info(f"Best Score: {self.best_score:.4f}")
        self.logger.info("Best Parameters:")
        for param, value in self.best_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info("="*60)
        
        return self.best_params, self.best_score
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not self.study:
            return pd.DataFrame()
        
        df = self.study.trials_dataframe()
        return df.sort_values('value', ascending=False)


class EmbeddingAutoML:
    """
    Complete AutoML pipeline for embeddings.
    1. Model Selection
    2. Hyperparameter Optimization
    """
    
    def __init__(
        self,
        candidate_models: Optional[List[ModelType]] = None,
        use_pca_variants: bool = True,
        model_selection_time_budget: int = 3600,
        hyperopt_trials: int = 50,
        hyperopt_timeout: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize AutoML pipeline.
        
        Args:
            candidate_models: Models to consider (None = all available)
            use_pca_variants: Test PCA variants during selection
            model_selection_time_budget: Time budget for model selection
            hyperopt_trials: Number of hyperparameter optimization trials
            hyperopt_timeout: Timeout for hyperparameter optimization
            output_dir: Directory to save results
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
        
        self.logger = logging.getLogger(__name__)
        
        # Results
        self.selected_model_config = None
        self.optimized_params = None
        self.final_model = None
    
    def run(
        self,
        sentences: List[List[str]],
        skip_model_selection: bool = False,
        initial_model_config: Optional[EmbeddingConfig] = None
    ) -> Tuple[BaseEmbeddingModel, EvaluationMetrics]:
        """
        Run complete AutoML pipeline.
        
        Args:
            sentences: Training sentences
            skip_model_selection: If True, skip model selection phase
            initial_model_config: Starting config (required if skipping selection)
            
        Returns:
            Tuple of (final_model, evaluation_metrics)
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EMBEDDING AUTOML PIPELINE")
        self.logger.info("="*80)
        
        # Phase 1: Model Selection
        if not skip_model_selection:
            self.logger.info("\n📊 PHASE 1: MODEL SELECTION")
            self.logger.info("-"*80)
            
            selector = ModelSelector(
                candidate_models=self.candidate_models,
                use_pca_variants=self.use_pca_variants,
                time_budget=self.model_selection_time_budget
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
        
        # Phase 2: Hyperparameter Optimization
        self.logger.info("\n🔧 PHASE 2: HYPERPARAMETER OPTIMIZATION")
        self.logger.info("-"*80)
        
        optimizer = HyperparameterOptimizer(
            model_config=self.selected_model_config,
            n_trials=self.hyperopt_trials,
            timeout=self.hyperopt_timeout,
            study_name=f"hyperopt_{self.selected_model_config.model_type.value}",
            storage=f"sqlite:///{self.output_dir}/optuna_study.db"
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
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("AUTOML PIPELINE COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Selected Model: {final_config.model_type.value}")
        self.logger.info(f"Use PCA: {final_config.use_pca}")
        self.logger.info(f"Final Score: {final_metrics.intrinsic_score:.4f}")
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
            'final_metrics': final_metrics.to_dict()
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
        
        # Save based on model type
        if hasattr(self.final_model.model, 'save'):
            self.final_model.model.save(str(path))
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.final_model, f)
        
        self.logger.info(f"Model saved to {path}")


# Integration with existing EmbeddingTraining class
class EmbeddingTrainingAutoML:
    """
    Enhanced EmbeddingTraining with AutoML capabilities.
    Integrates with the existing pipeline.
    """
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        automl_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize with AutoML support.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year for training data
            end_year: Ending year for training data
            automl_config: AutoML configuration dictionary
        """
        from utils import LoggerFactory, normalize_disease_name
        
        self.logger = LoggerFactory.setup_logger(
            "embedding_automl_training",
            target_year=str(start_year),
            log_to_file=True
        )
        
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        
        # Paths
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
        
        self.logger.info(f"EmbeddingTrainingAutoML initialized for {self.disease_name}")
        self.logger.info(f"Years: {start_year}-{end_year}")
    
    def _load_corpus(self) -> Optional[pd.DataFrame]:
        """Load corpus (same as original class)."""
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
            
            self.logger.info(f"Loaded {len(df)} abstracts")
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
        """Prepare sentences for training."""
        df = self.corpus_df
        
        if df is None or df.empty:
            return []
        
        if year_filter and 'year_extracted' in df.columns:
            df = df[df['year_extracted'] <= year_filter]
        
        abstracts = df['summary'].dropna().tolist()
        self.logger.info(f"Preparing {len(abstracts)} abstracts for training")
        
        sentences = [abstract.split() for abstract in abstracts if abstract]
        sentences = [s for s in sentences if len(s) > 0]
        
        return sentences
    
    def run_automl(
        self,
        year_filter: Optional[int] = None,
        skip_model_selection: bool = False,
        force_model_type: Optional[ModelType] = None
    ) -> bool:
        """
        Run AutoML pipeline.
        
        Args:
            year_filter: Filter abstracts up to this year
            skip_model_selection: Skip model selection phase
            force_model_type: Force specific model type
            
        Returns:
            True if successful
        """
        try:
            # Prepare data
            sentences = self._prepare_sentences(year_filter or self.end_year)
            
            if not sentences:
                self.logger.error("No sentences available for training")
                return False
            
            if len(sentences) < 10:
                self.logger.warning(f"Corpus too small: {len(sentences)} sentences")
                return False
            
            # Initialize AutoML
            automl = EmbeddingAutoML(
                candidate_models=self.automl_config['candidate_models'],
                use_pca_variants=self.automl_config['use_pca_variants'],
                model_selection_time_budget=self.automl_config['model_selection_time_budget'],
                hyperopt_trials=self.automl_config['hyperopt_trials'],
                hyperopt_timeout=self.automl_config['hyperopt_timeout'],
                output_dir=self.models_path / f'{self.start_year}_{self.end_year}'
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
            
            self.logger.info("AutoML pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error in AutoML pipeline: {e}")
            return False
    
    def run_year_over_year(self, step: int = 1) -> bool:
        """
        Run AutoML year-over-year.
        
        Args:
            step: Year increment
            
        Returns:
            True if successful
        """
        for year in range(self.start_year, self.end_year + 1, step):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"AutoML for year range {self.start_year}-{year}")
            self.logger.info(f"{'='*80}\n")
            
            success = self.run_automl(year_filter=year)
            
            if not success:
                self.logger.error(f"Failed for year {year}")
                return False
        
        return True


# Example usage and testing
if __name__ == '__main__':
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example 1: Full AutoML Pipeline
    print("\n" + "="*80)
    print("EXAMPLE 1: Full AutoML Pipeline")
    print("="*80)
    
    automl_trainer = EmbeddingTrainingAutoML(
        disease_name="acute myeloid leukemia",
        start_year=2024,
        end_year=2025,
        automl_config={
            'candidate_models': [
                ModelType.WORD2VEC,
                ModelType.FASTTEXT,
                ModelType.PUBMEDBERT,  # Medical domain specific
            ],
            'use_pca_variants': True,
            'model_selection_time_budget': 1800,  # 30 minutes
            'hyperopt_trials': 20,
            'hyperopt_timeout': 900,  # 15 minutes
        }
    )
    
    success = automl_trainer.run_automl()
    
    if success:
        print("\n✅ AutoML completed successfully!")
    else:
        print("\n❌ AutoML failed!")
    
    # Example 2: Skip Model Selection (optimize specific model)
    print("\n" + "="*80)
    print("EXAMPLE 2: Optimize Specific Model (Word2Vec)")
    print("="*80)
    
    automl_trainer_w2v = EmbeddingTrainingAutoML(
        disease_name="acute myeloid leukemia",
        start_year=2024,
        end_year=2025,
        automl_config={
            'hyperopt_trials': 30,
            'hyperopt_timeout': 1200,
        }
    )
    
    success = automl_trainer_w2v.run_automl(
        skip_model_selection=True,
        force_model_type=ModelType.WORD2VEC
    )
    
    # Example 3: Direct API usage
    print("\n" + "="*80)
    print("EXAMPLE 3: Direct AutoML API")
    print("="*80)
    
    # Sample sentences
    sample_sentences = [
        ["leukemia", "cancer", "blood", "cells"],
        ["treatment", "chemotherapy", "radiation"],
        ["diagnosis", "symptoms", "fever", "fatigue"],
        # ... more sentences
    ]
    
    # Run AutoML
    automl = EmbeddingAutoML(
        candidate_models=[ModelType.WORD2VEC, ModelType.FASTTEXT],
        use_pca_variants=True,
        hyperopt_trials=10,
        output_dir=Path('./automl_demo')
    )
    
    model, metrics = automl.run(sample_sentences)
    
    print(f"\nFinal Metrics: {metrics.to_dict()}")
    print(f"Best model type: {automl.selected_model_config.model_type.value}")