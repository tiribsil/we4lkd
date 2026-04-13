import os
import gc
import itertools
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

import numpy as np
import pandas as pd
from scipy.stats import qmc
from gensim.models import KeyedVectors, Word2Vec, FastText
from sklearn.decomposition import PCA
from mittens import Mittens, GloVe

from utils import get_logger, normalize_disease_name


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


class BaseEmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.pca = None
        self.end_year = self.config.end_year
        self.logger = get_logger(self.__class__.__name__)
    
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


class GloVeModel(BaseEmbeddingModel):
    """GloVe embedding model using mittens."""
    
    def train(self, sentences: List[List[str]]) -> None:
        from collections import defaultdict
        
        window = self.config.custom_params.get('window', 5)
        epochs = self.config.custom_params.get('epochs', 15)
        learning_rate = self.config.custom_params.get('alpha', 0.05)
        min_count = self.config.custom_params.get('min_count', 2)
        max_vocab_size = self.config.custom_params.get('max_vocab_size', 50000)
        
        # 1. Build vocab
        word_counts = defaultdict(int)
        for sent in sentences:
            for w in sent:
                word_counts[w] += 1
                
        # Limit vocab by min_count and sort by frequency
        valid_words = [(w, c) for w, c in word_counts.items() if c >= min_count]
        valid_words.sort(key=lambda x: x[1], reverse=True)
        
        # Cap vocabulary size to avoid OOM
        vocab = [w for w, c in valid_words[:max_vocab_size]]
        w2i = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        self.logger.info(f"GloVe Vocab size: {V} words (capped at {max_vocab_size})")
        
        if V == 0:
            self.logger.warning("Empty vocabulary for GloVe.")
            return

        # 2. Build co-occurrence matrix expected by mittens
        X = np.zeros((V, V), dtype=np.float32)
        
        for sent in sentences:
            n = len(sent)
            for i, w1 in enumerate(sent):
                if w1 not in w2i:
                    continue
                id1 = w2i[w1]
                
                start = max(0, i - window)
                end = min(n, i + window + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    w2 = sent[j]
                    if w2 not in w2i:
                        continue
                    id2 = w2i[w2]
                    
                    dist = abs(i - j)
                    # GloVe's typical weighting is 1.0 / distance
                    X[id1, id2] += 1.0 / dist
                    
        self.logger.info(f"GloVe Co-occurrence built: {X.shape}")
        
        # 3. Train GloVe using mittens
        glove_model = GloVe(n=self.config.vector_size, max_iter=epochs, learning_rate=learning_rate)
        embeddings = glove_model.fit(X)
        self.logger.info(f"GloVe trained: {V} words, dims={self.config.vector_size}")
        
        # 4. Wrap in Gensim's KeyedVectors
        kv = KeyedVectors(vector_size=self.config.vector_size)
        kv.add_vectors(vocab, embeddings)
        self.model = kv

    def get_embeddings(self, sentences: Optional[List[str]] = None) -> np.ndarray:
        if sentences:
            embeddings = []
            for sentence in sentences:
                words = sentence.lower().split()
                # self.model is a KeyedVectors instance
                vectors = [self.model[w] for w in words if w in self.model]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.config.vector_size))
            embeddings = np.array(embeddings)
        else:
            embeddings = self.model.vectors
            
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


class CandidateModelTraining:
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        use_lhs: bool = True,
        num_combinations: int = 7,
        use_glove: bool = False
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        self.use_lhs = use_lhs
        self.num_combinations = num_combinations
        self.use_glove = use_glove
        
        self.model_combinations: Dict[str, List[Any]] = {}
        self.model_combinations.update({
            "w2v_berto_et_al": [200, 5, 2, 1, 15, 0.025, 15, 4, 0.75, 0.001],
        })

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
        
        if self.corpus_path.is_dir():
            csv_files = list(self.corpus_path.glob('*.csv'))
            if not csv_files: return None
        else:
            csv_files = [self.corpus_path]

        all_dfs = []
        for path in csv_files:
            try:
                all_dfs.append(pd.read_csv(path))
            except Exception:
                self.logger.warning(f"Standard read failed for {path.name}. Using rsplit fallback.")
                raw_data = []
                with open(path, 'r', encoding='utf-8') as f:
                    f.readline() # Skip header
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.rsplit(',', 1)
                        if len(parts) == 2:
                            raw_data.append({'summary': parts[0].strip().strip('"'), 'year_extracted': parts[1].strip()})
                if raw_data:
                    df_fallback = pd.DataFrame(raw_data)
                    df_fallback['year_extracted'] = pd.to_numeric(df_fallback['year_extracted'], errors='coerce')
                    all_dfs.append(df_fallback)

        if not all_dfs: return None
        df = pd.concat(all_dfs, ignore_index=True)
            
        if 'summary' not in df.columns:
            self.logger.error("Column 'summary' not found in corpus")
            return None
        
        if 'year_extracted' not in df.columns:
            if 'year' in df.columns:
                df['year_extracted'] = df['year']
            else:
                df['year_extracted'] = self.end_year # Fallback
        
        return df

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
                'vector_size': params[0], 
                'window': params[1], 
                'min_count': params[2],
                'sg': params[3], 
                'negative': params[4], 
                'alpha': params[5],
                'epochs': params[6], 
                'workers': params[7],
                'ns_exponent': params[8],
                'sample': params[9]
            }
        elif architecture == 'ft':
            model_type = ModelType.FASTTEXT
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'min_count': params[2],
                'sg': params[3], 'negative': params[4], 'alpha': params[5],
                'epochs': params[6], 'workers': params[7], 'min_n': params[8], 'max_n': params[9],
                'ns_exponent': params[10], 'sample': params[11]
            }
        elif architecture == 'glove':
            model_type = ModelType.GLOVE
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'min_count': params[2],
                'alpha': params[3], 'epochs': params[4]
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
            self.logger.info(f"Model exists: {model_filename}")
            return True

        # Preparar dados
        sentences = self._prepare_sentences(self.start_year, target_year)
        if not sentences:
            self.logger.error(f"No data: {self.start_year}-{target_year}")
            return False

        # Configurar e Treinar
        config = self._create_config_from_params(architecture, params, target_year)
        if not config:
            self.logger.error(f"Unknown architecture for {model_name}")
            return False

        try:
            self.logger.info(f"Training {model_name} ({target_year})")
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
        elif isinstance(model_instance.model, KeyedVectors):
            model_instance.model.save_word2vec_format(str(model_path), binary=True)
        else:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance.model, f)
        
        self.logger.info(f"Saved: {model_filename}")

    def _scale_and_cast(self, sample: np.ndarray, low: float, high: float, dtype: str):
        """Scale sample in [0,1) to [low, high] and cast according to dtype ('int'|'float'|'bin')."""
        val = low + sample * (high - low)
        if dtype == 'int':
            v = int(np.round(val))
            v = max(int(low), min(int(high), v))
            return v
        elif dtype == 'bin':
            # threshold at 0.5
            return int(val >= 0.5)
        else:
            return float(val)

    def _generate_lhs_samples(self, param_specs: List[tuple], n_samples: int) -> List[Dict[str, Any]]:
        """
        param_specs: list of tuples (name, low, high, dtype)
        dtype: 'int', 'float', 'bin'
        returns list of dicts mapping param name to sampled value
        """
        d = len(param_specs)
        sampler = qmc.LatinHypercube(d=d, seed=None)
        raw = sampler.random(n=n_samples)  # shape (n_samples, d)
        out = []
        for i in range(n_samples):
            row = {}
            for j, (name, low, high, dtype) in enumerate(param_specs):
                row[name] = self._scale_and_cast(raw[i, j], low, high, dtype)
            out.append(row)
        return out

    def _generate_grid_samples(self, param_specs: List[tuple], n_samples: int) -> List[Dict[str, Any]]:
        """
        Gera amostras de forma sistemática (grade simples).
        Se n_samples > 1, distribui os níveis entre os parâmetros.
        """
        variable_params = [(name, low, high, dtype) for name, low, high, dtype in param_specs if low != high]
        
        if not variable_params or n_samples == 1:
            # Retorna apenas os valores centrais/fixos
            base_row = {}
            for name, low, high, dtype in param_specs:
                if low == high:
                    base_row[name] = low
                else:
                    base_row[name] = self._scale_and_cast(0.5, low, high, dtype)
            return [base_row] * n_samples

        # Para uma grade simples que atenda exatamente n_samples:
        # Vamos variar um parâmetro de cada vez, ou distribuir n_samples níveis no parâmetro mais "relevante"
        # Para ser verdadeiramente uma grade, vamos distribuir n_samples igualmente.
        # Mas para simplificar e garantir n_samples modelos únicos, vamos variar o parâmetro 'epochs' ou similar.
        
        # Estratégia: Pegar o primeiro parâmetro variável e dar a ele n_samples níveis.
        # Os outros ficam no ponto médio.
        
        out = []
        for i in range(n_samples):
            row = {}
            # Nível normalizado entre 0 e 1
            level = i / (n_samples - 1) if n_samples > 1 else 0.5
            
            first_var_found = False
            for name, low, high, dtype in param_specs:
                if low == high:
                    row[name] = low
                elif not first_var_found:
                    row[name] = self._scale_and_cast(level, low, high, dtype)
                    first_var_found = True
                else:
                    # Ponto médio para os outros variáveis
                    row[name] = self._scale_and_cast(0.5, low, high, dtype)
            out.append(row)
            
        return out

    def _dict_to_list_params(self, architecture: str, hp: Dict[str, Any]) -> List[Any]:
        """Converte o dicionário do LHS para a lista ordenada esperada pelo _create_config_from_params."""
        if architecture == 'w2v':
            # Ordem: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, ns_exponent, sample]
            return [
                hp['vector_size'], hp['window'], hp['min_count'], hp['sg'], 
                hp['negative'], hp['alpha'], hp['epochs'], hp['workers'],
                hp['ns_exponent'], hp['sample']
            ]
        elif architecture == 'ft':
            # Ordem: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, min_n, max_n, ns_exponent, sample]
            return [
                hp['vector_size'], hp['window'], hp['min_count'], hp['sg'], 
                hp['negative'], hp['alpha'], hp['epochs'], hp['workers'],
                hp['min_n'], hp['max_n'], hp['ns_exponent'], hp['sample']
            ]
        elif architecture == 'glove':
            return [
                hp['vector_size'], hp['window'], hp['min_count'],
                hp['alpha'], hp['epochs']
            ]
        return []

    # ------------------------------------------------------------------
    # Analogies path: <project_root>/data/analogies.txt
    # (two levels up from modularization/)
    # ------------------------------------------------------------------
    _ANALOGIES_PATH = Path(__file__).parent.parent / "data" / "analogies.txt"

    def _load_analogies(self) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Parse data/analogies.txt into a dict mapping section name to a list
        of 4-tuples (a, b, c, d).

        File format (lines starting with ':' open a new section):

            : grammar
            man men woman women
            ...

            : biomedical
            aspirin analgesic metformin antidiabetic
            ...

        Blank lines and lines starting with '#' are ignored.
        """
        path = self._ANALOGIES_PATH
        if not path.exists():
            raise FileNotFoundError(f"Analogies file not found at {path}. It is required for evaluating models.")

        sections: Dict[str, List[Tuple[str, str, str, str]]] = {}
        current_section: Optional[str] = None

        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(":"):
                    current_section = line[1:].strip().lower()
                    sections.setdefault(current_section, [])
                    continue
                if current_section is None:
                    continue  # skip orphan lines before any section header
                parts = line.split()
                if len(parts) == 4:
                    sections[current_section].append(
                        (parts[0], parts[1], parts[2], parts[3])
                    )
                else:
                    self.logger.warning(
                        f"Skipping malformed analogy line (expected 4 words): '{line}'"
                    )

        total = sum(len(v) for v in sections.values())
        self.logger.info(
            f"Loaded {total} analogy tuples from {len(sections)} section(s): "
            + ", ".join(f"{k}({len(v)})" for k, v in sections.items())
        )
        return sections

    def _load_model_wv(self, model_key: str) -> Optional[KeyedVectors]:
        """
        Load a trained model's KeyedVectors from disk.
        Supports Word2Vec / FastText .model files saved by gensim.
        Returns None if the file cannot be loaded.
        """
        model_dir = self.models_base_path / model_key
        model_filename = f"{model_key}_{self.start_year}_{self.end_year}.model"
        model_path = model_dir / model_filename

        if not model_path.exists():
            self.logger.warning(f"Model file not found for analogy eval: {model_path}")
            return None

        arch = model_key.split("_")[0]
        try:
            if arch == "w2v":
                wv = Word2Vec.load(str(model_path)).wv
            elif arch == "ft":
                wv = FastText.load(str(model_path)).wv
            elif arch == "glove":
                wv = KeyedVectors.load(str(model_path))
            else:
                self.logger.warning(f"Unknown architecture for loading: {arch}")
                return None
            return wv
        except Exception as e:
            self.logger.error(f"Failed to load model {model_key}: {e}")
            return None

    def _score_analogies(
        self,
        wv: KeyedVectors,
        analogies: List[Tuple[str, str, str, str]],
        topn: int = 10,
    ) -> float:
        """
        Evaluate a list of analogy 4-tuples against a KeyedVectors object.

        For each (a, b, c, d) we ask: b - a + c ≈ ?
        A hit is counted when `d` appears in the top-`topn` most similar words.
        Skips tuples where any word is not in the vocabulary.

        Returns accuracy in [0, 1] (0 if no valid tuple at all).
        """
        hits = 0
        total = 0
        vocab = set(wv.key_to_index.keys())

        for a, b, c, d in analogies:
            if not all(w in vocab for w in (a, b, c, d)):
                continue  # skip OOV tuples silently
            total += 1
            try:
                results = wv.most_similar(positive=[b, c], negative=[a], topn=topn)
                predicted_words = {r[0] for r in results}
                if d in predicted_words:
                    hits += 1
            except Exception:
                pass  # rare numerical edge cases

        return hits / total if total > 0 else 0.0

    def _filter_best_by_analogy(
        self,
        candidates: Dict[str, List[Any]],
        include_worst: bool = True,
    ) -> Dict[str, List[Any]]:
        """
        Rank candidate models using word-embedding analogy tasks and return
        only the **best-scoring model per architecture** (e.g., one w2v,
        one ft, one glove).

        Normalization Strategy:
          - For each strategy (grammar, biomedical), find the best accuracy among models of the same architecture.
          - Divide each model's accuracy by that maximum to get a normalized [0, 1] score.
          - Final score is the mean of these normalized scores.

        Parameters
        ----------
        candidates:
            Dict mapping model key → params list.
        include_worst:
            If True, also include the model with the lowest overall normalized score.

        Returns
        -------
        Dict with the best model per architecture (and optionally the worst overall).
        """
        self.logger.info("=== Analogy-based model filtering (Normalized) ===")

        # Load analogy sets from file
        analogy_sections = self._load_analogies()
        grammar_analogies = analogy_sections.get("grammar", [])
        biomedical_analogies = analogy_sections.get("biomedical", [])

        if not grammar_analogies and not biomedical_analogies:
            self.logger.warning(
                "No analogy data loaded; skipping filter (returning all candidates)"
            )
            return candidates

        # Group candidates by architecture prefix (first segment before '_')
        arch_groups: Dict[str, List[str]] = {}
        for key in candidates:
            arch = key.split("_")[0]
            arch_groups.setdefault(arch, []).append(key)

        best_per_arch: Dict[str, List[Any]] = {}
        all_scored: List[Tuple[float, str]] = []

        for arch, keys in arch_groups.items():
            self.logger.info(f"Evaluating {len(keys)} {arch.upper()} candidate(s)")

            raw_metrics: Dict[str, Dict[str, float]] = {}
            fallback_key = keys[0]  # kept if nothing can be loaded

            for key in keys:
                wv = self._load_model_wv(key)
                if wv is None:
                    continue

                g_acc = self._score_analogies(wv, grammar_analogies)
                b_acc = self._score_analogies(wv, biomedical_analogies)
                raw_metrics[key] = {
                    "grammar": g_acc,
                    "biomedical": b_acc
                }

                # free memory immediately
                del wv
                gc.collect()

            if not raw_metrics:
                self.logger.warning(
                    f"All {arch.upper()} models failed to load; keeping {fallback_key} as fallback"
                )
                best_per_arch[fallback_key] = candidates[fallback_key]
                continue

            # Find max scores for this architecture to normalize
            max_g = max((m["grammar"] for m in raw_metrics.values()), default=0.0)
            max_b = max((m["biomedical"] for m in raw_metrics.values()), default=0.0)

            # Compute normalized mean scores
            arch_scored: List[Tuple[float, str]] = []
            for key, metrics in raw_metrics.items():
                norm_g = metrics["grammar"] / max_g if max_g > 0 else 0.0
                norm_b = metrics["biomedical"] / max_b if max_b > 0 else 0.0
                mean_norm = (norm_g + norm_b) / 2.0

                self.logger.info(
                    f"  {key}: grammar={metrics['grammar']:.3f} (norm={norm_g:.3f}), "
                    f"biomedical={metrics['biomedical']:.3f} (norm={norm_b:.3f}), "
                    f"final={mean_norm:.3f}"
                )
                arch_scored.append((mean_norm, key))
                all_scored.append((mean_norm, key))

            # Pick best for this architecture
            best_score, best_key = max(arch_scored, key=lambda x: x[0])
            self.logger.info(
                f"Best {arch.upper()}: {best_key} (score={best_score:.3f})"
            )
            best_per_arch[best_key] = candidates[best_key]

        if include_worst and all_scored:
            worst_score, worst_key = min(all_scored, key=lambda x: x[0])
            self.logger.info(
                f"Overall worst model identification: {worst_key} (score={worst_score:.3f})"
            )
            best_per_arch[worst_key] = candidates[worst_key]

        self.logger.info(
            f"Analogy filter: {len(candidates)} → {len(best_per_arch)} model(s) kept"
        )
        return best_per_arch


    def run(self) -> Dict[str, List[Any]]:
        """
        Executa LHS (Latin Hypercube Sampling) e treina modelos em paralelo.
        Salva os modelos em disco e popula self.model_combinations para uso futuro.
        """
        self.logger.info("=== Starting Candidate Training ===")
        
        # 1. Preparar sentenças internamente
        sentences = self._prepare_sentences(self.start_year, self.end_year)
        if not sentences:
            self.logger.error("No sentences found.")
            return {}

        n_sets = 7 # Amostras por família

        # 2. Definição dos Espaços de Parâmetros
        w2v_specs = [
            ('vector_size', 100, 100, 'int'), ('window', 3, 10, 'int'),
            ('min_count', 1, 5, 'int'), ('sg', 1, 1, 'int'), # Force SG
            ('negative', 10, 20, 'int'), ('alpha', 0.01, 0.05, 'float'),
            ('epochs', 50, 150, 'int'), ('workers', 4, 4, 'int'),
            ('ns_exponent', -1.0, 0.5, 'float'), ('sample', 1e-5, 1e-3, 'float'),
        ]

        ft_specs = w2v_specs + [ # Herda specs do w2v e adiciona específicos
            ('min_n', 2, 3, 'int'), ('max_n', 4, 6, 'int'),
        ]

        glove_specs = [
            ('vector_size', 100, 100, 'int'), ('window', 3, 10, 'int'),
            ('min_count', 1, 5, 'int'), ('alpha', 0.01, 0.1, 'float'),
            ('epochs', 50, 150, 'int')
        ]

        # 3. Gerar Amostras (LHS ou Grid)
        w2v_sets = []
        ft_sets = []
        glove_sets = []
        
        if self.use_lhs:
            w2v_sets = self._generate_lhs_samples(w2v_specs, self.num_combinations)
            ft_sets = self._generate_lhs_samples(ft_specs, self.num_combinations)
            if self.use_glove:
                glove_sets = self._generate_lhs_samples(glove_specs, self.num_combinations)
        else:
            w2v_sets = self._generate_grid_samples(w2v_specs, self.num_combinations)
            if self.use_glove:
                glove_sets = self._generate_grid_samples(glove_specs, self.num_combinations)

        # 4. Construir lista de tarefas
        tasks = []
        
        # Helper interno para preparar tarefa
        def add_tasks(arch, param_sets):
            for idx, hp in enumerate(param_sets, start=1):
                if arch == 'ft' and hp['max_n'] < hp['min_n']:
                    hp['max_n'], hp['min_n'] = hp['min_n'], hp['max_n']
                
                # Converte dict para lista ordenada usada pelo sistema
                params_list = self._dict_to_list_params(arch, hp)
                key = f"{arch}_comb{idx}"
                tasks.append((key, arch, params_list))

        add_tasks('w2v', w2v_sets)
        add_tasks('ft', ft_sets)
        add_tasks('glove', glove_sets)

        berto_params = self.model_combinations["w2v_berto_et_al"]
        tasks.append(("w2v_berto_et_al", "w2v", berto_params))

        # 5. Worker para execução paralela
        def _worker(task_data):
            model_key, arch, params = task_data
            try:
                # Verificar se o modelo já existe
                model_output_dir = self.models_base_path / model_key
                model_filename = f"{model_key}_{self.start_year}_{self.end_year}.model"
                model_path = model_output_dir / model_filename
                
                if model_path.exists():
                    self.logger.info(f"Model already exists, skipping training: {model_filename}")
                    return (model_key, params, None)

                # Usa a infraestrutura existente para criar config compatível
                config = self._create_config_from_params(arch, params, self.end_year)
                if not config:
                    return None

                # Treina usando ModelFactory (garante compatibilidade de Wrapper)
                model_instance = ModelFactory.create_model(config)
                model_instance.train(sentences)
                return (model_key, params, model_instance)
            except Exception as e:
                self.logger.error(f"Error training {model_key}: {e}")
                return None

        # 6. Executar e Salvar
        # Limita workers para evitar OOM (Out of Memory) já que modelos são pesados
        max_workers = min((os.cpu_count() or 4), 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(_worker, t): t[0] for t in tasks}
            
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                result = future.result()
                
                if result:
                    m_key, m_params, m_instance = result
                    
                    if m_instance is not None:
                        self._save_trained_model(m_instance, m_key, self.start_year, self.end_year)
                        self.logger.info(f"Finished & Saved: {m_key}")
                    
                    self.model_combinations[m_key] = m_params
                else:
                    self.logger.warning(f"Failed task: {key}")

        # Filter to best model per architecture using analogy tasks
        self.model_combinations = self._filter_best_by_analogy(
            self.model_combinations, include_worst=True
        )

        return self.model_combinations
