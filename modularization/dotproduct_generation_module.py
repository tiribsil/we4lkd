import os
import re
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText, KeyedVectors
from sklearn.decomposition import TruncatedSVD
from chembl_webresource_client.new_client import new_client
from functools import lru_cache
from utils import *
import warnings
import pickle


def _get_chembl_client():
    """Lazy import of ChEMBL client to avoid import-time errors."""
    try:
        from chembl_webresource_client.new_client import new_client
        return new_client
    except Exception as e:
        warnings.warn(f"ChEMBL client unavailable: {e}")
        return None


class ModelType(Enum):
    """Supported embedding model types."""
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"


class ValidationModule:
    """
    Module for validating embedding models by computing similarities
    between therapeutic compounds and disease embeddings over time.
    Supports Word2Vec, FastText and GloVe models.
    """
    
    # Blacklist padrão de biomoléculas genéricas
    DEFAULT_BIOMOLECULE_BLACKLIST = frozenset({
        'thymidine', 'deoxycytidine', 'uridine', 'cytidine', 'adenosine', 
        'guanine', 'cytosine', 'thymine', 'aminoacids', 'glutathione', 
        'arginine', 'lysine', 'valine', 'citrulline', 'leucine', 'isoleucine',
        'cholesterol', 'histamine', 'folicacid', 'cholecalciferol', 
        'retinoicacid', 'nicotinicacid', 'alpha-tocopherol', 'lithium', 
        'magnesium', 'oxygen', 'nitrogen', 'platinum', 'hydrogenperoxide', 
        'radium', 'potassium', 'agar', 'hemin', 'phorbol12-myristate13-acetate', 
        'methylcellulose(4000cps)', 'insulin', 'triphosphate', 
        'histaminedihydrochloride', 'water', 'carbon'
    })
    
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
        biomolecule_blacklist: Optional[Set[str]] = None,
        use_chembl: bool = True
    ):
        """
        Initialize validation module.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year of the corpus
            end_year: Ending year of the corpus
            biomolecule_blacklist: Set of generic molecules to exclude
            use_chembl: Whether to attempt loading ChEMBL data (default: True)
        """
        self.logger = LoggerFactory.setup_logger("validation", str(end_year), log_to_file=True, log_file=f'logs/{end_year}.log')
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
        
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        self.use_chembl = use_chembl
        
        self.embedding_method = 'da'  # ['da', 'avg']
        
        # Configurar blacklist
        self.biomolecule_blacklist = (
            biomolecule_blacklist if biomolecule_blacklist 
            else self.DEFAULT_BIOMOLECULE_BLACKLIST
        )
        
        # Paths
        self.base_path = Path(f'./data/{self.disease_name}') 
        self.model_directory = Path(f'{self.base_path}/models')
        self.validation_path = Path(f'{self.base_path}/validation/compound_history')
        self.compound_list_path = Path(f'{self.base_path}/corpus/compounds_in_corpus.txt')
        self.whitelist_cache_path = Path('./data/compound_whitelist.txt')
        self.synonyms_path = Path('./data/pubchem_data/CID-Synonym-filtered')
        self.titles_path = Path('./data/pubchem_data/CID-Title')
        
        # Criar diretórios
        self.validation_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._therapeutic_compounds = None
        self._models_cache = {}
        self._model_types_by_year = {}  # Armazena tipo de modelo por ano
        self._chembl_client = None
        
        self.logger.info(f"ValidationModule initialized for {self.disease_name}")
        self.logger.info(f"Years: {start_year}-{end_year}")
        self.logger.info(f"ChEMBL enabled: {use_chembl}")
        self._detect_available_models()

    def _get_chembl_client_safe(self):
        """Get ChEMBL client with error handling."""
        if not self.use_chembl:
            return None
            
        if self._chembl_client is None:
            self._chembl_client = _get_chembl_client()
            if self._chembl_client is None:
                self.logger.warning("ChEMBL client unavailable - will skip ChEMBL data")
        return self._chembl_client

    def _detect_available_models(self) -> None:
        """Detecta modelos disponíveis no diretório e seus tipos."""
        self.logger.info(f"Detecting models in {self.model_directory}...")
        
        available_models = {}
        pattern = re.compile(r'^(\w+)_\d{4}_(\d{4})\.model$')
        
        # Mapear nomes de arquivo para ModelType
        valid_model_names = {mt.value for mt in ModelType}
        
        for model_file in self.model_directory.glob('**/*.model'):
            match = pattern.match(model_file.name)
            if match:
                model_type = match.group(1).lower()
                year = int(match.group(2))
                
                if model_type in valid_model_names:
                    available_models[year] = {
                        'type': model_type,
                        'path': model_file
                    }
                    self.logger.info(f"  Year {year}: {model_type} model found")
                else:
                    self.logger.warning(f"  Year {year}: unsupported model type '{model_type}'")
        
        self._model_types_by_year = available_models
        
        if not available_models:
            self.logger.warning("No models found in directory!")
        else:
            self.logger.info(f"Total models detected: {len(available_models)}")

    @lru_cache(maxsize=1)
    def _load_chembl_drugs(self) -> Set[str]:
        """Carrega lista de small molecule drugs do ChEMBL (cached)."""
        client = self._get_chembl_client_safe()
        if client is None:
            self.logger.warning("Skipping ChEMBL data loading")
            return set()
        
        self.logger.info("Loading small molecule drugs from ChEMBL...")
        drug_names_set = set()
        
        try:
            molecule = client.molecule
            
            # Filtrar apenas small molecules aprovadas ou em fase avançada
            approved_drugs_query = molecule.filter(
                max_phase__in=[2, 3, 4],
                molecule_type='Small molecule'
            ).only(['pref_name', 'synonyms'])
            
            count = 0
            for drug in approved_drugs_query:
                # Nome preferencial
                if drug.get('pref_name'):
                    normalized = drug['pref_name'].lower().replace(' ', '')
                    drug_names_set.add(normalized)
                
                # Sinônimos
                for synonym in drug.get('synonyms', []):
                    if synonym:
                        normalized = synonym.lower().replace(' ', '')
                        drug_names_set.add(normalized)
                
                count += 1
                if count % 5000 == 0:
                    self.logger.info(f"Processed {count} ChEMBL records...")
            
            self.logger.info(f"Loaded {len(drug_names_set)} drug names from ChEMBL")
            return drug_names_set
            
        except Exception as e:
            self.logger.error(f"Error loading ChEMBL data: {e}")
            self.logger.info("Continuing without ChEMBL data...")
            return set()

    def _load_pubchem_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados do PubChem."""
        if not self.synonyms_path.exists() or not self.titles_path.exists():
            raise FileNotFoundError(
                f"PubChem data not found. Please run preprocessing first."
            )
        
        self.logger.info("Loading PubChem data...")
        
        synonyms_df = pd.read_csv(
            self.synonyms_path,
            sep='\t',
            header=None,
            names=['cid', 'synonym'],
            dtype={'cid': str, 'synonym': str}
        )
        
        titles_df = pd.read_csv(
            self.titles_path,
            sep='\t',
            header=None,
            names=['cid', 'title'],
            dtype={'cid': str, 'title': str}
        )
        
        self.logger.info(f"Loaded {len(synonyms_df)} synonyms and {len(titles_df)} titles")
        return synonyms_df, titles_df

    def get_therapeutic_compounds(self) -> Set[str]:
        """
        Cria whitelist de compostos terapêuticos.
        Usa cache em arquivo para evitar reprocessamento.
        """
        # Verificar cache
        if self.whitelist_cache_path.exists():
            self.logger.info(f"Loading whitelist from cache: {self.whitelist_cache_path}")
            with open(self.whitelist_cache_path, 'r', encoding='utf-8') as f:
                whitelist = {line.strip() for line in f if line.strip()}
            self.logger.info(f"Loaded {len(whitelist)} compounds from cache")
            return whitelist
        
        self.logger.info("Cache not found. Generating whitelist...")
        
        # Carregar dados ChEMBL (com fallback se indisponível)
        chembl_drugs = self._load_chembl_drugs()
        if not chembl_drugs:
            self.logger.warning("No ChEMBL data available - will use only PubChem data")
        
        # Carregar dados PubChem
        try:
            synonyms_df, titles_df = self._load_pubchem_data()
        except FileNotFoundError as e:
            self.logger.error(str(e))
            return set()
        
        # Se temos dados ChEMBL, fazer matching
        if chembl_drugs:
            # Normalizar sinônimos PubChem
            self.logger.info("Mapping ChEMBL drugs to PubChem CIDs...")
            synonyms_df['synonym_normalized'] = (
                synonyms_df['synonym']
                .str.lower()
                .str.replace(r'\s+', '', regex=True)
            )
            synonyms_df.dropna(subset=['synonym_normalized', 'cid'], inplace=True)
            
            # Criar DataFrame com termos ChEMBL
            chembl_df = pd.DataFrame(
                list(chembl_drugs),
                columns=['chembl_term_normalized']
            )
            
            # Match com PubChem
            matched_cids_df = pd.merge(
                chembl_df,
                synonyms_df,
                left_on='chembl_term_normalized',
                right_on='synonym_normalized',
                how='inner'
            )
            
            unique_cids = matched_cids_df['cid'].unique()
            self.logger.info(f"Found {len(unique_cids)} unique CIDs")
            
            # Buscar títulos canônicos
            self.logger.info("Fetching canonical titles...")
            therapeutic_titles_df = titles_df[titles_df['cid'].isin(unique_cids)]
        else:
            # Sem ChEMBL, usar todos os títulos PubChem
            self.logger.info("Using all PubChem titles as therapeutic compounds")
            therapeutic_titles_df = titles_df
        
        # Normalizar títulos
        normalized_titles = (
            therapeutic_titles_df['title']
            .str.lower()
            .str.replace(r'\s+', '', regex=True)
        )
        
        whitelist = set(normalized_titles.dropna().unique())
        self.logger.info(f"Created whitelist with {len(whitelist)} compounds")
        
        # Aplicar blacklist
        filtered_whitelist = whitelist - self.biomolecule_blacklist
        removed = len(whitelist) - len(filtered_whitelist)
        
        self.logger.info(f"Removed {removed} generic biomolecules")
        self.logger.info(f"Final whitelist: {len(filtered_whitelist)} compounds")
        
        # Salvar cache
        self.logger.info(f"Saving whitelist to cache: {self.whitelist_cache_path}")
        self.whitelist_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.whitelist_cache_path, 'w', encoding='utf-8') as f:
            for compound in sorted(filtered_whitelist):
                f.write(f"{compound}\n")
        
        return filtered_whitelist

    def _load_model(self, year: int) -> Optional[Tuple[object, str]]:
        """
        Carrega modelo para um ano específico (com cache).
        
        Returns:
            Tuple (model, model_type) ou None se não encontrado
        """
        # Verificar cache
        if year in self._models_cache:
            return self._models_cache[year]
        
        # Verificar se modelo existe
        if year not in self._model_types_by_year:
            self.logger.warning(f"No model found for year {year}")
            return None
        
        model_info = self._model_types_by_year[year]
        model_type = model_info['type']
        model_path = model_info['path']
        
        try:
            # Carregar baseado no tipo
            if model_type == ModelType.WORD2VEC.value:
                model = Word2Vec.load(str(model_path))
                
            elif model_type == ModelType.FASTTEXT.value:
                model = FastText.load(str(model_path))
                
            elif model_type == ModelType.GLOVE.value:
                model = KeyedVectors.load(str(model_path))
                
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None
            
            # Cachear
            result = (model, model_type)
            if len(self._models_cache) < 5:
                self._models_cache[year] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading model for year {year}: {e}")
            return None

    def get_embedding(
        self,
        word: str,
        model: object,
        model_type: str,
        method: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Obtém embedding de uma palavra.
        
        Args:
            word: Palavra para buscar embedding
            model: Modelo de embeddings
            model_type: Tipo do modelo
            method: Método de extração ('da' ou 'avg')
        
        Returns:
            Array numpy com embedding ou None
        """
        method = method or self.embedding_method
        
        # --- Função auxiliar: detectar se o vocabulário usa underscores ---
        def _uses_underscore_vocab(vocab_keys: List[str]) -> bool:
            """Verifica se o modelo usa tokens compostos com underscore."""
            if not vocab_keys:
                return False
            sample_size = min(1000, len(vocab_keys))
            sample = vocab_keys[:sample_size]
            underscore_ratio = sum("_" in w for w in sample) / sample_size
            return underscore_ratio > 0.05  # heurística: 5% dos tokens contêm "_"
        
        def _generate_word_variants(word: str, uses_underscore: bool) -> List[str]:
            """Gera variantes possíveis de uma palavra/frase."""
            variants = [word]
            
            # Variantes com underscore/espaço
            if uses_underscore:
                variants.append(word.replace(" ", "_"))
                variants.append(word.replace("-", "_"))
            else:
                variants.append(word.replace("_", " "))
                variants.append(word.replace("-", " "))
            
            # Variantes adicionais
            variants.append(word.replace("_", ""))  # sem separador
            variants.append(word.replace(" ", ""))  # sem separador
            variants.append(word.replace("-", ""))  # sem separador
            
            # Remove duplicatas preservando ordem
            seen = set()
            unique_variants = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    unique_variants.append(v)
            
            return unique_variants
        
        # Obter vocabulário
        vocab = model.wv if hasattr(model, 'wv') else model
        
        # Get vocabulary keys properly for KeyedVectors
        if hasattr(vocab, 'index_to_key'):
            vocab_keys = vocab.index_to_key
        elif hasattr(vocab, 'key_to_index'):
            vocab_keys = list(vocab.key_to_index.keys())
        else:
            # Fallback for older gensim versions
            vocab_keys = list(vocab.vocab.keys()) if hasattr(vocab, 'vocab') else []

        # Detecta automaticamente se o modelo usa underscores
        uses_underscore = _uses_underscore_vocab(vocab_keys)

        # Gera variantes possíveis do termo
        variants = _generate_word_variants(word, uses_underscore)

        # --- Método de acesso direto ---
        if method == 'da':
            for variant in variants:
                if variant in vocab.key_to_index:
                    self.logger.debug(f"Found '{word}' as '{variant}' in vocab")
                    return vocab[variant]
            
            for variant in variants:
                # Procura tokens que contenham a variante
                partial_matches = [k for k in vocab_keys if variant.lower() in k.lower()]
                if partial_matches:
                    best_match = partial_matches[0]
                    return vocab[best_match]
            
            return None

        # --- Método de média dos termos contendo a palavra ---
        elif method == 'avg':
            matching_tokens = []
            for variant in variants:
                matching_tokens.extend(
                    [key for key in vocab_keys if variant in key]
                )
            if not matching_tokens:
                return None

            embeddings = vocab[matching_tokens]
            return np.mean(embeddings, axis=0)
        
        return None

    def _filter_compounds_by_vocab(self, compounds: Set[str]) -> List[str]:
        """Filtra compostos presentes no vocabulário do modelo final."""
        if self.end_year not in self._model_types_by_year:
            self.logger.warning(f"Final model for year {self.end_year} not found")
            return list(compounds)
        
        self.logger.info("Filtering compounds by final model vocabulary...")
        
        try:
            result = self._load_model(self.end_year)
            if result is None:
                return list(compounds)
            
            model, model_type = result
            
            # Obter vocabulário uma vez
            vocab_set = self._get_vocab_set(model, model_type)
            
            if not vocab_set:
                self.logger.warning("Empty vocabulary")
                return list(compounds)
            
            # Filtrar em batch
            filtered = []
            compounds_list = list(compounds)
            
            self.logger.info(f"Checking {len(compounds_list)} compounds against vocab of size {len(vocab_set)}")
            
            for compound in tqdm(
                compounds_list,
                desc="Filtering compounds",
                unit="compound",
                ncols=80,
                leave=False
            ):
                # Gerar variantes do composto
                variants = self._generate_compound_variants(compound, vocab_set)
                
                # Verificar se alguma variante está no vocab
                if any(v in vocab_set for v in variants):
                    filtered.append(compound)
            
            self.logger.info(
                f"Filtered: {len(compounds)} → {len(filtered)} compounds "
                f"({len(compounds) - len(filtered)} not in vocab)"
            )
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering compounds: {e}")
            return list(compounds)

    def _get_vocab_set(self, model: object, model_type: str) -> Set[str]:
        """
        Extrai vocabulário como set para lookup rápido.
        
        Args:
            model: Modelo de embeddings
            model_type: Tipo do modelo
        
        Returns:
            Set com todas as palavras do vocabulário
        """
        vocab = model.wv if hasattr(model, 'wv') else model
        
        if hasattr(vocab, 'index_to_key'):
            return set(vocab.index_to_key)
        elif hasattr(vocab, 'key_to_index'):
            return set(vocab.key_to_index.keys())
        elif hasattr(vocab, 'vocab'):
            return set(vocab.vocab.keys())
        
        return set()

    def _generate_compound_variants(self, compound: str, vocab_sample: Set[str]) -> List[str]:
        """
        Gera variantes possíveis de um composto baseado no vocabulário.
        
        Args:
            compound: Nome do composto
            vocab_sample: Amostra do vocabulário para detectar padrões
        
        Returns:
            Lista de variantes possíveis
        """
        # Detectar se vocab usa underscores
        uses_underscore = any('_' in w for w in list(vocab_sample)[:1000] if len(w) > 5)
        
        variants = [compound]
        
        # Variantes com underscore/espaço
        if uses_underscore:
            variants.append(compound.replace(" ", "_"))
            variants.append(compound.replace("-", "_"))
        else:
            variants.append(compound.replace("_", " "))
            variants.append(compound.replace("-", " "))
        
        # Variantes sem separador
        variants.append(compound.replace("_", ""))
        variants.append(compound.replace(" ", ""))
        variants.append(compound.replace("-", ""))
        
        # Remove duplicatas preservando ordem
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen and v:  # Não adicionar strings vazias
                seen.add(v)
                unique_variants.append(v)
        
        return unique_variants

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 50) -> str:
        """Sanitiza nome para uso em arquivo."""
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', name)
        return sanitized[:max_length]

    def _compute_metrics(
        self,
        compound_embedding: np.ndarray,
        disease_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas de similaridade entre embeddings.
        
        Returns:
            Dict com dot_product, normalized_dot_product e euclidean_distance
        """
        dot_product = np.dot(compound_embedding, disease_embedding)
        euclidean_distance = np.linalg.norm(compound_embedding - disease_embedding)
        
        norm_compound = np.linalg.norm(compound_embedding)
        norm_disease = np.linalg.norm(disease_embedding)
        
        if norm_compound > 0 and norm_disease > 0:
            normalized_dot_product = dot_product / (norm_compound * norm_disease)
        else:
            normalized_dot_product = 0.0
        
        return {
            'dot_product': float(dot_product),
            'normalized_dot_product': float(normalized_dot_product),
            'euclidean_distance': float(euclidean_distance)
        }

    def _compute_derived_metrics(self, compound_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Calcula métricas derivadas (delta e score) para um composto.
        """
        normalized_values = np.array(compound_data.get("normalized_dot_product", []), dtype=float)
        euclidean_distances = np.array(compound_data.get("euclidean_distance", []), dtype=float)
        
        # Delta normalized dot product
        if len(normalized_values) > 1:
            delta_values = np.insert(np.diff(normalized_values), 0, np.nan)
        else:
            delta_values = np.array([np.nan])
        
        compound_data["delta_normalized_dot_product"] = delta_values.tolist()

        # Score combinado
        if len(normalized_values) > 0 and len(euclidean_distances) > 0:
            safe_delta = np.nan_to_num(delta_values, nan=0.0)
            score = normalized_values * (1 + 10 * safe_delta) / (euclidean_distances + 1e-9)
            compound_data["score"] = score.tolist()
        else:
            compound_data["score"] = []

        return compound_data

    def generate_compound_histories(self) -> bool:
        """
        Gera históricos de similaridade para todos os compostos.
        
        Returns:
            True se sucesso, False caso contrário
        """
        # Obter compostos terapêuticos
        self.logger.info("Loading therapeutic compounds...")
        all_compounds = self.get_therapeutic_compounds()
        
        if not all_compounds:
            self.logger.error("Could not load therapeutic compounds")
            return False
        
        # Filtrar por vocabulário
        compounds = self._filter_compounds_by_vocab(all_compounds)
        
        if not compounds:
            self.logger.error("No compounds found in model vocabulary")
            return False
        
        self.logger.info(f"Processing {len(compounds)} compounds...")
        
        # Inicializar dicionário para todos os compostos
        compound_histories = {}
        for compound in compounds:
            compound_histories[compound] = {
                'year': [],
                'model_type': [],
                'dot_product': [],
                'normalized_dot_product': [],
                'euclidean_distance': []
            }
        
        # Processar cada ano
        for year in range(self.start_year, self.end_year + 1):
            self.logger.info(f"Processing year {year}...")
            
            # Carregar modelo
            result = self._load_model(year)
            if result is None:
                self.logger.warning(f"Model for year {year} not found. Skipping.")
                continue
            
            model, model_type = result
            
            # Obter embedding da doença
            disease_embedding = self.get_embedding(
                self.disease_name,
                model,
                model_type,
                method='da'
            )
            
            if disease_embedding is None:
                self.logger.error(
                    f"Disease '{self.disease_name}' not found in model vocabulary for year {year}"
                )
                continue
            
            # Processar cada composto
            processed = 0
            for compound in compounds:
                compound_embedding = self.get_embedding(compound, model, model_type)
                
                if compound_embedding is None:
                    continue
                
                # Calcular métricas
                metrics = self._compute_metrics(compound_embedding, disease_embedding)
                
                # Armazenar
                compound_histories[compound]['year'].append(year)
                compound_histories[compound]['model_type'].append(model_type)
                compound_histories[compound]['dot_product'].append(metrics['dot_product'])
                compound_histories[compound]['normalized_dot_product'].append(
                    metrics['normalized_dot_product']
                )
                compound_histories[compound]['euclidean_distance'].append(
                    metrics['euclidean_distance']
                )
                
                processed += 1
            
            self.logger.info(
                f"Year {year} ({model_type}): processed {processed}/{len(compounds)} compounds"
            )
        
        # Calcular métricas derivadas e salvar
        self.logger.info("Computing derived metrics and saving histories...")
        saved_count = 0
        
        for compound in compounds:
            # Verificar se há dados
            if not compound_histories[compound]['year']:
                continue
            
            # Calcular métricas derivadas
            compound_histories[compound] = self._compute_derived_metrics(
                compound_histories[compound]
            )
            
            # Salvar CSV
            filename = self._sanitize_filename(compound)
            output_path = Path(f'{self.validation_path}/{filename}.csv')
            
            try:
                df = pd.DataFrame(compound_histories[compound])
                df.to_csv(
                    output_path,
                    columns=[
                        'year',
                        'model_type',
                        'normalized_dot_product',
                        'delta_normalized_dot_product',
                        'euclidean_distance',
                        'score'
                    ],
                    index=False
                )
                saved_count += 1
                
            except Exception as e:
                self.logger.error(f"Error saving {filename}: {e}")
        
        self.logger.info(f"Saved {saved_count} compound histories")
        return saved_count > 0

    def get_top_compounds(
        self,
        metric: str = 'score',
        top_n: int = 20,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Obtém top compostos baseado em uma métrica.
        
        Args:
            metric: Métrica para ranking ('score', 'normalized_dot_product', etc.)
            top_n: Número de compostos a retornar
            year: Ano específico (ou último ano se None)
        
        Returns:
            DataFrame com top compostos
        """
        year = year or self.end_year
        
        # Ler todos os CSVs
        all_data = []
        
        for csv_file in self.validation_path.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                
                # Filtrar por ano
                df_year = df[df['year'] == year]
                
                if df_year.empty or metric not in df_year.columns:
                    continue
                
                # Extrair nome do composto
                compound_name = csv_file.stem
                
                all_data.append({
                    'compound': compound_name,
                    'model_type': df_year['model_type'].values[0],
                    metric: df_year[metric].values[0]
                })
                
            except Exception as e:
                self.logger.warning(f"Error reading {csv_file}: {e}")
        
        if not all_data:
            self.logger.warning("No data found for ranking")
            return pd.DataFrame()
        
        # Criar DataFrame e ordenar
        result_df = pd.DataFrame(all_data)
        result_df = result_df.sort_values(by=metric, ascending=False).head(top_n)
        
        return result_df

    def run(self) -> bool:
        """
        Executa pipeline de validação completo.
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            self.logger.info("=== Starting Validation Pipeline ===")
            self.logger.info(f"Disease: {self.disease_name}")
            self.logger.info(f"Year range: {self.start_year}-{self.end_year}")
            
            # Gerar históricos
            success = self.generate_compound_histories()
            
            if success:
                self.logger.info("=== Validation completed successfully ===")
                
                # Mostrar top compostos
                self.logger.info("\nTop 10 compounds by score:")
                top_df = self.get_top_compounds(metric='score', top_n=10)
                for idx, row in top_df.iterrows():
                    self.logger.info(
                        f"  {idx+1}. {row['compound']} ({row['model_type']}): {row['score']:.4f}"
                    )
            else:
                self.logger.error("=== Validation failed ===")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error in validation pipeline: {e}")
            return False
    
    def debug_vocabulary(self, year: int, search_term: Optional[str] = None) -> None:
        """
        Método de debug para inspecionar o vocabulário de um modelo.
        
        Args:
            year: Ano do modelo
            search_term: Termo opcional para buscar no vocabulário
        """
        result = self._load_model(year)
        if result is None:
            self.logger.error(f"Could not load model for year {year}")
            return
        
        model, model_type = result
        vocab = model.wv if hasattr(model, 'wv') else model
        
        if hasattr(vocab, 'index_to_key'):
            vocab_keys = vocab.index_to_key
        elif hasattr(vocab, 'key_to_index'):
            vocab_keys = list(vocab.key_to_index.keys())
        else:
            vocab_keys = []
        
        if search_term:
            search_lower = search_term.lower()
            matches = [w for w in vocab_keys if search_lower in w.lower()]
            self.logger.info(f"Words containing '{search_term}': {matches[:20]}")


if __name__ == '__main__':
    validator = ValidationModule(
        disease_name="acute myeloid leukemia",
        start_year=1990,
        end_year=1990
    )
    
    success = validator.run()
    
    exit(0 if success else 1)