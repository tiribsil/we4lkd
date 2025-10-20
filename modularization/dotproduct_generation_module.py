import os
import re
from pathlib import Path
from typing import Set, List, Dict, Optional, Literal, Tuple
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from chembl_webresource_client.new_client import new_client
from functools import lru_cache
from utils import *
import warnings



class ValidationModule:
    """
    Module for validating embedding models by computing similarities
    between therapeutic compounds and disease embeddings over time.
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
        biomolecule_blacklist: Optional[Set[str]] = None
    ):
        """
        Initialize validation module.
        
        Args:
            disease_name: Name of the disease
            start_year: Starting year of the corpus
            end_year: Ending year of the corpus
            model_type: Type of embedding model ('w2v' or 'ft')
            embedding_method: Method to extract embeddings ('da'=direct, 'avg'=average)
            biomolecule_blacklist: Set of generic molecules to exclude
        """
        self.logger = setup_logger("validation", log_to_file=False)
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        
        self.model_type = 'w2v' #['w2v', 'ft']

        self.embedding_method = 'da' #['da', 'avg']
        
        # Configurar blacklist
        self.biomolecule_blacklist = (
            biomolecule_blacklist if biomolecule_blacklist 
            else self.DEFAULT_BIOMOLECULE_BLACKLIST
        )
        
        # Configurar caminhos
        self.base_path = Path('.') / 'data' / self.disease_name
        self.combination = '15' if self.model_type == 'w2v' else '16'
        
        self.model_directory = self.base_path / 'models' / f'{self.model_type}_combination{self.combination}'
        self.validation_path = self.base_path / 'validation' / self.model_type / 'compound_history'
        self.compound_list_path = self.base_path / 'corpus' / 'compounds_in_corpus.txt'
        self.whitelist_cache_path = Path('./data/compound_whitelist.txt')
        
        # Criar diretórios
        self.validation_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._therapeutic_compounds = None
        self._models_cache = {}
        
        self.logger.info(f"ValidationModule initialized for {self.disease_name}")
        self.logger.info(f"Model type: {self.model_type.upper()}, Years: {start_year}-{end_year}")

    @lru_cache(maxsize=1)
    def _load_chembl_drugs(self) -> Set[str]:
        """Carrega lista de small molecule drugs do ChEMBL (cached)."""
        self.logger.info("Loading small molecule drugs from ChEMBL...")
        drug_names_set = set()
        
        try:
            molecule = new_client.molecule
            
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
            return set()

    def _load_pubchem_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados do PubChem."""
        synonyms_path = Path('./data/pubchem_data/CID-Synonym-filtered/CID-Synonym-filtered')
        titles_path = Path('./data/pubchem_data/CID-Title/CID-Title')
        
        if not synonyms_path.exists() or not titles_path.exists():
            raise FileNotFoundError(
                f"PubChem data not found. Please run preprocessing first."
            )
        
        self.logger.info("Loading PubChem data...")
        
        # Usar dtypes específicos para economizar memória
        synonyms_df = pd.read_csv(
            synonyms_path,
            sep='\t',
            header=None,
            names=['cid', 'synonym'],
            dtype={'cid': str, 'synonym': str}
        )
        
        titles_df = pd.read_csv(
            titles_path,
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
        
        # Carregar dados ChEMBL
        chembl_drugs = self._load_chembl_drugs()
        if not chembl_drugs:
            self.logger.error("Could not load ChEMBL data")
            return set()
        
        # Carregar dados PubChem
        try:
            synonyms_df, titles_df = self._load_pubchem_data()
        except FileNotFoundError as e:
            self.logger.error(str(e))
            return set()
        
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
        
        # Criar diretório se não existir (garantir que ./data existe)
        self.whitelist_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.whitelist_cache_path, 'w', encoding='utf-8') as f:
            # Salvar em ordem alfabética para consistência (igual à função original)
            for compound in sorted(filtered_whitelist):
                f.write(f"{compound}\n")
        
        return filtered_whitelist

    def _load_model(self, year: int) -> Optional[object]:
        """Carrega modelo para um ano específico (com cache)."""
        # Verificar cache
        if year in self._models_cache:
            return self._models_cache[year]
        
        model_path = self.model_directory / f'model_{self.start_year}_{year}.model'
        
        if not model_path.exists():
            return None
        
        try:
            if self.model_type == 'w2v':
                model = Word2Vec.load(str(model_path))
            else:
                model = FastText.load(str(model_path))
            
            # Cachear (limitar tamanho do cache)
            if len(self._models_cache) < 5:
                self._models_cache[year] = model
            
            return model
        except Exception as e:
            self.logger.error(f"Error loading model for year {year}: {e}")
            return None

    def get_embedding(
        self,
        word: str,
        model: object,
        method: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Obtém embedding de uma palavra.
        
        Args:
            word: Palavra para buscar embedding
            model: Modelo Word2Vec ou FastText
            method: Método de extração ('da' ou 'avg')
        
        Returns:
            Array numpy com embedding ou None
        """
        method = method or self.embedding_method
        
        # Acesso direto
        if method == 'da':
            if word in model.wv.key_to_index:
                return model.wv[word]
            return None
        
        # Média de embeddings contendo a palavra
        elif method == 'avg':
            # Buscar todas as palavras que contêm o termo
            matching_tokens = [
                key for key in model.wv.index_to_key 
                if word in key
            ]
            
            if not matching_tokens:
                return None
            
            # Calcular média
            embeddings = model.wv[matching_tokens]
            return np.mean(embeddings, axis=0)
        
        return None

    def _filter_compounds_by_vocab(self, compounds: Set[str]) -> List[str]:
        """Filtra compostos presentes no vocabulário do modelo final."""
        final_model_path = self.model_directory / f'model_{self.start_year}_{self.end_year}.model'
        
        if not final_model_path.exists():
            self.logger.warning(f"Final model not found at {final_model_path}")
            return list(compounds)
        
        self.logger.info("Filtering compounds by final model vocabulary...")
        
        try:
            if self.model_type == 'w2v':
                model = Word2Vec.load(str(final_model_path))
            else:
                model = FastText.load(str(final_model_path))
            
            filtered = [
                compound for compound in compounds
                if compound in model.wv.key_to_index
            ]
            
            self.logger.info(
                f"Filtered: {len(compounds)} → {len(filtered)} compounds "
                f"({len(compounds) - len(filtered)} not in vocab)"
            )
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering compounds: {e}")
            return list(compounds)

    @staticmethod
    def _sanitize_filename(name: str, max_length: int = 50) -> str:
        """Sanitiza nome para uso em arquivo."""
        # Remover caracteres inválidos
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
        # Dot product
        dot_product = np.dot(compound_embedding, disease_embedding)
        
        # Distância euclidiana
        euclidean_distance = np.linalg.norm(compound_embedding - disease_embedding)
        
        # Cosine similarity (normalized dot product)
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

    def _compute_derived_metrics(
        self,
        compound_data: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Calcula métricas derivadas (delta e score).
        
        Args:
            compound_data: Dict com year, normalized_dot_product, euclidean_distance
        
        Returns:
            Dict atualizado com delta e score
        """
        normalized_values = np.array(compound_data['normalized_dot_product'])
        euclidean_distances = np.array(compound_data['euclidean_distance'])
        
        # Delta normalized dot product
        if len(normalized_values) > 0:
            delta_values = np.diff(normalized_values, prepend=normalized_values[0])
            compound_data['delta_normalized_dot_product'] = delta_values.tolist()
        else:
            compound_data['delta_normalized_dot_product'] = []
        
        # Score combinado
        if len(normalized_values) > 0 and len(delta_values) > 0:
            score = normalized_values * (1 + delta_values) / (euclidean_distances + 1e-9)
            compound_data['score'] = score.tolist()
        else:
            compound_data['score'] = []
        
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
            key = f'{compound}_comb{self.combination}'
            compound_histories[key] = {
                'year': [],
                'dot_product': [],
                'normalized_dot_product': [],
                'euclidean_distance': []
            }
        
        # Processar cada ano
        for year in range(self.start_year, self.end_year + 1):
            self.logger.info(f"Processing year {year}...")
            
            # Carregar modelo
            model = self._load_model(year)
            if model is None:
                self.logger.warning(f"Model for year {year} not found. Skipping.")
                continue
            
            # Obter embedding da doença
            disease_embedding = self.get_embedding(
                self.disease_name,
                model,
                method='da'
            )
            
            if disease_embedding is None:
                self.logger.error(
                    f"Disease '{self.disease_name}' not found in model vocabulary"
                )
                continue
            
            # Processar cada composto
            processed = 0
            for compound in compounds:
                compound_embedding = self.get_embedding(compound, model)
                
                if compound_embedding is None:
                    continue
                
                # Calcular métricas
                metrics = self._compute_metrics(compound_embedding, disease_embedding)
                
                # Armazenar
                key = f'{compound}_comb{self.combination}'
                compound_histories[key]['year'].append(year)
                compound_histories[key]['dot_product'].append(metrics['dot_product'])
                compound_histories[key]['normalized_dot_product'].append(
                    metrics['normalized_dot_product']
                )
                compound_histories[key]['euclidean_distance'].append(
                    metrics['euclidean_distance']
                )
                
                processed += 1
            
            self.logger.info(f"Year {year}: processed {processed}/{len(compounds)} compounds")
        
        # Calcular métricas derivadas e salvar
        self.logger.info("Computing derived metrics and saving histories...")
        saved_count = 0
        
        for compound in compounds:
            key = f'{compound}_comb{self.combination}'
            
            # Verificar se há dados
            if not compound_histories[key]['year']:
                continue
            
            # Calcular métricas derivadas
            compound_histories[key] = self._compute_derived_metrics(
                compound_histories[key]
            )
            
            # Salvar CSV
            filename = self._sanitize_filename(key)
            output_path = self.validation_path / f'{filename}.csv'
            
            try:
                df = pd.DataFrame(compound_histories[key])
                df.to_csv(
                    output_path,
                    columns=[
                        'year',
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
                compound_name = csv_file.stem.replace(f'_comb{self.combination}', '')
                
                all_data.append({
                    'compound': compound_name,
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
            self.logger.info(f"Model type: {self.model_type.upper()}")
            self.logger.info(f"Year range: {self.start_year}-{self.end_year}")
            
            # Gerar históricos
            success = self.generate_compound_histories()
            
            if success:
                self.logger.info("=== Validation completed successfully ===")
                
                # Mostrar top compostos
                self.logger.info("\nTop 10 compounds by score:")
                top_df = self.get_top_compounds(metric='score', top_n=10)
                for idx, row in top_df.iterrows():
                    self.logger.info(f"  {idx+1}. {row['compound']}: {row['score']:.4f}")
            else:
                self.logger.error("=== Validation failed {e}===")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error in validation pipeline: {e}")
            return False


if __name__ == '__main__':
    validator = ValidationModule(disease_name="acute myeloid leukemia",
                                 start_year=2024,
                                 end_year = 2025)
    
    success = validator.run()
    
    exit(0 if success else 1)