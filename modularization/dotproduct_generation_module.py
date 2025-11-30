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
from functools import lru_cache
from utils import *
import warnings
import collections

# Tenta importar o cliente ChEMBL, mas não falha se não existir
try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    new_client = None

class ModelType(Enum):
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"

class ValidationModule:
    """
    Módulo para validar modelos de embeddings calculando similaridades
    entre compostos terapêuticos e embeddings de doenças ao longo do tempo.
    Agora suporta subpastas específicas de modelos e gera rankings (Top N).
    """
    
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
        model_subfolder: str,
        start_year: int,
        end_year: int,
        biomolecule_blacklist: Optional[Set[str]] = None,
        use_chembl: bool = True,
        top_n_to_save: int = 50
    ):
        """
        Args:
            disease_name: Nome da doença
            model_subfolder: Nome da pasta dentro de models/ (ex: 'w2v_fixed')
            start_year: Ano inicial da análise
            end_year: Ano final da análise
            biomolecule_blacklist: Set de moléculas a ignorar
            use_chembl: Se deve usar ChEMBL para whitelist
            top_n_to_save: Quantos compostos salvar nos arquivos de ranking
        """
        self.logger = LoggerFactory.setup_logger("validation", f"{model_subfolder}_{end_year}", log_to_file=True, log_file=f'logs/{model_subfolder}_{end_year}.log')
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.disease_name = normalize_disease_name(disease_name)
        self.model_subfolder = model_subfolder
        self.start_year = start_year
        self.end_year = end_year
        self.use_chembl = use_chembl
        self.top_n_to_save = top_n_to_save
        
        self.embedding_method = 'da'  # direct access
        
        self.biomolecule_blacklist = (
            biomolecule_blacklist if biomolecule_blacklist 
            else self.DEFAULT_BIOMOLECULE_BLACKLIST
        )
        
        # --- Configuração de Caminhos ---
        self.base_path = Path(f'./data/{self.disease_name}') 
        
        # Caminho dos modelos: data/disease/models/w2v_fixed/
        self.model_directory = Path(f'{self.base_path}/models/{self.model_subfolder}')
        
        # Caminho de validação base: data/disease/validation/w2v_fixed/
        self.validation_base = Path(f'{self.base_path}/validation/{self.model_subfolder}')
        self.history_path = self.validation_base / 'compound_history'
        self.top_n_path = self.validation_base / 'top_n_compounds'
        
        # Caminhos auxiliares
        self.whitelist_cache_path = Path('./data/compound_whitelist.txt')
        self.synonyms_path = Path('./data/pubchem_data/CID-Synonym-filtered')
        self.titles_path = Path('./data/pubchem_data/CID-Title')
        
        # Criar diretórios
        self.history_path.mkdir(parents=True, exist_ok=True)
        self.top_n_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._models_cache = {}
        self._model_files_by_year = {} 
        self._chembl_client = None
        
        self.logger.info(f"ValidationModule initialized for {self.disease_name}")
        self.logger.info(f"Model Subfolder: {self.model_subfolder}")
        self._detect_available_models()

    def _detect_available_models(self) -> None:
        """
        Detecta modelos na subpasta especificada.
        Espera padrão: nome_anoInic_anoFim.model (ex: w2v_fixed_1956_1967.model)
        """
        self.logger.info(f"Detecting models in {self.model_directory}...")
        
        if not self.model_directory.exists():
            self.logger.error(f"Model directory does not exist: {self.model_directory}")
            return

        # Regex flexível para capturar os anos no final do arquivo
        # Captura qualquer coisa + _ + 4 digitos + _ + 4 digitos + .model
        pattern = re.compile(r'.*_(\d{4})_(\d{4})\.model$')
        
        available_models = {}
        
        for model_file in self.model_directory.glob('*.model'):
            match = pattern.match(model_file.name)
            if match:
                # O "ano do modelo" é considerado o ano final do intervalo de treinamento
                file_end_year = int(match.group(2))
                
                # Determina tipo básico (assumindo word2vec se não especificado diferente no nome)
                model_type_str = model_file.stem.lower()
                if "fasttext" in model_type_str:
                    m_type = ModelType.FASTTEXT
                elif "glove" in model_type_str:
                    m_type = ModelType.GLOVE
                else:
                    m_type = ModelType.WORD2VEC
                
                available_models[file_end_year] = {
                    'type': m_type,
                    'path': model_file
                }
                self.logger.debug(f"  Mapped {file_end_year} -> {model_file.name} ({m_type.value})")
        
        self._model_files_by_year = available_models
        
        if not available_models:
            self.logger.warning("No models matching pattern found in directory!")
        else:
            self.logger.info(f"Total models detected: {len(available_models)}")

    # ... [Métodos _get_chembl_client_safe, _load_chembl_drugs, _load_pubchem_data, get_therapeutic_compounds mantidos iguais] ...
    # (Omitindo para brevidade, assuma que são idênticos ao anterior, focando nas mudanças)
    
    def _get_chembl_client_safe(self):
        if not self.use_chembl: return None
        if self._chembl_client is None and new_client: self._chembl_client = new_client
        return self._chembl_client

    @lru_cache(maxsize=1)
    def _load_chembl_drugs(self) -> Set[str]:
        """
        Carrega lista de 'Small molecule drugs' do ChEMBL.
        Refinado para buscar apenas moléculas pequenas em fases clínicas 2, 3 ou 4.
        """
        client = self._get_chembl_client_safe()
        if not client:
            return set()
        
        self.logger.info("Fetching 'Small molecule drugs' list from ChEMBL (refined search)...")
        drug_names_set = set()
        
        try:
            molecule = client.molecule
            
            # Filtro refinado conforme o código original funcional
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
            
            self.logger.info(f"Loaded {len(drug_names_set)} drug names/synonyms from ChEMBL")
            return drug_names_set
            
        except Exception as e:
            self.logger.error(f"Error loading ChEMBL data: {e}")
            return set()

    def get_therapeutic_compounds(self) -> Set[str]:
        """
        Cria ou carrega a whitelist de compostos terapêuticos.
        Se o cache não existir, gera a partir do cruzamento ChEMBL + PubChem.
        """
        # 1. Tentar carregar do cache
        if self.whitelist_cache_path.exists():
            self.logger.info(f"Loading whitelist from cache: {self.whitelist_cache_path}")
            with open(self.whitelist_cache_path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f if line.strip()}
        
        self.logger.info("Cache not found. Generating whitelist from data sources...")
        
        # 2. Carregar dados do ChEMBL
        chembl_drug_names = self._load_chembl_drugs()
        if not chembl_drug_names:
            self.logger.warning("Could not obtain drug list from ChEMBL. Aborting whitelist generation.")
            return set()
        
        # 3. Carregar dados do PubChem (CSV)
        self.logger.info("Loading PubChem data with Pandas...")
        try:
            if not self.synonyms_path.exists() or not self.titles_path.exists():
                raise FileNotFoundError("PubChem files (CID-Synonym-filtered or CID-Title) not found.")

            synonyms_df = pd.read_csv(
                self.synonyms_path, sep='\t', header=None, 
                names=['cid', 'synonym'], dtype={'cid': str}
            )
            titles_df = pd.read_csv(
                self.titles_path, sep='\t', header=None, 
                names=['cid', 'title'], dtype={'cid': str}
            )
        except Exception as e:
            self.logger.error(f"Error loading PubChem data: {e}")
            return set()

        # 4. Mapear ChEMBL -> PubChem CIDs
        self.logger.info("Mapping ChEMBL names to PubChem CIDs...")
        
        # Criar DF com termos normalizados do ChEMBL
        chembl_df = pd.DataFrame(list(chembl_drug_names), columns=['chembl_term_normalized'])
        
        # Normalizar sinônimos do PubChem
        synonyms_df['synonym_normalized'] = (
            synonyms_df['synonym']
            .str.lower()
            .str.replace(r'\s+', '', regex=True)
        )
        synonyms_df.dropna(subset=['synonym_normalized', 'cid'], inplace=True)
        
        # Merge (Inner Join)
        matched_cids_df = pd.merge(
            chembl_df,
            synonyms_df,
            left_on='chembl_term_normalized',
            right_on='synonym_normalized',
            how='inner'
        )
        
        unique_matched_cids = matched_cids_df['cid'].unique()
        self.logger.info(f"Found {len(unique_matched_cids)} unique CIDs corresponding to therapeutic compounds.")
        
        # 5. Obter Títulos Canônicos
        self.logger.info("Fetching canonical titles...")
        therapeutic_titles_df = titles_df[titles_df['cid'].isin(unique_matched_cids)]
        
        # Normalizar títulos finais
        normalized_titles = (
            therapeutic_titles_df['title']
            .str.lower()
            .str.replace(r'\s+', '', regex=True)
        )
        
        final_whitelist_set = set(normalized_titles.dropna().unique())
        
        # 6. Aplicar Blacklist
        filtered_whitelist = final_whitelist_set - self.biomolecule_blacklist
        
        removed_count = len(final_whitelist_set) - len(filtered_whitelist)
        self.logger.info(f"Removed {removed_count} generic compounds (blacklist).")
        self.logger.info(f"Final whitelist contains {len(filtered_whitelist)} compounds.")
        
        # 7. Salvar Cache
        self.logger.info(f"Saving whitelist to cache: {self.whitelist_cache_path}")
        self.whitelist_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.whitelist_cache_path, 'w', encoding='utf-8') as f:
                for compound in sorted(list(filtered_whitelist)):
                    f.write(f"{compound}\n")
        except Exception as e:
            self.logger.error(f"Error saving cache file: {e}")
        
        return filtered_whitelist

    def _load_model(self, year: int) -> Optional[Tuple[object, str]]:
        if year in self._models_cache:
            return self._models_cache[year]
        
        if year not in self._model_files_by_year:
            # Tentar encontrar o ano mais próximo se não houver exato? 
            # Por enquanto, loga aviso e retorna None
            self.logger.debug(f"No model found explicitly for year {year}")
            return None
        
        model_info = self._model_files_by_year[year]
        model_type_enum = model_info['type']
        model_path = model_info['path']
        
        try:
            self.logger.debug(f"Loading model: {model_path.name}")
            if model_type_enum == ModelType.WORD2VEC:
                model = Word2Vec.load(str(model_path))
            elif model_type_enum == ModelType.FASTTEXT:
                model = FastText.load(str(model_path))
            elif model_type_enum == ModelType.GLOVE:
                model = KeyedVectors.load(str(model_path))
            else:
                return None
            
            # Cache strategy: keep small cache
            result = (model, model_type_enum.value)
            if len(self._models_cache) > 3:
                # Remove oldest
                first_key = next(iter(self._models_cache))
                del self._models_cache[first_key]
            
            self._models_cache[year] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}: {e}")
            return None

    # ... [Métodos get_embedding, _filter_compounds_by_vocab, _get_vocab_set, _generate_compound_variants, _sanitize_filename mantidos] ...
    
    def get_embedding(self, word: str, model, model_type: str, method='da'):
        # (Código idêntico ao original para extração de embedding)
        vocab = model.wv if hasattr(model, 'wv') else model
        if word in vocab.key_to_index:
            return vocab[word]
        # Tentativa com variantes simples
        if word.replace(" ", "_") in vocab.key_to_index:
            return vocab[word.replace(" ", "_")]
        return None

    def _filter_compounds_by_vocab(self, compounds: Set[str]) -> List[str]:
        # Usa o último ano disponível para validar vocabulário
        last_year = max(self._model_files_by_year.keys()) if self._model_files_by_year else self.end_year
        
        model_res = self._load_model(last_year)
        if not model_res: return list(compounds)
        
        model, mtype = model_res
        vocab = model.wv if hasattr(model, 'wv') else model
        
        valid = []
        for c in compounds:
            # Verificação simplificada
            if self.get_embedding(c, model, mtype) is not None:
                valid.append(c)
        return valid

    def _compute_metrics(self, comp_emb, dis_emb):
        dot = np.dot(comp_emb, dis_emb)
        norm_c = np.linalg.norm(comp_emb)
        norm_d = np.linalg.norm(dis_emb)
        cosine = dot / (norm_c * norm_d) if (norm_c > 0 and norm_d > 0) else 0.0
        dist = np.linalg.norm(comp_emb - dis_emb)
        return {'dot_product': dot, 'normalized_dot_product': cosine, 'euclidean_distance': dist}

    def _compute_derived_metrics(self, compound_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        # Idêntico ao original
        normalized_values = np.array(compound_data.get("normalized_dot_product", []), dtype=float)
        eucl_values = np.array(compound_data.get("euclidean_distance", []), dtype=float)
        
        if len(normalized_values) > 1:
            delta_values = np.insert(np.diff(normalized_values), 0, 0.0) # Primeiro delta é 0
        else:
            delta_values = np.array([0.0])
            
        compound_data["delta_normalized_dot_product"] = delta_values.tolist()

        if len(normalized_values) > 0:
            # Score heurístico
            score = normalized_values * (1 + 10 * delta_values) / (eucl_values + 1e-9)
            compound_data["score"] = score.tolist()
        else:
            compound_data["score"] = []

        return compound_data

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', '_', name)[:50]

    def generate_compound_histories(self) -> bool:
        """
        Gera os arquivos CSV de histórico para cada composto.
        """
        all_compounds = self.get_therapeutic_compounds()
        if not all_compounds:
            self.logger.warning("No whitelist found, skipping generation.")
            return False

        compounds = self._filter_compounds_by_vocab(all_compounds)
        self.logger.info(f"Processing {len(compounds)} compounds for history generation...")
        
        # Estrutura em memória: history[compound] = {colunas...}
        compound_histories = {c: {
            'year': [], 'model_type': [], 
            'dot_product': [], 'normalized_dot_product': [], 'euclidean_distance': []
        } for c in compounds}
        
        years_processed = 0
        
        # Iterar sobre anos ordenados
        sorted_years = sorted([y for y in self._model_files_by_year.keys() if self.start_year <= y <= self.end_year])
        
        for year in sorted_years:
            res = self._load_model(year)
            if not res: continue
            
            model, m_type = res
            dis_emb = self.get_embedding(self.disease_name, model, m_type)
            
            if dis_emb is None:
                self.logger.warning(f"Disease '{self.disease_name}' not in vocab for year {year}")
                continue
                
            years_processed += 1
            
            for compound in compounds:
                c_emb = self.get_embedding(compound, model, m_type)
                if c_emb is None: continue
                
                metrics = self._compute_metrics(c_emb, dis_emb)
                
                hist = compound_histories[compound]
                hist['year'].append(year)
                hist['model_type'].append(m_type)
                hist['dot_product'].append(metrics['dot_product'])
                hist['normalized_dot_product'].append(metrics['normalized_dot_product'])
                hist['euclidean_distance'].append(metrics['euclidean_distance'])

        # Salvar arquivos
        saved_count = 0
        self.logger.info("Computing derived metrics and saving CSVs...")
        
        for compound, data in tqdm(compound_histories.items(), desc="Saving histories"):
            if not data['year']: continue
            
            final_data = self._compute_derived_metrics(data)
            df = pd.DataFrame(final_data)
            
            fname = self._sanitize_filename(compound)
            out = self.history_path / f'{fname}.csv'
            df.to_csv(out, index=False)
            saved_count += 1
            
        return saved_count > 0

    def generate_yearly_rankings(self):
        """
        Lê todos os CSVs de histórico gerados, agrega por ano e salva os rankings (Top N).
        Substitui a lógica que existia no LatentKnowledgeReport.
        """
        self.logger.info("Generating Top N rankings for all years...")
        
        metrics_to_rank = ['score', 'normalized_dot_product', 'delta_normalized_dot_product', 'euclidean_distance']
        
        # Estrutura para agrupar dados: yearly_data[year][metric] = list of (value, compound_name)
        yearly_data = collections.defaultdict(lambda: collections.defaultdict(list))
        
        csv_files = list(self.history_path.glob('*.csv'))
        
        # Leitura Otimizada: Ler cada arquivo uma vez e distribuir nos buckets de ano
        for csv_file in tqdm(csv_files, desc="Aggregating rankings"):
            try:
                df = pd.read_csv(csv_file)
                compound_name = self._sanitize_filename(csv_file.stem).replace('_', ' ') # Recuperar nome legível se possível
                
                for _, row in df.iterrows():
                    year = int(row['year'])
                    
                    for metric in metrics_to_rank:
                        if metric in row:
                            val = row[metric]
                            if pd.notna(val):
                                yearly_data[year][metric].append((val, compound_name))
                                
            except Exception as e:
                self.logger.error(f"Error reading {csv_file}: {e}")

        # Processar e Salvar Rankings
        for year, metrics_dict in yearly_data.items():
            year_dir = self.top_n_path / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            for metric, values in metrics_dict.items():
                # Ordenar
                # Para distância euclidiana, menor é melhor. Para outros, maior é melhor.
                reverse = (metric != 'euclidean_distance')
                
                # Ordena
                values.sort(key=lambda x: x[0], reverse=reverse)
                
                # Pega Top N
                top_items = values[:self.top_n_to_save]
                
                # Salva
                out_df = pd.DataFrame([{'chemical_name': name, metric: val} for val, name in top_items])
                out_file = year_dir / f'top_{self.top_n_to_save}_{metric}.csv'
                out_df.to_csv(out_file, index=False)
                
        self.logger.info(f"Rankings generated in {self.top_n_path}")

    def run(self) -> bool:
        try:
            self.logger.info("=== Starting Validation Pipeline ===")
            
            # Passo 1: Gerar Históricos
            if self.generate_compound_histories():
                self.logger.info("Compound histories generated.")
                
                # Passo 2: Gerar Rankings
                self.generate_yearly_rankings()
                
                self.logger.info("=== Validation pipeline complete ===")
                return True
            else:
                self.logger.error("Failed to generate histories.")
                return False
                
        except Exception as e:
            self.logger.exception(f"Fatal error: {e}")
            return False

if __name__ == '__main__':
    # Exemplo de uso
    validator = ValidationModule(
        disease_name="acute myeloid leukemia",
        model_subfolder="w2v_fixed", # O usuário especifica qual variante quer processar
        start_year=1950,
        end_year=2020
    )
    validator.run()
