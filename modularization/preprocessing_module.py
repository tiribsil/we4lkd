import os
import re
import gzip
import shutil
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple, Set
import spacy
import nltk
import pandas as pd
from functools import lru_cache
from Bio import Entrez
from nltk.tokenize import word_tokenize
from utils import *

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)



class Preprocessing:
    def __init__(self, disease_name: str, relevant_spacy_entity_types: Optional[List[str]] = None,
                 target_year: Optional[int] = None, incremental: bool = True):
        load_dotenv()

        self.logger = LoggerFactory.setup_logger("preprocessing", str(target_year), log_to_file=False)

        self.disease_name = normalize_disease_name(disease_name)
        self.spacy_model_name = "en_ner_bc5cdr_md"
        self.relevant_spacy_entity_types = relevant_spacy_entity_types or ["CHEMICAL"]
        self.mapped_entity_type = "pharmacologic_substance"
        self.batch_size = 1000
        self.target_year = target_year
        self.incremental = incremental

        #Paths
        self.data_dir = Path(f"./data/{self.disease_name}/corpus")
        self.output_csv = Path(f"{self.data_dir}/ner_table.csv")
        self.input_file = Path(f"{self.data_dir}/aggregated_abstracts/aggregated_corpus.txt")
        self.clean_papers_path = Path(f'./data/{self.disease_name}/corpus/clean_abstracts')
        self.aggregated_abstracts_path = Path(f'./data/{self.disease_name}/corpus/aggregated_abstracts')
        self.processed_years_file = Path(f'{self.clean_papers_path}/processed_years.txt')

        self.pubchem_data_dir = Path("./data/pubchem_data/")
        self.cid_title_gz = self.pubchem_data_dir / "CID-Title.gz"
        self.cid_synonym_filtered_gz = self.pubchem_data_dir / "CID-Synonym-filtered.gz"
        self.cid_title_file = self.pubchem_data_dir / "CID-Title"
        self.cid_synonym_filtered_file = self.pubchem_data_dir / "CID-Synonym-filtered"

        self.nlp = None
        self.load_spacy_model()

        self.email = os.getenv("ENTREZ_EMAIL")
        self.api_key = os.getenv("ENTREZ_KEY")
        Entrez.email = self.email
        Entrez.api_key = self.api_key

        self._download_nltk_resources()
        
        # Configurações padrão
        self.nature_filtered_words = frozenset([
            'foreword', 'prelude', 'commentary', 'workshop', 'conference', 'symposium',
            'comment', 'retract', 'correction', 'erratum', 'memorial'
        ])
        self.typo_corrections = default_typo_corrections()
        self.units_and_symbols_list = default_units_and_symbols()
        self._compiled_regex = self._compile_regex_patterns()

        self._stopwords = None
        self._compound_replacement_map = None
        self.base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"

        
    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Compila todos os regex patterns uma única vez."""
        return {
            'html_tags': re.compile(r'<[^>]+>'),
            'urls': re.compile(r'https?://\S+'),
            'punctuation': re.compile(r'[;:\(\)\[\]\{\}.,"!#$&\'*?@\\\^`|~]'),
            'whitespace': re.compile(r'\s+'),
            'units': re.compile('(%s)' % '|'.join(map(re.escape, self.units_and_symbols_list)))
        }

    @property
    def stopwords(self):
        """Lazy loading de stopwords."""
        if self._stopwords is None:
            self._stopwords = frozenset(nltk.corpus.stopwords.words('english'))
        return self._stopwords

    def _download_nltk_resources(self):
        """Download de recursos NLTK."""
        resources = ['wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4', 'stopwords']
        for resource in resources:
            nltk.download(resource, quiet=True)

    # ============================================
    # --------- Processamento Incremental --------
    # ============================================

    def _get_processed_years(self) -> Set[int]:
        """Retorna conjunto de anos já processados."""
        if not self.processed_years_file.exists():
            return set()
        
        with open(self.processed_years_file, 'r') as f:
            return set(int(line.strip()) for line in f if line.strip().isdigit())

    def _mark_year_as_processed(self, year: int):
        """Marca um ano como processado."""
        self.processed_years_file.parent.mkdir(parents=True, exist_ok=True)
        
        processed_years = self._get_processed_years()
        processed_years.add(year)
        
        with open(self.processed_years_file, 'w') as f:
            for y in sorted(processed_years):
                f.write(f"{y}\n")

    def _get_years_to_process(self) -> List[int]:
        """Determina quais anos precisam ser processados."""
        if not self.incremental:
            # Modo full: processar tudo
            return []
        
        # Verificar arquivos agregados disponíveis
        available_years = set()
        for file in self.aggregated_abstracts_path.glob('results_file_*.txt'):
            try:
                year = int(file.stem.split('_')[-1])
                available_years.add(year)
            except ValueError:
                continue
        
        processed_years = self._get_processed_years()
        
        # Anos que precisam ser processados
        years_to_process = available_years - processed_years
        
        # Se target_year especificado, forçar reprocessamento
        if self.target_year and self.target_year in available_years:
            years_to_process.add(self.target_year)
        
        return sorted(years_to_process)

    def _extract_year_from_filename(self, file_path: Path) -> Optional[int]:
        """Extrai ano do nome do arquivo results_file_XXXX.txt"""
        try:
            # Padrão: results_file_2020.txt -> 2020
            year_str = file_path.stem.split('_')[-1]
            return int(year_str)
        except (ValueError, IndexError):
            self.logger.warning(f"Could not extract year from filename: {file_path.name}")
            return None

    # ============================================
    # -------------- NER Extraction --------------
    # ============================================

    def load_spacy_model(self) -> bool:
        """Load the spaCy NER model."""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            pipes_to_disable = ['parser', 'lemmatizer', 'textcat', 'tagger', 'attribute_ruler']
            available_pipes = self.nlp.pipe_names
            
            for pipe in pipes_to_disable:
                if pipe in available_pipes and pipe != 'ner':
                    self.nlp.disable_pipe(pipe)
            
            self.logger.info(f"'{self.spacy_model_name}' loaded. Active pipes: {self.nlp.pipe_names}. GPU: {spacy.prefer_gpu()}")
            return True
        except OSError:
            self.logger.error(f"Model '{self.spacy_model_name}' not found. Please, download it first.")
        except Exception as e:
            self.logger.exception(f"Unexpected error loading spaCy model: {e}")
        return False

    def _read_abstracts_from_file(self, file_path: Path) -> List[str]:
        """Read abstracts from a specific file."""
        if not file_path.exists():
            return []

        texts = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                abstract = parts[1] if len(parts) == 2 else parts[0]
                texts.append(abstract)

        return texts

    def _read_abstracts(self, years: Optional[List[int]] = None) -> List[str]:
        """Read abstracts, optionally filtered by years."""
        if years:
            # Processar apenas anos específicos
            all_texts = []
            for year in years:
                year_file = Path(f'{self.aggregated_abstracts_path}/results_file_{year}.txt')
                texts = self._read_abstracts_from_file(year_file)
                self.logger.info(f"Read {len(texts)} abstracts from year {year}")
                all_texts.extend(texts)
            return all_texts
        else:
            # Processar corpus completo
            if not self.input_file.exists():
                self.logger.warning(f"File not found: {self.input_file}")
                return []

            self.logger.info(f"Reading {self.input_file}...")
            return self._read_abstracts_from_file(self.input_file)

    def _extract_entities_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Extract named entities from a batch of texts."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=50, n_process=1):
            for ent in doc.ents:
                if ent.label_ in self.relevant_spacy_entity_types:
                    results.append({
                        "token": ent.text.lower(),
                        "entity": self.mapped_entity_type
                    })
        return results

    def process_abstracts(self, years: Optional[List[int]] = None) -> List[Dict[str, str]]:
        """Process abstracts and return extracted entities."""
        texts = self._read_abstracts(years)
        if not texts:
            return []

        ner_results = []
        total_texts = len(texts)
        self.logger.info(f"Processing {total_texts} abstracts in batches of {self.batch_size}...")

        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._extract_entities_batch(batch)
            ner_results.extend(batch_results)
            
            if (i // self.batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed batch {i // self.batch_size + 1}/{(total_texts // self.batch_size) + 1}")

        return ner_results

    def save_ner_table(self, ner_results: List[Dict[str, str]], append: bool = False) -> bool:
        """Save the extracted entities into a CSV file."""
        if not ner_results:
            self.logger.warning("No NER results to save.")
            return False

        unique_results = {(r['token'], r['entity']) for r in ner_results}
        new_df = pd.DataFrame([{'token': t, 'entity': e} for t, e in unique_results])
        
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        if append and self.output_csv.exists():
            # Modo incremental: carregar existente e mesclar
            existing_df = pd.read_csv(self.output_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['token', 'entity'], keep='first')
            combined_df.to_csv(self.output_csv, index=False)
            self.logger.info(f"{len(combined_df)} unique NER tokens (including {len(new_df)} new) saved at {self.output_csv}")
        else:
            # Modo full: sobrescrever
            new_df.to_csv(self.output_csv, index=False)
            self.logger.info(f"{len(new_df)} unique NER tokens saved at {self.output_csv}")
        
        return True

    # ============================================
    # ---------------Text Cleaning ---------------
    # ============================================

    @lru_cache(maxsize=1)
    def _find_mesh_terms(self) -> List[str]:
        """Find MeSH terms (cached)."""
        handle = Entrez.esearch(db="mesh", term=f"\"{self.disease_name}\"[MeSH Terms]", retmax="20")
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    @lru_cache(maxsize=10)
    def _get_mesh_details(self, mesh_id: str) -> str:
        """Get MeSH details (cached)."""
        handle = Entrez.efetch(db="mesh", id=mesh_id, retmode="xml")
        records = handle.read()
        handle.close()
        return records

    @staticmethod
    def _get_mesh_disease_synonyms(disease_details: str) -> List[str]:
        """Extract synonyms from MeSH details."""
        lines = disease_details.splitlines()
        synonyms, start = [], False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Entry Terms:'):
                start = True
                continue
            if start:
                synonyms.append(line.lower())
        return synonyms

    @lru_cache(maxsize=1)
    def _get_disease_regex(self) -> Tuple[Optional[re.Pattern], str]:
        """Get compiled regex for disease synonyms (cached)."""
        mesh_ids = self._find_mesh_terms()
        if not mesh_ids:
            return None, ""
        
        disease_details = self._get_mesh_details(mesh_ids[0])
        synonyms = self._get_mesh_disease_synonyms(disease_details)
        
        if synonyms:
            regex_pattern = r'(?i)({})'.format('|'.join(map(re.escape, synonyms)))
            compiled_regex = re.compile(regex_pattern)
            canonical_name = self.disease_name.lower().replace(' ', '_')
            return compiled_regex, canonical_name
        
        return None, ""

    def _create_compound_replacement_map(self) -> List[Tuple[str, str]]:
        """Creates a sorted list of (term, replacement) for compound normalization."""
        if self._compound_replacement_map is not None:
            return self._compound_replacement_map

        self.logger.info("Creating compound replacement map...")

        # Define paths for PubChem data
        synonyms_path = self.cid_synonym_filtered_file
        titles_path = self.cid_title_file
        ner_table_path = self.output_csv

        # Check for required files
        if not all([synonyms_path.exists(), titles_path.exists(), ner_table_path.exists()]):
            self.logger.warning("Missing required files for compound normalization (synonyms, titles, or NER table). Skipping.")
            return []

        # Read data using Pandas
        try:
            synonyms_df = pd.read_csv(synonyms_path, sep='\t', header=None, names=['cid', 'synonym'], on_bad_lines='warn')
            titles_df = pd.read_csv(titles_path, sep='\t', header=None, names=['cid', 'title'], on_bad_lines='warn')
            ner_df = pd.read_csv(ner_table_path)
        except Exception as e:
            self.logger.error(f"Error reading data for compound map: {e}")
            return []

        # Create the full compound map
        compound_map_df = pd.merge(synonyms_df, titles_df, on='cid')
        compound_map_df['replacement'] = compound_map_df['title'].str.lower().str.replace(r'\s+', '', regex=True)
        compound_map_df['term'] = compound_map_df['synonym'].str.lower()
        compound_map_df = compound_map_df[['term', 'replacement']].dropna()
        compound_map_df = compound_map_df[compound_map_df['term'].str.len() > 1]
        compound_map_df = compound_map_df[compound_map_df['term'] != compound_map_df['replacement']]
        compound_map_df = compound_map_df.drop_duplicates()

        # Filter map using terms from the NER table
        self.logger.info("Filtering compound map using NER table...")
        corpus_chemical_terms = set(ner_df['token'].str.lower().unique())
        filtered_map_df = compound_map_df[compound_map_df['term'].isin(corpus_chemical_terms)]

        # Collect and sort the map by term length (descending)
        replacement_list = list(filtered_map_df.itertuples(index=False, name=None))
        replacement_list.sort(key=lambda x: len(x[0]), reverse=True)

        self.logger.info(f"Collected {len(replacement_list)} relevant compound terms for replacement.")
        self._compound_replacement_map = replacement_list
        return self._compound_replacement_map

    def _replace_compounds_in_text(self, text: str, replacement_map: List[Tuple[str, str]]) -> str:
        """Replaces compound synonyms in a single text."""
        if not text or not replacement_map:
            return text
        for term, replacement in replacement_map:
            pattern = r'(?i)\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, replacement, text)
        return text

    def _apply_compound_replacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies compound replacement to the 'summary' column of a DataFrame."""
        replacement_map = self._create_compound_replacement_map()
        if not replacement_map:
            self.logger.warning("No compound replacement map available. Skipping replacement.")
            return df

        self.logger.info("Applying compound synonym replacement...")
        df['summary'] = df['summary'].apply(lambda x: self._replace_compounds_in_text(x, replacement_map))
        return df

    def _summary_preprocessing_pandas(self, text: str) -> str:
        """Preprocessing de texto usando regex pré-compilados."""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).strip()
        text = self._compiled_regex['html_tags'].sub('', text)
        text = self._compiled_regex['urls'].sub('', text)
        text = self._compiled_regex['punctuation'].sub('', text)
        text = self._compiled_regex['whitespace'].sub(' ', text)
        
        disease_regex, canonical_name = self._get_disease_regex()
        if disease_regex:
            text = disease_regex.sub(canonical_name, text)
        
        return text.lower()

    def _clean_word(self, word: str) -> str:
        """Limpa uma palavra individual."""
        if not word or len(word) <= 1:
            return ''
        
        word_lower = word.lower()
        
        if word_lower in self.stopwords:
            return ''
        
        word_clean = self._compiled_regex['units'].sub('', word_lower).strip()
        
        return word_clean if len(word_clean) > 1 else ''

    def _words_preprocessing_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessamento de palavras usando operações vetorizadas."""
        df['word'] = df['word'].replace(self.typo_corrections)
        df['word'] = df['word'].apply(self._clean_word)
        df = df[df['word'].str.len() > 1].copy()
        
        return df

    def _process_year_summaries(self, year: int, chunk_size: int = 10000) -> pd.DataFrame:
        """Processa summaries de um ano específico e mantém o campo year_extracted corretamente."""
        year_file = Path(f'{self.aggregated_abstracts_path}/results_file_{year}.txt')

        if not year_file.exists():
            self.logger.warning(f"File not found: {year_file}")
            return pd.DataFrame()

        self.logger.info(f"Processing year {year}...")

        # Lê os summaries do arquivo do ano
        summaries = []
        with open(year_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|', 1)
                summary = parts[1] if len(parts) == 2 else parts[0]
                summaries.append(summary)

        df = pd.DataFrame({'summary': summaries})
        df['year_extracted'] = year

        # Limpeza inicial
        self.logger.info(f"Cleaning {len(df)} summaries from year {year}...")
        df['summary'] = df['summary'].apply(self._summary_preprocessing_pandas)
        df = df[df['summary'].str.len() > 0].copy()

        # Aplicar normalização de compostos
        df = self._apply_compound_replacement(df)

        # Tokenização
        df['words'] = df['summary'].str.split()

        # Explode mantendo o índice do resumo original
        df_exploded = df.explode('words').rename(columns={'words': 'word'}).reset_index(drop=False)

        # Limpeza de palavras
        df_exploded['word'] = df_exploded['word'].replace(self.typo_corrections)
        df_exploded['word'] = df_exploded['word'].apply(self._clean_word)
        df_exploded = df_exploded[df_exploded['word'].str.len() > 1]

        # Reagrupar de volta por índice original
        df_clean = (
            df_exploded.groupby('index', as_index=False)
            .agg({
                'word': lambda x: ' '.join(x),
                'year_extracted': 'first'
            })
            .rename(columns={'word': 'summary'})
        )

        # Filtros finais
        df_clean = df_clean[df_clean['summary'].str.len() > 10]
        df_clean = df_clean.drop_duplicates(subset='summary', keep='first').reset_index(drop=True)

        self.logger.info(f"Year {year}: {len(df_clean)} clean abstracts")
        return df_clean[['summary', 'year_extracted']]

    def clean_and_normalize_incremental(self, years_to_process: Optional[List[int]] = None):
        """Limpa e normaliza abstracts de forma incremental."""
        self.logger.info('Initializing incremental abstract cleaning...')
        
        if years_to_process is None:
            years_to_process = self._get_years_to_process()
        
        if not years_to_process:
            self.logger.info("No new years to process. Using full processing mode.")
            return self.clean_and_normalize()
        
        self.logger.info(f"Processing years: {years_to_process}")
        
        # Processar cada ano
        yearly_results = []
        for year in years_to_process:
            df_year = self._process_year_summaries(year)
            if not df_year.empty:
                yearly_results.append(df_year)
                self._mark_year_as_processed(year)
        
        if not yearly_results:
            self.logger.warning("No data processed from target years")
            return
        
        # Combinar novos resultados
        df_new = pd.concat(yearly_results, ignore_index=True)
        df_new = df_new.drop_duplicates(subset='summary', keep='first')
        
        # Arquivo de saída principal
        output_file = Path(f'{self.clean_papers_path}/clean_abstracts.csv')
        
        if output_file.exists() and self.incremental:
            # Modo incremental: carregar existente e combinar
            self.logger.info("Loading existing clean abstracts...")
            df_existing = pd.read_csv(output_file)
            
            # Garantir que df_existing tem year_extracted
            if 'year_extracted' not in df_existing.columns:
                self.logger.warning("Existing data missing year_extracted column. Cannot merge properly.")
                self.logger.info("Saving new data separately...")
                df_new.to_csv(output_file, index=False)
            else:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset='summary', keep='first')
                
                self.logger.info(f"Combined: {len(df_existing)} existing + {len(df_new)} new = {len(df_combined)} total")
                
                df_combined.to_csv(output_file, index=False)
        else:
            # Primeira execução ou modo full
            self.logger.info(f"Saving {len(df_new)} clean abstracts...")
            self.clean_papers_path.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(output_file, index=False)
        
        self.logger.info(f'Incremental cleaning completed. Output: {output_file}')

    def clean_and_normalize(self, chunk_size: int = 10000):
        """Limpa e normaliza abstracts usando processamento em chunks (modo full)."""
        self.logger.info('Initializing full abstract cleaning...')

        txt_files = list(self.aggregated_abstracts_path.glob('*.txt'))
        
        if not txt_files:
            self.logger.error(f"No .txt files found in {self.aggregated_abstracts_path}")
            return
        
        all_data = []
        
        for txt_file in txt_files:
            self.logger.info(f"Reading {txt_file.name}...")
            
            # Extrair ano do nome do arquivo
            year = self._extract_year_from_filename(txt_file)
            
            chunk_data = []
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('|', 1)
                    summary = parts[1] if len(parts) == 2 else parts[0]
                    chunk_data.append({'summary': summary, 'year_extracted': year})
                    
                    if len(chunk_data) >= chunk_size:
                        all_data.extend(chunk_data)
                        chunk_data = []
                
                if chunk_data:
                    all_data.extend(chunk_data)
        
        df = pd.DataFrame(all_data)
        self.logger.info(f"Loaded {len(df)} abstracts")

        # Preprocessing vetorizado
        self.logger.info("Cleaning summaries...")
        df['summary'] = df['summary'].apply(self._summary_preprocessing_pandas)
        df = df[df['summary'].str.len() > 0]

        # Aplicar normalização de compostos
        df = self._apply_compound_replacement(df)

        # Tokenização
        self.logger.info("Tokenizing...")
        df['words'] = df['summary'].str.split()
        df_exploded = df.explode('words').rename(columns={'words': 'word'})
        
        # Preservar year_extracted
        df_exploded = df_exploded[['word', 'year_extracted']].copy()

        # Limpeza de palavras
        self.logger.info("Cleaning words...")
        df_exploded = self._words_preprocessing_pandas(df_exploded)

        # Reagrupar mantendo year_extracted
        self.logger.info("Regrouping...")
        df_clean = df_exploded.groupby(df_exploded.index).agg({
            'word': lambda x: ' '.join(x),
            'year_extracted': 'first'
        }).reset_index(drop=True)
        
        df_clean.columns = ['summary', 'year_extracted']
        
        df_clean = df_clean[df_clean['summary'].str.len() > 10]
        df_clean = df_clean.drop_duplicates(subset='summary', keep='first')
        df_clean = df_clean[['summary', 'year_extracted']].reset_index(drop=True)

        # Salvar
        self.logger.info('Saving clean abstracts...')
        self.clean_papers_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(f'{self.clean_papers_path}/clean_abstracts.csv')
        
        df_clean.to_csv(output_file, index=False)

        self.logger.info(f'Full cleaning completed. Saved {len(df_clean)} clean abstracts to {output_file}')

    # ============================================
    # --------- Download PubChem archives --------
    # ============================================

    def download_if_missing(self, file_name: str, target_dir: Path) -> Optional[Path]:
        """Download and extract PubChem file if missing."""
        target_dir.mkdir(parents=True, exist_ok=True)

        gz_path = target_dir / file_name
        extracted_path = gz_path.with_suffix('')
        url = f"{self.base_url}{file_name}"

        if extracted_path.exists():
            self.logger.info(f"{extracted_path.name} already exists. Skipping download.")
            return extracted_path

        if not gz_path.exists():
            self.logger.info(f"Downloading: {url}")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (block_size * 100) == 0:
                            progress = (downloaded / total_size) * 100
                            self.logger.info(f"Download progress: {progress:.1f}%")
                
                self.logger.info(f"Download completed: {gz_path}")
            except Exception as e:
                self.logger.error(f"Error downloading '{file_name}': {e}")
                if gz_path.exists():
                    gz_path.unlink()
                return None
        else:
            self.logger.info(f"Compressed file found: {gz_path}")

        try:
            self.logger.info(f"Extracting '{gz_path.name}' ...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1024*1024)
            self.logger.info(f"Extracted to: {extracted_path}")
        except Exception as e:
            self.logger.error(f"Error extracting '{gz_path}': {e}")
            if extracted_path.exists():
                extracted_path.unlink()
            return None

        return extracted_path

    def run(self, force_full: bool = False) -> bool:
        """Main execution pipeline."""
        try:
            # Download PubChem data (se necessário)
            self.logger.info("=== Checking PubChem data ===")
            
            # Download and extract CID-Title
            result_title = self.download_if_missing(self.cid_title_gz.name, self.pubchem_data_dir)
            if result_title is None:
                self.logger.warning(f"Failed to download or extract {self.cid_title_gz.name}")
                return False
            self.cid_title_file = result_title # Update to the extracted path

            # Download and extract CID-Synonym-filtered
            result_synonym = self.download_if_missing(self.cid_synonym_filtered_gz.name, self.pubchem_data_dir)
            if result_synonym is None:
                self.logger.warning(f"Failed to download or extract {self.cid_synonym_filtered_gz.name}")
                return False
            self.cid_synonym_filtered_file = result_synonym # Update to the extracted path

            # NER extraction
            self.logger.info("=== Starting NER table generation ===")
            if not self.nlp:
                self.logger.error("spaCy model not loaded")
                return False

            years_to_process = None if force_full or not self.incremental else self._get_years_to_process()
            
            if years_to_process:
                self.logger.info(f"Processing NER for years: {years_to_process}")
                ner_data = self.process_abstracts(years=years_to_process)
                if ner_data:
                    self.save_ner_table(ner_data, append=True)
            else:
                self.logger.info("Processing NER for full corpus")
                ner_data = self.process_abstracts()
                if ner_data:
                    self.save_ner_table(ner_data, append=False)

            # Text cleaning
            self.logger.info("=== Starting preprocessing pipeline ===")
            
            if force_full or not self.incremental:
                self.clean_and_normalize()
            else:
                self.clean_and_normalize_incremental(years_to_process)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error in pipeline: {e}")
            return False


if __name__ == "__main__":

    preprocessing_module = Preprocessing(
        disease_name="acute myeloid leukemia",
        incremental=True
    )
    
    # Para forçar processamento completo: force_full=True
    success = preprocessing_module.run(force_full=False)
    
    exit(0 if success else 1)