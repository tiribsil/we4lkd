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

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
import pyspark.sql # Added to resolve NameError for pyspark.sql.DataFrame

# UDF helper function for summary preprocessing
@F.udf(returnType=StringType())
def spark_summary_preprocessing_udf(text: str, html_tags_pattern: str, urls_pattern: str, punctuation_pattern: str, whitespace_pattern: str, disease_regex_pattern: Optional[str], canonical_name: str) -> str:
    if text is None or not text:
        return ""
    
    text = str(text).strip()
    text = re.sub(html_tags_pattern, '', text)
    text = re.sub(urls_pattern, '', text)
    text = re.sub(punctuation_pattern, '', text)
    text = re.sub(whitespace_pattern, ' ', text)
    
    if disease_regex_pattern:
        text = re.sub(disease_regex_pattern, canonical_name, text)
    
    return text.lower()

# UDF helper function for word cleaning
@F.udf(returnType=StringType())
def spark_clean_word_udf(word: str, stopwords_list: List[str], units_pattern: str) -> str:
    if not word or len(word) <= 1:
        return ''
    
    word_lower = word.lower()
    
    if word_lower in stopwords_list:
        return ''
    
    word_clean = re.sub(units_pattern, '', word_lower).strip()
    
    return word_clean if len(word_clean) > 1 else ''

spark_session = None

def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession."""
    global spark_session
    if spark_session is None:
        print("Initializing SparkSession...")
        spark_session = SparkSession.builder \
            .appName("PreprocessingPipeline") \
            .config("spark.executor.memory", "16g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.local.dir", "/tmp/spark-temp") \
            .getOrCreate()
        print("SparkSession initialized.")
    return spark_session

class Preprocessing:
    def __init__(self, disease_name: str, relevant_spacy_entity_types: Optional[List[str]] = None,
                 target_year: Optional[int] = None, incremental: bool = True):
        load_dotenv()

        self.logger = setup_logger("preprocessing", log_to_file=False)

        self.disease_name = normalize_disease_name(disease_name)
        self.spacy_model_name = "en_ner_bc5cdr_md"
        self.relevant_spacy_entity_types = relevant_spacy_entity_types or ["CHEMICAL"]
        self.mapped_entity_type = "pharmacologic_substance"
        self.batch_size = 1000
        self.target_year = target_year
        self.incremental = incremental

        # Spark Session
        self.spark = self._init_spark_session()

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

        
    def _compile_regex_patterns(self) -> Dict[str, str]:
        """Compila todos os regex patterns uma única vez e retorna como strings."""
        return {
            'html_tags': r'<[^>]+>',
            'urls': r'https?://\S+',
            'punctuation': r'[;:\(\)\[\]\{\}.,"!#$&\'*?@\\\^`|~]',
            'whitespace': r'\s+',
            'units': r'(%s)' % '|'.join(map(re.escape, self.units_and_symbols_list))
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
        """Creates a sorted list of (term, replacement) for compound normalization using PySpark."""
        if self._compound_replacement_map is not None:
            return self._compound_replacement_map

        self.logger.info("Creating compound replacement map using PySpark...")

        # Define paths for PubChem data
        synonyms_path = str(self.cid_synonym_filtered_file)
        titles_path = str(self.cid_title_file)
        ner_table_path = str(self.output_csv)
        whitelist_path = str(Path(f'./data/compound_whitelist.txt'))

        # Check for required files
        if not all([Path(synonyms_path).exists(), Path(titles_path).exists(), Path(ner_table_path).exists(), Path(whitelist_path).exists()]):
            self.logger.warning("Missing required files for compound normalization (synonyms, titles, NER table, or whitelist). Skipping.")
            return []

        try:
            # Read data using Spark
            synonyms_df_spark = self.spark.read.option("sep", "\t").csv(synonyms_path).toDF('cid', 'synonym')
            titles_df_spark = self.spark.read.option("sep", "\t").csv(titles_path).toDF('cid', 'title')
            ner_df_spark = self.spark.read.option("header", "true").option("sep", ",").csv(ner_table_path)
            whitelist_df_spark = self.spark.read.option("header", "false").option("sep", "\n").csv(whitelist_path).toDF('term')
        except Exception as e:
            self.logger.error(f"Error reading data for compound map with Spark: {e}")
            return []

        # Create the full compound map
        compound_map_df_spark = synonyms_df_spark.join(titles_df_spark, "cid") \
            .withColumn("replacement", F.regexp_replace(F.lower(F.col("title")), r'\\s+', '')) \
            .select(F.lower(F.col("synonym")).alias("term"), "replacement") \
            .where(F.length(F.trim(F.col("term"))) > 1) \
            .where(F.col("term") != F.col("replacement")) \
            .distinct()

        # Filter map using terms from the NER table AND compound whitelist
        self.logger.info("Filtering compound map using NER table and compound whitelist...")
        corpus_chemical_terms_spark = ner_df_spark.select(F.lower(F.col('token')).alias('ner_term')).distinct()
        
        # Union with whitelist terms
        whitelist_terms_spark = whitelist_df_spark.select(F.lower(F.col('term')).alias('ner_term')).distinct()
        all_chemical_terms_spark = corpus_chemical_terms_spark.unionByName(whitelist_terms_spark).distinct()

        filtered_map_df_spark = compound_map_df_spark.join(
            F.broadcast(all_chemical_terms_spark),
            compound_map_df_spark.term == all_chemical_terms_spark.ner_term,
            'inner'
        ).select("term", "replacement")

        # Collect and sort the map by term length (descending)
        self.logger.info("Collecting filtered compound map for replacement...")
        compound_map_rows = filtered_map_df_spark.collect()
        replacement_list = [(row.term, row.replacement) for row in compound_map_rows]
        replacement_list.sort(key=lambda x: len(x[0]), reverse=True)

        self.logger.info(f"Collected {len(replacement_list)} relevant compound terms for replacement.")
        self._compound_replacement_map = replacement_list
        return self._compound_replacement_map



    def _apply_compound_replacement(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Applies compound replacement using a broadcasted list and a UDF, mirroring the more efficient old implementation."""
        replacement_list = self._create_compound_replacement_map()
        if not replacement_list:
            self.logger.warning("No compound replacement map available. Skipping replacement.")
            return df

        self.logger.info(f"Applying {len(replacement_list)} compound synonym replacements using a broadcasted UDF...")

        # Broadcast the list to make it efficiently available to the UDF on all executors.
        broadcasted_replacements = self.spark.sparkContext.broadcast(replacement_list)

        # Define a Python function to perform the series of replacements.
        def replace_compounds_in_text(text: str) -> str:
            if text is None:
                return ''
            # Access the broadcasted list of replacements.
            for term, replacement in broadcasted_replacements.value:
                # Use re.sub for case-insensitive, whole-word replacement.
                pattern = r'(?i)\\b' + re.escape(term) + r'\\b'
                text = re.sub(pattern, replacement, text)
            return text

        # Register the Python function as a Spark UDF.
        replace_udf = F.udf(replace_compounds_in_text, StringType())

        # Apply the UDF to the summary column.
        return df.withColumn('summary', replace_udf(F.col('summary')))





    def _words_preprocessing_spark(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Preprocessamento de palavras usando operações PySpark."""
        # Replace typo corrections
        typo_corrections_broadcast = self.spark.sparkContext.broadcast(self.typo_corrections)

        @F.udf(returnType=StringType())
        def apply_typo_corrections_udf(word: str) -> str:
            if word is None:
                return ''
            return typo_corrections_broadcast.value.get(word, word)

        df = df.withColumn('word', apply_typo_corrections_udf(F.col('word')))

        # Apply spark_clean_word_udf
        df = df.withColumn('word', spark_clean_word_udf(
            F.col('word'),
            F.lit(list(self.stopwords)), # Pass stopwords as a list
            F.lit(self._compiled_regex['units'])
        ))
        
        # Filter words by length
        df = df.filter(F.length(F.col('word')) > 1)
        
        return df

    def _process_year_summaries(self, year: int) -> pyspark.sql.DataFrame:
        """Processa summaries de um ano específico e mantém o campo year_extracted corretamente usando PySpark."""
        year_file = str(Path(f'{self.aggregated_abstracts_path}/results_file_{year}.txt'))

        if not Path(year_file).exists():
            self.logger.warning(f"File not found: {year_file}")
            return self.spark.createDataFrame([], schema="summary STRING, year_extracted INT")

        self.logger.info(f"Processing year {year} using PySpark...")

        # Read summaries from the year file using Spark
        # Assuming the file format is "title|summary" or "summary"
        df_raw = self.spark.read.option("sep", "|").csv(year_file).toDF('title_or_summary', 'summary_col')

        # Determine if the file has a title column or just summaries
        if 'summary_col' in df_raw.columns:
            df = df_raw.withColumn('summary', F.col('summary_col'))
        else:
            df = df_raw.withColumn('summary', F.col('title_or_summary'))
        
        df = df.select('summary').withColumn('year_extracted', F.lit(year))

        # Initial cleaning
        self.logger.info(f"Cleaning summaries from year {year} using PySpark...")
        disease_regex, canonical_name = self._get_disease_regex()
        df = df.withColumn('summary', spark_summary_preprocessing_udf(
            F.col('summary'),
            F.lit(self._compiled_regex['html_tags']),
            F.lit(self._compiled_regex['urls']),
            F.lit(self._compiled_regex['punctuation']),
            F.lit(self._compiled_regex['whitespace']),
            F.lit(disease_regex.pattern if disease_regex else None),
            F.lit(canonical_name)
        ))
        df = df.filter(F.length(F.col('summary')) > 0)

        # Apply compound normalization
        df = self._apply_compound_replacement(df)

        # Tokenization
        df = df.withColumn('words', F.split(F.col('summary'), ' '))

        # Explode words, keeping original index for regrouping
        # Add a unique ID to each row before exploding to simulate Pandas' reset_index(drop=False) behavior
        df = df.withColumn("original_index", F.monotonically_increasing_id())
        df_exploded = df.withColumn('word', F.explode('words')).select('original_index', 'word', 'year_extracted')

        # Word cleaning
        df_exploded = self._words_preprocessing_spark(df_exploded)

        # Regroup back by original index
        window_spec = Window.partitionBy('original_index', 'year_extracted').orderBy('original_index')
        df_clean = df_exploded.withColumn('summary_list', F.collect_list('word').over(window_spec)) \
                              .groupBy('original_index', 'year_extracted') \
                              .agg(F.concat_ws(' ', F.first('summary_list')).alias('summary')) \
                              .select('summary', 'year_extracted')

        # Final filters
        df_clean = df_clean.filter(F.length(F.col('summary')) > 10)
        df_clean = df_clean.dropDuplicates(subset=['summary'])

        self.logger.info(f"Year {year}: {df_clean.count()} clean abstracts")
        return df_clean.select('summary', 'year_extracted')

    def clean_and_normalize_incremental(self, years_to_process: Optional[List[int]] = None):
        """Limpa e normaliza abstracts de forma incremental usando PySpark."""
        self.logger.info('Initializing incremental abstract cleaning using PySpark...')
        
        if years_to_process is None:
            years_to_process = self._get_years_to_process()
        
        if not years_to_process:
            self.logger.info("No new years to process. Using full processing mode.")
            return self.clean_and_normalize()
        
        self.logger.info(f"Processing years: {years_to_process}")
        
        # Processar cada ano
        yearly_results_spark = []
        for year in years_to_process:
            df_year_spark = self._process_year_summaries(year)
            if df_year_spark.count() > 0: # Check if DataFrame is not empty
                yearly_results_spark.append(df_year_spark)
                self._mark_year_as_processed(year)
        
        if not yearly_results_spark:
            self.logger.warning("No data processed from target years")
            return
        
        # Combinar novos resultados
        df_new_spark = yearly_results_spark[0]
        for i in range(1, len(yearly_results_spark)):
            df_new_spark = df_new_spark.unionByName(yearly_results_spark[i])
        
        df_new_spark = df_new_spark.dropDuplicates(subset=['summary'])
        
        # Arquivo de saída principal
        output_file = str(Path(f'{self.clean_papers_path}/clean_abstracts.csv'))
        
        if Path(output_file).exists() and self.incremental:
            # Modo incremental: carregar existente e combinar
            self.logger.info("Loading existing clean abstracts with PySpark...")
            df_existing_spark = self.spark.read.option("header", "true").option("sep", ",").csv(output_file)
            
            # Garantir que df_existing tem year_extracted
            if 'year_extracted' not in df_existing_spark.columns:
                self.logger.warning("Existing data missing year_extracted column. Cannot merge properly.")
                self.logger.info("Saving new data separately...")
                df_new_spark.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_file)
            else:
                df_combined_spark = df_existing_spark.unionByName(df_new_spark)
                df_combined_spark = df_combined_spark.dropDuplicates(subset=['summary'])
                
                self.logger.info(f"Combined: {df_existing_spark.count()} existing + {df_new_spark.count()} new = {df_combined_spark.count()} total")
                
                df_combined_spark.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_file)
        else:
            # Primeira execução ou modo full
            self.logger.info(f"Saving {df_new_spark.count()} clean abstracts with PySpark...")
            self.clean_papers_path.mkdir(parents=True, exist_ok=True)
            df_new_spark.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_file)
        
        self.logger.info(f'Incremental cleaning completed. Output: {output_file}')

    def clean_and_normalize(self):
        """Limpa e normaliza abstracts usando PySpark (modo full)."""
        self.logger.info('Initializing full abstract cleaning using PySpark...')

        txt_files = list(self.aggregated_abstracts_path.glob('*.txt'))
        
        if not txt_files:
            self.logger.error(f"No .txt files found in {self.aggregated_abstracts_path}")
            return
        
        all_dfs_spark = []
        
        for txt_file in txt_files:
            self.logger.info(f"Reading {txt_file.name} with PySpark...")
            
            # Extract year from filename
            year = self._extract_year_from_filename(txt_file)
            
            # Read summaries from the file using Spark
            df_raw = self.spark.read.option("sep", "|").csv(str(txt_file)).toDF('title_or_summary', 'summary_col')

            if 'summary_col' in df_raw.columns:
                df = df_raw.withColumn('summary', F.col('summary_col'))
            else:
                df = df_raw.withColumn('summary', F.col('title_or_summary'))
            
            df = df.select('summary').withColumn('year_extracted', F.lit(year))
            all_dfs_spark.append(df)
        
        if not all_dfs_spark:
            self.logger.warning("No data loaded from aggregated abstracts.")
            return

        df_combined_spark = all_dfs_spark[0]
        for i in range(1, len(all_dfs_spark)):
            df_combined_spark = df_combined_spark.unionByName(all_dfs_spark[i])

        self.logger.info(f"Loaded {df_combined_spark.count()} abstracts")

        # Preprocessing vectorized
        self.logger.info("Cleaning summaries using PySpark...")
        disease_regex, canonical_name = self._get_disease_regex()
        df_combined_spark = df_combined_spark.withColumn('summary', spark_summary_preprocessing_udf(
            F.col('summary'),
            F.lit(self._compiled_regex['html_tags']),
            F.lit(self._compiled_regex['urls']),
            F.lit(self._compiled_regex['punctuation']),
            F.lit(self._compiled_regex['whitespace']),
            F.lit(disease_regex.pattern if disease_regex else None),
            F.lit(canonical_name)
        ))
        df_combined_spark = df_combined_spark.filter(F.length(F.col('summary')) > 0)

        # Apply compound normalization
        df_combined_spark = self._apply_compound_replacement(df_combined_spark)

        # Tokenization
        self.logger.info("Tokenizing...")
        df_combined_spark = df_combined_spark.withColumn('words', F.split(F.col('summary'), ' '))
        
        # Add a unique ID to each row before exploding to simulate Pandas' reset_index(drop=False) behavior
        df_combined_spark = df_combined_spark.withColumn("original_index", F.monotonically_increasing_id())
        df_exploded_spark = df_combined_spark.withColumn('word', F.explode('words')).select('original_index', 'word', 'year_extracted')

        # Word cleaning
        self.logger.info("Cleaning words using PySpark...")
        df_exploded_spark = self._words_preprocessing_spark(df_exploded_spark)

        # Regroup maintaining year_extracted
        self.logger.info("Regrouping...")
        window_spec = Window.partitionBy('original_index', 'year_extracted').orderBy('original_index')
        df_clean_spark = df_exploded_spark.withColumn('summary_list', F.collect_list('word').over(window_spec)) \
                                          .groupBy('original_index', 'year_extracted') \
                                          .agg(F.concat_ws(' ', F.first('summary_list')).alias('summary')) \
                                          .select('summary', 'year_extracted')
        
        df_clean_spark = df_clean_spark.filter(F.length(F.col('summary')) > 10)
        df_clean_spark = df_clean_spark.dropDuplicates(subset=['summary'])

        # Save
        self.logger.info('Saving clean abstracts with PySpark...')
        self.clean_papers_path.mkdir(parents=True, exist_ok=True)
        output_file = str(Path(f'{self.clean_papers_path}/clean_abstracts.csv'))
        
        df_clean_spark.coalesce(1).write.mode('overwrite').option("header", "true").csv(output_file)

        self.logger.info(f'Full cleaning completed. Saved {df_clean_spark.count()} clean abstracts to {output_file}')

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

    def _init_spark_session(self) -> SparkSession:
        """Initializes and returns a SparkSession."""
        return get_spark_session()

    def _stop_spark_session(self):
        """Stops the SparkSession."""
        global spark_session
        if spark_session:
            self.logger.info("Stopping SparkSession...")
            spark_session.stop()
            spark_session = None
            self.logger.info("SparkSession stopped.")

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
        # finally:
        #     self._stop_spark_session()


if __name__ == "__main__":

    preprocessing_module = Preprocessing(
        disease_name="acute myeloid leukemia",
        incremental=True
    )
    
    # Para forçar processamento completo: force_full=True
    success = preprocessing_module.run(force_full=False)
    
    exit(0 if success else 1)