import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import re

from Bio import Entrez

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from src.utils import *

import nltk
from nltk.tokenize import word_tokenize

from functools import reduce

def ss():
    spark = SparkSession.builder \
        .appName("LargeDataProcessingApp")\
        .config("spark.executor.memory", "64g")\
        .config("spark.driver.memory", "32g")\
        .config("spark.driver.maxResultSize", "8g")\
        .config("spark.sql.shuffle.partitions", "1000")\
        .config("spark.sql.adaptive.enabled", "true")\
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")\
        .config("spark.local.dir", "/tmp/spark-temp")\
        .getOrCreate()
    return spark


def dataframes_from_txt(summaries_path, nature_filtered_words: list = None):
    # This part of your code is perfect. It correctly finds and sorts the files.
    filenames = sorted([str(x) for x in Path(summaries_path).glob('*.txt')])
    if not filenames:
        print("Warning: No text files found in the specified path.")
        # Return an empty DataFrame with the expected schema
        schema = "title STRING, summary STRING, year_of_first_appearance STRING"
        return ss().createDataFrame([], schema)

    all_dfs_with_duplicates = []

    for file_path in filenames:
        try:
            # Using 'year_of_file' is good, but let's rename the final column for clarity
            year_of_file = Path(file_path).stem.split('_')[-1]
            int(year_of_file)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract year from filename {Path(file_path).name}. Skipping.")
            continue

        title_doesnt_have_nature_filtered_words = reduce(
            lambda acc, word: acc & (F.locate(word, F.lower(F.col('title'))) == F.lit(0)),
            nature_filtered_words,
            F.lit(True)
        )

        df = ss().read \
            .option('header', 'false') \
            .option('lineSep', '\n') \
            .option('sep', '|') \
            .option('quote', '') \
            .csv(file_path) \
            .withColumnRenamed('_c0', 'title') \
            .withColumnRenamed('_c1', 'summary') \
            .withColumn('year', F.lit(year_of_file)) \
            .where(F.col('title').isNotNull() & F.col('summary').isNotNull()) \
            .where(title_doesnt_have_nature_filtered_words)

        all_dfs_with_duplicates.append(df)

    if not all_dfs_with_duplicates:
        schema = "title STRING, summary STRING, year_of_first_appearance STRING"
        return ss().createDataFrame([], schema)

    unioned_df = reduce(lambda df1, df2: df1.union(df2), all_dfs_with_duplicates)

    final_df = unioned_df.groupBy("title", "summary") \
        .agg(F.min("year").alias("filename"))

    final_df = final_df.withColumn('id', F.monotonically_increasing_id())

    return final_df


def read_table_file(file_path, sep, has_header):
    return ss()\
        .read\
        .option('header', has_header)\
        .option('sep', sep)\
        .csv(file_path)


def to_csv(df, target_folder, num_files=1, sep=','):
    """Saves a PySpark Dataframe into .csv file.
    Args:
        sep: separator used in the .csv file, default is comma (',');
        df: object of the DataFrame;
        target_folder: path where the .csv is going to be saved;
        num_files: number of .csv files to be created, default is 1.
    """

    return df\
        .coalesce(num_files)\
        .write\
        .mode('overwrite')\
        .option('header', 'true')\
        .option('sep', sep)\
        .format('csv')\
        .save(target_folder)


def rename_csv_in_folder(directory: str, new_filename: str):
    try:
        # Cria um objeto Path para o diretório
        p = Path(directory)
        if not p.is_dir():
            print(f"Erro: O diretório '{directory}' não foi encontrado.")
            return

        # Encontra o primeiro arquivo .csv de forma eficiente
        csv_file = next(p.glob('*.csv'), None)

        if csv_file:
            # O novo caminho será dentro do mesmo diretório, mas com o novo nome
            destination_path = csv_file.with_name(new_filename)

            print(f"Arquivo encontrado: '{csv_file.name}'")
            print(f"Renomeando para: '{destination_path}'")

            # Renomeia o arquivo
            csv_file.rename(destination_path)
        else:
            print(f"Aviso: Nenhum arquivo .csv foi encontrado no diretório '{directory}'.")

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


def tokenize(data):
    """Tokenizes a sentence

        Args:
        data: a sentence (string).
    """
    if data is None:
        return ['']
    else:
        return word_tokenize(data)


def find_mesh_terms(disease_name: str, entrez_email: str):
    """
    Searches for MeSH terms related to a disease name.
    Args:
        disease_name (str): The name of the disease to search for.
        entrez_email (str): Email address for Entrez API.

    Returns:
        list: A list of MeSH IDs.
    """
    Entrez.email = entrez_email
    handle = Entrez.esearch(db="mesh", term=f"\"{disease_name}\"[MeSH Terms] ", retmax="20")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def get_mesh_details(mesh_id: str, entrez_email: str):
    """
    Fetches detailed information for a given MeSH ID.
    Args:
        mesh_id (str): The MeSH ID to fetch details for.
        entrez_email (str): Email address for Entrez API.

    Returns:
        str: XML records for the MeSH ID.
    """
    Entrez.email = entrez_email
    handle = Entrez.efetch(db="mesh", id=mesh_id, retmode="xml")
    records = handle.read()
    handle.close()
    return records


def get_mesh_disease_synonyms(disease_details):
    lines = disease_details.splitlines()
    synonyms = []
    start = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Entry Terms:'):
            start = True
            continue
        if start:
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                term = ' '.join(reversed(parts))
            else:
                term = line
            synonyms.append(term.lower())
    return synonyms


def summary_column_preprocessing(column, target_disease: str, entrez_email: str):
    """Executes initial preprocessing in a PySpark text column. It removes some unwanted regex from the text.
    Args:
        column: the name of the column to be processed.
        target_disease (str): The name of the target disease.
        entrez_email (str): Email address for Entrez API.
    """
    disease_details = get_mesh_details(find_mesh_terms(target_disease, entrez_email), entrez_email)

    disease_synonyms = get_mesh_disease_synonyms(disease_details)
    canonical_name = target_disease.lower().replace(' ', '_')

    # Cria um regex para procurar os sinônimos da doença.
    regex = r'(?i)({})'.format('|'.join(disease_synonyms))

    # Remove os espaços em branco do início e fim da coluna.
    column = F.trim(column)

    # Remove tags HTML e URLs.
    column = F.regexp_replace(column, r'<[^>]+>', '')
    column = F.regexp_replace(column, r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '')

    # Remove caracteres especiais.
    column = F.regexp_replace(column, r'[;:\(\)\[\]\{\}.,"!#$&\'*?@\\\^`|~]', '')

    # Normaliza o espaço em branco.
    column = F.regexp_replace(column, r'\s+', ' ')

    # Troca aqueles termos que são sinônimos da doença por um nome único.
    column = F.regexp_replace(column, regex, canonical_name)

    # Troca sinônimos de compostos por nomes únicos.

    # Deixa tudo minúsculo.
    column = F.lower(column)

    return column


def words_preprocessing(df, column='word', typo_corrections: dict = None, units_and_symbols_list: list = None):
    """Corrige alguns erros de digitação, normaliza alguns símbolos e remove palavras irrelevantes."""

    units_and_symbols_expr = '(%s)' % '|'.join(units_and_symbols_list)

    def __keep_only_compound_numbers():
        return F.when(
            F.regexp_replace(F.lower(F.col(column)), r'\d+', '') == F.lit(''),
            F.lit('')
        ).otherwise(F.lower(F.col(column)))

    return df\
            .replace(typo_corrections, subset=column)\
            .withColumn(column, F.regexp_replace(F.col(column), units_and_symbols_expr, ''))\
            .withColumn(column, __keep_only_compound_numbers())\
            .withColumn(column, F.trim(F.col(column)))\
            .where(F.length(F.col(column)) > F.lit(1))\
            .where(~F.col(column).isin(nltk.corpus.stopwords.words('english')))

def clean_and_normalize_abstracts(target_disease: str, normalized_target_disease: str, batch_size: int = 500, entrez_email: str = "tirs@estudante.ufscar.br", nature_filtered_words: list = None, typo_corrections: dict = None, units_and_symbols_list: list = None):
    if nature_filtered_words is None:
        nature_filtered_words = [
            'foreword', 'prelude', 'commentary', 'workshop', 'conference', 'symposium',
            'comment', 'retract', 'correction', 'erratum', 'memorial'
        ]
    if typo_corrections is None:
        typo_corrections = {
            'mol-ecule': 'molecule',
            '‑': '-',
            '‒': '-',
            '–': '-',
            '—': '-',
            '¯': '-',
            'à': 'a',
            'á': 'a',
            'â': 'a',
            'ã': 'a',
            'ä': 'a',
            'å': 'a',
            'ç': 'c',
            'è': 'e',
            'é': 'e',
            'ê': 'e',
            'ë': 'e',
            'í': 'i',
            'î': 'i',
            'ï': 'i',
            'ñ': 'n',
            'ò': 'o',
            'ó': 'o',
            'ô': 'o',
            'ö': 'o',
            '×': 'x',
            'ø': 'o',
            'ú': 'u',
            'ü': 'u',
            'č': 'c',
            'ğ': 'g',
            'ł': 'l',
            'ń': 'n',
            'ş': 's',
            'ŭ': 'u',
            'і': 'i',
            'ј': 'j',
            'а': 'a',
            'в': 'b',
            'н': 'h',
            'о': 'o',
            'р': 'p',
            'с': 'c',
            'т': 't',
            'ӧ': 'o',
            '⁰': '0',
            '⁴': '4',
            '⁵': '5',
            '⁶': '6',
            '⁷': '7',
            '⁸': '8',
            '⁹': '9',
            '₀': '0',
            '₁': '1',
            '₂': '2',
            '₃': '3',
            '₅': '5',
            '₇': '7',
            '₉': '9',
        }
    if units_and_symbols_list is None:
        units_and_symbols_list = [
            '/μm', '/mol', '°c', '≥', '≤', '<', '>', '±', '%', '/mumol',
            'day', 'month', 'year', '·', 'week', 'days',
            'weeks', 'years', '/µl', 'μg', 'u/mg',
            'mg/m', 'g/m', 'mumol/kg', '/week', '/day', 'm²', '/kg', '®',
            'ﬀ', 'ﬃ', 'ﬁ', 'ﬂ', '£', '¥', '©', '«', '¬', '®', '°', '±', '²', '³',
            '´', '·', '¹', '»', '½', '¿',
            '׳', 'ᇞ​', '‘', '’', '“', '”', '•', '˂', '˙', '˚', '˜' , '…', '‰', '′',
            '″', '‴', '€',
            '™', 'ⅰ', '↑', '→', '↓', '∗', '∙', '∝', '∞', '∼', '≈', '≠', '≤', '≥', '≦', '≫', '⊘',
            '⊣', '⊿', '⋅', '═', '■', '▵', '⟶', '⩽', '⩾', '、', '气', '益', '粒', '肾', '补',
            '颗', '', '', '', '', '，'
        ]
    """
    Cleans and normalizes aggregated abstract texts, replacing synonyms with canonical names.

    Args:
        target_disease (str): The name of the disease for which abstracts are being cleaned.
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        batch_size (int): The number of texts to process in each spaCy pipeline batch (used in process_abstracts_from_file, though not directly in this main function, it's a dependency). Defaults to 500.
    """

    # Downloads NLTK data files if they are not already present.
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

    clean_papers_path = f'./data/{normalized_target_disease}/corpus/clean_abstracts'
    aggregated_abstracts_path = f'./data/{normalized_target_disease}/corpus/aggregated_abstracts'

    # Creates pyspark session.
    ss()

    # Creates a window that will be used to group the words by article.
    w2 = Window.partitionBy(F.col('filename'), F.col('id')).orderBy(F.col('pos'))

    print('Preprocessing text for Word2Vec models.')

    # Creates synonym table from PubChem data (cid | synonym).
    synonyms = read_table_file('data/pubchem_data/CID-Synonym-filtered', '\t', 'false')
    synonyms = synonyms.withColumnRenamed("_c0", "cid") \
        .withColumnRenamed("_c1", "synonym")

    # Creates title table from PubChem data (cid | title). Will be used to normalize synonyms.
    titles = read_table_file('data/pubchem_data/CID-Title', '\t', 'false')
    titles = titles.withColumnRenamed("_c0", "cid") \
        .withColumnRenamed("_c1", "title")

    # Reads the NER table, which contains all chemical-related words/phrases found in the corpus.
    ner_df = read_table_file(f'./data/{normalized_target_disease}/corpus/ner_table.csv', ',', 'true')
    if 'token' not in ner_df.columns:
        print('NER table not found or is invalid. Have you run step 3?')
        return

    # Create a distinct set of lowercased chemical terms/phrases from the corpus for filtering.
    corpus_chemical_terms = ner_df.select(F.lower(F.col('token')).alias('ner_term')).distinct()

    # Create the full compound map from PubChem data.
    compound_map_df = synonyms.join(titles, "cid") \
        .withColumn("replacement", F.regexp_replace(F.lower(F.col("title")), r'\s+', '')) \
        .select(F.lower(F.col("synonym")).alias("term"), "replacement") \
        .where(F.length(F.trim(F.col("term"))) > 1) \
        .where(F.col("term") != F.col("replacement")) \
        .distinct()

    print('Filtering compound map using NER table...')

    # Filter the compound map to include only terms that are also present in the NER table.
    filtered_compound_map_df = compound_map_df.join(
        F.broadcast(corpus_chemical_terms),
        compound_map_df.term == corpus_chemical_terms.ner_term,
        'inner'
    ).select("term", "replacement")

    # Collect the much smaller, filtered map to the driver.
    # Sort by term length descending to replace longer terms first.
    print('Collecting filtered compound map for replacement...')
    compound_map_rows = filtered_compound_map_df.collect()
    compound_map_rows.sort(key=lambda x: len(x.term), reverse=True)
    print(f'Collected {len(compound_map_rows)} relevant compound terms for replacement.')

    # Loads the aggregated abstracts from the corpus.
    df = dataframes_from_txt(aggregated_abstracts_path, nature_filtered_words)
    print('Before any preprocessing:')
    df.show(truncate=False)

    # Apply initial preprocessing (removes HTML, special characters, etc.).
    df = df.withColumn('summary', summary_column_preprocessing(F.col('summary'), target_disease, entrez_email))

    # Replace multi-word compounds with their single-word equivalents BEFORE tokenizing.
    # Create a list of (term, replacement) tuples from the collected rows.
    # The list is already sorted by term length (descending) to ensure longer matches are replaced first.
    replacement_list = [(row.term, row.replacement) for row in compound_map_rows]

    # Broadcast the list to make it efficiently available to the UDF on all executors.
    broadcasted_replacements = ss().sparkContext.broadcast(replacement_list)

    # Define a Python function to perform the series of replacements.
    def replace_compounds_in_text(text: str) -> str:
        if text is None:
            return ''
        # Access the broadcasted list of replacements.
        for term, replacement in broadcasted_replacements.value:
            # Use re.sub for case-insensitive, whole-word replacement.
            pattern = r'(?i)\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, replacement, text)
        return text

    # Register the Python function as a Spark UDF.
    replace_udf = F.udf(replace_compounds_in_text, F.StringType())

    print('Replacing compound synonyms in full text using UDF...')
    # Apply the UDF to the summary column.
    df = df.withColumn('summary', replace_udf(F.col('summary')))

    # Tokenize the abstracts by splitting on spaces.
    df = df.select('id', 'filename', F.posexplode(F.split(F.col('summary'), ' ')).alias('pos', 'word'))

    # Perform final word-level cleaning (removes stopwords, etc.).
    # Then, reassemble the tokens into a final summary string.
    df = words_preprocessing(df, column='word', typo_corrections=typo_corrections, units_and_symbols_list=units_and_symbols_list) \
        .withColumn('summary', F.collect_list('word').over(w2)) \
        .groupby('id', 'filename') \
        .agg(
        F.concat_ws(' ', F.max(F.col('summary'))).alias('summary')
    )

    print('Final DataFrame:')
    df = df.withColumn('id', F.monotonically_increasing_id())
    df.show(n=60, truncate=False)

    # Writes the clean abstracts to a CSV file.
    print('Writing CSV...')
    to_csv(df, target_folder=clean_papers_path)
    rename_csv_in_folder(clean_papers_path, 'clean_abstracts.csv')

    df.printSchema()
    print('END!')


if __name__ == '__main__':
    clean_and_normalize_abstracts(get_target_disease(), get_normalized_target_disease())
