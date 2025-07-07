from Bio import Entrez
Entrez.email = "tirs@estudante.ufscar.br"

from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from src.target_disease import *

import nltk, os
from nltk.tokenize import word_tokenize

from functools import reduce

os.chdir(Path(__file__).resolve().parent.parent)

def ss():
    spark = SparkSession.builder \
        .appName("LargeDataProcessingApp")\
        .config("spark.executor.memory", "64g")\
        .config("spark.driver.memory", "32g")\
        .config("spark.sql.shuffle.partitions", "1000")\
        .config("spark.sql.adaptive.enabled", "true")\
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")\
        .config("spark.local.dir", "/tmp/spark-temp")\
        .getOrCreate()
    return spark


def dataframes_from_txt(summaries_path):
    filenames = sorted([str(x) for x in Path(summaries_path).glob('*.txt')])
    dataframes = []

    # Para cada arquivo no diretório...
    for file_path in filenames:
        # Pega o ano.
        try:
            year_of_file = Path(file_path).stem.split('_')[-1]
            int(year_of_file)  # Confirma que é um número
        except (IndexError, ValueError):
            print(f"Warning: Could not extract year from filename {Path(file_path).name}. Skipping.")
            continue

        NATURE_FILTERED_WORDS_IN_TITLE = [
            'foreword', 'prelude', 'commentary', 'workshop', 'conference', 'symposium',
            'comment', 'retract', 'correction', 'erratum', 'memorial'
        ]

        # Filtra os artigos com título contendo pelo menos uma dessas palavras.
        title_doesnt_have_nature_filtered_words = reduce(
            lambda acc, word: acc & (F.locate(word, F.col('title')) == F.lit(0)),
            NATURE_FILTERED_WORDS_IN_TITLE,
            F.lit(True)
        )

        # Cria uma tabela para cada arquivo com as colunas filename, title, summary e id.
        df = ss()\
            .read\
            .option('header', 'false')\
            .option('lineSep', '\n')\
            .option('sep', '|')\
            .option('quote', '')\
            .csv(file_path)\
            .withColumn('filename', F.lit(year_of_file))\
            .withColumnRenamed('_c0', 'title')\
            .withColumnRenamed('_c1', 'summary')\
            .where(title_doesnt_have_nature_filtered_words)\
            .withColumn('id', F.monotonically_increasing_id())

        # Coloca essa tabela na lista de tabelas.
        dataframes.append(df)

    # Junta tudo numa tabela só e retorna.
    return reduce(lambda df1, df2: df1.union(df2), dataframes)


def get_csv_in_folder(file_path):
    """Search for a .csv file in a given path. It must find just one .csv file - this constraint is tested with assert command.
    This is an auxiliar function used during reading .csv PySpark DataFrames.
    Args:
        file_path: path to the folder containg the .csv file.
    """

    files = os.listdir(file_path)
    files = filter(lambda x: x[-3:] == 'csv', files)
    files = list(files)

    assert len(files) == 1, files

    return os.path.join(file_path, files[0])


def read_table_file(file_path, sep, has_header):
    return ss()\
        .read\
        .option('header', has_header)\
        .option('sep', sep)\
        .csv(file_path)


def to_csv(df, target_folder, num_files=1, sep=','):
    """Saves a PySpark Dataframe into .csv file.
    Args:
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


def get_wordnet_pos(treebank_tag):
    """Returns WORDNET POS compliance to WORDNET lemmatization (ADJ, VERB, NOUN, ADV)"""

    if treebank_tag.startswith('J'):
            return 'a'
    elif treebank_tag.startswith('V'):
            return 'v'
    elif treebank_tag.startswith('N'):
            return 'n'
    elif treebank_tag.startswith('R'):
            return 'r'
    else:
    # As default pos in lemmatization is Noun
        return 'n'


def tokenize(data):
    """Tokenizes a sentence

        Args:
        data: a sentence (string).
    """
    if data is None:
        return ['']
    else:
        return word_tokenize(data)


def remove_last_digit(data):
    """Removes the last character of a string a sentence

        Args:
        data: a string.
    """

    return data[:-1]


def return_last_digit(data):
    """Toeknizes a sentence

        Args:
        data: a sentence (string).
    """

    try:
        aux = data[-1]

    except:
        aux = ''

    return aux


def find_mesh_terms(disease_name):
    handle = Entrez.esearch(db="mesh", term=f"\"{disease_name}\"[MeSH Terms] ", retmax="20")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def get_mesh_details(mesh_id):
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


def summary_column_preprocessing(column, target_disease):
    """Executes initial preprocessing in a PySpark text column. It removes some unwanted regex from the text.
    Args:
        target_disease:
        column: the name of the column to be processed.
    """
    disease_details = get_mesh_details(find_mesh_terms(target_disease))

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


def words_preprocessing(df, column='word'):
    """Corrige alguns erros de digitação, normaliza alguns símbolos e remove palavras irrelevantes."""

    fix_typos_dict = {
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

    units_and_symbols = [
        '/μm', '/mol', '°c', '≥', '≤', '<', '>', '±', '%', '/mumol',
        'day', 'month', 'year', '·', 'week', 'days',
        'weeks', 'years', '/µl', 'μg', 'u/mg',
        'mg/m', 'g/m', 'mumol/kg', '/week', '/day', 'm²', '/kg', '®',
        'ﬀ', 'ﬃ', 'ﬁ', 'ﬂ', '£', '¥', '©', '«', '¬', '®', '°', '±', '²', '³',
        '´', '·', '¹', '»', '½', '¿',
         '׳', 'ᇞ​', '‘', '’', '“', '”', '•',  '˂', '˙', '˚', '˜' ,'…', '‰', '′',
        '″', '‴', '€',
        '™', 'ⅰ', '↑', '→', '↓', '∗', '∙', '∝', '∞', '∼', '≈', '≠', '≤', '≥', '≦', '≫', '⊘',
        '⊣', '⊿', '⋅', '═', '■', '▵', '⟶', '⩽', '⩾', '、', '气', '益', '粒', '肾', '补',
        '颗', '', '', '', '', '，'
    ]

    units_and_symbols_expr = '(%s)' % '|'.join(units_and_symbols)

    def __keep_only_compound_numbers():
        return F.when(
            F.regexp_replace(F.lower(F.col(column)), r'\d+', '') == F.lit(''),
            F.lit('')
        ).otherwise(F.lower(F.col(column)))

    return df\
            .replace(fix_typos_dict, subset=column)\
            .withColumn(column, F.regexp_replace(F.col(column), units_and_symbols_expr, ''))\
            .withColumn(column, __keep_only_compound_numbers())\
            .withColumn(column, F.trim(F.col(column)))\
            .where(F.length(F.col(column)) > F.lit(1))\
            .where(~F.col(column).isin(nltk.corpus.stopwords.words('english')))


def main():
    target_disease = get_target_disease()
    normalized_target_disease = get_normalized_target_disease()

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
    synonyms = synonyms.withColumnRenamed("_c0", "cid")\
                        .withColumnRenamed("_c1", "synonym")

    # Creates title table from PubChem data (cid | title). Will be used to normalize synonyms.
    titles = read_table_file('data/pubchem_data/CID-Title', '\t', 'false')
    titles = titles.withColumnRenamed("_c0", "cid")\
                    .withColumnRenamed("_c1", "title")

    # Reads the NER table.
    ner_df = read_table_file(f'./data/{normalized_target_disease}/corpus/ner_table.csv', ',', 'true')
    if not ner_df:
        print('NER table not found. Have you run step 3?')
        return

    print('NER table:')
    ner_df.show(truncate=False)

    # Joins the synonyms with their titles. Now we have a table with (synonym | synonym_title).
    synonyms = synonyms\
            .withColumn('synonym', F.regexp_replace(F.lower(F.col('synonym')), r'\s+', ''))\
            .groupby('synonym')\
            .agg(F.min('cid').alias('cid'))\
            .join(titles, 'cid')\
            .withColumn('synonym_title', F.regexp_replace(F.lower(F.col('title')), r'\s+', ''))\
            .select('synonym', 'synonym_title')

    # Removes synonyms that are the same as their titles.
    # Joins the synonyms with the NER table to filter out words that don't show up in the corpus.
    # This is done as a performance optimization, the PubChem tables are huge.
    synonyms = synonyms\
                .where(F.col('synonym') != F.col('synonym_title'))\
                .join(ner_df, F.col('synonym') == F.col('token'), 'inner')\
                .drop(*('token', 'entity'))

    # Loads the aggregated abstracts from the corpus, creates an (id | filename | summary) table.
    cleaned_documents = dataframes_from_txt(aggregated_abstracts_path)
    print('Before any preprocessing:')
    cleaned_documents.show(truncate=False)

    # Preprocesses each line and separates each word into a different line in the table.
    # This preprocessing step is basically the removal of special characters and strings, along with synonym normalization.
    cleaned_documents = cleaned_documents\
                        .withColumn('summary', summary_column_preprocessing(F.col('summary'), target_disease))\
                        .select('id', 'filename', F.posexplode(F.split(F.col('summary'), ' ')).alias('pos', 'word'))

    print('After summary_column_preprocessing:')
    cleaned_documents.show(truncate=False)

    # Preprocesses again, this time removing stopwords and normalizing the words.
    # Reverts the posexplode so that each abstract is in a single line again.
    cleaned_documents = words_preprocessing(cleaned_documents)\
                        .withColumn('summary', F.collect_list('word').over(w2))\
                        .groupby('id', 'filename')\
                        .agg(
                            F.concat_ws(' ', F.max(F.col('summary'))).alias('summary')
                        )

    print('After words_preprocessing:')
    cleaned_documents.show(truncate=False)

    # Separates the words again.
    df = cleaned_documents\
        .select(
            'id',
            'filename',
            F.posexplode(F.split(F.col('summary'), ' ')).alias('pos', 'word')
        )

    df = df\
        .withColumnRenamed('word', 'word_n')

    df.show(n=60, truncate=False)


    # Joins the word table with the synonyms table.
    # If the word is a synonym, there will be something in the title column. Otherwise, it will be null.
    df = df\
        .join(synonyms, F.col('synonym') == F.lower(F.col('word_n')), 'left')\
        .distinct()

    df.show(truncate=False)

    # Swaps synonyms for their titles.
    matched_synonyms = df\
        .where(F.col('synonym_title').isNotNull())\
        .select(F.col('synonym'), F.col('synonym_title'))\
        .distinct()\
        .where(F.col('synonym') != F.col('synonym_title'))

    # Aggregates the words to form the whole abstracts again, but now they're preprocessed.
    df = df\
        .withColumn('word', F.coalesce(F.col('synonym_title'), F.col('word_n')))\
        .drop(*('synonym', 'synonym_title', 'word_n'))\
        .withColumn('summary', F.collect_list('word').over(w2))\
        .groupby('id', 'filename')\
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
    main()
