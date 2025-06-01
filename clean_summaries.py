from Bio import Entrez
Entrez.email = "tirs@estudante.ufscar.br"

from pathlib import Path
import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.window import Window
import pyspark.sql.types as T
from target_disease import target_disease

import string
import nltk, os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from functools import reduce

def ss():
    sc = SparkContext.getOrCreate()
    return SparkSession(sc)

def dataframes_from_txt(summaries_path):
    filenames = sorted([str(x) for x in Path(summaries_path).glob('*.txt')])
    dataframes = []

    # Para cada arquivo no diretório...
    for file_path in filenames:
        # Pega o ano.
        year_of_file = file_path\
            .replace(os.path.join(summaries_path, 'results_file_1900_'), '')\
            .replace('.txt', '')

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

def read_csv_table_files(file_path, sep=','):
    full_path = file_path

    if file_path[-3:] != 'csv':
        file_path = get_csv_in_folder(file_path)

    return ss()\
        .read\
        .option('header', 'true')\
        .option('sep', sep)\
        .csv(full_path)

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

def summary_column_preprocessing(column):
    """Executes initial preprocessing in a PySpark text column. It removes some unwanted regex from the text.
    Args:
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

if __name__ == '__main__':
    # Baixando conjuntos relevantes de palavras.
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

    folder_name = target_disease.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
    CLEAN_PAPERS_PATH = f'./data/clean_results/{folder_name}/'
    SYNONYM_ENTITIES = [x.lower() for x in ['Drug', 'Clinical_Drug', 'Pharmacologic_Substance']]

    # Cria a sessão do pyspark.
    ss()

    # Cria janelas de agregação que definem critérios para agrupamento de linhas dos dataframes.
    w1 = Window.partitionBy(F.col('summary')).orderBy(F.col('filename'))
    w2 = Window.partitionBy(F.col('filename'), F.col('id')).orderBy(F.col('pos'))

    print('Preprocessing text for Word2Vec models')

    print('Replace synonyms: ' + str(True) + '\n')

    #####################################################################
    # PASSO 1
    # se for desejado substituir os compostos/drogas a partir de dados do PubChem:
        # cria a tabela de sinonimos. A primeira coluna contém o sinônimo do composto e a segunda coluna contém o nome (título) do composto ao qual aquele sinônimo se refere.
        # a coluna com os nomes dos sinônimos é transformada (sofre processamento), enquanto o título é apenas transformado em letras minúsculas
        # o grau da tabela original não é alterado. Ou seja, mantém-se a proporção 1 linha = 1 sinônimo

    # se o processamento do texto estiver sendo feito para treinamento de futuros modelos Word2Vec:
        # também é realizada a leitura do arquivo de texto que contém as palavras mais comuns do inglês. Esse arquivo é transformado em um DataFrame, removendo-se aquelas palavras selecionadas para o processo de validação
        # o DataFrame de palavras em inglês será usado para remover tais palavras do texto, antes do treinamento dos modelos.
    #####################################################################

    # Cria tabela de sinônimos (cid | sinônimo).
    synonyms = read_csv_table_files('./data/synonyms')
    synonyms = synonyms

    # Cria tabela de nomes (cid | nome).
    titles = read_csv_table_files('./data/titles.csv', sep='|')
    exit(0) # ADAPTANDO ATÉ AQUI!!-------------------------------------------------------------------------------

    # Cria tabela NER, que identifica o que cada termo é.
    ner_df = read_csv_table_files('./ner/')\
            .where(F.col('entity').isin(SYNONYM_ENTITIES))

    print('ner_df:')
    ner_df.show(truncate=False)


    ## se a normalização de sinônimos for ser realizada para futuro treinamento de modelos Word2vec, o Dataframe de sinônimos deve ser unido (join) ao Dataframe de palavras comuns do inglês,
    ## pois elas serão removidas do texto:

    # Junta as tabelas de sinônimo e nome.
    synonyms = synonyms\
            .withColumn('synonym', F.regexp_replace(F.lower(F.col('synonym')), r'\s+', ''))\
            .groupby('synonym')\
            .agg(F.min('cid').alias('cid'))\
            .join(titles, 'cid')\
            .withColumn('synonym_title', F.regexp_replace(F.lower(F.col('title')), r'\s+', ''))\
            .select('synonym', 'synonym_title')

    ## independentemente de qual o futuro modelo a ser treinado, se houver noralização de sinônimos, o Dataframe de sinônimos é unido com o NER,
    ## para que haja a normalização apenas de palavras identificadas como drogas/compostos/fármacos:

    # Filtra compostos específicos (devem ter dado algum problema).
    synonyms = synonyms\
                .filter(F.col('synonym_title') != 'methyl(9r,10s,11s,12r,19r)-11-acetyloxy-12-ethyl-4-[(13s,15r,17s)-17-ethyl-17-hydroxy-13-methoxycarbonyl-1,11-diazatetracyclo[13.3.1.04,12.05,10]nonadeca-4(12),5,7,9-tetraen-13-yl]-8-formyl-10-hydroxy-5-methoxy-8,16-diazapentacyclo[10.6.1.01,9.02,7.016,19]nonadeca-2,4,6,13-tetraene-10-carboxylate')\
                .filter(F.col('synonym_title') != 'methyl(1r,10s,11r,12r,19r)-11-acetyloxy-12-ethyl-4-[(13s,15r,17s)-17-ethyl-17-hydroxy-13-methoxycarbonyl-1,11-diazatetracyclo[13.3.1.04,12.05,10]nonadeca-4(12),5,7,9-tetraen-13-yl]-10-hydroxy-5-methoxy-8-methyl-8,16-diazapentacyclo[10.6.1.01,9.02,7.016,19]nonadeca-2,4,6,13-tetraene-10-carboxylate')

    # Tira dos sinônimos as linhas que contêm nome duplicado (sinônimo = nome), junta o tipo NER na tabela.
    synonyms = synonyms\
                .where(F.col('synonym') != F.col('synonym_title'))\
                .join(ner_df, F.col('synonym') == F.col('token'), 'inner')\
                .drop(*('token', 'entity'))

    #####################################################################
    # PASSO 2
    # cria o DataFrame de artigos limpos/processados. Cada linha dessa tabela equivale a um artigo.
    # a tabela tem três colunas: filename, id, summary
    #       "filename" é o nome do arquivo de onde o artigo foi retirado (results_aggregated), ou seja, é o ano de publicação do artigo.
    #       "id" é uma coluna serial, apenas para contagem/identificação
    #       "summary" é o próprio texto (título e/ou prefácio do artigo) limpo/processado.
    #####################################################################

    # Carrega os abstracts e salva como uma tabela (filename | id | summary), cada linha representa um artigo.
    cleaned_documents = dataframes_from_txt('./results_aggregated')
    print('Abstracts originais:')
    cleaned_documents.show(truncate=False)

    # Pré-processa cada linha e separa cada palavra numa linha diferente da tabela.
    # Esse pré-processamento é basicamente a remoção de caracteres e strings especiais e substituição de sinônimos.
    cleaned_documents = cleaned_documents\
                        .withColumn('summary', summary_column_preprocessing(F.col('summary')))\
                        .select('id', 'filename', F.posexplode(F.split(F.col('summary'), ' ')).alias('pos', 'word'))

    print('Após summary_column_preprocessing:')
    cleaned_documents.show(truncate=False)

    # Pré-processa mais uma vez, corrigindo erros de digitação e normalizando algumas coisas.
    # Concatena as palavras do mesmo abstract numa linha só, revertendo o posexplode.
    cleaned_documents = words_preprocessing(cleaned_documents)\
                        .withColumn('summary', F.collect_list('word').over(w2))\
                        .groupby('id', 'filename')\
                        .agg(
                            F.concat_ws(' ', F.max(F.col('summary'))).alias('summary')
                        )

    print('Após words_preprocessing:')
    cleaned_documents.show(truncate=False)

    #####################################################################
    # PASSO 3
    # se o texto estiver sendo processado para modelos BERT, NÃO é realizada a lemmatização dos verbos e advérbios
    # caso contrário, é realizada a lemmatização <- (onde?)
    # em ambos os casos, o texto (summary) é tokenizado nos espaços em branco, formando uma linha do Dataframe para cada token
    # esse novo Dataframe - com os tokens - será utilizado para união (join) com o dataframe de sinônimos
    #####################################################################

    # Separa as palavras em linhas diferentes de novo.
    df = cleaned_documents\
        .select(
            'id',
            'filename',
            F.posexplode(F.split(F.col('summary'), ' ')).alias('pos', 'word')
        )

    print('Após primeiro posexplode:')
    df.show(truncate=False)

    # Renomeia a coluna "word" para "word_n".
    df = df\
        .withColumnRenamed('word', 'word_n')

    df.show(n=60, truncate=False)

    # Junta com a tabela de sinônimos.
    # Assim, sempre que a palavra for um sinônimo de x, o campo synonym_title será x, caso contrário, null.
    df = df\
        .join(synonyms, F.col('synonym') == F.lower(F.col('word_n')), 'left')\
        .distinct()

    df.show(truncate=False)

    # Troca os sinônimos pelos termos normalizados.
    matched_synonyms = df\
        .where(F.col('synonym_title').isNotNull)\
        .select(F.col('synonym'), F.col('synonym_title'))\
        .distinct()\
        .where(F.col('synonym') != F.col('synonym_title'))

    # Junta as palavras e forma os abstracts pré-processados.
    df = df\
        .withColumn('word', F.coalesce(F.col('synonym_title'), F.col('word_n')))\
        .drop(*('synonym', 'synonym_title', 'word_n'))\
        .withColumn('summary', F.collect_list('word').over(w2))\
        .groupby('id', 'filename')\
        .agg(
            F.concat_ws(' ', F.max(F.col('summary'))).alias('summary')
        )

    print('Final - após possível normalização de sinônimos:')
    df = df.withColumn('id', F.monotonically_increasing_id())
    df.show(n=60, truncate=False)

    # Escreve os .csv.
    print('Escrevendo csv')
    to_csv(df, target_folder=CLEAN_PAPERS_PATH)
    to_csv(matched_synonyms, target_folder=MATCHED_SYNONYMS_PATH)

    df.printSchema()
    print('END!')