import string

import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.window import Window
import pyspark.sql.types as T
from target_disease import target_disease

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
    """Toeknizes a sentence

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

def summary_column_preprocessing(column):
    """Executes intial preprocessing in a PySpark text column. It removes some unwanted regex from the text.
    Args:
        column: the name of the column to be processed.
    """

    aml_synonyms = [
        'acute[- ]?myeloid[- ]?leukemia',
        'acute myelocytic leukemia',
        'acute myelogenous leukemia',
        'acute granulocytic leukemia',
        'acute non-lymphocytic leukemia',
        'acute mylogenous leukemia',
        'acute myeloid leukemia',
        'acute nonlymphoblastic leukemia',
        'acute myeloblastic leukemia',

        # subtipos de AML:
        'acute erythroid leukemia',
        'acute myelomonocytic leukemia',
        'acute monocytic leukemia',
        'acute megakaryoblastic leukemia',
        'acute promyelocytic leukemia',
    ]

    # Cria um regex para procurar os sinônimos da AML.
    regex = r'(?i)({})'.format('|'.join(aml_synonyms))

    # Remove os espaços em branco do início e fim da coluna.
    column = F.trim(column)

    # Remove tags HTML e URLs.
    column = F.regexp_replace(column, r'<[^>]+>', '')
    column = F.regexp_replace(column, r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '')

    # Remove caracteres especiais.
    column = F.regexp_replace(column, r'[;:\(\)\[\]\{\}.,"!#$&\'*?@\\\^`|~]', '')

    # Normaliza o espaço em branco.
    column = F.regexp_replace(column, r'\s+', ' ')

    # Normaliza termos para leukemia.
    column = F.regexp_replace(column, r'(?i)(leukaemia)', 'leukemia')
    column = F.regexp_replace(column, r'(?i)(leukaemic)', 'leukemic')

    # Troca aqueles termos que são sinônimos de AML por AML.
    column = F.regexp_replace(column, regex, 'AML')

    # Troca alguns sinônimos mais complexos por AML.
    column = F.regexp_replace(column, r'(?i)(acute myeloid or (lymphoblastic|lymphoid) leukemia[s]?)', 'AML or ALL')
    column = F.regexp_replace(column, r'(?i)(acute myeloid and (lymphoblastic|lymphoid) leukemia[s]?)', 'AML and ALL')
    column = F.regexp_replace(column, r'(?i)(acute myeloid and chronic lymphocytic leukemia[s]?)', 'AML and CLL')

    # Troca sinônimos de compostos pelo nome base.
    # cytarabine
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside triphosphate )', ' cytarabinetriphosphate ')
    column = F.regexp_replace(column, r'(?i)(cytosine arabinoside triphosphate )', 'cytarabinetriphosphate ')
    column = F.regexp_replace(column, r'(?i)(\(cytosine arabinoside triphosphate\))', '(cytarabinetriphosphate)')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside triphosphate.)', ' cytarabinetriphosphate.')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside triphosphate,)', ' cytarabinetriphosphate,')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside monophosphate )', ' cytarabinemonophosphate')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside monophosphate.)', ' cytarabinemonophosphate.')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside monophosphate,)', ' cytarabinemonophosphate,')
    column = F.regexp_replace(column, r" arabinocytidine 5' phosphate|arabinofuranosylcytosine 5'-triphosphate|Ara-CTP", " cytarabine5phosphate")
    column = F.regexp_replace(column, r"(?i)(1-beta-D|1-β-d|1β-d|1 beta-D)-Arabinofuranosylcytosine 5'-triphosphate", "cytarabine5phosphate")
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside | \[Ara-C\] )', ' cytarabine ')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside\.)', ' cytarabine.')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside,)', ' cytarabine,')
    column = F.regexp_replace(column, r'(?i)( cytosine arabinoside:)', ' cytarabine:')
    column = F.regexp_replace(column, r'(?i)(\(cytosine arabinoside\))', '(cytarabine)')
    column = F.regexp_replace(column, r'(?i)(\(ara[-]?c\))', '(cytarabine)')
    column = F.regexp_replace(column, r'(?i)(\(ara[-]?c, )', '(cytarabine, ')
    column = F.regexp_replace(column, r'(?i)(\(ara[-]?c )', '(cytarabine ')
    column = F.regexp_replace(column, r'(?i)( ara[-]?c\))', ' cytarabine)')
    column = F.regexp_replace(column, r'(?i)( ara[-]?c/)', ' cytarabine/)')
    column = F.regexp_replace(column, r'(?i)(/ara[-]?c\.)', '/cytarabine.')
    column = F.regexp_replace(column, r'(?i)(/ara[-]?c )', '/cytarabine ')
    column = F.regexp_replace(column, r'(?i)(ara[-]?c-induced)', 'cytarabine-induced')
    column = F.regexp_replace(column, r'(?i)(ara[-]?c-treated)', 'cytarabine-treated')
    column = F.regexp_replace(column, r'(?i)(ara[-]?c-based)', 'cytarabine-based')
    column = F.regexp_replace(column, r'(?i)(ara[-]?c-resistant)', 'cytarabine-resistant')
    column = F.regexp_replace(column, r'(?i)(/ara[-]?c/)', '/cytarabine/')
    column = F.regexp_replace(column, r'(HDAra-C|HD-Ara-C)', 'high-dose cytarabine')
    column = F.regexp_replace(column, r'(LDAra-C|LD-Ara-C|LDAC)', 'low-dose cytarabine')
    column = F.regexp_replace(column, r'(IDAra-C|ID-Ara-C)', 'intermediate-dose cytarabine')
    column = F.regexp_replace(column, r'\(Ara-C;', '(cytarabine;')
    column = F.regexp_replace(column, r'\[3H\]Ara-C', '([3H]cytarabine')
    column = F.regexp_replace(column, r' Arabinocytidine', ' Cytarabine')
    column = F.regexp_replace(column, r"(?i)(1-beta-D|1-β-d|1β-d|1 beta-D)-Arabinofuranosyl[-]?cytosine ", "cytarabine ")
    column = F.regexp_replace(column, r"(?i)(1-beta-D|1-β-d|1β-d|1 beta-D)-Arabinofuranosyl[-]?cytosine,", "cytarabine,")
    column = F.regexp_replace(column, r"(?i)(1-beta-D|1-β-d|1β-d|1 beta-D)-Arabinofuranosyl[-]?cytosine.", "cytarabine.")
    column = F.regexp_replace(column, r'(?i)( Arabinofuranosylcytosine | Ara-C )', " cytarabine ")
    column = F.regexp_replace(column, r'(?i)( Arabinofuranosylcytosine,)', " cytarabine,")
    column = F.regexp_replace(column, r'(?i)( Arabinofuranosylcytosine\.)', " cytarabine.")
    column = F.regexp_replace(column, r' arabinocytidine ', ' cytarabine ')
    column = F.regexp_replace(column, r' arabinocytidine,', ' cytarabine,')
    column = F.regexp_replace(column, r' arabinocytidine:', ' cytarabine:')
    column = F.regexp_replace(column, r' arabinocytidine\.', ' cytarabine.')
    column = F.regexp_replace(column, r'147-94-4', 'cytarabine')
    column = F.regexp_replace(column, r'\(Cytosar-U\)|\(Cytosar\)', '(cytarabine)')
    column = F.regexp_replace(column, r'(?i)\(Cytosar-U,', '(cytarabine,')
    column = F.regexp_replace(column, r'(?i) cytosar\)', ' cytarabine)')
    column = F.regexp_replace(column, r' aracytin[e]? ', ' cytarabine ')
    column = F.regexp_replace(column, r' aracytin[e]?\.', ' cytarabine.')
    column = F.regexp_replace(column, r' aracytin[e]?,', ' cytarabine,')
    column = F.regexp_replace(column, r'\+aracytine\+', '+cytarabine+')
    column = F.regexp_replace(column, r'-aracytine ', '-cytarabine ')
    column = F.regexp_replace(column, r'-aracytine\.', '-cytarabine.')
    column = F.regexp_replace(column, r'-aracytine:', '-cytarabine:')
    column = F.regexp_replace(column, r'(?i)(Cytosine beta-D-arabinoside)', 'cytarabine')
    column = F.regexp_replace(column, r'(?i)(Cytosine-1-beta-D-arabinofuranoside)', 'cytarabine')
    column = F.regexp_replace(column, r'(?i)(beta-ara c)', 'cytarabine')
    column = F.regexp_replace(column, r'(?i)(liposomal cytarabine)', 'liposomalcytarabine')
    column = F.regexp_replace(column, r'(?i)(depocyte)', 'liposomalcytarabine')
    column = F.regexp_replace(column, r"4'-thio-ara-C", "4-thio-cytarabine")
    column = F.regexp_replace(column, r"NSC[ -]63878", "cytarabine")
    column = F.regexp_replace(column, r"ofcytarabine", "of cytarabine")

    # daunorubicin:
    column = F.regexp_replace(column, r'[Dd]aunomycin \(DAU\)|[Dd]aunorubicin \(DAU\)|daunorubicin hydrochloride \(DAU\)', 'daunorubicin (daunorubicin)')
    column = F.regexp_replace(column, r'NSC[ -]?82151|Rubomycin C|[^R]DNX[^B]', 'daunorubicin')
    column = F.regexp_replace(column, r'daunomycin-', 'daunorubicin-')
    column = F.regexp_replace(column, r'Daunomycin-', 'Daunorubicin-')
    column = F.regexp_replace(column, r'(?i)( daunorubicine )', ' daunorubicin ')
    column = F.regexp_replace(column, r'\(DNR\)', '(daunorubicin)')
    column = F.regexp_replace(column, r'\(daunomycin ', '(daunorubicin ')
    column = F.regexp_replace(column, r'rubidomicine|rubidomicin', 'daunorubicin')
    column = F.regexp_replace(column, r'leukaemomycin C', 'daunorubicin')
    column = F.regexp_replace(column, r'(Cerubidine)', '(daunorubicin)')
    column = F.regexp_replace(column, r'[Ll]iposomal daunorubicin', 'liposomaldaunorubicin')
    column = F.regexp_replace(column, r'(?i)daunoxome', 'daunorubicin')
    column = F.regexp_replace(column, r'LDL-daunomycin|LDL:daunomycin', 'low-density-lipoproteins-daunorubicin')
    column = F.regexp_replace(column, r'\(daunomycin\)', '(daunorubicin)')
    column = F.regexp_replace(column, r'\(daunomycin,', '(daunorubicin,')
    column = F.regexp_replace(column, r'LDL-daunomycin', 'low-density-lipoproteins-daunorubicin')
    column = F.regexp_replace(column, r'[\[]?3H[\]]?[-]?daunomycin', '3H-daunorubicin')
    column = F.regexp_replace(column, r' daunomycin\.', ' daunorubicin.')
    column = F.regexp_replace(column, r'13-dihydrodaunomycin', '13-dihydrodaunorubicin')
    column = F.regexp_replace(column, r'cys-aconytil-daunomycin', 'cys-aconytil-daunorubicin')
    column = F.regexp_replace(column, r'\[daunomycin\]', '[daunorubicin]')
    column = F.regexp_replace(column, r'porphyrin-daunomycin[a-z]', 'Por-(daunorubicin)')

    # azacitidine:
    column = F.regexp_replace(column, r'5-AZA|5-azaC|5-AZAC|5-Aza|5-ACR|5-AC', 'azacitidine')
    column = F.regexp_replace(column, r" 5[']?[-]?azac[yi]tidine ", ' azacitidine ')
    column = F.regexp_replace(column, r' 5-azac[yi]tidine-', ' azacitidine-')
    column = F.regexp_replace(column, r' 5-azac[yi]tidine\.', ' azacitidine.')
    column = F.regexp_replace(column, r' 5-azac[yi]tidine;', ' azacitidine:')
    column = F.regexp_replace(column, r' 5-azac[yi]tidine,', ' azacitidine,')
    column = F.regexp_replace(column, r' 5-azac[yi]tidine\)', ' azacitidine)')
    column = F.regexp_replace(column, r'(?i)(5-aza-CR)', 'azacitidine')
    column = F.regexp_replace(column, r'(?i)( azac[yi]tidine )', ' azacitidine ')
    column = F.regexp_replace(column, r'\(AZA\)', '(azacitidine)')
    column = F.regexp_replace(column, r'(?i)(vidaza)', 'azacitidine')

    # gemtuzumab-ozogamicin:
    column = F.regexp_replace(column, r"(?i)(gemtuzumab[- ]?ozogam[yi]cin) \(GO\)", 'gemtuzumab-ozogamicin (gemtuzumab-ozogamicin)')
    column = F.regexp_replace(column, r"(?i)(gemtuzumab[- ]?ozogam[yi]cin) \(GO,", 'gemtuzumab-ozogamicin (gemtuzumab-ozogamicin,')
    column = F.regexp_replace(column, r"(?i)(gemtuzumab[- ]?ozogam[yi]cin) \(GO;", 'gemtuzumab-ozogamicin (gemtuzumab-ozogamicin;')
    column = F.regexp_replace(column, r"(?i)(gemtuzumab ozogam[yi]cin)", 'gemtuzumab-ozogamicin')
    column = F.regexp_replace(column, r"CMA-676|FLASI-GO", 'gemtuzumab-ozogamicin')
    column = F.regexp_replace(column, r"(?i)(my[o]?lotarg)", 'gemtuzumab-ozogamicin')

    # midostaurin:
    column = F.regexp_replace(column, r"(?i)(4'-N-benzoyl staurosporine|4'-N-Benzoylstaurosporine|N-Benzoylstaurosporine)", 'midostaurin')
    column = F.regexp_replace(column, r"(?i)(rydapt)", 'midostaurin')
    column = F.regexp_replace(column, r"PKC[ -]?412|CGP[ -]?41251|CAS 120685-11-2", 'midostaurin')

    # CPX-351 (ou vyxeos):
    column = F.regexp_replace(column, r"(?i)(cpx[- ]?351)", 'vyxeos')
    column = F.regexp_replace(column, r"(?i)(vyxeos liposomal)", 'vyxeos')
    column = F.regexp_replace(column, r"(?i)(Daunorubicin[ ]?\/[ ]?cytarabine liposome|liposomaldaunorubicin/cytarabine)", 'vyxeos')

    # ivosidenib:
    column = F.regexp_replace(column, r"(?i)(tibsovo)", 'ivosidenib')
    column = F.regexp_replace(column, r"(?i)(ag120)", 'ivosidenib')

    # venetoclax:
    column = F.regexp_replace(column, r"(?i)(ABT[ -]?199|GDC[ -]?0199|venclyxto|venclexta)", 'venetoclax')

    # enasidenib:
    column = F.regexp_replace(column, r"(?i)(ag[ -]?221|idhifa|cc[ -]?90007)", 'enasidenib')

    # gilteritinib:
    column = F.regexp_replace(column, r"Xospata|ASP2215", 'gilteritinib')

    # glasdegib:
    column = F.regexp_replace(column, r"\(DAU\)", '(glasdegib)')
    column = F.regexp_replace(column, r"(DAURISMO|PF[ -][0]?4449913|PF[ -]913)", 'glasdegib')

    # arsenic trioxide:
    column = F.regexp_replace(column, r"(?i)(arsenic trioxide)", 'arsenictrioxide')
    column = F.regexp_replace(column, r"As2[O0]3,", 'arsenictrioxide,')
    column = F.regexp_replace(column, r"As2[O0]3\)", 'arsenictrioxide)')
    column = F.regexp_replace(column, r"As2[O0]3\.", 'arsenictrioxide.')
    column = F.regexp_replace(column, r"As2[O0]3$", 'arsenictrioxide.')
    column = F.regexp_replace(column, r"As2[O0]3 ", 'arsenictrioxide ')
    column = F.regexp_replace(column, r"As2[O0]3", 'arsenictrioxide')
    column = F.regexp_replace(column, r"As\(2\)[O0]\(3\) ", 'arsenictrioxide ')
    column = F.regexp_replace(column, r" ATO ", ' arsenictrioxide ')
    column = F.regexp_replace(column, r"\(ATO\)", '(arsenictrioxide)')
    column = F.regexp_replace(column, r"\(ATO[,;]", '(arsenictrioxide,')
    column = F.regexp_replace(column, r" ATO-", ' arsenictrioxide-')
    column = F.regexp_replace(column, r"(?i)(trisenoxt|trisenox)", 'arsenictrioxide')
    column = F.regexp_replace(column, r"(?i)(arsenic\(III\) oxide)", 'arsenictrioxide')

    # cyclophosphamide:
    column = F.regexp_replace(column, r"(?i)(methylerythritol cyclophosphane)", 'methylerythritolcyclophosphamide')
    column = F.regexp_replace(column, r"(?i)(cyclophosphane)", 'cyclophosphamide')
    column = F.regexp_replace(column, r"cyclophosphamide-endoxan|Cyklofosfamid|NSC[ -]?26271|Genoxal", 'cyclophosphamide')
    column = F.regexp_replace(column, r"(?i)(endoxane|endoxan|cyclophosphamidum)", 'cyclophosphamide')
    column = F.regexp_replace(column, r"(?i)( neosar\))", ' cyclophosphamide)')

    # dexamethasone:
    column = F.regexp_replace(column, r"(?i)(decadron|dexasone|dexason|Dextenza|Dexycu)", 'dexamethasone')
    column = F.regexp_replace(column, r"Maxidex|dexasone|Hexadrol|Oradexon|Fortecortin", 'dexamethasone')
    column = F.regexp_replace(column, r"[Ii]ntratympanic dexamethazone", 'intratympanicdexamethasone')
    column = F.regexp_replace(column, r"(?i)(de[sx]ametha[sz]one|Desametasone)", 'dexamethasone')
    column = F.regexp_replace(column, r"dexamethasone \(DMS\)", 'dexamethasone (dexamethasone)')

    # idarubicin:
    column = F.regexp_replace(column, r"NSC-256439|Zavedos", 'idarubicin')
    column = F.regexp_replace(column, r"\(4-demethoxydauno(mycin|rubicin)\)", '(idarubicin)')
    column = F.regexp_replace(column, r"\(4-demethoxydaunorubicin;", '(idarubicin;')
    column = F.regexp_replace(column, r" 4-demethoxydauno(mycin|rubicin) ", ' idarubicin ')
    column = F.regexp_replace(column, r" 4-demethoxydauno(mycin|rubicin)\.", ' idarubicin.')
    column = F.regexp_replace(column, r" 4-demethoxydauno(mycin|rubicin),", ' idarubicin,')
    column = F.regexp_replace(column, r"\[14-14C\]4-demethoxydaunorubicin HCl", '[14-14C]idarubicinhcl')
    column = F.regexp_replace(column, r"(?i)(Idarubicine)", 'idarubicin')
    column = F.regexp_replace(column, r"(?i)(idarubicin hcl)", 'idarubicin')
    column = F.regexp_replace(column, r"(?i)(4-DMD[N]?R )", 'idarubicin ')
    column = F.regexp_replace(column, r"(?i)(4-DMD[N]?R[.;])", 'idarubicin,')
    column = F.regexp_replace(column, r"(?i)(4-DMD[N]?R\.)", 'idarubicin.')
    column = F.regexp_replace(column, r"(?i)(4-DMD[N]?R\))", 'idarubicin)')
    column = F.regexp_replace(column, r"(?i)(4-DMD[N]?R-)", 'idarubicin')

    # mitoxantrone:
    column = F.regexp_replace(column, r"mitoxantron,cytarabine", 'mitoxantrone, cytarabine')
    column = F.regexp_replace(column, r"(?i)(mito[zx]ant[h]?rone|novantrone)", 'mitoxantrone')
    column = F.regexp_replace(column, r"NSC[ -]?301739", 'mitoxantrone')
    column = F.regexp_replace(column, r"CL[ -]232[,]?315", 'mitoxantronehydrochloride')
    column = F.regexp_replace(column, r"mitoxantron ", 'mitoxantrone ')
    column = F.regexp_replace(column, r"mitoxantron\.", 'mitoxantrone.')
    column = F.regexp_replace(column, r"(?i)(mitoxantrone \(mit\))", 'mitoxantrone (mitoxantrone)')
    column = F.regexp_replace(column, r"(?i)(mitoxantrone \(mit,)", 'mitoxantrone (mitoxantrone,')
    column = F.regexp_replace(column, r"(?i)(mitoxantrone) \((MTX|MIP|MX)\)", 'mitoxantrone (mitoxantrone)')
    column = F.regexp_replace(column, r"(?i)(mitoxantrone hydrochloride) \((MIT|MTO)\)", 'mitoxantronehydrochloride (mitoxantronehydrochloride)')
    column = F.regexp_replace(column, r"(?i)(mitoxantrone hydrochloride)", 'mitoxantronehydrochloride')
    column = F.regexp_replace(column, r"(?i)(1,4-dihydroxy-5,8-bis[ ]?\(\([ ]?\(2-\[\(2-hydroxyethyl\)amino\]ethyl\)[ -]?amino\)\)-9,10-anthracenedione dihydrochloride)", 'mitoxantrone')

    # pemigatinib:
    column = F.regexp_replace(column, r"PEMAZYRE", 'pemigatinib')

    # prednisone:
    column = F.regexp_replace(column, r"1-dehydrocortisone|[Dd]eltasone|meticorten|NSC[ -]?10023", 'prednisone')
    column = F.regexp_replace(column, r"(?i)(ultracorten-H|ultracorten H|ultracortene|ultracorten)", 'prednisone')

    # rituximab:
    column = F.regexp_replace(column, r"Rituxan", 'Rituximab')
    column = F.regexp_replace(column, r"rituxan|\[RTX-(EU|US)\]|Truxima|CT-P10", 'rituximab')
    column = F.regexp_replace(column, r"(?i)(MabThera)", 'rituximab')
    column = F.regexp_replace(column, r"rituximab/Rituximab", 'rituximab rituximab')

    # thioguanine:
    column = F.regexp_replace(column, r"[Tt]ioguanine|NSC-752", 'thioguanine')
    column = F.regexp_replace(column, r"(?i)(6-thioguanine| 6 thioguanine)", 'thioguanine')
    column = F.regexp_replace(column, r"daunorubicin-cytarabine-6 thioguanine", 'daunorubicin cytarabine thioguanine')
    column = F.regexp_replace(column, r"\(6TG\)", '(thioguanine)')
    column = F.regexp_replace(column, r" 6TG ", ' thioguanine ')
    column = F.regexp_replace(column, r" 6TG\.", ' thioguanine.')
    column = F.regexp_replace(column, r" 6TG,", ' thioguanine,')
    column = F.regexp_replace(column, r"6TG-", 'thioguanine-')
    column = F.regexp_replace(column, r"2-amino-6-mercaptopurine|6-mercaptoguanine", 'thioguanine')
    column = F.regexp_replace(column, r"6[ -]?TG[Rr]", 'thioguanine resistance')
    column = F.regexp_replace(column, r"(?i)(thioguanine \(tg\))", 'thioguanine (thioguanine)')

    # vincristine:
    column = F.regexp_replace(column, r"vincrystine|NSC[ -]67574", 'vincristine')
    column = F.regexp_replace(column, r"(?i)(vincristine sulfate)", 'vincristinesulfate')
    column = F.regexp_replace(column, r"\[3H\][-]?vincristine|3Hvincristine", '3H-vincristine')
    column = F.regexp_replace(column, r"\[3H\][-]?VCR|3HVCR", '3H-vincristine')
    column = F.regexp_replace(column, r"[Vv]incristine \(VCR\)", 'vincristine (vincristine)')
    column = F.regexp_replace(column, r"CAS 57-22-7", 'vincristine')
    column = F.regexp_replace(column, r"vincristine,cytarabine", 'vincristine, cytarabine')

    # C-1027:
    column = F.regexp_replace(column, r"(?i)(lidamycin\(LDM\))", 'c-1027 (c-1027)')
    column = F.regexp_replace(column, r"lidamycin", 'c-1027')

    # glyceryl behenate:
    column = F.regexp_replace(column, r"glyceryl behenate", 'glycerylbehenate')
    column = F.regexp_replace(column, r"[Cc]ompritol 888 ATO", 'glycerylbehenate')

    # decitabine:
    column = F.regexp_replace(column, r"2'-deoxy-(beta-D|beta-d|β-d|β-D)-5-azacytidine", 'decitabine')
    column = F.regexp_replace(column, r"2'-deoxy-(beta-L|beta-l|β-l|β-L)-5-azacytidine", 'l-decitabine')

    # daunomycinone:
    column = F.regexp_replace(column, r'daunomycin aglycone', 'daunomycinone')
    column = F.regexp_replace(column, r'13-dihydrodaunomycinone', 'feudomycinonea')

    # valrubicin:
    column = F.regexp_replace(column, r'AD32|AD 32|AD-32', 'valrubicin')
    column = F.regexp_replace(column, r'(?i)(N-Trifluoroacetyladriamycin[ -]14-valerate)', 'valrubicin')

    # carmustine:
    column = F.regexp_replace(column, r'NSC-409962', 'carmustine')
    column = F.regexp_replace(column, r'BCNU|BCNU-NSC', 'carmustine')

    # dextromethorphan:
    column = F.regexp_replace(column, r'DXM[S]?', 'dextromethorphan')

    # docetaxel:
    column = F.regexp_replace(column, r'NSC[ -]?628503|RP[ -]?56976', 'docetaxel')

    # dactinomycin:
    column = F.regexp_replace(column, r'[Aa]ctinomycin[ -]D', 'dactinomycin')

    # Deixa tudo minúsculo.
    column = F.lower(column)

    return column

def words_preprocessing(df, column='word'):
    """Corrige alguns erros de digitação, normaliza alguns símbolos e remove palavras irrelevantes."""

    fix_typos_dict = {
        'citarabine': 'cytarabine',
	    'hdara-c': 'high-dose cytarabine',
	    'no-arac': 'n4-octadecyl-1-beta-d-arabinofuranosylcytosine',
	    'ara-c-ab': 'arac-agarose-bead',
	    'anhydro-ara-fc': "2,2'-anhydro-1-beta-d-arabinofuranosyl-5-fluorocytosine",
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

    # Cria tabela de sinônimos (cid | sinônimo), filtra alguns compostos que deram problema.
    synonyms = read_csv_table_files('./data/synonyms')
    synonyms = synonyms\
                .filter(F.col('cid') != "122172881")\
                .filter(F.col('cid') != "11104792")

    # Cria tabela de nomes (cid | nome), filtra alguns compostos que deram problema.
    titles = read_csv_table_files('./data/titles.csv', sep='|')
    titles = titles\
            .filter(F.col('cid') != "122172881")\
            .filter(F.col('cid') != "11104792")

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