import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import re

from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd

from src.utils import *

def get_embedding_of_word(word, model, method):
    """
    Gets the word embedding of a given word from a Word2Vec or FastText model.
    Args:
        word: Desired word to get the embedding of.
        model: Word2Vec or FastText model to get the embedding from.
        method: Must be either 'da' (direct access) or 'avg' (average of embeddings).

    Returns:
        The output embedding if method is 'da' and the word is in the model's vocabulary.
        The average vector of all words containing the substring 'word' if method is 'avg'.
    """

    if method not in ['da', 'avg']:
        print('Method must be either "da" or "avg".')
        return None

    # If it's the chosen method, tries to directly access the word embedding.
    if method == 'da':
        if word in model.wv.key_to_index: return model.wv[word]
        return None

    else:
        # Gets all tokens in the model's vocabulary that contain the substring 'word'.
        tokens_containing_the_word = [key for key in model.wv.index_to_key if word in key]

        if not tokens_containing_the_word:
            return None

        # Gets the embeddings of all tokens that contain the substring 'word'.
        output_embeddings = model.wv[tokens_containing_the_word]

        # Returns the average of the embeddings.
        return np.mean(output_embeddings, axis=0)


import pandas as pd
from chembl_webresource_client.new_client import new_client

def load_drugs_from_chembl_refined():
    """
    Versão refinada que busca APENAS 'Small molecules' no ChEMBL.
    """
    print("Buscando lista de 'Small molecule drugs' do ChEMBL (busca refinada)...")
    drug_names_set = set()
    try:
        molecule = new_client.molecule
        
        # --- MUDANÇA 1: Adicionado filtro 'molecule_type' para aumentar a especificidade ---
        # Isso remove elementos, anticorpos, peptídeos grandes, etc., focando em drogas clássicas.
        approved_drugs_query = molecule.filter(
            max_phase__in=[2, 3, 4], 
            molecule_type='Small molecule' 
        ).only(['pref_name', 'synonyms'])

        for i, drug in enumerate(approved_drugs_query):
            if drug['pref_name']:
                drug_names_set.add(drug['pref_name'].lower().replace(' ', ''))
            for synonym in drug.get('synonyms', []):
                if synonym:
                    drug_names_set.add(synonym.lower().replace(' ', ''))
            if (i + 1) % 5000 == 0:
                print(f"Processados {i+1} registros de 'Small molecules' do ChEMBL...")

        print(f"Carregados {len(drug_names_set)} nomes/sinônimos de 'Small molecules' do ChEMBL.")
        return drug_names_set
    except Exception as e:
        print(f"Ocorreu um erro ao conectar com a API do ChEMBL: {e}")
        return set()

def get_therapeutic_compounds(normalized_target_disease: str, biomolecule_blacklist: set = None):
    """
    Creates a whitelist of therapeutic compounds with enhanced filters to remove
    biological noise and focus on specific treatments.

    Args:
        normalized_target_disease (str): The normalized name of the target disease.
        biomolecule_blacklist (set): A set of biomolecules to exclude from the whitelist.
                                     Defaults to a predefined set of common biomolecules.

    Returns:
        filtered_whitelist (set): A set of normalized therapeutic compound names.
    """

    cache_path = f'./data/compound_whitelist.txt'

    if os.path.exists(cache_path):
        print(f"Carregando whitelist a partir do arquivo de cache: {cache_path}")
        with open(cache_path, 'r') as f:
            # Lê cada linha, remove espaços/quebras de linha e adiciona ao conjunto
            filtered_whitelist = {line.strip() for line in f}
        print(f"Whitelist carregada com {len(filtered_whitelist)} compostos.")
        return filtered_whitelist

    # Se não existir, a função continua...
    print("Arquivo de cache não encontrado. Gerando whitelist a partir das fontes de dados...")


    pubchem_syn_path = "./data/pubchem_data/CID-Synonym-filtered"
    pubchem_title_path = "./data/pubchem_data/CID-Title"


    # Usa a nova função de busca refinada
    chembl_drug_names = load_drugs_from_chembl_refined()
    if not chembl_drug_names:
        print("Não foi possível obter a lista de medicamentos do ChEMBL. Abortando.")
        return set()

    # O carregamento e merge com Pandas continua o mesmo
    print("Carregando arquivos da PubChem com Pandas...")
    try:
        synonyms_df = pd.read_csv(pubchem_syn_path, sep='\t', header=None, names=['cid', 'synonym'], dtype={'cid': str})
        titles_df = pd.read_csv(pubchem_title_path, sep='\t', header=None, names=['cid', 'title'], dtype={'cid': str})
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}. Verifique os caminhos.")
        return set()

    print("Mapeando nomes do ChEMBL para CIDs da PubChem...")
    chembl_df = pd.DataFrame(list(chembl_drug_names), columns=['chembl_term_normalized'])
    synonyms_df['synonym_normalized'] = synonyms_df['synonym'].str.lower().str.replace(r'\s+', '', regex=True)
    synonyms_df.dropna(subset=['synonym_normalized', 'cid'], inplace=True)
    matched_cids_df = pd.merge(chembl_df, synonyms_df, left_on='chembl_term_normalized', right_on='synonym_normalized', how='inner')
    unique_matched_cids = matched_cids_df['cid'].unique()
    print(f"Encontrados {len(unique_matched_cids)} CIDs únicos correspondentes a compostos terapêuticos.")

    print("Buscando títulos canônicos para os CIDs encontrados...")
    therapeutic_titles_df = titles_df[titles_df['cid'].isin(unique_matched_cids)]
    normalized_titles = therapeutic_titles_df['title'].str.lower().str.replace(r'\s+', '', regex=True)
    final_whitelist_set = set(normalized_titles.dropna().unique())
    print(f"Whitelist (antes da exclusão) criada com {len(final_whitelist_set)} nomes canônicos.")

    if biomolecule_blacklist is None:
        biomolecule_blacklist = {
            'thymidine', 'deoxycytidine', 'uridine', 'cytidine', 'adenosine', 'guanine', 'cytosine', 'thymine',
            'aminoacids', 'glutathione', 'arginine', 'lysine', 'valine', 'citrulline', 'leucine', 'isoleucine',
            'cholesterol', 'histamine', 'folicacid', 'cholecalciferol', 'retinoicacid', 'nicotinicacid', 'alpha-tocopherol',
            'lithium', 'magnesium', 'oxygen', 'nitrogen', 'platinum', 'hydrogenperoxide', 'radium', 'potassium',
            'agar', 'hemin', 'phorbol12-myristate13-acetate', 'methylcellulose(4000cps)',
            'insulin', 'triphosphate', 'histaminedihydrochloride', 'water', 'carbon'
    }

    # Filtra a whitelist final removendo os itens da blacklist
    filtered_whitelist = final_whitelist_set - biomolecule_blacklist
    
    print(f"Removidos {len(final_whitelist_set) - len(filtered_whitelist)} compostos genéricos da whitelist.")
    print(f"Whitelist final e filtrada contém {len(filtered_whitelist)} compostos.")

    # Salva o conjunto final no arquivo de cache para uso futuro
    print(f"Salvando whitelist gerada em: {cache_path}")
    
    # Garante que o diretório de dados exista antes de tentar salvar o arquivo
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with open(cache_path, 'w') as f:
        # Salva em ordem alfabética para consistência
        for compound in sorted(list(filtered_whitelist)):
            f.write(f"{compound}\n")
    
    return filtered_whitelist

def generate_compound_dot_products_csv(normalized_target_disease: str, model_type: str = 'w2v', embedding_method: str = 'da', biomolecule_blacklist: set = None):
    """
    Generates CSV files containing historical dot products and Euclidean distances
    between compound embeddings and the target disease embedding.

    Args:
        normalized_target_disease (str): The normalized name of the target disease.
        model_type (str): The type of word embedding model to use ('w2v' or 'ft'). Defaults to 'w2v'.
        embedding_method (str): The method to get word embeddings ('da' for direct access or 'avg' for average of embeddings). Defaults to 'da'.
        biomolecule_blacklist (set): A set of biomolecules to exclude from the whitelist.
                                     Defaults to a predefined set of common biomolecules.
    """

    # Sets the model type and parameter combination.
    if model_type not in ['w2v', 'ft']:
        print('Invalid validation type, has to be either "w2v" or "ft".')
        return
    combination = '15' if model_type == 'w2v' else '16'

    model_directory_path = f'./data/{normalized_target_disease}/models/{model_type}_combination{combination}/'
    compound_history_path = f'./data/{normalized_target_disease}/validation/{model_type}/compound_history/'
    compound_list_path = f'./data/{normalized_target_disease}/corpus/compounds_in_corpus.txt'

    os.makedirs(compound_history_path, exist_ok=True)

    # Loads all compounds present in the corpus.
    print('Gathering all compounds mentioned in the corpus.')
    all_compounds_in_corpus = get_therapeutic_compounds(normalized_target_disease)

    if not all_compounds_in_corpus:
        print(f"Could not load or generate compound list. Have you run step 4?")
        return

    # Loads each of the year range trained models.
    print(f'Loading models...')
    models = []
    try:
        models = sorted([f.path for f in os.scandir(model_directory_path) if f.name.endswith('.model')])
    except FileNotFoundError: pass
    if not models:
        print('No models found. Have you run step 5 with the same model type?')
        return

    # Filters the list of compounds to include only those that are present in the final model's vocabulary.
    start_year, end_year = get_corpus_year_range(normalized_target_disease)
    final_model_path = f'{model_directory_path}/model_{start_year}_{end_year}.model'
    if os.path.exists(final_model_path):
        if model_type == 'w2v': final_model = Word2Vec.load(final_model_path)
        else: final_model = FastText.load(final_model_path)
        
        compounds_in_model_vocab = [compound for compound in all_compounds_in_corpus if compound in final_model.wv.key_to_index]
        
        print(f"Original compound list size: {len(all_compounds_in_corpus)}")
        print(f"Filtered compound list size (in model vocab): {len(compounds_in_model_vocab)}")
        
        all_compounds_in_corpus = compounds_in_model_vocab
    else:
        print(f"Final model not found at {final_model_path}. Skipping vocabulary filtering.")

    # Initializes a dictionary to store various metrics for each compound.
    # Each metric is a way to measure the relationship between the compound and the target disease. TODO add a delta variation of each metric.
    dictionary_for_all_compounds = {}
    for c in all_compounds_in_corpus:
        dictionary_for_all_compounds.update({
            f'{c}_comb{combination}': {
                'year': [],
                'dot_product': [],
                'normalized_dot_product': [],
                'delta_normalized_dot_product': [],
                'euclidian_distance': [],
                'score': [],
            }
        }
        )

    # List of years until which the models were trained.
    start_year, end_year = get_corpus_year_range(normalized_target_disease)
    
    # For each year until which a model was trained...
    for current_year in range(start_year, end_year + 1):
        print(f'Loading model trained until {current_year}.')

        # Loads the model that was trained until year y.
        model_path = f'{model_directory_path}/model_{start_year}_{current_year}.model'
        if not os.path.exists(model_path):
            print(f'Model trained until {current_year} not found. Skipping.')
            continue

        if model_type == 'w2v': model = Word2Vec.load(model_path)
        else: model = FastText.load(model_path)

        # Gets the word embedding of the target disease.
        target_disease_embedding = get_embedding_of_word(normalized_target_disease, model, method='da')
        if target_disease_embedding is None:
            print(f"CRITICAL: Target disease '{normalized_target_disease}' not found in the model's vocabulary.")
            continue

        # Tries to get the embedding of each compound.
        for compound in all_compounds_in_corpus:
            #print(f'Accessing the output embedding of {compound}.')
            compound_we = get_embedding_of_word(compound, model, method='da')
            if compound_we is None:
                # This should not happen in the last model, as all compounds were validated against the corpus.
                if current_year == end_year: 
                    print(f"Compound '{compound}' not found in the model's vocabulary even though it is mentioned in the corpus.")
                continue

            # Computes the dot product and the Euclidean distance between the compound's embedding and the target disease's embedding.
            dot_product = np.dot(compound_we, target_disease_embedding)
            euclidean_distance = np.linalg.norm(compound_we - target_disease_embedding)

            # Normalizes the dot product by using the cosine similarity.
            # This is a more stable metric that is not dependent on the maximum value of the entire time series.
            norm_compound = np.linalg.norm(compound_we)
            norm_disease = np.linalg.norm(target_disease_embedding)
            if norm_compound > 0 and norm_disease > 0:
                normalized_dot_product = (dot_product / (norm_compound * norm_disease)).item()
            else:
                normalized_dot_product = 0.0

            # Fills out the corresponding dictionary entry with the results.
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['year'].append(current_year)
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['dot_product'].append(dot_product.item())
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['normalized_dot_product'].append(normalized_dot_product)
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['euclidian_distance'].append(euclidean_distance.item())

    print('Generating historical record for each compound...')
    for c in all_compounds_in_corpus:
        key = f'{c}_comb{combination}'
        key_filename = re.sub(r'[\\/*?:"<>|]', '_', key)
        key_filename = key_filename[:50]

        # Gets the dot products for the current compound. Will be used for further calculations.
        dot_products = dictionary_for_all_compounds[key]['dot_product']

        # Skips the compound if it was never iterated upon (i.e., no dot products were computed).
        # This SHOULD NOT happen, as the compounds were validated against the corpus. TODO
        if not dot_products:
            # print(f"Aviso: Lista de dados para '{key}' está vazia. Pulando cálculos.")
            continue

        # The normalized_dot_product is now calculated as cosine similarity inside the year loop.
        normalized_values = np.array(dictionary_for_all_compounds[key]['normalized_dot_product'])

        # Computes the delta normalized dot product, which is the difference between the current and previous normalized dot product.
        # This metric shows the change in the relationship between the compound and the target disease over time.
        if normalized_values.size > 0:
            delta_values = np.diff(normalized_values, prepend=normalized_values[0])
            dictionary_for_all_compounds[key]['delta_normalized_dot_product'] = delta_values.tolist()
        else:
            delta_values = np.array([])
            dictionary_for_all_compounds[key]['delta_normalized_dot_product'] = []

        euclidian_distances = np.array(dictionary_for_all_compounds[key]['euclidian_distance'])

        # Computes a score to measure the overall closeness to the target disease
        if normalized_values.size > 0:
            score = normalized_values * (1 + (10 * delta_values)) / (euclidian_distances + 1e-9)
            dictionary_for_all_compounds[key]['score'] = score.tolist()
        else:
            dictionary_for_all_compounds[key]['score'] = []


        # Writes the compound entry in the dictionary to a CSV file.
        pd.DataFrame.from_dict(data=dictionary_for_all_compounds[key]).to_csv(
            f'{compound_history_path}/{key_filename}.csv',
            columns=['year', 'normalized_dot_product', 'delta_normalized_dot_product', 'euclidian_distance', 'score'],
            index=False
        )

    print('End :)')


if __name__ == '__main__':
    generate_compound_dot_products_csv(get_normalized_target_disease())
