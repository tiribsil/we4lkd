import os
import re

from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd

from src.utils import *

os.chdir(Path(__file__).resolve().parent.parent)


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


def get_compounds(normalized_target_disease):
    pubchem_path = 'data/pubchem_data/CID-Title'
    clean_abstracts_path = f'./data/{normalized_target_disease}/corpus/clean_abstracts/clean_abstracts.csv'

    pubchem_titles = pd.read_csv(
        pubchem_path,
        sep='\t',
        header=None,
        usecols=[1],
        names=['Title']
    )
    pubchem_normalized_set = set(
        pubchem_titles['Title'].str.lower().str.replace(r'\s+', '', regex=True).dropna()
    )

    try:
        clean_df = pd.read_csv(clean_abstracts_path)
    except FileNotFoundError:
        return []

    word_counts = clean_df['summary'].str.split(expand=True).stack().value_counts()

    frequent_words = word_counts[word_counts >= 5]
    corpus_words_set = set(frequent_words.index)
    print(f"Found {len(corpus_words_set)} unique words with frequency >= 5 in the corpus.")

    validated_normalized_compounds = list(pubchem_normalized_set & corpus_words_set)

    print(f"Total validated compounds: {len(validated_normalized_compounds)}")

    with open(f'./data/{normalized_target_disease}/corpus/compounds_in_corpus.txt', 'w', encoding='utf-8') as f:
        for compound in sorted(validated_normalized_compounds):
            f.write(f"{compound}\n")

    return validated_normalized_compounds


def main():
    normalized_target_disease = get_normalized_target_disease()

    # Sets the model type and parameter combination.
    model_type = 'w2v'
    if model_type not in ['w2v', 'ft']:
        print('Invalid validation type, has to be either "w2v" or "ft".')
        return
    combination = '15' if model_type == 'w2v' else '16'

    model_directory_path = f'./data/{normalized_target_disease}/models/{model_type}_combination{combination}/'
    compound_history_path = f'./data/{normalized_target_disease}/validation/{model_type}/compound_history/'
    compound_list_path = f'./data/{normalized_target_disease}/corpus/compounds_in_corpus.txt'

    os.makedirs(compound_history_path, exist_ok=True)

    # Loads all compounds present in the corpus.
    if os.path.exists(compound_list_path):
        print('List of compounds in corpus already exists, loading it.')
        with open(compound_list_path, 'r', encoding='utf-8') as f:
            all_compounds_in_corpus = [line.strip() for line in f if line.strip()]
    else:
        print('Gathering all compounds mentioned in the corpus.')
        all_compounds_in_corpus = get_compounds(normalized_target_disease)
    if not all_compounds_in_corpus:
        print(f"Clean abstracts file not found. Have you run step 4?")
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

        # Tries to get the embedding of each compound.
        for compound in all_compounds_in_corpus:
            print(f'Accessing the output embedding of {compound}.')
            compound_we = get_embedding_of_word(compound, model, method='da')
            if compound_we is None:
                # This should not happen in the last model, as all compounds were validated against the corpus.
                print(f"Compound '{compound}' not found in the model's vocabulary.")
                continue

            # Computes the dot product and the Euclidean distance between the compound's embedding and the target disease's embedding.
            dot_product = np.dot(compound_we, target_disease_embedding).item()
            euclidean_distance = np.linalg.norm(compound_we - target_disease_embedding).item()

            # Fills out the corresponding dictionary entry with the results.
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['year'].append(current_year)
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['dot_product'].append(dot_product)
            dictionary_for_all_compounds[f'{compound}_comb{combination}']['euclidian_distance'].append(euclidean_distance)


    print('Generating historical record for each compound...')
    for c in all_compounds_in_corpus:
        key = f'{c}_comb{combination}'
        key_filename = re.sub(r'[\\/*?:"<>|]', '_', key)

        # Gets the dot products for the current compound. Will be used for further calculations.
        dot_products = dictionary_for_all_compounds[key]['dot_product']

        # Skips the compound if it was never iterated upon (i.e., no dot products were computed).
        # This SHOULD NOT happen, as the compounds were validated against the corpus. TODO
        if not dot_products:
            print(f"Aviso: Lista de dados para '{key}' está vazia. Pulando cálculos.")
            continue

        # Gets the normalized dot products and stores them in the dictionary.
        maximum = np.max(dot_products)
        if maximum > 0:
            normalized_values = [x / maximum for x in dot_products]
            dictionary_for_all_compounds[key]['normalized_dot_product'] = normalized_values
        else:
            # Should not happen, but this prevents division by zero.
            normalized_values = [0.0] * len(dot_products)
            dictionary_for_all_compounds[key]['normalized_dot_product'] = normalized_values

        # Computes the delta normalized dot product, which is the difference between the current and previous normalized dot product.
        # This metric shows the change in the relationship between the compound and the target disease over time.
        if normalized_values:
            delta_values = np.diff(np.array(normalized_values), prepend=0.0).tolist()
            dictionary_for_all_compounds[key]['delta_normalized_dot_product'] = delta_values
        else:
            dictionary_for_all_compounds[key]['delta_normalized_dot_product'] = []

        # Writes the compound entry in the dictionary to a CSV file.
        print(f'Writing file for {key}')
        pd.DataFrame.from_dict(data=dictionary_for_all_compounds[key]).to_csv(
            f'{compound_history_path}/{key_filename}.csv',
            columns=['year', 'normalized_dot_product', 'delta_normalized_dot_product', 'euclidian_distance'],
            index=False
        )

    print('End :)')


if __name__ == '__main__':
    main()