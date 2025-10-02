import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from gensim.models import Word2Vec, FastText
import pandas as pd
from src.utils import *

def train_word_embedding_models(normalized_target_disease: str, start_year: int, current_year: int, model_type: str = 'w2v', parameters_combination: list = None, min_count: int = 5, sg: int = 1, hs: int = 0, epochs: int = 15, min_corpus_size: int = 10):
    """
    Trains Word2Vec or FastText models year-over-year based on cleaned abstracts.

    Args:
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        model_type (str): The type of model to train ('w2v' for Word2Vec or 'ft' for FastText). Defaults to 'w2v'.
        parameters_combination (list): A list of parameter combinations for the embedding model. Should contain [vector_size, alpha, negative]. Defaults to [[100, 0.0025, 10], [200, 0.025, 15]].
        min_count (int): Ignores all words with total frequency lower than this. Defaults to 5.
        sg (int): Training algorithm: 1 for skip-gram, 0 for CBOW. Defaults to 1.
        hs (int): If 1, hierarchical softmax will be used for model training. If 0 (default), and `negative` is non-zero, negative sampling will be used. Defaults to 0.
        epochs (int): Number of iterations (epochs) over the corpus. Defaults to 15.
        min_corpus_size (int): Minimum number of abstracts required to train a model for a given year range. Defaults to 10.
    """

    w2v_path = f'./data/{normalized_target_disease}/models/w2v_combination15/'
    ft_path = f'./data/{normalized_target_disease}/models/ft_combination16/'

    os.makedirs(w2v_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)

    # Determines what kind of model will be trained.
    if model_type not in ['w2v', 'ft']:
        raise ValueError("model_type must be either 'w2v' or 'ft'")

    # Sets the parameters for the models.
    if parameters_combination is None:
        if model_type == 'w2v': parameters_combination = [[100, 0.0025, 10], [200, 0.025, 15]]
        else: parameters_combination = [[300, 0.0025, 5]]

    # Reads the DataFrame with the clean abstracts.
    print('Reading clean abstracts...')
    df = pd.read_csv(f'./data/{normalized_target_disease}/corpus/clean_abstracts/clean_abstracts.csv', names=['id', 'year_extracted', 'summary'], header=0)
    if df is None or df.empty:
        print("Clean abstracts file not found. Have you run step 4?")
        return
    print(df.head())

    print(f'Training model from {start_year} to {current_year}.')

    # Puts the abstracts published in the year range in a list.
    abstracts_in_range = df[df['year_extracted'] <= current_year]['summary'].dropna().to_list()
    print(f'{len(abstracts_in_range)} abstracts found in range.\n')

    # Splits the abstracts into words.
    abstracts_in_range = [x.split() for x in abstracts_in_range]

    # If there are not enough abstracts, vocabulary building will fail.
    # Skip this year if the corpus is too small.
    if len(abstracts_in_range) < min_corpus_size:
        print(f"Warning: Corpus for year {current_year} is too small ({len(abstracts_in_range)} abstracts). Skipping.")
        return

    # Trains the model with the abstracts in the year range.
    if model_type == 'w2v':
        model = Word2Vec(
            sentences=abstracts_in_range,
            sorted_vocab=True,
            min_count=min_count,
            sg=sg,
            hs=hs,
            epochs=epochs,
            vector_size=parameters_combination[1][0],
            alpha=parameters_combination[1][1],
            negative=parameters_combination[1][2]
        )
        model.save(f'{w2v_path}/model_{start_year}_{current_year}.model')

    else:
        model = FastText(
            sentences=abstracts_in_range,
            sorted_vocab=True,
            min_count=min_count,
            sg=sg,
            hs=hs,
            epochs=epochs,
            vector_size=parameters_combination[0][0],
            alpha=parameters_combination[0][1],
            negative=parameters_combination[0][2])
        model.save(f'{ft_path}/model_{start_year}_{current_year}.model')

    print('Finished model training.')

if __name__ == '__main__':
    start = input('Enter start year: ')
    current = input('Enter current year: ')
    train_word_embedding_models(get_normalized_target_disease(), start, current)
