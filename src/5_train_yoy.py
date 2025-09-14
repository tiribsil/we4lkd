import os

from gensim.models import Word2Vec, FastText
import pandas as pd
from src.utils import *

os.chdir(Path(__file__).resolve().parent.parent)

def main():
    normalized_target_disease = get_normalized_target_disease()

    w2v_path = f'./data/{normalized_target_disease}/models/w2v_combination15/'
    ft_path = f'./data/{normalized_target_disease}/models/ft_combination16/'

    os.makedirs(w2v_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)

    # Determines what kind of model will be trained.
    model_type = 'w2v'
    if model_type not in ['w2v', 'ft']:
        raise ValueError("model_type must be either 'w2v' or 'ft'")

    # Sets the parameters for the models.
    if model_type == 'w2v': parameters_combination = [[100, 0.0025, 10], [200, 0.025, 15]]
    else: parameters_combination = [[300, 0.0025, 5]]

    # Reads the DataFrame with the clean abstracts.
    print('Reading clean abstracts...')
    df = pd.read_csv(f'./data/{normalized_target_disease}/corpus/clean_abstracts/clean_abstracts.csv', names=['id', 'year_extracted', 'summary'], header=0)
    if df is None or df.empty:
        print("Clean abstracts file not found. Have you run step 4?")
        return
    print(df.head())

    start_year, end_year = get_corpus_year_range(normalized_target_disease)

    # Trains a model for each year range (from start_year to each year between start_year and end_year).
    for current_year in range(start_year, end_year + 1):
        print(f'Training model from {start_year} to {current_year}.')

        # Puts the abstracts published in the year range in a list.
        abstracts_in_range = df[df['year_extracted'] <= current_year]['summary'].dropna().to_list()
        print(f'{len(abstracts_in_range)} abstracts found in range.\n')

        # Splits the abstracts into words.
        abstracts_in_range = [x.split() for x in abstracts_in_range]

        # --- Gemini Edit Start ---
        # If there are not enough abstracts, vocabulary building will fail.
        # Skip this year if the corpus is too small.
        MINIMUM_CORPUS_SIZE = 10  # Set a reasonable threshold
        if len(abstracts_in_range) < MINIMUM_CORPUS_SIZE:
            print(f"Warning: Corpus for year {current_year} is too small ({len(abstracts_in_range)} abstracts). Skipping.")
            continue
        # --- Gemini Edit End ---

        # Trains the model with the abstracts in the year range.
        if model_type == 'w2v':
            model = Word2Vec(
                sentences=abstracts_in_range,
                sorted_vocab=True,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                vector_size=parameters_combination[1][0],
                alpha=parameters_combination[1][1],
                negative=parameters_combination[1][2]
            )
            model.save(f'{w2v_path}/model_{start_year}_{current_year}.model')

        else:
            model = FastText(
                sentences=abstracts_in_range,
                sorted_vocab=True,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                vector_size=parameters_combination[0][0],
                alpha=parameters_combination[0][1],
                negative=parameters_combination[0][2])
            model.save(f'{ft_path}/model_{start_year}_{current_year}.model')

    print('Finished model training.')

if __name__ == '__main__':
    main()