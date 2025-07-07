#######################################################
"""
    Esse script realiza a geração dos modelos Word2Vec a partir dos prefácios dos artigos
    já pré-processados/limpos/normalizados, presentes na pasta definida pela constante 
    CLEANED_DOCUMENTS_PATH.

    Os modelos sao treinados de forma incremental, por exemplo:
        Modelo 1: contempla artigos publicados entre 1900 e 1901;
        Modelo 2: contempla artigos publicados entre 1900 e 1902;
        Modelo 3: contempla artigos publicados entre 1900 e 1903;
        .
        .
        .
"""
#######################################################

import os, shutil

from gensim.models import Word2Vec, FastText
from pathlib import Path
import pandas as pd
from src.target_disease import *

os.chdir(Path(__file__).resolve().parent.parent)

def list_from_txt(file_path):
    """Creates a list of itens based on a .txt file, each line becomes an item.

    Args:
      file_path: the path where the .txt file was created.
    """
    
    strings_list = []
    with open (file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def clear_folder(dirpath):
    """ Clears all files from a folder, without deleting the folder.

        dirpath: the path of the folder.    
    """

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def main():
    normalized_target_disease = get_normalized_target_disease()

    w2v_path = f'./data/{normalized_target_disease}/models/w2v_combination15/'
    ft_path = f'./data/{normalized_target_disease}/models/ft_combination16/'

    print('Starting script')

    # CONSTANTS:
    model_type = 'w2v' # 'w2v' for Word2Vec or 'ft' for FastText
    if model_type not in ['w2v', 'ft']:
        raise ValueError("model_type must be either 'w2v' or 'ft'")

    # Cria as pastas para salvar os modelos e escolhe a combinação dos parâmetros.
    os.makedirs(w2v_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)

    if model_type == 'w2v': parameters_combination = [[100, 0.0025, 10], [200, 0.025, 15]]
    else: parameters_combination = [[300, 0.0025, 5]]

    # Pega os dados do arquivo.
    print('Reading DataFrame of papers')
    df = pd.read_csv(f'./data/{normalized_target_disease}/corpus/clean_abstracts/clean_abstracts.csv', names=['id', 'year_extracted', 'summary'])
    print(df.head())

    years = sorted(df['year_extracted'].unique().tolist())
    first_year = years[0]
    print(f'Years of the papers: {years}')

    # Pega os intervalos de tempo para treinar os modelos.
    # Isso é só para fazer o teste de quais compostos teriam sido recomendados com o tempo.
    ranges = [years[:i+1] for i in range(len(years))]

    # Para cada intervalo de tempo...
    for r in ranges:
        print('training model from {} to {}'.format(r[0], r[-1]))

        # Pega os abstracts que estão no intervalo.
        abstracts = df[df['year_extracted'].isin(r)]['summary'].dropna().to_list()
        print('number of abstracts: {}\n'.format(len(abstracts)))

        # Transforma cada elemento (string) numa lista de palavras.
        abstracts = [x.split() for x in abstracts]

        # Treina os modelos :)
        if model_type == 'w2v':
            model_comb15 = Word2Vec(
                # constant parameters:
                sentences=abstracts,
                sorted_vocab=True,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                # variable parameters:
                vector_size=parameters_combination[1][0],
                alpha=parameters_combination[1][1],
                negative=parameters_combination[1][2]
            )
            model_comb15.save(f'{w2v_path}/model_{first_year}_{r[-1]}.model')

        else:
            model = FastText(
                # constant parameters:
                sentences=abstracts,
                sorted_vocab=True,
                min_count=5,
                sg=1,
                hs=0,
                epochs=15,
                # variable parameters:
                vector_size=parameters_combination[0][0],
                alpha=parameters_combination[0][1],
                negative=parameters_combination[0][2])
            model.save(f'{ft_path}/model_{first_year}_{r[-1]}.model')

    print('END!')

if __name__ == '__main__':
    main()