import glob

import pandas as pd

from target_disease import target_disease, folder_name


def extract_cooccurrence_contexts(corpus_df, target_chemical, target_disease, window_size=20):
    """
    Extrai janelas de texto (pseudo-sentenças) onde um chemical e uma doença coocorrem.

    Args:
        corpus_df (pd.DataFrame): O DataFrame contendo os textos limpos na coluna 'summary'.
        target_chemical (str): A entidade CHEMICAL a ser procurada.
        target_disease (str): O nome da doença-alvo a ser procurada.
        window_size (int): O número de palavras a serem capturadas antes e depois do
                             target_chemical para formar a janela de contexto.

    Returns:
        list: Uma lista de strings, onde cada string é uma "sentença de evidência"
              contendo ambos os termos.
    """
    evidence_sentences = []

    # Processa cada resumo (linha) no DataFrame
    for summary in corpus_df['summary'].dropna():
        words = summary.split()

        # Encontra todos os índices onde o chemical aparece no resumo
        try:
            indices = [i for i, word in enumerate(words) if word == target_chemical]
        except ValueError:
            # Se o chemical não estiver no resumo, pula para o próximo
            continue

        # Para cada ocorrência do chemical, cria uma janela de contexto
        for index in indices:
            # Define o início e o fim da janela, cuidando dos limites do texto
            start = max(0, index - window_size)
            end = min(len(words), index + window_size + 1)

            # Extrai a janela de palavras
            context_window = words[start:end]

            # Verifica se a doença-alvo está na janela de contexto
            if target_disease in context_window:
                # Se ambos estiverem presentes, junta a janela em uma string e armazena
                evidence_sentence = " ".join(context_window)

                # Opcional: Adiciona "..." para indicar que é um trecho
                evidence_sentences.append(f"... {evidence_sentence} ...")

    # Retorna uma lista de sentenças únicas para evitar duplicatas
    return list(set(evidence_sentences))

if __name__ == '__main__':
    MODEL_TYPE = 'w2v' # Deve ser 'w2v' ou 'ft'
    if MODEL_TYPE not in ['w2v', 'ft']:
        raise ValueError("MODEL_TYPE deve ser 'w2v' ou 'ft'.")

    top_n_directory = f'./data/{folder_name}/{MODEL_TYPE}'

    metrics = [
        'dot_product_result_absolute',
        'softmax',
        'softmax_normalization',
        'softmax_standardization'
    ]

    candidate_compounds = []
    file_pattern = f'{top_n_directory}/top_*.csv'
    files = glob.glob(file_pattern)

    for file in files:
        df = pd.read_csv(file)
        candidate_compounds.extend(df['chemical_name'].tolist())

    candidate_compounds = list(set(candidate_compounds))
    clean_results_df = pd.read_csv(f'./data/{folder_name}/clean_results/clean_results.csv')

    for compound in candidate_compounds:
        sentences = extract_cooccurrence_contexts(clean_results_df, compound, folder_name, 20)
        if not sentences: continue
        print(sentences)



























