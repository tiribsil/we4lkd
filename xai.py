import glob
import json
import os
from time import sleep

import pandas as pd
from google import genai

from api_key import MY_API_KEY
from target_disease import target_disease, normalized_target_disease


def extract_co_occurrence_contexts(corpus_df, target_chemical, target_disease, window_size=20):
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

def _calculate_proximity_score(sentence, term1, term2):
    """
    Função auxiliar para calcular um score de proximidade baseado na distância
    entre dois termos em uma sentença.
    """
    words = sentence.split()
    try:
        # Encontra o índice da primeira ocorrência de cada termo
        idx1 = words.index(term1)
        idx2 = words.index(term2)

        # Calcula a distância absoluta entre os índices
        distance = abs(idx1 - idx2)

        # Converte a distância em um score (quanto menor a distância, maior o score)
        # Adicionamos 1 para evitar divisão por zero se a distância for 0.
        return 1.0 / (1.0 + distance)
    except ValueError:
        # Se um dos termos não for encontrado (o que é improvável mas seguro de tratar),
        # retorna um score baixo.
        return 0.0

def sort_by_rank(sentences_by_compound, target_disease_name):
    """
    Ordena as sentenças por rank de proximidade entre a doença alvo e o composto.
    A proximidade é medida pela distância entre as palavras na sentença.

    Args:
        sentences_by_compound (dict): Dicionário onde as chaves são compostos e os valores são listas de sentenças.
        target_disease_name (str): O nome da doença alvo.

    Returns:
        dict: Dicionário ordenado por rank.
    """
    sorted_sentences = {}
    for compound, sentences in sentences_by_compound.items():
        sorted_sentences[compound] = sorted(
            sentences,
            key=lambda s: _calculate_proximity_score(s, compound, target_disease_name),
            reverse=True
        )
    return sorted_sentences

def explanation_for_chemical(chemical, sentences, n_sentences=3):
    client = genai.Client(api_key=MY_API_KEY)

    prompt = f"""
    Context: You are a data scientist analyzing clues in old scientific papers to predict future drug discoveries. Below are excerpts from public medical papers, mentioning the compound "{chemical}" and the disease "{target_disease}".

    Instruction: Based ONLY on these clues, formulate a concise hypothesis about the potential therapeutic or biological relationship between "{chemical}" and "{target_disease}". Why might a scientist of that time, upon reading these excerpts, suspect a connection?
    
    Clues from the corpus:
    """
    for sentence in sentences[:n_sentences]:
        prompt += f"\n- {sentence}"

    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)

    return response.text

if __name__ == '__main__':
    MODEL_TYPE = 'w2v' # Deve ser 'w2v' ou 'ft'
    MODEL_YEAR = '2025'
    n_sentences = 3

    if MODEL_TYPE not in ['w2v', 'ft']:
        raise ValueError("MODEL_TYPE deve ser 'w2v' ou 'ft'.")

    sentences_dir_path = f'./data/{normalized_target_disease}/validation/per_compound/{MODEL_TYPE}_sentences'
    sentences_file_path = f'{sentences_dir_path}/sentences_{MODEL_YEAR}.json'
    os.makedirs(sentences_dir_path, exist_ok=True)

    sentences_by_compound = {}
    if os.path.exists(sentences_file_path):
        print(f"Arquivo de cache encontrado.")
        with open(sentences_file_path, 'r', encoding='utf-8') as f:
            sentences_by_compound = json.load(f)
    else:
        top_n_directory = f'./data/{normalized_target_disease}/{MODEL_TYPE}'

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
        clean_results_df = pd.read_csv(f'./data/{normalized_target_disease}/clean_results/clean_results.csv')

        print(f"Arquivo de cache não encontrado. Processando sentenças do zero...")

        for compound in candidate_compounds:
            sentences = extract_co_occurrence_contexts(clean_results_df, compound, normalized_target_disease, window_size=20)
            if len(sentences) < n_sentences: continue
            print(f'{len(sentences)} sentences for {compound}')
            sentences_by_compound[compound] = sentences

        sentences_by_compound = sort_by_rank(sentences_by_compound, normalized_target_disease)

        print(f"Salvando resultados processados em '{sentences_dir_path}'...")
        with open(sentences_file_path, 'w', encoding='utf-8') as f:
            json.dump(sentences_by_compound, f, indent=4, ensure_ascii=False)

    with open(f'{sentences_dir_path}/xai_{MODEL_YEAR}.txt', 'w', encoding='utf-8') as f:
        xai_output = ''
        for compound, sentences in sentences_by_compound.items():
            explanation = explanation_for_chemical(compound, sentences, n_sentences)
            xai_output += f'Explanation for {compound}:\n{explanation}\n'
            sleep(30)
        f.write(xai_output)
        print(xai_output)


























