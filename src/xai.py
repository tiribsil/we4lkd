import os
os.chdir(Path(__file__).resolve().parent.parent)

import glob
import itertools
import json
from time import sleep

import pandas as pd
from google import genai

from api_key import MY_API_KEY
from src.utils import *

def extract_co_occurrence_contexts(corpus_df, target_chemical, target_disease, year, window_size=20):
    """
    Extracts sentences from a corpus DataFrame where a target chemical and a target disease co-occur.
    Args:
        corpus_df (pd.DataFrame): The DataFrame containing the corpus.
        target_chemical (str): The chemical to search for in the summaries.
        target_disease (str): The disease to search for in the summaries.
        year (int): The year to filter the summaries by.
        window_size (int): The number of words before and after the chemical to include in the context window.
    Returns:
        list: A list of unique sentences where the chemical and disease co-occur.
    """
    evidence_sentences = []

    for index, line in corpus_df.dropna().iterrows():
        if int(line['filename']) > year: continue
        summary = line['summary']
        words = summary.split()

        try:
            indices = [i for i, word in enumerate(words) if word == target_chemical]
        except ValueError:
            continue

        for i in indices:
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            context_window = words[start:end]

            if target_disease in context_window:
                evidence_sentence = " ".join(context_window)

                evidence_sentences.append(f"... {evidence_sentence} ...")

    return list(set(evidence_sentences))

def _calculate_proximity_score(sentence, term1, term2):
    """
    Calculates a proximity score based on the distance between two terms in a sentence.
    """
    words = sentence.split()
    try:
        idx1 = words.index(term1)
        idx2 = words.index(term2)

        distance = abs(idx1 - idx2)

        return 1.0 / (1.0 + distance)
    except ValueError:
        return 0.0

def sort_by_rank(sentences_by_compound, target_disease_name):
    """
    Sorts the sentences for each compound based on a proximity score to the target disease name.
    Args:
        sentences_by_compound: Compound -> List of sentences dictionary.
        target_disease_name: The name of the target disease to calculate proximity scores with.
    Returns:
        dict: Sorted dictionary.
    """
    sorted_sentences = {}
    for compound, sentences in sentences_by_compound.items():
        sorted_sentences[compound] = sorted(
            sentences,
            key=lambda s: _calculate_proximity_score(s, compound, target_disease_name),
            reverse=True
        )
    return sorted_sentences

def sort_by_sentences(sentences_by_compound):
    """
    Sorts the compounds by the number of sentences they have, in descending order.

    Args:
        sentences_by_compound: Compound -> List of sentences dictionary.

    Returns:
        dict: Sorted dictionary.
    """
    return dict(sorted(sentences_by_compound.items(), key=lambda item: len(item[1]), reverse=True))

def explanation_for_chemical(target_disease: str, chemical: str, sentences: list, n_sentences: int, gemini_model_name: str):
    """
    Generates an explanation for the relationship between a chemical and a target disease based on provided sentences.
    Args:
        target_disease (str): The disease to analyze.
        chemical (str): The chemical to analyze.
        sentences (list): The sentences containing information about the chemical and disease.
        n_sentences (int): The number of sentences to be analyzed.
        gemini_model_name (str): The name of the Gemini model to use for generating explanations.

    Returns:
        str: A hypothesis about the potential therapeutic or biological relationship between the chemical and the disease.
    """
    client = genai.Client(api_key=MY_API_KEY)

    prompt = f"""
    Context: You are a data scientist analyzing clues in old scientific papers to predict future drug discoveries. Below are excerpts from public medical papers, mentioning the compound "{chemical}" and the disease "{target_disease}".

    Instruction: Based ONLY on these clues, formulate a concise hypothesis about the potential therapeutic or biological relationship between "{chemical}" and "{target_disease}". Why might a scientist of that time, upon reading these excerpts, suspect a connection?
    
    Clues from the corpus:
    """
    for sentence in sentences[:n_sentences]:
        prompt += f"\n- {sentence}"

    response = client.models.generate_content(model=gemini_model_name, contents=prompt)

    return response.text

def generate_xai_explanations(target_disease: str, normalized_target_disease: str, model_type: str = 'w2v', num_sentences_for_explanation: int = 3, num_compounds_for_xai: int = 10, co_occurrence_window_size: int = 20, gemini_model_name: str = 'gemini-2.5-flash-lite-preview-06-17', sleep_between_explanations: int = 30):
    """
    Generates XAI (Explainable AI) explanations for the relationship between top compounds and a target disease.

    Args:
        target_disease (str): The name of the target disease.
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        model_type (str): The type of word embedding model used ('w2v' or 'ft'). Defaults to 'w2v'.
        num_sentences_for_explanation (int): The number of top sentences to use for generating each explanation. Defaults to 3.
        num_compounds_for_xai (int): The number of top compounds to generate explanations for. Defaults to 10.
        co_occurrence_window_size (int): The number of words before and after the chemical to include in the context window for co-occurrence extraction. Defaults to 20.
        gemini_model_name (str): The name of the Gemini model to use for generating explanations. Defaults to 'gemini-2.5-flash-lite-preview-06-17'.
        sleep_between_explanations (int): Time in seconds to sleep between generating explanations to avoid API rate limits. Defaults to 30.
    """

    # Must be either 'w2v' or 'ft'.
    if model_type not in ['w2v', 'ft']:
        print('Model type must be either "w2v" or "ft".')
        return

    sentences_dir_path = f'./data/{normalized_target_disease}/validation/{model_type}/sentences'
    os.makedirs(sentences_dir_path, exist_ok=True)

    xai_path = f'./data/{normalized_target_disease}/validation/{model_type}/xai'
    os.makedirs(xai_path, exist_ok=True)

    # Gets the year range of the corpus for the target disease.
    start_year, end_year = get_corpus_year_range(normalized_target_disease)

    # For each year, finds sentences for each compound and pass them to the XAI model.
    for year in range(start_year, end_year + 1):
        sentences_file_path = f'{sentences_dir_path}/sentences_{year}.json'

        sentences_by_compound = {}
        if os.path.exists(sentences_file_path):
            print(f"Sentences found for {year}.")
            with open(sentences_file_path, 'r', encoding='utf-8') as f:
                sentences_by_compound = sort_by_sentences(json.load(f))
        else:
            print(f"Finding sentences for {year}...")

            top_n_directory = f'./data/{normalized_target_disease}/validation/{model_type}/top_n_compounds/{year}'

            candidate_compounds = []
            file_pattern = f'{top_n_directory}/top_*.csv'
            files = glob.glob(file_pattern)

            # Makes a list of the top compounds for every metric in that year.
            for file in files:
                df = pd.read_csv(file)
                candidate_compounds.extend(df['chemical_name'].tolist())

            candidate_compounds = list(set(candidate_compounds))
            clean_results_df = pd.read_csv(f'./data/{normalized_target_disease}/corpus/clean_abstracts/clean_abstracts.csv')

            # Extracts sentences for each compound in the candidate compounds list.
            for compound in candidate_compounds:
                sentences = extract_co_occurrence_contexts(clean_results_df, compound, normalized_target_disease, year, window_size=20)
                if len(sentences) < n_sentences: continue
                print(f'{len(sentences)} sentences for {compound}')
                sentences_by_compound[compound] = sentences

            sentences_by_compound = sort_by_sentences(sort_by_rank(sentences_by_compound, normalized_target_disease))

            with open(sentences_file_path, 'w', encoding='utf-8') as f:
                json.dump(sentences_by_compound, f, indent=4, ensure_ascii=False)

        if os.path.exists(f'{xai_path}/xai_{year}.md'):
            print(f"Explanation already exists for {year}. Skipping.")
            continue
        with open(f'{xai_path}/xai_{year}.md', 'w', encoding='utf-8') as f:
            xai_output = ''
            for compound, sentences in itertools.islice(sentences_by_compound.items(), n_compounds):
                explanation = explanation_for_chemical(target_disease, compound, sentences, n_sentences)
                xai_output += f'**Explanation for {compound}:**\n{explanation}\n\n'
                sleep(30)
            f.write(xai_output)
            print(xai_output)

if __name__ == '__main__':
    generate_xai_explanations(get_target_disease(), get_normalized_target_disease())
