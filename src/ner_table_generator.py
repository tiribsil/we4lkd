import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import spacy
import pandas as pd
from src.utils import *

def load_spacy_model(spacy_model_name: str):
    """
    Loads the spaCy NER model for biomedical named entity recognition.
    Args:
        spacy_model_name (str): The name of the spaCy model to load.
    Returns:
        The loaded spaCy model if successful.
    """

    spacy.require_gpu()
    try:
        nlp = spacy.load(spacy_model_name)
        print(f"'{spacy_model_name}' spaCy model loaded. Using GPU: {spacy.prefer_gpu()}")
        return nlp
    except OSError:
        print(f"'{spacy_model_name}' model not found. Please, download it.")
        return None


def process_abstracts_from_file(nlp_model, normalized_target_disease: str, relevant_spacy_entity_types: list, mapped_entity_type: str, batch_size: int):
    """
    Generates a table of named entities from abstracts.
    Args:
        nlp_model: SpaCy NER model.
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        relevant_spacy_entity_types (list): A list of spaCy entity types to extract.
        mapped_entity_type (str): The entity type to map relevant spaCy entities to.
        batch_size (int): The number of texts to process in each spaCy pipeline batch.

    Returns:
        ner_results: List of dictionaries with 'token' and 'entity' keys.
    """
    # relevant_spacy_entity_types = ['CHEMICAL'] # Now a parameter
    # mapped_entity_type = 'pharmacologic_substance' # Now a parameter

    input_folder_path = Path(f'./data/{normalized_target_disease}/corpus/aggregated_abstracts')
    if not input_folder_path.exists() or not input_folder_path.is_dir():
        return []

    file_to_process = f'{input_folder_path}/aggregated_corpus.txt'

    print(f"Reading {file_to_process}...")

    ner_results = []
    texts_to_process_batch = []
    
    # Goes through each line (abstract) in the file, processes the text in batches, and saves the NER results in a list.
    with open(file_to_process, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('|', 1)
            if len(parts) == 2:
                title, abstract = parts
                texts_to_process_batch.append(abstract)
            elif len(parts) == 1:
                texts_to_process_batch.append(parts[0])
            else:
                print(f"Warning: Line {line_number} in {file_to_process} has formatting issues: '{line[:100]}...'")
                continue

            if len(texts_to_process_batch) < batch_size: continue

            for doc in nlp_model.pipe(texts_to_process_batch, disable=["parser", "lemmatizer"]):
                for ent in doc.ents:
                    if ent.label_ in relevant_spacy_entity_types:
                        ner_results.append({
                            'token': ent.text.lower(),
                            'entity': mapped_entity_type
                        })
            texts_to_process_batch = []

    if texts_to_process_batch:
        for doc in nlp_model.pipe(texts_to_process_batch, disable=["parser", "lemmatizer"]):
            for ent in doc.ents:
                if ent.label_ in relevant_spacy_entity_types:
                    ner_results.append({
                        'token': ent.text.lower(),
                        'entity': mapped_entity_type
                    })

    return ner_results

def generate_ner_table(target_disease: str, normalized_target_disease: str, spacy_model_name: str = "en_ner_bc5cdr_md", relevant_spacy_entity_types: list = None, mapped_entity_type: str = 'pharmacologic_substance', batch_size: int = 500):
    """
    Generates a Named Entity Recognition (NER) table from aggregated abstracts.

    Args:
        target_disease (str): The name of the disease for which NER is being performed.
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        spacy_model_name (str): The name of the spaCy model to load. Defaults to "en_ner_bc5cdr_md".
        relevant_spacy_entity_types (list): A list of spaCy entity types to extract. Defaults to ['CHEMICAL'].
        mapped_entity_type (str): The entity type to map relevant spaCy entities to. Defaults to 'pharmacologic_substance'.
        batch_size (int): The number of texts to process in each spaCy pipeline batch. Defaults to 500.
    """
    if relevant_spacy_entity_types is None:
        relevant_spacy_entity_types = ['CHEMICAL']

    output_ner_csv_path = f'./data/{normalized_target_disease}/corpus/ner_table.csv'

    # Loads the spaCy NER model.
    nlp = load_spacy_model(spacy_model_name)
    if nlp is None:
        return False

    Path(f"./data/{normalized_target_disease}").mkdir(parents=True, exist_ok=True)

    # Generates NER data based on the latest aggregated abstracts file (contains every abstract from corpus).
    print(f"Starting NER table generation for {target_disease}.")
    ner_data = process_abstracts_from_file(nlp, normalized_target_disease, relevant_spacy_entity_types, mapped_entity_type, batch_size)
    if not ner_data:
        print(f"Aggregated abstracts not found. Have you run step 2?")
        return False

    # Creates a DataFrame from the NER data and removes duplicates.
    ner_df = pd.DataFrame(ner_data).drop_duplicates().reset_index(drop=True)

    print(f"{len(ner_df)} NER tokens found.")
    print(ner_df.head())

    ner_df.to_csv(output_ner_csv_path, index=False)
    print(f"NER table saved at {output_ner_csv_path}.")

    return True


if __name__ == '__main__':
    generate_ner_table(
        target_disease=get_target_disease(),
        normalized_target_disease=get_normalized_target_disease()
    )
