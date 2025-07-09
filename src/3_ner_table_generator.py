import os

import spacy
import pandas as pd
from pathlib import Path
from src.utils import *

os.chdir(Path(__file__).resolve().parent.parent)


target_disease = get_target_disease()
normalized_target_disease = get_normalized_target_disease()

def load_spacy_model():
    """
    Loads the spaCy NER model for biomedical named entity recognition.
    Returns:
        The loaded spaCy model if successful.
    """

    spacy.require_gpu()
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        print(f"'en_ner_bc5cdr_md' spaCy model loaded. Using GPU: {spacy.prefer_gpu()}")
        return nlp
    except OSError:
        print("'en_ner_bc5cdr_md' model not found. Please, download it.")
        return None


def process_abstracts_from_file(nlp_model):
    """
    Generates a table of named entities from abstracts.
    Args:
        nlp_model: SpaCy NER model.

    Returns:
        ner_results: List of dictionaries with 'token' and 'entity' keys.
    """
    relevant_spacy_entity_types = ['CHEMICAL']
    mapped_entity_type = 'pharmacologic_substance'

    input_folder_path = Path(f'./data/{normalized_target_disease}/corpus/aggregated_abstracts')
    if not input_folder_path.exists() or not input_folder_path.is_dir():
        return []

    all_filenames = sorted([str(x) for x in input_folder_path.glob('*.txt')])

    file_to_process = all_filenames[-1]

    print(f"Reading {file_to_process}...")

    ner_results = []
    texts_to_process_batch = []
    batch_size = 500

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

def main():
    output_ner_csv_path = f'./data/{normalized_target_disease}/corpus/ner_table.csv'

    # Loads the spaCy NER model.
    nlp = load_spacy_model()
    if nlp is None:
        return

    Path(f"./data/{normalized_target_disease}").mkdir(parents=True, exist_ok=True)

    # Generates NER data based on the latest aggregated abstracts file (contains every abstract from corpus).
    print(f"Starting NER table generation for {target_disease}.")
    ner_data = process_abstracts_from_file(nlp)
    if not ner_data:
        print(f"Aggregated abstracts not found. Have you run step 2?")
        return

    # Creates a DataFrame from the NER data and removes duplicates.
    ner_df = pd.DataFrame(ner_data).drop_duplicates().reset_index(drop=True)

    print(f"{len(ner_df)} NER tokens found.")
    print(ner_df.head())

    ner_df.to_csv(output_ner_csv_path, index=False)
    print(f"NER table saved at {output_ner_csv_path}.")


if __name__ == '__main__':
    main()