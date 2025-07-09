import string
from pathlib import Path


def get_target_disease():
    try:
        with open('target_disease.txt', 'r') as file:
            target_disease = file.read().strip()
    except FileNotFoundError:
        target_disease = None

    return target_disease

def set_target_disease(disease):
    with open('target_disease.txt', 'w') as file:
        file.write(disease)

def get_normalized_target_disease():
    return get_target_disease().lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')

def get_corpus_year_range(normalized_target_disease):
    aggregated_files = sorted(
        list(map(str, Path(f'./data/{normalized_target_disease}/corpus/aggregated_abstracts').glob('*.txt'))))

    year_range = int(Path(aggregated_files[0]).stem[-4:]), int(Path(aggregated_files[-1]).stem[-4:])
    return year_range[0], year_range[1]