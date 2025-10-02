import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from src.utils import *

def aggregate_abstracts_from_year(normalized_target_disease: str, target_year: int):
    """
    Aggregates raw abstract text files published during a certain year into a single file.

    Args:
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        target_year (int): Year from which abstracts will be considered.
    """
    source_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    destination_path = f'./data/{normalized_target_disease}/corpus/aggregated_abstracts'
    os.makedirs(destination_path, exist_ok=True)

    # Gets the names of all raw abstracts that came from the crawler.
    filenames = sorted(list(map(str, Path(source_path).glob('*.txt'))))
    if not filenames:
        print('No files found in raw_abstracts directory. Have you run the crawler first?')
        exit(0)

    # Gets all filenames that are from the target year.
    filenames_in_year = [f for f in filenames if int(Path(f).stem[:4]) == target_year]

    print(f'Aggregating {len(filenames_in_year)} papers from the year {target_year}.')

    # Merges the content of all files into a single file.
    output_file = f'{destination_path}/results_file_{target_year}.txt'
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for fname in filenames_in_year:
            with open(fname, 'r', encoding='utf-8') as f_in:
                content = f_in.read().strip().replace('\n', ' ')
                title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                f_out.write(f"{title}|{content}\n")

    full_corpus_file = f'{destination_path}/aggregated_corpus.txt'
    
    print(f'\nCreating a cumulative aggregation of all {len(filenames)} papers.')
    with open(full_corpus_file, 'w', encoding='utf-8') as f_out:
        for fname in filenames:
            with open(fname, 'r', encoding='utf-8') as f_in:
                content = f_in.read().strip().replace('\n', ' ')
                title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                f_out.write(f"{title}|{content}\n")
    
    print(f'Full corpus saved to {full_corpus_file}')

if __name__ == '__main__':
    y = input('Aggregate files from which year?')
    aggregate_abstracts_from_year(get_normalized_target_disease(), y)
