import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from src.utils import *

def aggregate_abstracts_by_year(normalized_target_disease: str, start_year: int, end_year: int):
    """
    Aggregates raw abstract text files into yearly files and a single cumulative corpus file.
    For each year from start_year to end_year, it creates a file with all abstracts from that year.

    Args:
        normalized_target_disease (str): The normalized name of the target disease, used for file paths.
        start_year (int): The first year to process.
        end_year (int): The last year to process.
    """
    source_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    destination_path = f'./data/{normalized_target_disease}/corpus/aggregated_abstracts'
    os.makedirs(destination_path, exist_ok=True)

    # Gets the names of all raw abstracts that came from the crawler.
    filenames = sorted(list(map(str, Path(source_path).glob('*.txt'))))
    if not filenames:
        print('No files found in raw_abstracts directory. Have you run the crawler first?')
        return False

    # Update results file for each year from start_year to end_year
    for year in range(start_year, end_year + 1):
        filenames_in_year = [f for f in filenames if int(Path(f).stem[:4]) == year]
        print(f'Aggregating {len(filenames_in_year)} papers from the year {year}.')

        output_file = f'{destination_path}/results_file_{year}.txt'
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for fname in filenames_in_year:
                with open(fname, 'r', encoding='utf-8') as f_in:
                    content = f_in.read().strip().replace('\n', ' ')
                    title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                    f_out.write(f"{title}|{content}\n")

    # Create the single aggregated corpus with all papers found so far
    full_corpus_file = f'{destination_path}/aggregated_corpus.txt'
    print(f'\nCreating a cumulative aggregation of all {len(filenames)} papers.')
    with open(full_corpus_file, 'w', encoding='utf-8') as f_out:
        for fname in filenames:
            with open(fname, 'r', encoding='utf-8') as f_in:
                content = f_in.read().strip().replace('\n', ' ')
                title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                f_out.write(f"{title}|{content}\n")
    
    print(f'Full corpus saved to {full_corpus_file}')
    return True

if __name__ == '__main__':
    start_y = input('Aggregate files from which year?')
    end_y = input('Until which year?')
    aggregate_abstracts_by_year(get_normalized_target_disease(), int(start_y), int(end_y))
