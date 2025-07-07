import os
from pathlib import Path
from src.target_disease import *

def main():
    normalized_target_disease = get_normalized_target_disease()

    source_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    destination_path = f'./data/{normalized_target_disease}/corpus/aggregated_abstracts'
    os.makedirs(destination_path, exist_ok=True)

    # Gets the names of all raw abstracts that came from the crawler.
    filenames = sorted(list(map(str, Path(source_path).glob('*.txt'))))
    if not filenames:
        print('No files found in raw_abstracts directory. Have you run the crawler first?')
        exit(0)

    # Gets the year range from the filenames.
    start_year = int(Path(filenames[0]).stem[:4])
    end_year = int(Path(filenames[-1]).stem[:4])

    # For each year in the range, aggregates all articles published up to that year.
    for year in range(start_year, end_year + 1):
        # Gets all filenames that are from the current year or earlier.
        filenames_in_range = [f for f in filenames if int(Path(f).stem[:4]) <= year]

        print(f'Aggregating {len(filenames_in_range)} papers from {start_year} to {year}.')

        # Merges the content of all files in the current year range into a single file.
        output_file = f'{destination_path}/results_file_{start_year}_{year}.txt'
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for fname in filenames_in_range:
                with open(fname, 'r', encoding='utf-8') as f_in:
                    content = f_in.read().strip().replace('\n', ' ')
                    title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                    f_out.write(f"{title}|{content}\n")

if __name__ == '__main__':
    main()