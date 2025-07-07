import os
from pathlib import Path
from src.target_disease import *

def main():
    normalized_target_disease = get_normalized_target_disease()

    source_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    destination_path = f'./data/{normalized_target_disease}/corpus/aggregated_abstracts'

    # Pega os nomes de todos os arquivos que vieram do crawler.
    filenames = sorted(list(map(str, Path(source_path).glob('*.txt'))))
    if not filenames:
        print('No files found in raw_abstracts directory.')
        exit(0)

    os.makedirs(destination_path, exist_ok=True)

    try:
        start_year = int(Path(filenames[0]).stem[:4])
        end_year = int(Path(filenames[-1]).stem[:4])
    except (ValueError, IndexError) as e:
        print(f"Error: Could not determine start/end year from filenames in {source_path}.")
        print("Filenames must start with 'YYYY_'. Example: '2023_my_article_title.txt'")
        print(f"Details: {e}")
        exit(1)

    # Para cada ano no intervalo...
    for year in range(start_year, end_year + 1):
        # Pega os nomes dos arquivos de artigos publicados ATÉ o ano atual (lógica cumulativa)
        filenames_in_range = [f for f in filenames if int(Path(f).stem[:4]) <= year]

        # Se não tiver nenhum nesse intervalo, vai para o próximo.
        if not filenames_in_range:
            continue

        print(f'Aggregating {len(filenames_in_range)} papers from {start_year} to {year}.')

        # Junta tudo em um só arquivo texto, que representa o conhecimento até aquele ano.
        # O nome do arquivo reflete o intervalo cumulativo.
        output_file = f'{destination_path}/results_file_{start_year}_{year}.txt'
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for fname in filenames_in_range:
                try:
                    with open(fname, 'r', encoding='utf-8') as f_in:
                        content = f_in.read().strip().replace('\n', ' ')
                        title = Path(fname).stem[5:].replace('_', ' ').capitalize()
                        f_out.write(f"{title}|{content}\n")

                except Exception as e:
                    print(f"  Warning: Could not process file {fname}. Error: {e}")


if __name__ == '__main__':
    main()