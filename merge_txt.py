import os, string
from pathlib import Path

from target_disease import target_disease

if __name__ == '__main__':
    folder_name = target_disease.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
    source_path = f'./data/raw_results/{folder_name}/'
    destination_path = f'./data/aggregated_results/{folder_name}/'

    # Pega os nomes de todos os arquivos que vieram do crawler.
    filenames = sorted(list(map(str, Path(source_path).glob('*.txt'))))
    if not filenames:
        print('No files found')
        exit(0)

    os.makedirs(destination_path, exist_ok=True)

    start_year = int(Path(filenames[0]).stem[:4])
    end_year = int(Path(filenames[-1]).stem[:4])

    # Para cada ano no intervalo...
    for year in range(start_year, end_year + 1):
        # Pega os nomes dos arquivos de artigos que são do ano atual.
        filenames_current_year = [f for f in filenames if Path(f).stem.startswith(str(year))]

        # Se não tiver nenhum desse ano, vai para o próximo.
        if not filenames_current_year:
            continue

        print(f'{len(filenames_current_year)} papers from {year}.\n')

        # Pega os prefácios de todos os artigos desse ano.
        abstracts = []
        for fname in filenames:
            with open(fname, encoding='utf-8') as infile:
                abstracts.extend(infile.readlines())

        # Junta tudo em um só arquivo texto.
        output_file = Path(destination_path) / f'results_file_{start_year}_{year}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for fname, abstract in zip(filenames, abstracts):
                f.write(f"{Path(fname).stem[5:]}|{abstract}")