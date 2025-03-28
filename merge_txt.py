import os
from pathlib import Path

if __name__ == '__main__':
    start_year = 1900
    end_year = 2025
    source_path = './results/'
    destination_path = './results_aggregated/'

    # Pega os nomes de todos os arquivos que vieram do crawler.
    filenames = list(map(str, Path(source_path).glob('**/*.txt')))
    os.makedirs(destination_path, exist_ok=True)

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
        output_file = Path(destination_path) / f'results_file_1900_{year}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for fname, abstract in zip(filenames, abstracts):
                f.write(f"{Path(fname).stem[5:]}|{abstract}")