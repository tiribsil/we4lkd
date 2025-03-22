from pathlib import Path
import os

if __name__ == '__main__':
    initial_year = 1900
    final_year = 2022
    source_path = './results/'
    destination_path = './results_aggregated/'
    filenames = [str(x) for x in Path(source_path).glob('**/*.txt')]

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    for y in range(initial_year, final_year+1):
        filtered_filenames = []
        for f in filenames:
            if f.split('\\')[-1].startswith(str(y)):
                filtered_filenames.append(f)

        print('number of papers in {}: {}\n'.format(y, len(filtered_filenames)))

        if len(filtered_filenames) == 0: continue

        abstract_list = []
        for fname in filtered_filenames:
            with open(fname, encoding='utf-8') as infile:
                for line in infile:
                    abstract_list.append(line)

        filename = destination_path + 'results_file_1900_{}.txt'.format(y)

        Path(filename).touch()
        with open(filename,'w+', encoding='utf-8') as f:
            for fname, abstract in zip(filtered_filenames, abstract_list):
                f.write(fname.split('\\')[2][5:-4] + '|') #em linux é /, windows \\
                f.write(abstract + '\n')