import string, os

from Bio import Entrez
from pathlib import Path
import pubchempy as pcp

def list_from_txt(file_path):
    strings_list = []
    with open(file_path, 'rt', encoding='utf-8') as list_file:
        for line in list_file: strings_list.append(line.rstrip('\n'))
    return strings_list

def search(paper_query):
    final_query = '{} AND English[Language]'.format(paper_query)
    Entrez.email = 'tirs@estudante.ufscar.br'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='999999',
                            retmode='xml',
                            term=final_query)
    found = Entrez.read(handle)
    handle.close()
    return found

def fetch_details(paper_ids):
    ids_string = ','.join(paper_ids)
    Entrez.email = 'tirs@estudante.ufscar.br'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids_string)
    found = Entrez.read(handle)
    handle.close()
    return found

def flat(round_list):
    flat_list = []
    for element in round_list:
        if type(element) is list:
            for item in element: flat_list.append(item)
        else: flat_list.append(element)
    return flat_list

def extend_search():
    compound_ids = [
        14888, 9444, 62770, 2907, 6253, 122640033, 5743, 90480031, 76970819, 636362, 71657455, 9829523, 51082, 5865,
        2723601, 249332, 49846579, 11422859
    ]

    substance_ids = [
        404336834
    ]

    extended = [pcp.Compound.from_cid(compound_ids[0]).synonyms]

    for c in compound_ids[1:]:
        extended.extend(pcp.Compound.from_cid(c).synonyms)

    for s in substance_ids:
        extended.extend(pcp.Substance.from_sid(s).synonyms)

    extended = flat(extended)
    remove_words = []
    for s in extended:
        if '[as the base]' in s or '[Poison]' in s or '[ISO]' in s or '[INN]' in s or 'USP/JAN' in s or '(JAN' in s or '[JAN' in s or 'Latin' in s or 'Spanish' in s or '(TN)' in s or '(INN)' in s or 'USAN' in s or 'JP17/USP' in s or 'Czech' in s or 'German' in s or '[CAS]' in s:
            remove_words.append(s)

    extended = [x for x in extended if x not in remove_words]
    extended = [x for x in extended if len(x) >= 3]
    extended = list(dict.fromkeys(extended))

    useless_strings = list_from_txt('./data/bad_search_strings.txt')
    for w in useless_strings:
        try:
            extended.remove(w)
        except ValueError:
            pass

    remove_words = ['Antibiotic U18496', 'D,L-Cyclophosphamide', 'N,O-propylen-phosphorescence-ester-diamid',
                    'UNII-6UXW23996M component CMSMOCZEIVJLDB-CQSZACIVSA-N']
    for w in remove_words:
        try:
            extended.remove(w)
        except ValueError:
            pass

    return extended

if __name__ == '__main__':
    destination_path = './results/'
    Path('./data/ids.txt').touch(exist_ok=True)
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    # Cria uma lista com todos os termos de busca do arquivo. (não vai mais existir esse arquivo)
    queries = list_from_txt('./data/queries.txt')
    paper_counter = 0

    # Insere os sinônimos dos compostos no fim da lista.
    synonyms = extend_search()
    queries.extend(synonyms)

    # Cria uma lista com os IDs de todos os artigos já obtidos e um conjunto de IDs de artigos obtidos.
    old_papers = list_from_txt('./data/ids.txt')
    ids = set(old_papers)

    # Para cada termo de busca...
    for query in queries[100:]:
        # Normaliza o termo de busca.
        query = query.encode('ascii', 'ignore').decode('ascii')
        print('searching for {}'.format(query))

        # Procura pelo termo de busca, salva os IDs dos artigos encontrados em id_list.
        id_list = list(search(query)['IdList'])
        # Mantém só os que ainda não estão no conjunto de IDs conhecidos, ou seja, os novos.
        id_list = [x for x in id_list if x not in ids]

        # Se não achar nada novo, pode ir para o próximo termo de busca.
        if not id_list:
            print('No new papers found\n')
            continue

        print('{} papers found\n'.format(len(id_list)))
        # Insere os novos IDs no conjunto.
        ids.update(id_list)

        # Pega os detalhes de cada artigo novo.
        papers = fetch_details(id_list)

        # Cria uma pasta com o nome do termo de busca, formatado para poder ser nome de pasta.
        folder_name = query.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
        Path(destination_path + '{}'.format(folder_name)).mkdir(parents=True, exist_ok=True)

        # Para cada artigo novo...
        for paper in papers['PubmedArticle']:
            article_year = ''
            article_abstract = ''

            # Se não tiver título, vai para o próximo
            try:
                article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                article_title_filename = article_title.lower().translate(
                    str.maketrans('', '', string.punctuation)).replace(' ', '_')
            except KeyError as e: continue

            # Se não tiver prefácio, vai para o próximo.
            try:
                article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
            except KeyError as e:
                if 'Abstract' in e.args: continue
                if 'Year' in e.args:
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'][
                                       'MedlineDate'][0:4]

            # Se não tiver ano válido, vai para o próximo.
            if len(article_year) != 4: continue

            # O nome do arquivo vai ser "{ano}_{nome_do_artigo}"...
            filename = '{}_{}'.format(article_year, article_title_filename)
            if len(filename) > 150: filename = filename[0:146]

            # e vai ser escrito naquela pasta criada antes do loop.
            path_name = destination_path + folder_name + '/{}.txt'.format(filename)
            path_name = path_name.encode('ascii', 'ignore').decode('ascii')

            # Escreve o arquivo.
            with open(path_name, "a", encoding='utf-8') as file:
                file.write(article_title + ' ' + article_abstract)

            paper_counter += 1

        # Coloca os IDs dos artigos novos no arquivo.
        with open('./data/ids.txt', 'a+', encoding='utf-8') as file:
            for new_id in id_list: file.write('\n' + str(new_id))

    print('Crawler finished with {} papers collected.'.format(len(old_papers) + paper_counter))