##################################################
## Collect abstracts from the PubMed search engine.
##################################################
## Author: Ti Ribeiro Silvério
## Copyright: Copyright 2025, [PAPER NAME]
## Email: tiribsil@gmail.com
##################################################

# IMPORTS:
import json, string, re, os, fnmatch
from Bio import Entrez
from pathlib import Path
import pubchempy as pcp

def list_from_txt(file_path):
    strings_list = []
    with open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            strings_list.append(line.rstrip('\n'))
    return strings_list

def search(query):
    final_query = '{} AND English[Language]'.format(query)  # especificando que os artigos devem ser apenas em inglês
    Entrez.email = 'tirs@estudante.ufscar.br'  # change here
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='999999',
                            retmode='xml',
                            term=final_query)
    results = Entrez.read(handle)
    handle.close()
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'tirs@estudante.ufscar.br'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

def flat(round_list):
    flat_list = []
    for element in round_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def extend_search():
    cids = [
        14888, 9444, 62770, 2907, 6253, 122640033, 5743, 90480031, 76970819, 636362, 71657455, 9829523, 51082, 5865,
        2723601, 249332, 49846579, 11422859
    ]

    sids = [
        404336834
    ]

    extended = []
    extended.append(pcp.Compound.from_cid(cids[0]).synonyms)

    for c in cids[1:]:
        extended.extend(pcp.Compound.from_cid(c).synonyms)

    for s in sids:
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

        except:
            pass

    remove_words = ['Antibiotic U18496', 'D,L-Cyclophosphamide', 'N,O-propylen-phosphorsaeure-ester-diamid',
                    'UNII-6UXW23996M component CMSMOCZEIVJLDB-CQSZACIVSA-N']
    for w in remove_words:
        try:
            extended.remove(w)

        except:
            pass

    return extended


# MAIN PROGRAM:
if __name__ == '__main__':
    destination_path = './results/'
    Path('./data/ids.txt').touch(exist_ok=True)
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    search_strings = list_from_txt('data/queries.txt')
    paper_counter = 0

    synonyms = extend_search()
    search_strings.extend(synonyms)

    ids = set()
    old_papers = list_from_txt('./data/ids.txt')
    if len(old_papers) > 0:
        ids = set(old_papers)

    for s in search_strings[100:]:
        s = s.encode('ascii', 'ignore').decode('ascii')
        print('searching for {}'.format(s))

        results = search(s)
        id_list = []
        id_list = results['IdList']
        id_list = [x for x in id_list if x not in ids]

        # se após a eliminação de IDs já coletados não sobrar nenhum ID em id_list, pula para o próximo termo de busca:
        papers_retrieved = len(id_list)
        if papers_retrieved == 0:
            print('No new papers found\n')
            continue

        print('{} papers found\n'.format(papers_retrieved))
        ids.update(id_list)

        papers = fetch_details(id_list)

        s = s.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
        Path(destination_path + '{}'.format(s)).mkdir(parents=True, exist_ok=True)

        for paper in papers['PubmedArticle']:
            article_title = ''
            article_title_filename = ''
            article_abstract = ''
            article_year = ''
            filename = ''
            path_name = ''

            try:
                article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                article_title_filename = article_title.lower().translate(
                    str.maketrans('', '', string.punctuation)).replace(' ', '_')

            except KeyError as e:
                if 'ArticleTitle' in e.args:
                    pass

            if article_title != '' and article_title != None:
                try:
                    article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']

                except KeyError as e:
                    if 'Abstract' in e.args:  # caso o artigo não tenha prefácio, continua o processamento, pois o título já foi extraído
                        article_abstract = ''
                        pass

                    elif 'Year' in e.args:
                        article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'][
                                           'MedlineDate'][0:4]

                if len(article_year) == 4:
                    filename = '{}_{}'.format(article_year, article_title_filename)

                    if len(filename) > 150:
                        filename = filename[0:146]

                    path_name = destination_path + s + '/{}.txt'.format(filename)
                    path_name = path_name.encode('ascii', 'ignore').decode('ascii')

                    # depois de pegar o título, resumo e data (e pulando o loop, quando não é possível), escrever o arquivo:
                    with open(path_name, "a", encoding='utf-8') as myfile:
                        myfile.write(article_title + ' ' + article_abstract)

                    paper_counter += 1

        with open('./data/ids.txt', 'a+', encoding='utf-8') as f:
            for id in id_list:
                f.write('\n' + str(id))

    print('Crawler finished with {} papers collected.'.format(len(old_papers) + paper_counter))