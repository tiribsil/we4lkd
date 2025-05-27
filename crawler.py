import string

from Bio import Entrez
from pathlib import Path
from google import genai
from api_key import MY_API_KEY
from target_disease import target_disease

def list_from_txt(file_path):
    strings_list = []
    try:
        with open(file_path, 'rt', encoding='utf-8') as list_file:
            for line in list_file: strings_list.append(line.rstrip('\n'))
    except FileNotFoundError: pass
    return strings_list

def search(paper_query):
    final_query = f'{paper_query} AND English[Language]'
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

def generate_query(disease):
    client = genai.Client(api_key=MY_API_KEY)

    #for m in client.models.list():
    #    for action in m.supported_actions:
    #        if action == "generateContent":
    #            print(m.name)

    prompt = f"""
    Generate a comprehensive and optimized PubMed search query to retrieve biomedical papers that will help a word embedding model learn meaningful relationships between {disease}, compounds, treatments, and related biological concepts.
        
    The query should:
    
    Include MeSH terms and common synonyms for {disease}
    Include known drugs, treatments, and drug classes related to {disease}
    Include relevant biological mechanisms, such as pathways, genes, proteins, and biomarkers
    Emphasize research discussing therapeutic, pharmacological, or mechanistic contexts (e.g., treatment, inhibition, activation, clinical trials, animal models)
    Use logical operators (AND, OR) and parentheses for clarity
    
    Output ONLY the final query in a format suitable for use directly in PubMed with no additional text or information.
    """
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    return response.text

if __name__ == '__main__':
    DESTINATION_DIR = './data/raw_results/'
    DOWNLOADED_PAPERS_IDS_FILE = './data/ids.txt'

    # Cria uma query com termos de busca relevantes.
    # target_disease = 'acute myeloid leukemia' # input('Enter a target disease: ')
    query = generate_query(target_disease)
    print(f'Query: {query}')
    paper_counter = 0

    # Cria uma lista com os IDs de todos os artigos já obtidos e um conjunto de IDs de artigos obtidos.
    old_papers = list_from_txt(DOWNLOADED_PAPERS_IDS_FILE)
    ids = set(old_papers)

    # Normaliza a query.
    query = query.encode('ascii', 'ignore').decode('ascii')
    print(f'Searching for papers...')

    # Procura pelo query, salva os IDs dos artigos encontrados em id_list.
    id_list = list(search(query)['IdList'])
    # Mantém só os que ainda não estão no conjunto de IDs conhecidos, ou seja, os novos.
    id_list = [x for x in id_list if x not in ids]

    # Se não achar nada novo, pode parar.
    if not id_list:
        print('No new papers found\n')
        exit(0)

    print(f'{len(id_list)} papers found\n')
    # Insere os novos IDs no conjunto.
    ids.update(id_list)

    # Pega os detalhes de cada artigo novo.
    papers = fetch_details(id_list)

    # Cria uma pasta com o nome da doença, formatado para poder ser nome de pasta.
    folder_name = target_disease.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
    Path(DESTINATION_DIR + '{}'.format(folder_name)).mkdir(parents=True, exist_ok=True)

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

        # O nome do arquivo será {ano}_{nome_do_artigo}...
        filename = '{}_{}'.format(article_year, article_title_filename)
        if len(filename) > 150: filename = filename[0:146]

        # e vai ser escrito naquela pasta criada antes do loop.
        path_name = DESTINATION_DIR + folder_name + '/{}.txt'.format(filename)
        path_name = path_name.encode('ascii', 'ignore').decode('ascii')

        # Escreve o arquivo.
        with open(path_name, "a", encoding='utf-8') as file:
            file.write(article_title + ' ' + article_abstract)

        paper_counter += 1

    # Coloca os IDs dos artigos novos no arquivo.
    with open(DOWNLOADED_PAPERS_IDS_FILE, 'a+', encoding='utf-8') as file:
        for new_id in id_list: file.write('\n' + str(new_id))

    print(f'Crawler finished with {len(old_papers) + paper_counter} papers collected.')