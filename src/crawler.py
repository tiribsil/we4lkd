import os
import time
from pathlib import Path

# This makes sure the script runs from the root of the project, so relative paths work correctly.
os.chdir(Path(__file__).resolve().parent.parent)

from Bio import Entrez

from src.utils import *

def list_from_file(file_path):
    """
    Reads a text file and returns a list of strings, each string being a line from the file.
    Args:
        file_path: Path to the text file.

    Returns:
        strings_list: List of strings from the file.
    """
    strings_list = []
    try:
        with open(file_path, 'rt', encoding='utf-8') as list_file:
            for line in list_file: strings_list.append(line.rstrip('\n'))
    except FileNotFoundError: pass
    return strings_list

def search(query: str, entrez_email: str, retmax_papers: str, start_year: int, end_year: int, retstart: int = 0):
    """
    Searches for papers in PubMed using the provided query within a date range.
    Args:
        query (str): The search query for PubMed.
        entrez_email (str): Email address for Entrez API.
        retmax_papers (str): Maximum number of papers to retrieve from PubMed.
        start_year (int): The starting year for the publication date range.
        end_year (int): The ending year for the publication date range.
        retstart (int): The starting index of the results to retrieve.

    Returns:
        found: A dictionary containing the search results.
    """
    date_filter = f"AND ((\"{start_year}/01/01\"[Date - Publication] : \"{end_year}/12/31\"[Date - Publication]))"
    final_query = f'({query} AND English[Language]) {date_filter}'

    Entrez.email = entrez_email
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=retmax_papers,
                            retstart=retstart,
                            retmode='xml',
                            term=final_query)
    found = Entrez.read(handle)
    handle.close()
    return found

def fetch_details(paper_ids: list, entrez_email: str):
    """
    Fetches detailed information about papers from PubMed using their IDs.
    Args:
        paper_ids (list): A list of PubMed paper IDs.
        entrez_email (str): Email address for Entrez API.

    Returns:
        found: A dictionary containing detailed information about the papers.
    """
    ids_string = ','.join(paper_ids)
    Entrez.email = entrez_email
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids_string)
    found = Entrez.read(handle)
    handle.close()
    return found

def get_synonyms_for_terms(terms: list):
    """
    Expands a list of terms to include their synonyms from PubChem data.
    Args:
        terms (list): A list of terms to find synonyms for.

    Returns:
        list: A list containing the original terms and all their found synonyms.
    """
    print("Reading PubChem data to find synonyms...")
    # Paths to the PubChem data files
    titles_path = 'data/pubchem_data/CID-Title'
    synonyms_path = 'data/pubchem_data/CID-Synonym-filtered'

    # 1. Read CID-Title and create a mapping from lowercase title to CID
    title_to_cid = {}
    try:
        with open(titles_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('	')
                if len(parts) == 2:
                    cid, title = parts
                    title_to_cid[title.lower()] = cid
    except FileNotFoundError:
        print(f"Warning: Title file not found at {titles_path}")
        return terms  # Return original terms if data is missing

    # 2. Read CID-Synonym-filtered and create a mapping from CID to synonyms
    cid_to_synonyms = {}
    try:
        with open(synonyms_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('	')
                if len(parts) == 2:
                    cid, synonym = parts
                    if cid not in cid_to_synonyms:
                        cid_to_synonyms[cid] = []
                    cid_to_synonyms[cid].append(synonym)
    except FileNotFoundError:
        print(f"Warning: Synonym file not found at {synonyms_path}")
        return terms  # Return original terms if data is missing

    # 3. Expand the initial list of terms with synonyms
    expanded_terms = set(terms)
    for term in terms:
        # The terms in topics_of_interest.txt are expected to be lowercase canonical names
        cid = title_to_cid.get(term.lower())
        if cid:
            synonyms = cid_to_synonyms.get(cid, [])
            for s in synonyms:
                expanded_terms.add(s)

    print(f"Expanded {len(terms)} terms to {len(expanded_terms)} with synonyms.")
    return list(expanded_terms)


def generate_query(disease, normalized_target_disease):
    """
    Generates a generalized PubMed search query from a file of topics.
    If the file doesn't exist, it creates it with the target disease.
    For each topic, it searches in Title/Abstract and MeSH terms.
    Args:
        disease: The name of the disease.

    Returns:
        The generated PubMed search query.
    """

    # TODO: fazer usar todos os sinônimos de cada termo em vez de simplesmente o termo

    topics_file = f'./data/{normalized_target_disease}/topics_of_interest.txt'
    if not os.path.exists(topics_file):
        with open(topics_file, 'w', encoding='utf-8') as f:
            f.write(disease + '\n')
    
    topics = list_from_file(topics_file)
    if len(topics) > 1: topics = get_synonyms_for_terms(topics)
    sub_queries = [f'("{topic}"[Title/Abstract] OR "{topic}"[MeSH Terms])' for topic in topics]
    query = " OR ".join(sub_queries)
    return f'({query})'

def run_pubmed_crawler(target_disease: str, normalized_target_disease: str,
                       start_year: int, end_year: int,
                       entrez_email: str = 'tirs@estudante.ufscar.br',
                       retmax_papers: str = '9998'):
    """
    Fetches and saves PubMed papers related to a target disease for a specific year range.

    Args:
        target_disease (str): The name of the disease to search for.
        normalized_target_disease (str): The normalized name of the target disease.
        start_year (int): The start year of the search range.
        end_year (int): The end year of the search range.
        entrez_email (str): Email address for Entrez API.
        retmax_papers (str): Maximum number of papers to retrieve from PubMed.
    """
    # This is for the final version of the program. For now, we will use a hardcoded target disease as it's easier to test.
    # target_disease = input('Enter a target disease: ')
    # normalized_target_disease = get_normalized_target_disease()

    set_target_disease(target_disease)
    print(f'Target disease: {target_disease}')

    raw_abstracts_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    downloaded_paper_ids_path = f'./data/{normalized_target_disease}/corpus/ids.txt'

    os.makedirs(raw_abstracts_path, exist_ok=True)

    # Generates a query to find relevant papers about the target disease.
    query = generate_query(target_disease, normalized_target_disease)
    print(f'Query: {query}')
    paper_counter = 0

    # From now on, everything will be tied to the target disease, being stored in a folder named after it inside the ./data folder.

    # Creates a list of IDs of already downloaded papers so we don't download them again.
    old_papers = list_from_file(downloaded_paper_ids_path)
    downloaded_paper_ids = set(old_papers)

    # Normalizes query to avoid issues with special characters.
    query = query.encode('ascii', 'ignore').decode('ascii')
    print(f'Searching for papers...')

    # Looks for papers matching the query and stores their IDs in a list.
    all_new_paper_ids = []
    start_index = 0
    while True:
        search_results = search(query, entrez_email, retmax_papers, start_year, end_year, retstart=start_index)
        batch_ids = list(search_results['IdList'])
        if not batch_ids:
            break  # No more papers found
        all_new_paper_ids.extend(batch_ids)
        start_index += len(batch_ids)
        time.sleep(1)  # Be respectful to the API

    new_paper_id_list = list(set(all_new_paper_ids))
    # Keeps only the IDs that are not already in the set of old papers.
    new_paper_id_list = [x for x in new_paper_id_list if x not in downloaded_paper_ids]

    # If nothing new was found, exits the program. All papers matching the query were already downloaded.
    if not new_paper_id_list:
        print('No new papers found\n')
        return 0

    print(f'{len(new_paper_id_list)} papers found\n')

    # Fetches the details of the new papers.
    papers = fetch_details(new_paper_id_list, entrez_email)

    # For each paper found...
    for paper in papers['PubmedArticle']:
        article_year = ''
        article_abstract = ''

        # If the paper doesn't have basic information, goes to the next one.
        try:
            article_title = paper['MedlineCitation']['Article']['ArticleTitle']
            article_title_filename = article_title.lower().translate(
                str.maketrans('', '', string.punctuation)).replace(' ', '_')
        except KeyError: continue

        # If the paper has no abstract, skips it.
        try:
            article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
            article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
        except KeyError as e:
            if 'Abstract' in e.args: continue
            if 'Year' in e.args:
                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'][
                                   'MedlineDate'][0:4]

        # If the paper doesn't have a valid year, skips it.
        if len(article_year) != 4: continue

        # The filename will be the year and the title of the article.
        # This is the file in which the raw abstract will be saved.
        filename = f'{article_year}_{article_title_filename}'
        if len(filename) > 150: filename = filename[0:146]

        # Sets up the destination directory.
        path_name = f'{raw_abstracts_path}/{filename}.txt'
        path_name = path_name.encode('ascii', 'ignore').decode('ascii')

        # Writes the file with the paper title and abstract.
        with open(path_name, "a", encoding='utf-8') as file:
            file.write(article_title + ' ' + article_abstract)

        paper_counter += 1

    # Updates the list of downloaded papers with the new ones.
    with open(downloaded_paper_ids_path, 'a+', encoding='utf-8') as file:
        for new_id in new_paper_id_list: file.write(str(new_id) + '\n')

    print(f'Crawler finished with {len(old_papers) + paper_counter} papers collected.')

    return paper_counter + len(old_papers)

if __name__ == '__main__':
    run_pubmed_crawler(get_target_disease(), get_normalized_target_disease())
