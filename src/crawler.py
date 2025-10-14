import os
import time
from pathlib import Path
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

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




def search_year(topic, year, entrez_email, retmax_papers, base_delay=0.5, max_retries=5):
    """
    Performs a PubMed search for a given topic and year, handling pagination and retries with exponential backoff.
    """
    print(f"Searching for topic '{topic}' in year {year}...")
    query = f'("{topic}"[Title/Abstract] OR "{topic}"[MeSH Terms])'
    query = query.encode('ascii', 'ignore').decode('ascii')
    
    year_ids = set()
    start_index = 0
    current_delay = base_delay
    retries = 0

    while True:
        try:
            search_results = search(query, entrez_email, retmax_papers, year, year, retstart=start_index)
            batch_ids = list(search_results['IdList'])
            
            if not batch_ids:
                break
            
            year_ids.update(batch_ids)
            start_index += len(batch_ids)

            if start_index >= 10000:
                print(f"Reached the 10,000 record limit for topic '{topic}' in year {year}.")
                break
            
            time.sleep(base_delay) # Consistent delay between successful batches
            current_delay = base_delay # Reset delay on success

        except Exception as e:
            if "HTTP Error 429" in str(e) and retries < max_retries:
                retries += 1
                current_delay *= 2  # Exponential backoff
                print(f"HTTP Error 429: Too Many Requests for topic '{topic}' in year {year}. Retrying in {current_delay:.2f} seconds (attempt {retries}/{max_retries})...")
                time.sleep(current_delay)
            else:
                print(f"An unrecoverable error occurred during search for topic '{topic}' in year {year}: {e}")
                break
    return year_ids


def run_pubmed_crawler(target_disease: str, normalized_target_disease: str,
                       start_year: int, end_year: int,
                       entrez_email: str = 'tirs@estudante.ufscar.br',
                       retmax_papers: str = '9999', expand_synonyms: bool = False):
    """
    Fetches and saves PubMed papers related to a target disease for a specific year range.
    This function iterates through a list of topics, and for each topic, it searches PubMed
    year by year to retrieve a comprehensive set of papers, bypassing the 9,999 record limit
    for a single query.

    Args:
        target_disease (str): The name of the disease to search for.
        normalized_target_disease (str): The normalized name of the target disease.
        start_year (int): The start year of the search range.
        end_year (int): The end year of the search range.
        entrez_email (str): Email address for Entrez API.
        retmax_papers (str): Maximum number of papers to retrieve per API request.
    """
    set_target_disease(target_disease)
    print(f'Target disease: {target_disease}')

    raw_abstracts_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    downloaded_paper_ids_path = f'./data/{normalized_target_disease}/corpus/ids.txt'
    topics_file = f'./data/{normalized_target_disease}/topics_of_interest.txt'

    os.makedirs(raw_abstracts_path, exist_ok=True)

    # Create topics file if it doesn't exist
    if not os.path.exists(topics_file):
        with open(topics_file, 'w', encoding='utf-8') as f:
            f.write(target_disease + '\n')

    # Get topics and expand with synonyms
    topics = list_from_file(topics_file)
    if len(topics) > 1 and expand_synonyms:
        topics = get_synonyms_for_terms(topics)
    
    # Load IDs of already downloaded papers
    old_papers = list_from_file(downloaded_paper_ids_path)
    downloaded_paper_ids = set(old_papers)

    all_new_paper_ids = set()

    # Use a ThreadPoolExecutor to parallelize the searches by year
    for topic in topics:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_year = {executor.submit(search_year, topic, year, entrez_email, retmax_papers): year for year in range(start_year, end_year + 1)}
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_ids = future.result()
                    all_new_paper_ids.update(year_ids)
                except Exception as exc:
                    print(f'{year} generated an exception: {exc}')


    # Filter out already downloaded papers
    new_paper_id_list = list(all_new_paper_ids - downloaded_paper_ids)

    if not new_paper_id_list:
        print('No new papers found\n')
        return 0

    print(f'\nFound {len(new_paper_id_list)} new papers in total.\n')
    paper_counter = 0

    # Fetch details in batches to avoid overly long query strings
    batch_size = 200
    for i in range(0, len(new_paper_id_list), batch_size):
        batch_ids = new_paper_id_list[i:i+batch_size]
        print(f"Fetching details for batch {i//batch_size + 1}...")
        try:
            papers = fetch_details(batch_ids, entrez_email)
        except Exception as e:
            print(f"Error fetching details for batch. Skipping. Error: {e}")
            continue

        for paper in papers['PubmedArticle']:
            article_year = ''
            article_abstract = ''

            try:
                article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                article_title_filename = article_title.lower().translate(
                    str.maketrans('', '', string.punctuation)).replace(' ', '_')
            except KeyError:
                continue

            try:
                article_abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
                article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
            except KeyError as e:
                if 'Abstract' in e.args:
                    continue
                try:
                    # Try to get year from MedlineDate if PubDate is incomplete
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['MedlineDate'].split(' ')[0]
                except KeyError:
                    continue # Skip if no year is found

            if not article_year or not article_year.isdigit() or len(article_year) != 4:
                continue

            filename = f'{article_year}_{article_title_filename}'
            if len(filename) > 150:
                filename = filename[:146]

            path_name = f'{raw_abstracts_path}/{filename}.txt'
            path_name = path_name.encode('ascii', 'ignore').decode('ascii')

            with open(path_name, "w", encoding='utf-8') as file:
                file.write(article_title + ' ' + article_abstract)

            paper_counter += 1
        
        time.sleep(0.5)


    # Update the list of downloaded papers with the new ones.
    with open(downloaded_paper_ids_path, 'a+', encoding='utf-8') as file:
        for new_id in new_paper_id_list:
            file.write(str(new_id) + '\n')

    print(f'Crawler finished. {paper_counter} new papers collected.')
    print(f'Total papers in corpus: {len(downloaded_paper_ids) + paper_counter}')

    return paper_counter

if __name__ == '__main__':
    run_pubmed_crawler(get_target_disease(), get_normalized_target_disease())
