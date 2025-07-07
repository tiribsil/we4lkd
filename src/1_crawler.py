import os
from time import sleep

from Bio import Entrez
from pathlib import Path
from google import genai

from api_key import MY_API_KEY
from src.target_disease import *
from lark import Lark, LarkError

# This makes sure the script runs from the root of the project, so relative paths work correctly.
os.chdir(Path(__file__).resolve().parent.parent)

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

def search(query):
    """
    Searches for papers in PubMed using the provided query.
    Args:
        query: The search query for PubMed.

    Returns:
        found: A dictionary containing the search results.
    """
    final_query = f'{query} AND English[Language]'
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
    """
    Fetches detailed information about papers from PubMed using their IDs.
    Args:
        paper_ids: A list of PubMed paper IDs.

    Returns:
        found: A dictionary containing detailed information about the papers.
    """
    ids_string = ','.join(paper_ids)
    Entrez.email = 'tirs@estudante.ufscar.br'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids_string)
    found = Entrez.read(handle)
    handle.close()
    return found

def follows_grammar(query: str) -> bool:
    """
    Checks if the provided PubMed query string follows the defined grammar rules.
    Args:
        query: The PubMed query string to be validated.

    Returns:
        bool: True if the query follows the grammar, False otherwise.
    """
    pubmed_grammar = r"""
        ?query: or_expr

        or_expr: and_expr (_OR and_expr)*

        and_expr: atom+

        atom: main_body (_LBRACKET tag_body _RBRACKET)?
            | _LPAREN or_expr _RPAREN

        main_body: QUOTED_STRING | unquoted_main_body

        unquoted_main_body: IDENTIFIER _ASTERISK?

        tag_body: IDENTIFIER+

        QUOTED_STRING: /"[^"]+"/

        _AND: "AND"
        _OR: "OR"
        _LPAREN: "("
        _RPAREN: ")"
        _LBRACKET: "["
        _RBRACKET: "]"

        _ASTERISK: "*"

        IDENTIFIER: /[a-zA-Z0-9_.:\/\-]+/

        %import common.WS
        %ignore WS
    """

    try:
        pubmed_parser = Lark(pubmed_grammar, start='query', parser='lalr')
    except Exception as e:
        print(f"Erro ao criar o parser (gramática inválida?): {e}")
        pubmed_parser = None

    if not pubmed_parser:
        print("Parser não foi inicializado corretamente.")
        return False
    if not query.strip():
        return False
    try:
        pubmed_parser.parse(query)
        return True
    except LarkError as e:
        print(query)
        print(f"--> Detalhe do erro do Lark: {e}")
        return False

def generate_query(disease):
    """
    Generates a PubMed search query for a given disease using Google GenAI. TODO: change to a local model.
    Args:
        disease: The name of the disease for which to generate the query.

    Returns:
        response.text: The generated PubMed search query.
    """
    client = genai.Client(api_key=MY_API_KEY)

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
    if not follows_grammar(response.text):
        prompt = f"""
        The following text is a PubMed query that contains syntax errors:
        "{response.text}"
        
        Correct this query so it adheres to PubMed query grammar, while preserving the original search intent as closely as possible.
        
        Your entire response should be *only* the corrected PubMed query string. Do not include any explanations, apologies, or any text other than the corrected query itself.
        """
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    if not follows_grammar(response.text):
        print('Query does not follow grammar rules. Exiting.')
        exit(1)
    return response.text

def canonize_disease_name(disease):
    """
    Generates the canonical name of a disease using Google GenAI. TODO: change to a local solution.
    Args:
        disease: The name of the disease to be canonized.

    Returns:
        response.text: The canonical name of the disease in lower case and without punctuation.
    """
    client = genai.Client(api_key=MY_API_KEY)

    prompt = f"""
    What is the canonical, most used and correct name of the disease "{disease}"?

    Output ONLY the name of the disease, without any additional text or information, in lower case and no punctuation.
    """
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)

    return response.text

def grammar_testing(iterations, target_disease):
    """
    Testing function. It tries to force errors in the grammar checking function by generating queries repeatedly.
    Args:
        iterations: Number of iterations to run the test.
        target_disease: Name of the target disease for which to generate queries.

    Returns:
        Nothing.
    """
    for i in range(iterations):
        query = generate_query(target_disease)
        print(query)
        sleep(30)

def main():
    # This is for the final version of the program. For now, we will use a hardcoded target disease as it's easier to test.
    target_disease = get_target_disease() # input('Enter a target disease: ')
    normalized_target_disease = get_normalized_target_disease()

    target_disease = canonize_disease_name(target_disease)
    set_target_disease(target_disease)
    print(f'Target disease: {target_disease}')

    raw_abstracts_path = f'./data/{normalized_target_disease}/corpus/raw_abstracts'
    downloaded_paper_ids_path = f'./data/{normalized_target_disease}/corpus/ids.txt'

    os.makedirs(raw_abstracts_path, exist_ok=True)

    # Generates a query to find relevant papers about the target disease.
    # There is a grammar checking function that will ensure the query is valid.
    query = generate_query(target_disease)
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
    new_paper_id_list = list(search(query)['IdList'])
    # Keeps only the IDs that are not already in the set of old papers.
    new_paper_id_list = [x for x in new_paper_id_list if x not in downloaded_paper_ids]

    # If nothing new was found, exits the program. All papers matching the query were already downloaded.
    if not new_paper_id_list:
        print('No new papers found\n')
        return

    print(f'{len(new_paper_id_list)} papers found\n')

    # Fetches the details of the new papers.
    papers = fetch_details(new_paper_id_list)

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
        for new_id in new_paper_id_list: file.write('\n' + str(new_id))

    print(f'Crawler finished with {len(old_papers) + paper_counter} papers collected.')

if __name__ == '__main__':
    main()