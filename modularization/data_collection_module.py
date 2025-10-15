import os
import string
import time
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataCollection:
    def __init__(self, disease_name: str, start_year: int, max_workers: int = 4):
        load_dotenv()

        self.logger = self.setup_logger("data_collection", log_to_file=False)

        self.disease_name = self.normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = datetime.now().year
        self.retmax_papers = 9998
        self.email = os.getenv("ENTREZ_EMAIL")
        self.api_key = os.getenv("ENTREZ_KEY")
        self.max_workers = max_workers


        Entrez.email = self.email
        Entrez.api_key = self.api_key

        self.raw_abstracts_path = os.path.join('.', 'data', self.disease_name, 'corpus', 'raw_abstracts')
        self.downloaded_paper_ids_path = os.path.join('.', 'data', self.disease_name, 'corpus', 'ids.txt')
        os.makedirs(self.raw_abstracts_path, exist_ok=True)

        self.paper_counter = 0

    def normalize_disease_name(self, disease_name: str) -> str:
        return disease_name.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')

    def list_from_file(self, file_path: str) -> list:
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def generate_query(self, disease_name: str) -> str:
        topics_file = os.path.join('.', 'data', disease_name, 'topics_of_interest.txt')
        if not os.path.exists(topics_file):
            with open(topics_file, 'w', encoding='utf-8') as f:
                f.write(disease_name + '\n')
        topics = self.list_from_file(topics_file)
        if len(topics) > 1:
            topics = self.get_synonyms_for_terms(topics)
        sub_queries = [f'("{topic}"[Title/Abstract] OR "{topic}"[MeSH Terms])' for topic in topics]
        return f'({" OR ".join(sub_queries)})'

    def get_synonyms_for_terms(self, terms: list) -> list:
        self.logger.info("Reading PubChem data to find synonyms...")
        titles_path = 'data/pubchem_data/CID-Title'
        synonyms_path = 'data/pubchem_data/CID-Synonym-filtered'

        title_to_cid, cid_to_synonyms = {}, {}

        try:
            with open(titles_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        cid, title = parts
                        title_to_cid[title.lower()] = cid
        except FileNotFoundError:
            self.logger.warning(f"Title file not found at {titles_path}")
            return terms

        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        cid, synonym = parts
                        cid_to_synonyms.setdefault(cid, []).append(synonym)
        except FileNotFoundError:
            self.logger.warning(f"Synonym file not found at {synonyms_path}")
            return terms

        expanded_terms = set(terms)
        for term in terms:
            cid = title_to_cid.get(term.lower())
            if cid:
                expanded_terms.update(cid_to_synonyms.get(cid, []))

        self.logger.info(f"Expanded {len(terms)} terms to {len(expanded_terms)} with synonyms.")
        return list(expanded_terms)

    def search(self, query: str, retstart: int = 0):
        date_filter = f'AND (("{self.start_year}/01/01"[Date - Publication] : "{self.end_year}/12/31"[Date - Publication]))'
        final_query = f'({query} AND English[Language]) {date_filter}'
        handle = Entrez.esearch(db='pubmed', sort='relevance', retmax=self.retmax_papers,
                                retstart=retstart, retmode='xml', term=final_query)
        found = Entrez.read(handle)
        handle.close()
        return found

    def fetch_details_chunk(self, paper_ids_chunk: list):
        ids_string = ','.join(paper_ids_chunk)
        handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids_string)
        papers = Entrez.read(handle)
        handle.close()
        time.sleep(0.35)  # protege contra limite de requisições
        return papers

    def fetch_details_parallel(self, paper_ids: list, chunk_size: int = 200):
        results = []
        chunks = [paper_ids[i:i + chunk_size] for i in range(0, len(paper_ids), chunk_size)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(self.fetch_details_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(future_to_chunk):
                try:
                    papers = future.result()
                    results.append(papers)
                except Exception as e:
                    self.logger.warning(f"Error fetching chunk: {e}")

        return results
    

    def aggregate_abstracts_by_year(self, start_year: int, end_year: int):
        """
        Aggregates raw abstract text files into yearly files and a single cumulative corpus file.
        For each year from start_year to end_year, it creates a file with all abstracts from that year.
        """
        source_path = Path(f'./data/{self.disease_name}/corpus/raw_abstracts')
        destination_path = Path(f'./data/{self.disease_name}/corpus/aggregated_abstracts')
        destination_path.mkdir(parents=True, exist_ok=True)

        # Get all raw abstracts
        filenames = sorted(source_path.glob('*.txt'))
        if not filenames:
            self.logger.info('No files found in raw_abstracts directory. Have you run the crawler first?')
            return False

        # Preprocess files once: read content and extract year + title
        processed_files = []
        for fname in filenames:
            stem = fname.stem
            year = int(stem[:4])
            title = stem[5:].replace('_', ' ').capitalize()
            content = fname.read_text(encoding='utf-8').strip().replace('\n', ' ')
            processed_files.append((year, title, content))

        # Aggregate per year
        for year in range(start_year, end_year + 1):
            yearly_content = [f"{title}|{content}" for y, title, content in processed_files if y == year]
            self.logger.info(f'Aggregating {len(yearly_content)} papers from the year {year}.')
            if yearly_content:
                (destination_path / f'results_file_{year}.txt').write_text('\n'.join(yearly_content), encoding='utf-8')

        # Aggregate full corpus
        full_corpus_content = [f"{title}|{content}" for _, title, content in processed_files]
        self.logger.info(f'\nCreating a cumulative aggregation of all {len(full_corpus_content)} papers.')
        full_corpus_file = destination_path / 'aggregated_corpus.txt'
        full_corpus_file.write_text('\n'.join(full_corpus_content), encoding='utf-8')
        self.logger.info(f'Full corpus saved to {full_corpus_file}')

        return True

    def run(self):
        start_time = time.time()
        self.logger.info(f'Target disease: {self.disease_name}')
        query = self.generate_query(self.disease_name).encode('ascii', 'ignore').decode('ascii')

        downloaded_paper_ids = set(self.list_from_file(self.downloaded_paper_ids_path))
        self.logger.info("Searching for papers...")

        all_new_paper_ids = set()
        start_index, batch_size = 0, self.retmax_papers

        while True:
            self.logger.info(f"Searching papers from index {start_index}...")
            search_results = self.search(query, retstart=start_index)
            batch_ids = set(search_results.get('IdList', []))
            if not batch_ids:
                self.logger.info("No new results found. Ending search.")
                break
            all_new_paper_ids.update(batch_ids)
            start_index += batch_size
            time.sleep(0.35)

        new_paper_id_list = list(all_new_paper_ids - downloaded_paper_ids)
        if not new_paper_id_list:
            self.logger.info("No new papers found.")
            return 0

        self.logger.info(f"{len(new_paper_id_list)} new papers found.")

        papers_chunks = self.fetch_details_parallel(new_paper_id_list)

        for papers in papers_chunks:
            for paper in papers['PubmedArticle']:
                try:
                    article_title = paper['MedlineCitation']['Article']['ArticleTitle']
                    article_title_filename = article_title.lower().translate(
                        str.maketrans('', '', string.punctuation)).replace(' ', '_')
                except KeyError:
                    continue

                abstract_texts = paper['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_texts, list):
                    article_abstract = ' '.join([str(a) for a in abstract_texts])
                else:
                    article_abstract = str(abstract_texts)

                try:
                    article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'].get('Year')
                    if not article_year:
                        article_year = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate'].get('MedlineDate', '')[:4]
                except KeyError:
                    continue

                if len(article_year) != 4:
                    continue

                filename = f"{article_year}_{article_title_filename[:146]}"
                filename = filename.encode('ascii', 'ignore').decode('ascii')
                path_name = os.path.join(self.raw_abstracts_path, f"{filename}.txt")

                with open(path_name, "w", encoding='utf-8') as file:
                    file.write(article_title + ' ' + article_abstract)

                self.paper_counter += 1

        with open(self.downloaded_paper_ids_path, 'a+', encoding='utf-8') as file:
            for new_id in new_paper_id_list:
                file.write(str(new_id) + '\n')
        
        total_papers = len(downloaded_paper_ids) + self.paper_counter
        self.logger.info(f"Crawler finished: {len(downloaded_paper_ids)} existing + {self.paper_counter} new papers = {total_papers} total.")

        self.aggregate_abstracts_by_year(self.start_year, self.end_year)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        self.logger.info(f"Total time spent in data collection: {minutes} min {seconds} sec")

        return

    @staticmethod
    def setup_logger(name: str = "logger", log_level: int = logging.INFO,
                     log_to_file: bool = False, log_file: str = "app.log",
                     max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3) -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_to_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger


if __name__ == '__main__':
    data_collection_module = DataCollection(disease_name="acute myeloid leukemia", start_year=2024)
    data_collection_module.run()
