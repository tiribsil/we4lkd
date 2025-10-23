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

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

class DataCollection:
    def __init__(self, disease_name: str, start_year: int, target_year: int, max_workers: int = 4, expand_synonyms: bool = False):
        load_dotenv()

        self.logger = self.setup_logger("data_collection", log_to_file=False)

        self.disease_name = self.normalize_disease_name(disease_name)
        self.start_year = start_year
        self.target_year = target_year
        self.retmax_papers = 9998
        self.email = os.getenv("ENTREZ_EMAIL")
        self.api_key = os.getenv("ENTREZ_KEY")
        self.max_workers = max_workers
        self.expand_synonyms = expand_synonyms

        Entrez.email = self.email
        Entrez.api_key = self.api_key

        #Paths
        self.raw_abstracts_path = Path(f'./data/{self.disease_name}/corpus/raw_abstracts')
        self.aggregated_path = Path(f'./data/{self.disease_name}/corpus/aggregated_abstracts')
        self.downloaded_paper_ids_path = Path(f'./data/{self.disease_name}/corpus/ids.txt')
        self.topics_file = Path(f'./data/{self.disease_name}/topics_of_interest.txt')
        self.titles_path = Path('data/pubchem_data/CID-Title/CID-Title')
        self.synonyms_path = Path('data/pubchem_data/CID-Synonym-filtered/CID-Synonym-filtered')
        self.corpus_file = Path(f'{self.aggregated_path}/aggregated_corpus.txt')
        
        self.raw_abstracts_path.mkdir(parents=True, exist_ok=True)
        self.aggregated_path.mkdir(parents=True, exist_ok=True)

        self.paper_counter = 0

    def normalize_disease_name(self, disease_name: str) -> str:
        return disease_name.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')

    def list_from_file(self, file_path: Path) -> list:
        if not file_path.exists():
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def generate_query(self, disease_name: str) -> str:
        
        if not self.topics_file.exists():
            self.topics_file.parent.mkdir(parents=True, exist_ok=True)
            self.topics_file.write_text(disease_name + '\n', encoding='utf-8')
        
        topics = self.list_from_file(self.topics_file)
        if len(topics) > 1 and self.expand_synonyms:
            topics = self.get_synonyms_for_terms(topics)

        self.logger.info(f'TOPICS: {topics}') 
        
        sub_queries = [f'("{topic}"[Title/Abstract] OR "{topic}"[MeSH Terms])' for topic in topics]
        return f'({" OR ".join(sub_queries)})'

    def get_synonyms_for_terms(self, terms: list) -> list:
        self.logger.info("Reading PubChem data to find synonyms...")
        title_to_cid, cid_to_synonyms = {}, {}

        try:
            with open(self.titles_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        cid, title = parts
                        title_to_cid[title.lower()] = cid
        except FileNotFoundError:
            self.logger.warning(f"Title file not found at {self.titles_path}")
            return terms

        try:
            with open(self.synonyms_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        cid, synonym = parts
                        cid_to_synonyms.setdefault(cid, []).append(synonym)
        except FileNotFoundError:
            self.logger.warning(f"Synonym file not found at {self.synonyms_path}")
            return terms

        expanded_terms = set(terms)
        for term in terms:
            cid = title_to_cid.get(term.lower())
            if cid:
                expanded_terms.update(cid_to_synonyms.get(cid, []))

        self.logger.info(f"Expanded {len(terms)} terms to {len(expanded_terms)} with synonyms.")
        return list(expanded_terms)

    """def search(self, query: str, retstart: int = 0):
        date_filter = f'AND (("{self.target_year}/01/01"[Date - Publication] : "{self.target_year}/12/31"[Date - Publication]))'
        final_query = f'({query} AND English[Language]) {date_filter}'
        
        try:
            handle = Entrez.esearch(db='pubmed', sort='relevance', retmax=self.retmax_papers,
                                    retstart=retstart, retmode='xml', term=final_query)
            found = Entrez.read(handle)
            handle.close()
            return found
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return {}"""

    def search(self, query: str, year: int, retstart: int = 0):
        """
        Faz a busca no PubMed apenas para o ano especificado.
        Usa o campo [DP] (Date of Publication) para evitar anos fora do intervalo.
        """
        # Filtro de data: apenas artigos publicados no ano exato
        date_filter = f'AND ("{year}"[DP])'
        final_query = f'({query} AND English[Language]) {date_filter}'
        
        try:
            handle = Entrez.esearch(
                db='pubmed',
                sort='relevance',
                retmax=self.retmax_papers,
                retstart=retstart,
                retmode='xml',
                term=final_query
            )
            found = Entrez.read(handle)
            handle.close()
            return found
        except Exception as e:
            self.logger.error(f"Error searching PubMed for year {year}: {e}")
            return {}

    def fetch_details_chunk(self, paper_ids_chunk: list):
        ids_string = ','.join(paper_ids_chunk)
        try:
            handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids_string)
            papers = Entrez.read(handle)
            handle.close()
            time.sleep(0.35)
            return papers
        except Exception as e:
            self.logger.error(f"Error fetching chunk: {e}")
            return None

    def fetch_details_parallel(self, paper_ids: list, chunk_size: int = 200):
        results = []
        chunks = [paper_ids[i:i + chunk_size] for i in range(0, len(paper_ids), chunk_size)]
        
        total_chunks = len(chunks)
        self.logger.info(f"Fetching {len(paper_ids)} papers in {total_chunks} chunks...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.fetch_details_chunk, chunk): idx 
                for idx, chunk in enumerate(chunks)
            }
            
            completed = 0
            for future in as_completed(future_to_chunk):
                completed += 1
                try:
                    papers = future.result()
                    if papers:
                        results.append(papers)
                    
                    if completed % 10 == 0 or completed == total_chunks:
                        self.logger.info(f"Progress: {completed}/{total_chunks} chunks fetched")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing chunk: {e}")

        return results

    def _get_affected_years(self, new_papers_count: int) -> set:
        """Identifica quais anos tiveram novos papers adicionados."""
        if new_papers_count == 0:
            return set()
        
        # Buscar apenas arquivos recém-adicionados (últimos N arquivos por modificação)
        all_files = sorted(
            self.raw_abstracts_path.glob('*.txt'),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        affected_years = set()
        for fname in all_files[:new_papers_count]:
            try:
                year_str = fname.stem[:4]
                if year_str.isdigit():
                    affected_years.add(int(year_str))
            except Exception:
                continue
        
        return affected_years

    def aggregate_abstracts_incremental(self, affected_years: set) -> bool:
        """Agrega apenas os anos afetados e atualiza corpus cumulativo."""
        if not affected_years:
            self.logger.info("No new papers to aggregate.")
            return True
        
        self.logger.info(f"Aggregating papers for years: {sorted(affected_years)}")
        
        # Processar apenas arquivos dos anos afetados
        yearly_papers = {year: [] for year in affected_years}
        new_corpus_entries = []
        
        for year in affected_years:
            year_files = sorted(self.raw_abstracts_path.glob(f'{year}_*.txt'))
            
            for fname in year_files:
                try:
                    stem = fname.stem
                    title = stem[5:].replace('_', ' ').capitalize()
                    content = fname.read_text(encoding='utf-8').strip().replace('\n', ' ')
                    
                    yearly_papers[year].append(f"{title}|{content}")
                    new_corpus_entries.append(f"{title}|{content}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {fname}: {e}")
                    continue
        
        # Atualizar arquivos anuais afetados
        for year in affected_years:
            papers = yearly_papers[year]
            if papers:
                self.logger.info(f'Aggregating {len(papers)} papers from year {year}.')
                yearly_file = Path(f'{self.aggregated_path}/results_file_{year}.txt')
                yearly_file.write_text('\n'.join(papers), encoding='utf-8')
        
        # Atualizar corpus cumulativo
        
        
        if not self.corpus_file.exists():
            # Primeira execução: criar corpus completo
            self.logger.info("Creating new cumulative corpus...")
            return self.aggregate_abstracts_full()
        else:
            # Append incremental
            if new_corpus_entries:
                self.logger.info(f'Appending {len(new_corpus_entries)} new papers to cumulative corpus.')
                with open(self.corpus_file, 'a', encoding='utf-8') as f:
                    f.write('\n' + '\n'.join(new_corpus_entries))
        
        return True

    def aggregate_abstracts_full(self) -> bool:
        """Agrega todos os abstracts (usado apenas na primeira execução)."""
        filenames = sorted(self.raw_abstracts_path.glob('*.txt'))
        if not filenames:
            self.logger.warning('No files found in raw_abstracts directory.')
            return False

        self.logger.info(f"Processing {len(filenames)} files for full aggregation...")
        
        # Processar todos os arquivos
        processed_files = []
        for fname in filenames:
            try:
                stem = fname.stem
                year_str = stem[:4]
                
                if not year_str.isdigit():
                    continue
                
                year = int(year_str)
                title = stem[5:].replace('_', ' ').capitalize()
                content = fname.read_text(encoding='utf-8').strip().replace('\n', ' ')
                processed_files.append((year, title, content))
                
            except Exception as e:
                self.logger.warning(f"Error processing {fname}: {e}")
                continue

        if not processed_files:
            self.logger.warning("No valid files processed.")
            return False

        # Agregar por ano
        yearly_papers = {}
        for year, title, content in processed_files:
            if self.start_year <= year <= self.target_year:
                yearly_papers.setdefault(year, []).append(f"{title}|{content}")

        # Salvar arquivos por ano
        for year in range(self.start_year, self.target_year + 1):
            papers = yearly_papers.get(year, [])
            if papers:
                self.logger.info(f'Aggregating {len(papers)} papers from year {year}.')
                yearly_file = Path(f'{self.aggregated_path}/results_file_{year}.txt')
                yearly_file.write_text('\n'.join(papers), encoding='utf-8')

        # Criar corpus completo
        full_corpus = [f"{title}|{content}" for _, title, content in processed_files]
        self.logger.info(f'Creating cumulative aggregation of all {len(full_corpus)} papers.')
        
        self.corpus_file.write_text('\n'.join(full_corpus), encoding='utf-8')
        self.logger.info(f'Full corpus saved to {self.corpus_file}')
        
        return True

    def aggregate_abstracts_by_year(self, start_year: int, end_year: int):
        """
        Método mantido para compatibilidade, mas agora usa agregação incremental.
        DEPRECATED: Use aggregate_abstracts_incremental() ou aggregate_abstracts_full().
        """
        # Detectar se há novos papers baseado no paper_counter
        if self.paper_counter > 0:
            affected_years = self._get_affected_years(self.paper_counter)
            return self.aggregate_abstracts_incremental(affected_years)
        else:
            if not self.corpus_file.exists():
                return self.aggregate_abstracts_full()
            return True

    def run(self):
        start_time = time.time()
        self.logger.info(f'Target disease: {self.disease_name}')
        self.logger.info(f'Year range: {self.start_year} to {self.target_year}')
        
        query = self.generate_query(self.disease_name).encode('ascii', 'ignore').decode('ascii')

        downloaded_paper_ids = set(self.list_from_file(self.downloaded_paper_ids_path))
        self.logger.info(f"Already have {len(downloaded_paper_ids)} papers downloaded.")
        
        all_new_paper_ids = set()
        
        # 🔁 Itera apenas sobre os anos solicitados
        for year in range(self.start_year, self.target_year + 1):
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Processing year: {year}")
            self.logger.info(f"{'='*50}")
            
            start_index = 0
            year_paper_ids = set()

            # Busca paginada por ano
            while True:
                self.logger.info(f"Searching papers from year {year}, index {start_index}...")
                search_results = self.search(query, year=year, retstart=start_index)
                batch_ids = set(search_results.get('IdList', []))
                
                if not batch_ids:
                    self.logger.info(f"No more results found for year {year}.")
                    break
                
                year_paper_ids.update(batch_ids)
                start_index += self.retmax_papers
                time.sleep(0.35)
            
            self.logger.info(f"Found {len(year_paper_ids)} total papers for year {year}")
            all_new_paper_ids.update(year_paper_ids)

        new_paper_id_list = list(all_new_paper_ids - downloaded_paper_ids)
        
        if not new_paper_id_list:
            self.logger.info("No new papers found.")
            if not self.corpus_file.exists():
                self.logger.info("Corpus file not found. Creating full aggregation...")
                self.aggregate_abstracts_full()
            return 0

        self.logger.info(f"{len(new_paper_id_list)} new papers found across all years.")

        papers_chunks = self.fetch_details_parallel(new_paper_id_list)

        for papers in papers_chunks:
            if not papers or 'PubmedArticle' not in papers:
                continue
                
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

                if not article_abstract or article_abstract == 'None':
                    continue

                try:
                    pub_date = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
                    article_year = pub_date.get('Year') or pub_date.get('MedlineDate', '')[:4]
                except KeyError:
                    continue

                if not article_year or len(article_year) != 4 or not article_year.isdigit():
                    continue

                article_year = int(article_year)

                # ✅ Filtro local de segurança (descarta anos fora do intervalo)
                if not (self.start_year <= article_year <= self.target_year):
                    continue

                filename = f"{article_year}_{article_title_filename[:146]}"
                filename = filename.encode('ascii', 'ignore').decode('ascii')
                path_name = Path(f'{self.raw_abstracts_path}/{filename}.txt')

                try:
                    path_name.write_text(article_title + ' ' + article_abstract, encoding='utf-8')
                    self.paper_counter += 1
                except Exception as e:
                    self.logger.error(f"Error saving paper: {e}")
                    continue

        # Salva IDs dos novos artigos
        if new_paper_id_list:
            with open(self.downloaded_paper_ids_path, 'a', encoding='utf-8') as file:
                file.write('\n'.join([str(id) for id in new_paper_id_list]) + '\n')
        
        total_papers = len(downloaded_paper_ids) + self.paper_counter
        self.logger.info(
            f"Crawler finished: {len(downloaded_paper_ids)} existing + "
            f"{self.paper_counter} new papers = {total_papers} total."
        )

        # Agregação incremental
        affected_years = self._get_affected_years(self.paper_counter)
        self.aggregate_abstracts_incremental(affected_years)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        self.logger.info(f"Total time spent in data collection: {minutes} min {seconds} sec")

        return self.paper_counter


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
    data_collection_module = DataCollection(
        disease_name="acute myeloid leukemia",
        start_year=2020,
        target_year=2020,
        expand_synonyms=True
    )
    data_collection_module.run()