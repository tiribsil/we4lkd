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
from utils import *

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

class DataCollection:
    def __init__(self, disease_name: str, target_year: int, max_workers: int = 4, 
                 expand_synonyms: bool = False, filter_synonyms: bool = True):
        load_dotenv()
        self.logger = LoggerFactory.setup_logger("data_collection", target_year=str(target_year), log_to_file=False)
        self.disease_name = self.normalize_disease_name(disease_name)
        self.target_year = target_year
        self.retmax_papers = 9998  # Máximo permitido pelo NCBI
        self.email = os.getenv("ENTREZ_EMAIL")
        self.api_key = os.getenv("ENTREZ_KEY")
        self.max_workers = max_workers
        self.expand_synonyms = expand_synonyms
        self.filter_synonyms = filter_synonyms
        
        Entrez.email = self.email
        Entrez.api_key = self.api_key
        
        # Paths
        self.raw_abstracts_path = Path(f'./data/{self.disease_name}/corpus/raw_abstracts')
        self.aggregated_path = Path(f'./data/{self.disease_name}/corpus/aggregated_abstracts')
        self.downloaded_paper_ids_path = Path(f'./data/{self.disease_name}/corpus/ids.txt')
        self.topics_file = Path(f'./data/{self.disease_name}/topics_of_interest.txt')
        self.titles_path = Path('data/pubchem_data/CID-Title')
        self.synonyms_path = Path('data/pubchem_data/CID-Synonym-filtered')
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

    def generate_query_for_topics(self, topics: list) -> str:
        self.logger.info(f'Generating query for {len(topics)} topics')
        sub_queries = [f'("{topic}"[Title/Abstract] OR "{topic}"[MeSH Terms])' for topic in topics]
        return f'({" OR ".join(sub_queries)})'

    def filter_relevant_synonyms(self, synonyms: list) -> list:
        """
        Filtra sinônimos para manter apenas os mais relevantes para pesquisa médica.
        """
        # Padrões a remover (muito genéricos ou não-médicos)
        exclude_patterns = [
            'element', 'isotope', 'lamp', 'sample', 'standard',
            'powder', 'lump', 'metal', 'specimen', 'catalog',
            'hollow cathode', 'polycrystalline', 'basis',
            'fertilizer', 'soil', 'water', 'lipstick', 'gloss'
        ]
        
        # Prefixos técnicos irrelevantes
        exclude_prefixes = [
            'AKOS', 'HMS', 'BRD-', 'CHEMBL', 'DTXSID', 'GTPL',
            'SCHEMBL', 'NCGC', 'MFCD', 'orb', 'STK', 'EN300',
            'BCPP', 'BP-', 'CCG-', 'SW', 'EX-A', 'DA-', 'TRA-',
            'ALBB-', 'MLS', 'SY', 'AC-', 'AB', 'NS', 'F0',
            'Z8', 'J1', 'K0', 'I1', 'PB', 'BL', 'FA', 'FL'
        ]
        
        filtered = []
        for synonym in synonyms:
            syn_lower = synonym.lower()
            
            # Remover sinônimos muito curtos (genéricos)
            if len(synonym) < 3:
                continue
            
            # Remover se contém padrões irrelevantes
            if any(pattern in syn_lower for pattern in exclude_patterns):
                continue
            
            # Remover se começa com prefixo técnico
            if any(synonym.startswith(prefix) for prefix in exclude_prefixes):
                continue
            
            # Remover se é apenas número ou código alfanumérico curto
            if synonym.replace('-', '').replace('_', '').isdigit():
                continue
            
            # Remover sinônimos com números muito longos (códigos de catálogo)
            if len([c for c in synonym if c.isdigit()]) > 8:
                continue
            
            filtered.append(synonym)
        
        return filtered

    def get_synonyms_for_terms(self, terms: list, filter_synonyms: bool = True) -> list:
        """
        Expande termos com sinônimos do PubChem.
        
        Args:
            terms: Lista de termos originais
            filter_synonyms: Se True, aplica filtros de relevância
        """
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
        raw_synonyms = []
        
        for term in terms:
            cid = title_to_cid.get(term.lower())
            if cid:
                raw_synonyms.extend(cid_to_synonyms.get(cid, []))
        
        if filter_synonyms and raw_synonyms:
            self.logger.info(f"Found {len(raw_synonyms)} raw synonyms, applying filters...")
            filtered_synonyms = self.filter_relevant_synonyms(raw_synonyms)
            expanded_terms.update(filtered_synonyms)
            self.logger.info(f"After filtering: {len(filtered_synonyms)} relevant synonyms kept")
        else:
            expanded_terms.update(raw_synonyms)
        
        self.logger.info(f"Final: {len(terms)} original terms → {len(expanded_terms)} total terms")
        return list(expanded_terms)

    def search_with_pagination(self, query: str, year: int) -> set:
        """
        Busca com paginação usando Web History para contornar limite de 9998.
        """
        date_filter = f'AND ("{year}"[DP])'
        final_query = f'({query} AND English[Language]) {date_filter}'
        
        all_ids = set()
        
        try:
            handle = Entrez.esearch(
                db='pubmed',
                sort='relevance',
                retmax=self.retmax_papers,
                retstart=0,
                retmode='xml',
                term=final_query,
                usehistory='y'
            )
            result = Entrez.read(handle)
            handle.close()
            
            total_count = int(result.get('Count', 0))
            webenv = result.get('WebEnv')
            query_key = result.get('QueryKey')
            
            self.logger.info(f"Found {total_count} total papers for year {year}")
            
            all_ids.update(result.get('IdList', []))
            
            if total_count > self.retmax_papers and webenv and query_key:
                self.logger.info(f"Paginating through {total_count} results...")
                
                for start in range(self.retmax_papers, total_count, self.retmax_papers):
                    remaining = total_count - start
                    fetch_size = min(self.retmax_papers, remaining)
                    
                    self.logger.info(f"Fetching results {start+1} to {start + fetch_size}...")
                    
                    try:
                        handle = Entrez.efetch(
                            db='pubmed',
                            rettype='uilist',
                            retmode='text',
                            retstart=start,
                            retmax=fetch_size,
                            webenv=webenv,
                            query_key=query_key
                        )
                        page_ids = handle.read().strip().split('\n')
                        handle.close()
                        
                        # Filtra IDs vazios
                        page_ids = [pid for pid in page_ids if pid]
                        all_ids.update(page_ids)
                        
                        self.logger.info(f"Retrieved {len(page_ids)} IDs (total: {len(all_ids)})")
                        time.sleep(0.35)
                        
                    except Exception as e:
                        self.logger.error(f"Pagination error at position {start}: {e}")
                        break
            
            time.sleep(0.35)
            return all_ids
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return set()

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

    def aggregate_abstracts(self) -> bool:
        """Agrega todos os abstracts baixados em arquivos por ano e corpus completo."""
        filenames = sorted(self.raw_abstracts_path.glob('*.txt'))
        
        if not filenames:
            self.logger.warning('No files found in raw_abstracts directory.')
            return False
        
        self.logger.info(f"Processing {len(filenames)} files for aggregation...")
        
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
            yearly_papers.setdefault(year, []).append(f"{title}|{content}")
        
        # Salvar arquivos por ano
        for year, papers in yearly_papers.items():
            self.logger.info(f'Aggregating {len(papers)} papers from year {year}')
            yearly_file = Path(f'{self.aggregated_path}/results_file_{year}.txt')
            yearly_file.write_text('\n'.join(papers), encoding='utf-8')
        
        # Criar corpus completo
        full_corpus = [f"{title}|{content}" for _, title, content in processed_files]
        self.logger.info(f'Creating cumulative corpus with {len(full_corpus)} papers')
        self.corpus_file.write_text('\n'.join(full_corpus), encoding='utf-8')
        self.logger.info(f'Corpus saved to {self.corpus_file}')
        
        return True

    def run(self):
        start_time = time.time()
        
        self.logger.info(f'Target disease: {self.disease_name}')
        self.logger.info(f'Target year: {self.target_year}')
        
        # 1. Carregar e expandir tópicos
        if not self.topics_file.exists():
            self.topics_file.parent.mkdir(parents=True, exist_ok=True)
            self.topics_file.write_text(self.disease_name + '\n', encoding='utf-8')
            
        topics = self.list_from_file(self.topics_file)
        if len(topics) > 1 and self.expand_synonyms:
            topics = self.get_synonyms_for_terms(topics, self.filter_synonyms)
        
        downloaded_paper_ids = set(self.list_from_file(self.downloaded_paper_ids_path))
        self.logger.info(f"Already have {len(downloaded_paper_ids)} papers downloaded")
        
        # 2. Dividir tópicos em chunks para evitar erro de excesso de operadores
        topic_chunk_size = 100  # Ajustável
        topic_chunks = [topics[i:i + topic_chunk_size] for i in range(0, len(topics), topic_chunk_size)]
        
        self.logger.info(f"Processing {len(topics)} topics in {len(topic_chunks)} chunks.")
        
        all_paper_ids = set()
        
        # 3. Iterar sobre os chunks e buscar
        for i, chunk in enumerate(topic_chunks):
            self.logger.info("="*60)
            self.logger.info(f"SEARCHING CHUNK {i+1}/{len(topic_chunks)} FOR YEAR {self.target_year}")
            self.logger.info("="*60)
            
            query = self.generate_query_for_topics(chunk).encode('ascii', 'ignore').decode('ascii')
            chunk_paper_ids = self.search_with_pagination(query, self.target_year)
            
            if chunk_paper_ids:
                self.logger.info(f"Chunk {i+1} found {len(chunk_paper_ids)} papers.")
                all_paper_ids.update(chunk_paper_ids)
            else:
                self.logger.warning(f"Chunk {i+1} found no papers.")
        
        if not all_paper_ids:
            self.logger.warning("No papers found in search across all chunks")
            return 0
        
        self.logger.info(f"Search completed: {len(all_paper_ids)} unique paper IDs collected across all chunks")
        
        new_paper_id_list = list(all_paper_ids - downloaded_paper_ids)
        
        if not new_paper_id_list:
            self.logger.info("No new papers to download")
            
            if not self.corpus_file.exists():
                self.logger.info("Creating corpus aggregation...")
                self.aggregate_abstracts()
            return 0
        
        self.logger.info(f"Downloading {len(new_paper_id_list)} new papers...")
        
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
                    self.logger.info(f"Discarding paper due to missing article title. Paper ID: {paper.get('MedlineCitation', {}).get('PMID')}")
                    continue
                
                abstract_texts = paper['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_texts, list):
                    article_abstract = ' '.join([str(a) for a in abstract_texts])
                else:
                    article_abstract = str(abstract_texts)
                

                
                try:
                    pub_date = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
                    article_year = pub_date.get('Year') or pub_date.get('MedlineDate', '')[:4]
                except KeyError:
                    self.logger.debug(f"Discarding paper due to missing publication date. Paper ID: {paper.get('MedlineCitation', {}).get('PMID')}")
                    continue
                
                if not article_year or len(article_year) != 4 or not article_year.isdigit():
                    self.logger.info(f"Discarding paper due to invalid article year ({article_year}). Paper ID: {paper.get('MedlineCitation', {}).get('PMID')}")
                    continue
                
                article_year = int(article_year)
                
                # Aceita apenas papers do target_year
                #if article_year != self.target_year: continue
                
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
            f"Download finished: {len(downloaded_paper_ids)} existing + "
            f"{self.paper_counter} new = {total_papers} total papers"
        )
        
        # Agregação
        self.aggregate_abstracts()
        
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        self.logger.info(f"Total time: {minutes} min {seconds} sec")
        
        return self.paper_counter


if __name__ == '__main__':
    # Exemplo 1: Com todos os sinônimos (padrão antigo)
    # data_collection_module = DataCollection(
    #     disease_name="acute myeloid leukemia",
    #     target_year=2020,
    #     expand_synonyms=True,
    #     filter_synonyms=False  # Usa TODOS os sinônimos
    # )
    
    # Exemplo 2: Com sinônimos filtrados (RECOMENDADO)
    data_collection_module = DataCollection(
        disease_name="acute myeloid leukemia",
        target_year=2020,
        expand_synonyms=True,
        filter_synonyms=True  # Filtra sinônimos irrelevantes
    )
    
    data_collection_module.run()
