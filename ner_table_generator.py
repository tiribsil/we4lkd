import spacy
import pandas as pd
from pathlib import Path
from target_disease import target_disease, folder_name

FINAL_PROCESSING_YEAR = 2025

RELEVANT_SPACY_ENTITY_TYPES = ['CHEMICAL']
MAPPED_ENTITY_TYPE = 'pharmacologic_substance'

OUTPUT_NER_CSV_PATH = f'./data/{folder_name}/ner_table.csv'
INPUT_ABSTRACTS_PATH = f'./data/{folder_name}/aggregated_results'

def load_spacy_model():
    spacy.require_gpu()
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        print(f"Modelo spaCy 'en_ner_bc5cdr_md' carregado. Usando GPU: {spacy.prefer_gpu()}")
        return nlp
    except OSError:
        print("AVISO: Modelo 'en_ner_bc5cdr_md' não encontrado. Tentando 'en_core_web_sm'.")
        try:
            nlp = spacy.load("en_core_web_sm")
            print(f"Modelo spaCy 'en_core_web_sm' carregado. Usando GPU: {spacy.prefer_gpu()}")
            print("AVISO: RELEVANT_SPACY_ENTITY_TYPES pode precisar de ajuste para 'en_core_web_sm'.")
            return nlp
        except OSError:
            print("ERRO CRÍTICO: Nenhum modelo spaCy funcional encontrado.")
            return None


def get_year_from_filename(filename_str):
    try:
        parts = filename_str.split('_')
        # Espera-se 'results_file_YYYY_YYYY.txt' ou 'results_file_YYYY.txt' (se for um único ano)
        # Vamos pegar o último ano mencionado no nome do arquivo para o filtro
        if len(parts) >= 4 and parts[-1].replace('.txt', '').isdigit():  # Formato results_file_START_END.txt
            return int(parts[-1].replace('.txt', ''))
        elif len(parts) >= 3 and parts[-1].replace('.txt', '').isdigit():  # Formato results_file_YEAR.txt
            return int(parts[-1].replace('.txt', ''))
        return None  # Formato de nome de arquivo não esperado
    except ValueError:
        return None


def process_abstracts_from_files(nlp_model):
    input_folder_path = Path(INPUT_ABSTRACTS_PATH)
    if not input_folder_path.exists() or not input_folder_path.is_dir():
        print(f"ERRO: Pasta de entrada não encontrada: {input_folder_path}")
        return []

    all_filenames = sorted([str(x) for x in input_folder_path.glob('*.txt')])

    filenames_to_process = []
    for f_path_str in all_filenames:
        file_year = get_year_from_filename(Path(f_path_str).name)
        if file_year is not None and file_year <= FINAL_PROCESSING_YEAR:
            filenames_to_process.append(f_path_str)
        elif file_year is None:
            print(f"AVISO: Não foi possível determinar o ano do arquivo {Path(f_path_str).name}. Incluindo por padrão.")
            filenames_to_process.append(f_path_str)  # Inclui se não puder determinar o ano

    if not filenames_to_process:
        print(f"AVISO: Nenhum arquivo .txt encontrado em {input_folder_path} até o ano {FINAL_PROCESSING_YEAR}.")
        return []

    print(
        f"Processando {len(filenames_to_process)} arquivos .txt até o ano {FINAL_PROCESSING_YEAR} de {input_folder_path}")

    ner_results = []
    texts_to_process_batch = []
    batch_size = 500  # Ajuste conforme necessário para otimizar uso de memória/velocidade

    for file_path_str in filenames_to_process:
        file_path = Path(file_path_str)
        print(f"Lendo arquivo: {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('|', 1)  # Dividir apenas no primeiro '|'
                    if len(parts) == 2:
                        title, abstract = parts
                        texts_to_process_batch.append(title)
                        texts_to_process_batch.append(abstract)
                    elif len(parts) == 1:  # Apenas título ou abstract
                        texts_to_process_batch.append(parts[0])
                    else:  # Linha mal formatada
                        print(f"  AVISO: Linha {line_number} em {file_path.name} mal formatada: '{line[:100]}...'")
                        continue

                    if len(texts_to_process_batch) >= batch_size:
                        for doc in nlp_model.pipe(texts_to_process_batch, disable=["parser", "lemmatizer"]):
                            for ent in doc.ents:
                                if ent.label_ in RELEVANT_SPACY_ENTITY_TYPES:
                                    ner_results.append({
                                        'token': ent.text.lower(),
                                        'entity': MAPPED_ENTITY_TYPE
                                    })
                        texts_to_process_batch = []  # Limpar lote
            print(f"  Arquivo {file_path.name} lido.")
        except Exception as e:
            print(f"  ERRO ao processar o arquivo {file_path.name}: {e}")
            continue

    if texts_to_process_batch:  # Processar o último lote restante
        for doc in nlp_model.pipe(texts_to_process_batch, disable=["parser", "lemmatizer"]):
            for ent in doc.ents:
                if ent.label_ in RELEVANT_SPACY_ENTITY_TYPES:
                    ner_results.append({
                        'token': ent.text.lower(),
                        'entity': MAPPED_ENTITY_TYPE
                    })

    return ner_results


def main():
    nlp = load_spacy_model()
    if nlp is None:
        return

    Path(f"./data/{folder_name}").mkdir(parents=True, exist_ok=True)

    print(f"Iniciando processamento NER para a doença: {target_disease} (pasta: {INPUT_ABSTRACTS_PATH})")
    print(f"Processando arquivos até o ano (inclusive): {FINAL_PROCESSING_YEAR}")

    ner_data = process_abstracts_from_files(nlp)

    if not ner_data:
        print("Nenhuma entidade NER relevante foi extraída.")
        return

    ner_df = pd.DataFrame(ner_data).drop_duplicates().reset_index(drop=True)

    if ner_df.empty:
        print("DataFrame NER vazio após remoção de duplicatas.")
        return

    print(f"\nTotal de {len(ner_df)} entidades NER únicas encontradas (tipo: {MAPPED_ENTITY_TYPE}).")
    print("Amostra da tabela NER gerada:")
    print(ner_df.head())

    try:
        ner_df.to_csv(OUTPUT_NER_CSV_PATH, index=False)
        print(f"\nTabela NER salva com sucesso em: {OUTPUT_NER_CSV_PATH}")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo CSV: {e}")


if __name__ == '__main__':
    if not Path("target_disease.py").exists():
        print("ERRO: O arquivo 'target_disease.py' não foi encontrado.")
    else:
        main()