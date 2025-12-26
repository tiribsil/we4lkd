## Passo a passo do que acontece no módulo de coleta de dados (data_collection) [INTERNO]

Este módulo (Fase 2) realiza a extração massiva de dados do PubMed. Recebe o nome da doença e o ano alvo. A execução principal ocorre via **run()**.

1. **Geração de Query**:
   - Lê os tópicos expandidos na Fase 1.
   - Constrói uma query complexa combinando o nome da doença com os tópicos de interesse e seus sinônimos (obtidos via tabelas da PubChem).

2. **Busca e Download**:
   - Utiliza as APIs `esearch` e `efetch` do NCBI para localizar e baixar abstracts de artigos publicados no ano especificado.
   - Garante que apenas artigos com abstracts disponíveis sejam processados.

3. **Armazenamento Bruto**:
   - Salva cada abstract individualmente em `data/{disease}/corpus/raw_abstracts/{year}/`.
   - Mantém um controle de IDs para evitar downloads duplicados em execuções incrementais.

4. **Gerenciamento de Corpus**:
   - No final da coleta de um ano, agrupa os arquivos individuais em um formato consolidado para facilitar o processamento em larga escala (PySpark).
