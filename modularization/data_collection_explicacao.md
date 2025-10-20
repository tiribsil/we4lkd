## Passo a passo do que acontece no módulo de coleta de dados (crawler) [INTERNO]

É passada para a classe no nome da doença, primeiro e último ano a serem buscados. A pipeline começa com a função **run()**, que agrega todas as outras funções.

1. Primeiro é gerada uma query que será enviada para a API do PubMed.
Nessa função ocorre:
    - Criação de uma lista de tópicos de interesse a serem pesquisados nos artigos
    - Expansão da query com a busca de sinônimos - é feita com o download de arquivos do PubChem, que possuem CID, o nome da doença e sinônimos para seus compostos, de forma filtrada. Assim, a lista de tópicos de interesse cresce com os sinônimos dos tópicos originais.
    - Depois, com os tópicos, criamos subqueries para serem também buscadas na PubMed

2. Procuramos artigos relacionados com os tópicos encontrados e seus sinônimos no PubMed e salvamos seus ids para identificação única.

3. - Caso seja a primeira execução (não temos corpus), é feita uma agregação total dos abstracts. É organizado de acordo com seu ano, seu título e seu conteúdo, depois agregado por ano **DE PUBLICAÇÃO**.
   - Caso não seja a primeira execução e tenhamos um corpus inicial, recuperamos as informações apenas dos novos papers (identificados a partir de novos ids) e fazemos sua agregação da mesma forma (ano, seu título e seu conteúdo). Depois salvamos o paper e agregamos ao corpus anterior.

4. Também identificamos quais anos tiveram novos papers, ao buscar arquivos que foram recém-adicionados. Com isso, agregamos os abstracts de forma incremental apenas para os anos afetados, e processamos apenas o que precisa.

