## Passo a passo do que acontece no módulo de producao dos produtos escalares[INTERNO]

É passada para a classe no nome da doença, primeiro e último ano a serem buscados. A pipeline começa com a função **run()**, que agrega todas as outras funções.

1. São gerados os históricos de similaridade dos compostos. Para isso cria uma whitelist com nome de drogas com moléculas pequenas e aprovadas pela FDA - ChemBL e também os compostos do PubChem e faz o match (df ChemBL + df PubMed).

2. com os compostos do chembl e pubmed, filtramos os que, desses, aparecem no vocabulário do modelo gerado.  Depois, pegamos os embeddings da doença e de cada composto, e calculamos as métricas de similaridade e derivadas entre os embeddings da doença e dos compostos.

3. Depois encontramos os compostos com maior similaridade com a doença.