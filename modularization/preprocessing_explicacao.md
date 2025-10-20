## Passo a passo do que acontece no módulo de preprocessamento [INTERNO]

É passada para a classe no nome da doença, o ano de interesse e também o modo de processamento (incremental ou não). A pipeline começa com a função **run()**, que agrega todas as outras funções.

1. Para preprocessamento utilizamos dados da PubChem, que possui informações de compostos de interesse. Para isso fazemos download de 2 tabelas com esses dados. CID-Title e CID-Synonym-filtered.

2. Caso o modo incremental não seja escolhido, processamos os abstracts apenas de alguns anos, se não, processamos de todos. No processamento dos batches ocorre a extração de entidades (NER) como compostos, tratamentos, sintomas, etc de cada um dos abstracts do corpus e os salva em uma tabela.

3. Depois disso ocorre a limpeza e normalização dos dados, com remoçao de stopwords, correção de typos, etc. Depois temos uma tokenização simples e limpeza de cada um desses tokens e depois o reagrupamento dos tokens em um sumário completo. **deve ter formas melhores de fazer isso (muito custo?)**
