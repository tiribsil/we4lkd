## Passo a passo do que acontece no módulo de preprocessamento (preprocessing) [INTERNO]

Este módulo (Fase 2) limpa e normaliza o corpus textual. Recebe o nome da doença e o ano de referência. Inicia com a função **run()**.

1. **Consolidação e Limpeza Básica**:
   - Une os abstracts coletados em um único fluxo de processamento.
   - Aplica correções de encoding, remoção de caracteres especiais e normalização de espaços.

2. **Extração de Entidades (NER)**:
   - Utiliza o modelo `en_ner_bc5cdr_md` do spaCy para identificar entidades químicas e doenças.
   - Gera uma tabela de NER que auxilia na identificação de termos a serem normalizados.

3. **Normalização de Compostos (PubChem)**:
   - Crucial para o treinamento: substitui sinônimos variados de uma substância química pelo seu nome canônico (CUI/Name) definido pela PubChem.
   - Isso garante que o Word2Vec aprenda um único vetor para o mesmo composto, independentemente de como ele foi escrito no artigo.

4. **Tokenização e Filtragem**:
   - Remove stopwords gerais e específicas do domínio médico.
   - Corrige "typos" comuns e padroniza termos científicos.
   - O resultado é salvo em `data/{disease}/corpus/clean_abstracts/`, pronto para a fase de treinamento.
