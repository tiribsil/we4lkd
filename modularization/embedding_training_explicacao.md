## Passo a passo do que acontece no módulo de treinamento (embedding_training) [INTERNO]

Este módulo é responsável por transformar o texto limpo em vetores numéricos. Ele consolida o treinamento de modelos específicos e a busca por hiperparâmetros. Inicia com **run()**.

1. **Definição de Hiperparâmetros (LHS)**:
   - Utiliza *Latin Hypercube Sampling* para explorar o espaço de parâmetros (epochs, vector_size, window, ns_exponent, etc.) de forma eficiente.
   - Gera diversas combinações de modelos (Word2Vec CBOW, Word2Vec Skip-gram e FastText).

2. **Treinamento Iterativo**:
   - Para cada combinação de hiperparâmetros, treina o modelo utilizando os abstracts limpos até o ano de corte do desenvolvimento (`model_dev_end_year`).
   - Os modelos são salvos separadamente para posterior seleção.

3. **Avaliação Primal**:
   - Realiza testes básicos de sanidade nos modelos (ex: verificar se termos médicos comuns estão próximos no espaço vetorial).

4. **Treinamento de Modelo Específico**:
   - Também permite treinar um modelo único com parâmetros fixos, utilizado em fases de expansão ou após a seleção do "melhor" modelo.
