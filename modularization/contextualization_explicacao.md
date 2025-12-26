## Passo a passo do que acontece no módulo de contextualização (contextualization) [EXTERNO]

Este módulo (Fase 5) é a camada de Inteligência Artificial Explicável (XAI), responsável por interpretar os resultados numéricos. Inicia com **analyze_batch()**.

1. **Seleção de Alvos Atuais**:
   - Recupera os compostos que estão no Top-N de similaridade no ano mais recente (as "apostas" atuais do modelo).

2. **Interface com LLM (BioMistral/Mistral)**:
   - Envia o par (Doença, Composto) para um Modelo de Linguagem de Grande Porte especializado em biomedicina.
   - Solicita que o modelo explique a possível conexão biológica ou mecanismo de ação, baseando-se em seu conhecimento interno de farmacologia e medicina.

3. **Geração de Hipóteses**:
   - O resultado é um texto em linguagem natural que ajuda o pesquisador a entender *por que* aquele composto pode ser um tratamento viável, mesmo que a literatura ainda não tenha uma prova definitiva.

4. **Exportação**:
   - Salva as análises em formato JSON para fácil integração em dashboards ou sistemas de revisão.
