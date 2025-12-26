## Passo a passo do que acontece no módulo de métricas (metric_generation) [INTERNO]

Este módulo calcula a similaridade entre a doença alvo e todos os compostos químicos identificados no corpus. Começa com **run()**.

1. **Cálculo de Similaridade (Produto Escalar)**:
   - Para um determinado modelo treinado, calcula a similaridade de cosseno (ou dot product) entre o vetor da doença alvo e os vetores de todos os compostos químicos presentes no vocabulário.

2. **Geração de Rankings**:
   - Para cada ano de análise, ordena os compostos pela força da relação semântica com a doença.
   - Salva os rankings históricos que mostram como a "proximidade" de um composto com a doença evoluiu ao longo do tempo.

3. **Validação de Top-N**:
   - Identifica os compostos que entraram no Top-N (ex: Top 20, Top 50) de maior similaridade em cada período.

4. **Exportação**:
   - Gera arquivos CSV em `data/{disease}/validation/{model_name}/compound_history/` contendo o histórico completo de pontuação para análise de tendência.
