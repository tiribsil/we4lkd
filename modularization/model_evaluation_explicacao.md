## Passo a passo do que acontece no módulo de avaliação (model_evaluation) [INTERNO]

Este módulo (Fase 4) realiza o teste de estresse final no melhor modelo selecionado. Inicia com **run()**.

1. **Treinamento Incremental**:
   - Treina o melhor modelo ano a ano no período de teste. A cada passo, o modelo recebe apenas os dados disponíveis até aquele momento histórico.

2. **Avaliação de Desempenho**:
   - Calcula as estatísticas detalhadas de antecipação (Média, Mediana, Desvio Padrão, Moda).
   - Identifica as "maiores antecipações" (casos onde o modelo previu uma conexão décadas antes do reporte oficial).

3. **Geração de Rankings de Teste**:
   - Salva os Top-N de cada ano do período de teste, que serão a base para as previsões futuras e para o relatório final.
