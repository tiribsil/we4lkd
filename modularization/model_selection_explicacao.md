## Passo a passo do que acontece no módulo de seleção de modelos (model_selection) [EXTERNO]

Este módulo (Fase 3) decide qual das arquiteturas candidatas é a melhor para prever descobertas na doença alvo. Inicia com **select_best_model()**.

1. **Definição do Ground Truth**:
   - Utiliza a classe `GroundTruthGenerator` para escanear o corpus histórico e determinar o ano real em que compostos foram reportados como tratamentos na literatura.

2. **Cálculo da Antecipação (Years Early)**:
   - Para cada modelo candidato:
     - Verifica em que ano o modelo colocou um composto "descoberto no futuro" em seu Top-N de similaridade.
     - Compara o ano da "previsão latente" do modelo com o ano real da descoberta.

3. **Métrica Final (Score)**:
   - Calcula a média de anos de antecipação. O melhor modelo é aquele que, em média, "previu" as relações terapêuticas com o maior tempo de antecedência em relação à publicação oficial.

4. **Persistência**:
   - Salva o nome do melhor modelo no checkpoint, que será utilizado exclusivamente para as fases finais de avaliação e relatório.
