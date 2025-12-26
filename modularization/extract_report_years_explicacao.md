## Passo a passo do que acontece no módulo de Ground Truth (extract_report_years) [INTERNO]

Este módulo é fundamental para a validação histórica. Ele descobre quando uma relação "composto-doença" deixou de ser latente e se tornou conhecimento estabelecido.

1. **Definição de Palavras-Chave Terapêuticas**:
   - Possui listas de termos positivos (ex: "efficacy", "clinical trial") e negativos (ex: "toxicity", "side effect") para filtrar o contexto.

2. **Varredura de Corpus**:
   - Escaneia todo o corpus textual em busca de co-ocorrências entre um composto e a doença em um contexto terapêutico positivo.

3. **Determinação do Ano de Reporte**:
   - Identifica o primeiro ano em que a evidência textual atinge um limite crítico (threshold) de menções. Este se torna o "Ano de Descoberta" oficial para aquele composto.

4. **Sistema de Cache**:
   - Como este processo é computacionalmente pesado, os resultados são salvos em `ground_truth_cache` para evitar reprocessamento em execuções futuras da seleção de modelos.
