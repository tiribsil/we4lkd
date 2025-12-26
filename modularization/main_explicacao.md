## Passo a passo do que acontece no roteiro principal (main) [EXTERNO]

O arquivo `main.py` é o orquestrador central de todo o sistema. Ele garante que as fases sejam executadas na ordem correta e gerencia a continuidade através de checkpoints.

1. **Configuração Global**:
   - Define os parâmetros gerais da execução (nome da doença, splits de anos para treino/validação/teste).
   - Inicializa o sistema de logs centralizado.

2. **Execução por Fases**:
   - **Fase 1 (Expansão)**: Abrange tópicos de pesquisa.
   - **Fase 2 (Desenvolvimento)**: Coleta, limpa e treina dezenas de modelos candidatos.
   - **Fase 3 (Seleção)**: Identifica qual modelo é o mais "vidente" historicamente.
   - **Fase 4 (Avaliação)**: Consolida a prova de conceito e gera o relatório LaTeX oficial.
   - **Fase 5 (Sumário)**: Interpreta as descobertas latentes mais recentes via IA.

3. **Sistema de Checkpoints**:
   - Antes de iniciar qualquer fase, verifica o arquivo `artifacts/{disease}_pipeline_checkpoint.json`. Se uma fase já foi concluída, ela é pulada, permitindo retomar execuções longas após interrupções.

4. **Orquestração de Dados**:
   - Garante que o output de um módulo (ex: corpus limpo) esteja disponível no local esperado pelo módulo seguinte (ex: treinador).
