## Passo a passo do que acontece no módulo de expansão de tópicos (topic_expansion) [INTERNO]

Este módulo é responsável pela Fase 1 da pipeline, onde expandimos os termos de busca para garantir um corpus rico e contextualizado. A classe recebe o nome da doença, o número máximo de tópicos e o limite de novos tópicos por iteração. Começa com a função **run()**.

1. **Inicialização**: Verifica se já existe um arquivo de tópicos inicial (`topics_of_interest.txt`). Se não existir, utiliza apenas o nome da doença como ponto de partida.

2. **Loop de Expansão**:
   - Para cada tópico atual, realiza uma busca no PubMed para identificar termos frequentemente associados (co-ocorrência).
   - Utiliza um processo iterativo para descobrir novos candidatos a "tópicos de interesse" que aparecem recorrentemente na literatura daquela doença.

3. **Filtragem**: Novos tópicos são validados para evitar redundâncias e garantir que sejam termos médicos/químicos relevantes (utilizando critérios de frequência e relevância).

4. **Persistência**: O resultado final é uma lista consolidada de tópicos salva em `data/{disease}/topics_of_interest.txt`, que servirá de base para a coleta de dados massiva na Fase 2.
