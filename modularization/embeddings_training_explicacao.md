# Documentação: Sistema de Avaliação de Embeddings para Análise de Similaridade

## Sumário Executivo

Este documento detalha o sistema de avaliação de embeddings desenvolvido para tarefas de análise de similaridade em textos biomédicos. O sistema implementa uma abordagem **não supervisionada** para avaliar a qualidade de diferentes modelos de embeddings (Word2Vec, FastText, GloVe, LSA, BERT variants) sem necessidade de dados rotulados.

---

## 1. Contexto e Objetivo

### 1.1 Problema
O projeto busca encontrar o melhor modelo de embedding para representar abstracts científicos sobre doenças, com foco em **tarefas de similaridade** entre textos. A avaliação deve ser:
- **Não supervisionada** (sem labels)
- **Focada em similaridade** (não em classificação)
- **Comparável** entre modelos diferentes

### 1.2 Solução
Um sistema de avaliação multidimensional que combina:
- Métricas de clustering (estrutura dos dados)
- Métricas de cobertura vocabular
- Métricas de consistência de similaridade
- Métricas de preservação de vizinhança

---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                  PIPELINE DE AVALIAÇÃO                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. TREINAMENTO                                         │
│     └─> sentences → model.train() → embeddings          │
│                                                         │
│  2. AVALIAÇÃO (EmbeddingEvaluator)                      │
│     ├─> Clustering Metrics                              │
│     ├─> Vocabulary Coverage                             │
│     ├─> Similarity Consistency                          │
│     ├─> Neighborhood Preservation                       │
│     └─> Rank Correlation                                │
│                                                         │
│  3. AGREGAÇÃO                                           │
│     └─> Intrinsic Score (métrica final)                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Métricas Detalhadas

### 3.1 MÉTRICAS DE CLUSTERING

Avaliam a **estrutura dos embeddings** através de agrupamento automático.

#### 3.1.1 Silhouette Score
```python
silhouette_score(embeddings, cluster_labels)
```

**O que mede:** Quão bem separados estão os clusters.

**Como funciona:**
- Para cada ponto, calcula:
  - **a:** distância média para pontos do mesmo cluster
  - **b:** distância média para pontos do cluster mais próximo
  - **silhouette = (b - a) / max(a, b)**
- Retorna a média de todos os pontos

**Interpretação:**
- **+1:** Clusters perfeitamente separados
- **0:** Clusters sobrepostos
- **-1:** Pontos no cluster errado

**Por que usar:**
- Avalia se embeddings similares ficam próximos
- Detecta se há estrutura natural nos dados
- Valores altos indicam boa capacidade de distinção

**Normalização no código:**
```python
silhouette_norm = (silhouette + 1) / 2  # [-1,1] → [0,1]
```

---

#### 3.1.2 Calinski-Harabasz Index
```python
calinski_harabasz_score(embeddings, cluster_labels)
```

**O que mede:** Razão entre dispersão inter-cluster e intra-cluster.

**Como funciona:**
- **Numerador:** Variância entre clusters (quão separados)
- **Denominador:** Variância dentro de clusters (quão compactos)
- **CH = (B / W) × ((n - k) / (k - 1))**
  - B = dispersão entre clusters
  - W = dispersão dentro de clusters
  - n = número de pontos
  - k = número de clusters

**Interpretação:**
- **Valores altos:** Clusters bem definidos e compactos
- **Sem limite superior** (por isso normalizamos)

**Por que usar:**
- Recompensa clusters densos e bem separados
- Não sofre com clusters de tamanhos desiguais
- Complementa silhouette (usa variância, não distâncias)

**Normalização no código:**
```python
calinski_norm = min(calinski / 1000, 1.0)  # Cap em 1.0
```

---

#### 3.1.3 Davies-Bouldin Index
```python
davies_bouldin_score(embeddings, cluster_labels)
```

**O que mede:** Razão média entre distâncias intra-cluster e inter-cluster.

**Como funciona:**
- Para cada cluster i, encontra o cluster j mais similar
- **DB_i = (S_i + S_j) / d(c_i, c_j)**
  - S_i = distância média intra-cluster i
  - d(c_i, c_j) = distância entre centróides
- **DB = média de DB_i para todos os clusters**

**Interpretação:**
- **0:** Clusters perfeitamente separados
- **Valores altos:** Clusters sobrepostos
- **MENOR é melhor** (inverso dos outros)

**Por que usar:**
- Penaliza clusters que se sobrepõem
- Sensível a clusters mal formados
- Baseado em centróides (geometria clara)

**Normalização no código:**
```python
davies_bouldin_norm = 1 / (1 + davies_bouldin)  # Inverter
```

---

### 3.2 MÉTRICAS DE COBERTURA

Avaliam a **capacidade do modelo** de representar o vocabulário.

#### 3.2.1 Vocabulary Coverage
```python
vocab_coverage = len(vocabulary) / total_words
```

**O que mede:** Proporção de palavras únicas do corpus presentes no modelo.

**Como funciona:**
- **vocabulary:** palavras que o modelo conhece
- **total_words:** palavras únicas no corpus
- **coverage = |vocabulary| / |total_words|**

**Interpretação:**
- **1.0:** Modelo conhece todas as palavras
- **0.5:** Modelo conhece metade do vocabulário
- **Baixo:** Muitas palavras OOV (out-of-vocabulary)

**Por que usar:**
- Detecta se o modelo foi bem treinado
- Modelos pré-treinados podem ter baixa cobertura
- Importante para domínios específicos (medicina)

---

#### 3.2.2 OOV Handling
```python
def _calculate_oov_handling(model_type):
    if model_type == FASTTEXT: return 0.8
    if model_type in TRANSFORMERS: return 1.0
    return 0.3
```

**O que mede:** Capacidade de lidar com palavras desconhecidas.

**Como funciona:**
- **FastText (0.8):** Usa subwords (ex: "cardiology" → "card", "io", "logy")
- **Transformers (1.0):** Tokenização por subwords (sempre gera embedding)
- **Word2Vec/GloVe (0.3):** Apenas palavras exatas (OOV = zero vector)

**Por que usar:**
- Crucial para textos científicos (neologismos, siglas)
- Modelos com boa OOV generalizam melhor
- Complementa vocabulary coverage

---

### 3.3 MÉTRICAS DE CONSISTÊNCIA DE SIMILARIDADE

Avaliam se **diferentes métricas de distância concordam** sobre o que é similar.

#### 3.3.1 Similarity Consistency
```python
def _calculate_similarity_consistency(embeddings, n_samples=100):
    # Calcula 3 tipos de similaridade
    cosine_sim = cosine(embeddings)
    euclidean_sim = 1 / (1 + euclidean_dist)
    dot_product = dot(embeddings)
    
    # Retorna consistência (baixo desvio padrão = alta consistência)
    return (1/(1+std(cosine)), 1/(1+std(euclidean)), 1/(1+std(dot)))
```

**O que mede:** Concordância entre métricas de similaridade.

**Como funciona:**
1. Amostra pares de embeddings
2. Calcula similaridade por:
   - **Cosseno:** ângulo entre vetores (normalizado)
   - **Euclidiana:** distância euclidiana invertida
   - **Dot product:** produto escalar
3. Mede desvio padrão de cada métrica
4. **Consistência = 1 / (1 + std)**

**Interpretação:**
- **Alta consistência:** Todas as métricas rankeiam similaridades parecido
- **Baixa consistência:** Métricas discordam (embeddings problemáticos)

**Por que usar:**
- Garante que similaridade é robusta à métrica escolhida
- Detecta embeddings com propriedades geométricas ruins
- Importante quando não sabemos qual métrica usar a priori

**Exemplo:**
```
Doc A e B:
- Cosseno: 0.9 (muito similar)
- Euclidiana: 0.3 (pouco similar)
→ Baixa consistência = embedding ruim
```

---

#### 3.3.2 Neighborhood Preservation
```python
def _calculate_neighborhood_preservation(embeddings):
    # k vizinhos mais próximos por cosseno
    nbrs_cosine = NearestNeighbors(metric='cosine')
    indices_cosine = nbrs_cosine.kneighbors(embeddings)
    
    # k vizinhos mais próximos por euclidiana
    nbrs_euclidean = NearestNeighbors(metric='euclidean')
    indices_euclidean = nbrs_euclidean.kneighbors(embeddings)
    
    # Overlap entre vizinhanças
    overlaps = [len(set(cos) & set(euc)) / k for cos, euc in zip(...)]
    return mean(overlaps)
```

**O que mede:** Se vizinhos próximos são os mesmos em diferentes métricas.

**Como funciona:**
1. Para cada embedding, encontra k vizinhos mais próximos:
   - Por similaridade de cosseno
   - Por distância euclidiana
2. Calcula **overlap** (interseção / k)
3. Retorna média dos overlaps

**Interpretação:**
- **1.0:** Mesmos vizinhos em ambas as métricas
- **0.5:** 50% de sobreposição
- **0.0:** Vizinhanças completamente diferentes

**Por que usar:**
- Garante que "similaridade local" é preservada
- Crucial para tarefas de retrieval (buscar documentos similares)
- Detecta distorções geométricas nos embeddings

**Exemplo prático:**
```
Documento sobre "leucemia":
- Cosseno: vizinhos = [câncer, quimioterapia, medula]
- Euclidiana: vizinhos = [câncer, quimioterapia, medula]
→ Alta preservação = bom embedding

vs.

- Cosseno: vizinhos = [câncer, quimioterapia, medula]
- Euclidiana: vizinhos = [diabetes, hipertensão, gripe]
→ Baixa preservação = embedding ruim
```

---

#### 3.3.3 Rank Correlation
```python
def _calculate_rank_correlation(embeddings, n_samples=50):
    # Escolhe ponto de referência
    reference = embeddings[random_idx]
    
    # Calcula distâncias por 3 métricas
    cosine_dists = 1 - cosine_similarity(samples, reference)
    euclidean_dists = euclidean(samples, reference)
    dot_dists = -dot_product(samples, reference)
    
    # Correlação de Spearman (ranking)
    corr_ce = spearmanr(cosine_dists, euclidean_dists)
    corr_cd = spearmanr(cosine_dists, dot_dists)
    corr_ed = spearmanr(euclidean_dists, dot_dists)
    
    return mean([abs(corr_ce), abs(corr_cd), abs(corr_ed)])
```

**O que mede:** Concordância no **ranking de distâncias** entre métricas.

**Como funciona:**
1. Escolhe um embedding de referência
2. Calcula distância desse ponto para todos os outros em 3 métricas
3. Usa **Spearman correlation** para comparar rankings
4. Retorna média das correlações

**Spearman vs Pearson:**
- **Pearson:** Mede correlação linear (valores absolutos)
- **Spearman:** Mede correlação monotônica (rankings)
- Usamos Spearman porque importa a **ordem**, não a distância exata

**Interpretação:**
- **1.0:** Rankings idênticos em todas as métricas
- **0.0:** Rankings não correlacionados

**Por que usar:**
- Complementa neighborhood preservation (global vs local)
- Detecta se embeddings preservam ordem de similaridade
- Importante para ranking tasks (retrieval, recomendação)

**Exemplo:**
```
Referência: "leucemia aguda"

Ranking por Cosseno:
1. leucemia crônica (0.95)
2. câncer sanguíneo (0.85)
3. quimioterapia (0.75)

Ranking por Euclidiana:
1. leucemia crônica (2.1)
2. câncer sanguíneo (3.5)
3. quimioterapia (4.2)

→ Rankings idênticos = alta correlação = bom embedding
```

---

### 3.4 MÉTRICAS AGREGADAS

Combinam múltiplas dimensões em scores únicos.

#### 3.4.1 Similarity Score
```python
similarity_score = (
    0.25 * cosine_consistency +
    0.25 * euclidean_consistency +
    0.20 * dot_product_consistency +
    0.15 * neighborhood_preservation +
    0.15 * rank_correlation
)
```

**O que mede:** Qualidade geral para tarefas de similaridade.

**Pesos justificados:**
- **25% cosseno:** Métrica mais comum em NLP
- **25% euclidiana:** Métrica natural para espaços vetoriais
- **20% dot product:** Usado em attention mechanisms
- **15% vizinhança:** Importante mas já capturado parcialmente
- **15% ranking:** Complementa vizinhança (global vs local)

---

#### 3.4.2 Intrinsic Score (FINAL)
```python
intrinsic_score = (
    0.15 * silhouette_normalized +
    0.10 * calinski_normalized +
    0.10 * davies_bouldin_normalized +
    0.10 * vocab_coverage +
    0.15 * oov_handling +
    0.40 * similarity_score
)
```

**O que mede:** Qualidade intrínseca do embedding (métrica final de seleção).

**Pesos justificados:**

**Clustering (35%):**
- 15% Silhouette: Principal métrica de separação
- 10% Calinski-Harabasz: Compactação de clusters
- 10% Davies-Bouldin: Penalização de sobreposição

**Cobertura (25%):**
- 10% Vocabulary Coverage: Modelo conhece as palavras
- 15% OOV Handling: Modelo generaliza para novas palavras

**Similaridade (40%):**
- Maior peso porque é o **objetivo principal**
- Engloba 5 sub-métricas complementares

---

## 4. Pipeline de Avaliação

### 4.1 Fluxo de Execução

```python
# 1. TREINAMENTO
model = ModelFactory.create_model(config)
model.train(sentences)

# 2. EXTRAÇÃO DE EMBEDDINGS
embeddings = model.get_embeddings()

# 3. AVALIAÇÃO
evaluator = EmbeddingEvaluator(n_clusters=10, k_neighbors=10)
metrics = evaluator.evaluate(embeddings, vocabulary, total_words, model_type)

# 4. RESULTADO
print(f"Intrinsic Score: {metrics.intrinsic_score}")
```

### 4.2 Processo de Seleção de Modelo

```
FASE 1: SCREENING RÁPIDO
├─> Testa todos os modelos sem PCA
├─> Ranqueia por intrinsic_score
└─> Seleciona top-3

FASE 2: REFINAMENTO
├─> Para cada top-3:
│   ├─> Testa sem PCA
│   └─> Testa com PCA (50 componentes)
└─> Seleciona melhor configuração

FASE 3: OTIMIZAÇÃO DE HIPERPARÂMETROS
├─> Usa Optuna (Bayesian Optimization)
├─> Search space específico por modelo
├─> Maximiza intrinsic_score
└─> Retorna parâmetros ótimos

FASE 4: TREINAMENTO FINAL
└─> Treina modelo com configuração ótima
```

---

## 5. Modelos Suportados

### 5.1 Modelos Clássicos

| Modelo | OOV Handling | Melhor Para | Limitações |
|--------|--------------|-------------|------------|
| **Word2Vec** | ❌ Ruim (0.3) | Vocabulário fechado | Palavras OOV = zero vector |
| **FastText** | ✅ Bom (0.8) | Textos com neologismos | Maior memória |
| **GloVe** | ❌ Ruim (0.3) | Co-ocorrências globais | Pré-treinado (baixa cobertura em domínios específicos) |
| **LSA** | ❌ Ruim (0.3) | Análise semântica latente | Não captura ordem das palavras |

### 5.2 Modelos Transformer

| Modelo | OOV Handling | Melhor Para | Domínio |
|--------|--------------|-------------|---------|
| **BioBERT** | ✅ Excelente (1.0) | Textos biomédicos | PubMed + PMC |
| **PubMedBERT** | ✅ Excelente (1.0) | Abstracts científicos | PubMed |
| **SciBERT** | ✅ Excelente (1.0) | Artigos científicos | Semantic Scholar |
| **Bio_ClinicalBERT** | ✅ Excelente (1.0) | Notas clínicas | MIMIC-III |
| **SBERT** | ✅ Excelente (1.0) | Propósito geral | NLI datasets |

---

## 6. Normalização e Escala

### 6.1 Por que Normalizar?

Métricas têm escalas diferentes:
- Silhouette: [-1, 1]
- Calinski-Harabasz: [0, ∞)
- Davies-Bouldin: [0, ∞)

Normalização permite:
- Comparação justa entre métricas
- Agregação com pesos
- Interpretação uniforme (0 = ruim, 1 = ótimo)

### 6.2 Estratégias de Normalização

```python
# 1. Min-Max (bounded metrics)
silhouette_norm = (silhouette + 1) / 2  # [-1,1] → [0,1]

# 2. Capping (unbounded metrics)
calinski_norm = min(calinski / 1000, 1.0)  # Cap em 1.0

# 3. Inverse (lower is better)
davies_bouldin_norm = 1 / (1 + davies_bouldin)

# 4. Logistic (smooth transformation)
consistency = 1 / (1 + std)  # ∞ → [0,1]
```

---

## 7. Considerações Práticas

### 7.1 Tamanho Mínimo de Dataset

```python
min_samples = max(n_clusters, k_neighbors + 1)
if len(embeddings) < min_samples:
    return zeros  # Não é possível avaliar
```

**Recomendações:**
- **Mínimo absoluto:** 20 documentos
- **Recomendado:** 100+ documentos
- **Ideal:** 1000+ documentos

### 7.2 Escolha de Hiperparâmetros

**n_clusters:** Número de clusters para avaliação
- **Padrão:** 10
- **Regra prática:** √(n_samples) ou n_samples/100

**k_neighbors:** Vizinhos para análise local
- **Padrão:** 10
- **Regra prática:** log(n_samples) ou 5-15

### 7.3 Trade-offs

| Aspecto | Word2Vec/FastText | Transformers |
|---------|-------------------|--------------|
| **Velocidade** | ⚡ Rápido (minutos) | 🐌 Lento (horas) |
| **Memória** | 💾 Baixa (< 1GB) | 💾 Alta (4-8GB) |
| **Qualidade** | 📊 Boa | 📊 Excelente |
| **OOV** | ❌ Limitado | ✅ Robusto |
| **Customização** | ✅ Total | ⚠️ Fine-tuning complexo |

---

## 8. Interpretação de Resultados

### 8.1 Scores de Referência

| Intrinsic Score | Interpretação | Ação |
|-----------------|---------------|------|
| **> 0.80** | 🟢 Excelente | Usar em produção |
| **0.70 - 0.80** | 🟡 Bom | Considerar otimização |
| **0.60 - 0.70** | 🟠 Regular | Revisar configuração |
| **< 0.60** | 🔴 Ruim | Trocar modelo ou aumentar dados |

### 8.2 Análise de Componentes

```python
# Exemplo de resultado
{
    'intrinsic_score': 0.75,
    'similarity_score': 0.82,  # ✅ Ótimo para similaridade
    'silhouette': 0.45,        # 🟡 Clusters moderados
    'vocab_coverage': 0.65,    # 🟠 Cobertura limitada
    'oov_handling': 0.8        # ✅ Bom OOV (FastText)
}
```

**Diagnóstico:**
- Modelo funciona bem para similaridade
- Pode ter problemas com clustering
- Vocabulário pode ser limitado
- FastText compensa OOV

**Ação:** Manter modelo, considerar treinar mais para melhorar cobertura

---

## 9. Exemplos de Uso

### 9.1 Seleção Automática de Modelo

```python
# Define candidatos
candidates = [
    ModelType.WORD2VEC,
    ModelType.FASTTEXT,
    ModelType.BIOBERT,
]

# Seleção automática
selector = SequentialModelSelector(
    candidate_models=candidates,
    use_pca_variants=True
)

best_config, metrics = selector.select_best_model(sentences)
print(f"Melhor modelo: {best_config.model_type}")
print(f"Score: {metrics.intrinsic_score}")
```

### 9.2 Otimização de Hiperparâmetros

```python
# Define modelo base
config = EmbeddingConfig(
    model_type=ModelType.FASTTEXT,
    use_pca=False,
    vector_size=300
)

# Otimiza hiperparâmetros
optimizer = SequentialHyperparameterOptimizer(
    model_config=config,
    n_trials=50,
    timeout=1800  # 30 minutos
)

best_params, best_score = optimizer.optimize(sentences)
print(f"Melhores parâmetros: {best_params}")
```

### 9.3 Pipeline Completo (AutoML)

```python
# AutoML completo
automl = SequentialEmbeddingAutoML(
    candidate_models=[...],
    hyperopt_trials=30,
    output_dir='./results'
)

# Executa pipeline
final_model, metrics = automl.run(sentences)

# Salva modelo
automl.save_model('./best_model.pkl')
```

---

## 10. Conclusão

### 10.1 Pontos-Chave

1. **Avaliação não supervisionada:** Não precisa de labels
2. **Foco em similaridade:** Otimizado para tarefas de retrieval
3. **Multidimensional:** Combina estrutura, cobertura e consistência
4. **Robusto:** Funciona com diferentes tipos de modelos
5. **Automatizado:** Pipeline completo de seleção e otimização

### 10.2 Quando Usar Cada Métrica Isoladamente

- **Silhouette:** Quando há estrutura natural (documentos sobre tópicos diferentes)
- **Consistency:** Quando não sabe qual métrica de distância usar
- **Neighborhood:** Para tarefas de retrieval (buscar documentos similares)
- **Rank Correlation:** Para recomendação (ordenar por relevância)
- **OOV Handling:** Para textos com muitos neologismos/jargões

### 10.3 Limitações

- ❌ Não avalia qualidade semântica (apenas geométrica)
- ❌ Não detecta bias nos embeddings
- ❌ Assume que similaridade textual = similaridade semântica
- ❌ Não avalia performance em downstream tasks específicas

### 10.4 Próximos Passos

Para validação adicional:
1. **Extrinsic evaluation:** Testar em task específica (classificação, Q&A)
2. **Human evaluation:** Comparar similaridades com julgamento humano
3. **Analogy tests:** Word2Vec analogies (rei - homem + mulher = rainha)
4. **Semantic similarity:** Comparar com benchmarks (STS, SICK)

---

## Referências

- **Clustering Metrics:** Scikit-learn documentation
- **Word Embeddings:** Mikolov et al. (2013), Pennington et al. (2014)
- **Transformers:** Devlin et al. (2018), Lee et al. (2020)
- **Evaluation:** Schnabel et al. (2015) - "Evaluation methods for unsupervised word embeddings"