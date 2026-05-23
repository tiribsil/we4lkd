---

# WE4LKD — Automated Longitudinal Framework for Latent Knowledge Discovery in Biomedical Literature

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An end-to-end, automated pipeline for **Latent Knowledge Discovery (LKD)** in biomedical literature. The framework identifies non-obvious semantic relationships between chemical compounds and target diseases, anticipating future therapeutic discoveries before they are formally published.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Key Contributions](#key-contributions)
- [The Framework: How It Works](#the-framework-how-it-works)
- [Evaluation Metrics](#evaluation-metrics)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## About The Project

The exponential growth of biomedical literature makes it impossible for researchers to manually track all advancements. This framework provides a fully automated pipeline that analyzes decades of PubMed abstracts to identify *latent knowledge*—implicit semantic relationships between chemical compounds and a target disease that may foreshadow future therapeutic breakthroughs.

Building upon [Tshitoyan et al. (2019)](https://www.nature.com/articles/s41586-019-1335-8) and [Berto et al. (2024)](https://doi.org/10.1016/j.eswa.2024.123566), the framework extends the word-embedding-based discovery paradigm by introducing:

- **Longitudinal corpus partitioning** with temporal bias controls.
- **LLM-augmented ground truth** built from the corpus itself, eliminating data leakage.
- **Orthogonal Procrustes alignment** for consistent cross-year semantic comparisons.
- **Explainable AI (XAI) contextualization** via domain-specific LLMs that transform vector distances into natural-language pharmacological hypotheses.

The framework's efficacy is validated on an **Acute Myeloid Leukemia (AML) case study**, where it identified promising therapeutic compounds up to **19 years before their clinical recognition**.

---

## Key Contributions

- **End-to-end discovery pipeline** — fully automated, from raw PubMed search to explainable drug candidate reports, with no human-in-the-loop dependencies.
- **Domain-adaptive query generation (Topic Expansion)** — dynamically broadens search terms using compound-disease vector proximity to prevent temporal anachronisms and data leakage.
- **Longitudinal semantic mapping** — applies Orthogonal Procrustes alignment to word embeddings trained year-over-year, tracking each compound's semantic trajectory over seven decades.
- **LLM-curated ground truth** — uses a local quantized LLM (Qwen2.5-14B) as a zero-shot classifier to verify compound–disease therapeutic relationships directly from the corpus, eliminating reliance on external databases.
- **Novel LKD evaluation metrics** — proposes and computes TDG, NDG, LKD, Weighted LKD, Temporal Hit@K, AUC Gain, and LKD-Composite scores to rigorously evaluate predictive performance.
- **Contextualization layer** — uses BioMistral-7B to generate evidence-based pharmacological hypotheses for top-ranked candidates.

---

## The Framework: How It Works

The pipeline is anchored to the pathology's first recorded publication in the literature (**T_base**). The subsequent timeline is partitioned into:

| Segment | Purpose |
|---|---|
| T_base → T_exp | **Topic Expansion**: corpus seeding and search query refinement |
| T_exp → T_train | **Development**: data collection, preprocessing, LHS-sampled model training |
| T_train → T_sel | **Selection**: year-over-year model evaluation against LLM ground truth |
| T_sel → T_now | **Testing**: incremental inference on the best model; report generation |
| Concurrent | **Knowledge Base**: LLM constructs the ground truth from corpus co-occurrences |

The main entry point is `modularization/main.py`, which orchestrates the `PreliminaryPipeline` class across 5 phases, with full checkpointing support.

---

### Phase 1 — Expansion (`topic_expansion.py`)

Implements **Iterative Topic Expansion** between T_base and T_exp. Starting from the disease's canonical name, the module runs repeated cycles of:

1. **Data Acquisition** (`data_collection.py`) — PubMed search for the current year.
2. **Text Normalization** (`preprocessing.py`) — NER-based entity extraction, PubChem synonym normalization.
3. **Embedding Training** (`embedding_training.py`) — trains a fixed Word2Vec model on the cumulative corpus.
4. **Metric Generation** (`metric_generation.py`) — computes cosine similarity scores between compounds and the disease.
5. **Feedback / Reporting** (`reporting.py`) — ranks top compounds by proximity; the highest-scoring new terms are appended to `topics_of_interest.txt` as expanded search queries.

This iterative loop continues until a predefined topic cardinality (`max_topics`) is reached, ensuring the corpus represents the full medical semantic domain of the disease.

---

### Phase 2 — Development & Training (`data_collection.py`, `preprocessing.py`, `embedding_training.py`, `metric_generation.py`)

Covers the period T_exp → T_train (training split). Key sub-steps:

1. **Data Collection** — crawls PubMed year-by-year using the expanded topic set. Supports pagination via NCBI Web History and parallel abstract fetching.
2. **Preprocessing** — full-corpus NLP pipeline powered by **Apache Spark** (`PySpark`):
   - HTML/URL/punctuation stripping.
   - Disease synonym normalization via MeSH + regex.
   - Compound synonym normalization via PubChem CID-Title and CID-Synonym-filtered cross-reference.
   - Stopword and unit removal; tokenization.
   - Incremental update mode for efficiency.
3. **Candidate Model Training** — explores the hyperparameter space using **Latin Hypercube Sampling (LHS)** to generate `num_combinations` candidate configurations for Word2Vec and FastText. An analogy-based filter (`data/analogies.txt`) selects the best configuration per architecture before passing them to the selection phase.
4. **Metric Generation** — for each candidate model, computes per-compound, per-year: cosine similarity, Euclidean distance, Δ-similarity, and a composite heuristic `score`. Generates Top-N compound rankings and per-compound history CSVs.

---

### Phase 3 — Selection (`model_selection.py`)

Covers T_train → T_sel (validation split).

- Uses **`ModelSelector`** to iteratively train each LHS candidate year-by-year across the validation window.
- For each candidate, **`ValidationModule`** generates yearly Top-N rankings.
- **`ModelEvaluator`** computes the **Mean Lead Time** (average years a compound appears in the top-N before its LLM-confirmed first report year).
- The candidate with the highest mean lead time is selected as `best_model`.

The **Ground Truth** is built once by **`GroundTruthGenerator`** (`extract_report_years.py`), which uses a local Qwen2.5-14B GGUF model as a zero-shot classifier to scan corpus abstracts and confirm the earliest year each compound was validated as a direct therapeutic agent for the disease.

---

### Phase 4 — Evaluation (`model_evaluation.py`, `reporting.py`)

Covers T_sel → T_now (test split). Uses the best selected model to:

1. **Incremental Training** — retrains `best_model` year-by-year on the test period.
2. **Metric Generation** — computes compound rankings for each test year.
3. **Performance Calculation** — computes all LKD metrics (TDG, NDG, LKD, Weighted LKD, Temporal Hit@K, AUC Gain, LKD-Composite) and logs detailed per-compound anticipation results.
4. **Visual Report Generation** (`reporting.py`) — produces:
   - Historical similarity trajectories per compound.
   - Aligned PCA trajectory plots (Procrustes-aligned, disease as origin).
   - Ranking convergence plots (rank evolution relative to the literature report year).
   - Lead-time vs. rank scatter plots.
   - Discovery timeline plots for high-anticipation compounds.
   - Anticipation CSV report (`anticipation_report_<year>.csv`).

---

### Phase 5 — Summary & Contextualization (`contextualization.py`)

The XAI layer. Takes the top-N compounds from the current year and uses **BioMistral-7B** (quantized GGUF, CPU-only via `llama-cpp-python`) to generate:

- Compound identification and primary clinical uses.
- Mechanism of action summary (molecular targets/pathways).
- Theoretical or evidence-based connection to the target disease.
- Plausibility assessment (High / Moderate / Low / Speculative / Implausible).
- Key safety considerations.

Results are exported as a structured JSON file (`contextualization_results_<year>.json`).

---

## Evaluation Metrics

The framework computes a novel suite of LKD-specific metrics:

| Metric | Description |
|---|---|
| **TDG** (Time-to-Discovery Gain) | `t*(c) − t_pred(c)`: years the model anticipated compound `c` relative to its first literature report. Positive = early prediction. |
| **NDG** (Normalized Discovery Gain) | TDG normalized by corpus span: `TDG(c) / (t*(c) − t0)`. Enables cross-disease comparison. |
| **LKD** | Mean TDG restricted to correctly anticipated compounds only. |
| **Weighted LKD** | `mean(TDG(c) × 1/rank)`: penalizes early predictions that rank poorly. |
| **Temporal Hit@K** | Fraction of future ground-truth compounds appearing in top-K predictions, averaged over all test years. |
| **AUC Gain** | Area under the model's cumulative discovery curve minus a random (linear) baseline. |
| **Emergence Score** | Max year-over-year delta in normalized similarity, measuring the speed of semantic convergence. |
| **LKD-Composite** | `0.8 × NDG + 0.2 × Hit@10` — unified scalar for model selection and comparison. |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Java (required for PySpark)
- NCBI Entrez API key and email (set via `.env`)
- `en_ner_bc5cdr_md` spaCy model
- PubChem flat files: `CID-Title` and `CID-Synonym-filtered` (downloaded automatically by `preprocessing.py`)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <remote_repository>
   cd we4lkd
   ```

2. **Set up the environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_ner_bc5cdr_md
   ```

3. **Configure credentials** — create a `.env` file at the project root:
   ```
   ENTREZ_EMAIL=your@email.com
   ENTREZ_KEY=your_ncbi_api_key
   ```

---

## Usage

1. **Set the target disease** in `modularization/main.py`:
   ```python
   pipeline = PreliminaryPipeline(
       disease_name='acute myeloid leukemia',
       max_topics=9,
       max_new_topics=9,
       train_val_test_split=[0.6, 0.2, 0.2]
   )
   ```

2. **Run the pipeline:**
   ```sh
   cd modularization
   python main.py
   ```

3. **Checkpointing** — the pipeline saves progress to `artifacts/<disease>_pipeline_checkpoint.json`. If interrupted, it resumes automatically from the last completed phase.

4. **Outputs** are written to `data/<disease>/`:
   - `corpus/` — raw and clean abstracts.
   - `models/` — trained Word2Vec/FastText models.
   - `validation/<model>/` — compound history CSVs, Top-N rankings.
   - `reports/` — performance CSVs.
   - `plots/<model>/` — all PDF visualizations.
   - `ground_truth_cache/` — LLM-verified ground truth cache.
   - `contextualization_results_<year>.json` — XAI output.

---

## Project Structure

```
.
├── modularization/                      # Core pipeline logic
│   ├── main.py                          # Entry point — PreliminaryPipeline orchestrator
│   │
│   ├── topic_expansion.py               # Phase 1: Iterative Topic Expansion loop
│   │
│   ├── data_collection.py               # Phase 2: PubMed crawler (Entrez API, pagination)
│   ├── preprocessing.py                 # Phase 2: PySpark NLP pipeline (NER, PubChem normalization)
│   ├── embedding_training.py            # Phase 2: Word2Vec/FastText/GloVe + LHS hyperparameter search
│   ├── metric_generation.py             # Phase 2 & 3: Cosine/Euclidean/score computation, Top-N rankings
│   │
│   ├── model_selection.py               # Phase 3: Candidate evaluation; selects best model by mean lead time
│   ├── extract_report_years.py          # Phase 3 & 4: LLM-based ground truth generator (Qwen2.5-14B)
│   │
│   ├── model_evaluation.py              # Phase 4: Incremental test inference + full LKD metric suite
│   ├── reporting.py                     # Phase 4: Visual report generator (PCA, timelines, convergence plots)
│   │
│   ├── contextualization.py             # Phase 5: BioMistral-7B XAI hypothesis generator
│   │
│   ├── llm_utils.py                     # Shared: GGUF model manager (auto-download via HuggingFace Hub)
│   ├── utils.py                         # Shared: logging, checkpoint I/O, disease name normalization
│   │
│   └── calculate_performance_metrics.py # Standalone CLI tool: recompute metrics from existing rankings
│
├── data/                                # Generated data (gitignored)
│   ├── <disease>/                       # Per-disease outputs (corpus, models, validation, reports, plots)
│   ├── pubchem_data/                    # PubChem flat files (CID-Title, CID-Synonym-filtered)
│   ├── compound_whitelist.txt           # ChEMBL × PubChem therapeutic compound whitelist (cached)
│   └── analogies.txt                    # Word analogy evaluation sets (grammar + biomedical sections)
│
├── artifacts/                           # Pipeline checkpoints (JSON)
├── logs/                                # Execution logs
├── models/                              # Downloaded GGUF LLM weights
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Acknowledgments

This project extends and systematizes the methodologies presented in:

- [Tshitoyan et al., *Nature* (2019)](https://www.nature.com/articles/s41586-019-1335-8) — Unsupervised word embeddings capture latent knowledge from materials science literature.
- [Berto et al., *Expert Systems with Applications* (2024)](https://doi.org/10.1016/j.eswa.2024.123566) — Adaptation of the LBD methodology to medicine.

LLM components use:
- [BioMistral-7B](https://arxiv.org/abs/2402.10373) — open-source biomedical LLM for contextualization.
- [Qwen2.5-14B-Instruct](https://arxiv.org/abs/2412.15115) — high-performance instruction-tuned LLM for ground truth extraction.
