# How Fast Is Fashion

**A temporal trend-intelligence pipeline for fast-fashion retail.** Collects monthly snapshots of retailer imagery, normalizes garments into structured attributes via multi-modal LLM perception, assigns trend labels through two parallel signals (rule engine + fashion-CLIP retrieval — a third, a fine-tuned LoRA classifier, is deliberately deferred past v1.0), tracks trend lifecycles over a rolling 24-month window, and ranks inventory against user preference profiles with lifecycle-aware scoring.

Primary data window: `2024-02` through `2026-02`. Primary retailer for v1: **Zara**.

> **Full design spec:** [ARCHITECTURE.md](ARCHITECTURE.md) — module boundaries, data contracts, revision log.

---

## Why this exists

Fashion is temporal. A recommendation that shows you "quiet luxury" items *after* the trend has peaked is worse than no recommendation — it reinforces declining inventory. Retail recsys rarely encode *when* a trend is rising, peaking, or dying. This project tests whether explicit trend-lifecycle awareness improves recommendation quality vs. attribute-match alone.

---

## Architecture at a glance

```
Raw retail snapshots
  → LLM perception (image + text → structured attributes)
  → Two parallel trend signals (v1.0 scope):
       • Rule engine (symbolic, over normalized attributes)
       • FashionCLIP + Qdrant k-NN (vector retrieval over labeled refs)
       • Student LoRA (PaliGemma, fine-tuned) — deferred past v1.0, see below
  → Monthly aggregation → Lifecycle classification
  → Ranking (attribute-only baseline vs. lifecycle-aware, synthetic profiles — real user data also deferred past v1.0)
  → Insight layer + Streamlit interface
```

---

## Status

| Stage | Module | Status |
|---|---|---|
| Data ingestion | `src/fashion_forensics/mining/` | shipped |
| Teacher labeling (multi-modal LLM) | `src/fashion_forensics/normalization/` | shipped |
| Attribute schema | `configs/trend_rules.yaml` | shipped |
| Rule engine | `src/fashion_forensics/nlp/trend_engine.py` | shipped |
| **Retrieval: fashion-CLIP + Qdrant k-NN** | `src/fashion_forensics/retrieval/` | **shipped (v4)** |
| Automated reference-image curator (scrape/review/promote) | `src/fashion_forensics/curation.py` | shipped |
| Time-series aggregation | `src/fashion_forensics/nlp/time_series_aggregation.py` | shipped |
| Lifecycle classifier | `src/fashion_forensics/nlp/lifecycle_classifier.py` | shipped |
| Ranking engine (A/B) | `src/fashion_forensics/nlp/ranking_engine.py` | shipped — synthetic profiles only; real user data deferred past v1.0 |
| Streamlit app (Fashion / Lab / Curate) | `src/fashion_forensics/app/` | shipped |
| Student LoRA classifier | `src/fashion_forensics/training/lora.py` | scaffold only — **deliberately deferred past v1.0** (two signals validate the core hypothesis; the reference corpus is also too small to fine-tune on as-is, e.g. `office_siren` has only 28 labeled images) |

See the [ARCHITECTURE.md revision log](ARCHITECTURE.md#12-architecture-revision-log) for the full history (v1–v17).

---

## Quickstart — retrieval pipeline end-to-end

The fashion-CLIP + Qdrant k-NN trend classifier (§3.5) is the most complete stage. To reproduce:

```bash
# 1. Clone and set up a Python 3.11+ virtualenv
python3.11 -m venv .venv
.venv/bin/pip install -e '.[training,dev]'

# 2. Configure env (copy and fill in LLM / Langfuse keys if needed)
cp .env.example .env

# 3. Labeled reference images live under:
#    data/02_reference_corpus/labeled/{basics,mob_wife,office_siren,quiet_luxury}/
#    Drop more images into any subfolder; the pipeline picks them up automatically.

# 4. Embed the reference corpus into local Qdrant
.venv/bin/python scripts/embed_reference_corpus.py

# 5. Run LOOCV evaluation, log to MLflow
.venv/bin/python scripts/eval_trend_classifier.py --k 5 --vote shepard

# 6. Inspect results — narrative notebook with inline images + confusion matrices
.venv/bin/jupyter lab notebooks/03_trend_classification.ipynb
```

MLflow UI: `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`

---

## Notebooks

| Notebook | What it covers |
|---|---|
| [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Monthly coverage, sample visualization, data QA |
| [02_attributes_and_model.ipynb](notebooks/02_attributes_and_model.ipynb) | Teacher labeling, prompt-iteration lab |
| [03_trend_classification.ipynb](notebooks/03_trend_classification.ipynb) | fashion-CLIP + Qdrant k-NN: LOOCV eval, voting ablation, UMAP, calibration diagrams, reference-curation investigation |
| [04_ranking_evaluation.ipynb](notebooks/04_ranking_evaluation.ipynb) | Ranking engine (§3.9): weight sensitivity, Model A vs. B cross-profile comparison |

---

## Tech stack

- **Python 3.11+**, package layout via hatchling
- **fashion-CLIP** (`patrickjohncyh/fashion-clip`) — domain-tuned CLIP checkpoint
- **Qdrant** (local on-disk) — vector DB with named-vector support for hybrid search
- **transformers / torch** (MPS / CUDA / CPU)
- **MLflow** (sqlite backend) — experiment tracking for retrieval ablations and training runs
- **Langfuse** — LLM observability for the perception stage
- **DuckDB + pandas / polars** — analytical data layer
- **Streamlit** — interface (v1)
- **ruff / pytest** — lint + tests, gated in CI (see [.github/workflows/ci.yml](.github/workflows/ci.yml))

---

## Development

```bash
# Lint + format
.venv/bin/ruff check
.venv/bin/ruff format

# Tests
.venv/bin/pytest tests/

# CI runs all three on every push and pull request to main
```

---

## License

TBD.
