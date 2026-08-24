# How Fast Is Fashion

**A temporal trend-intelligence pipeline for fast-fashion retail.** Collects monthly snapshots of retailer imagery, normalizes garments into structured attributes via multi-modal LLM perception, assigns trend labels through three parallel signals (rule engine + fashion-CLIP + LoRA), tracks trend lifecycles over a rolling 24-month window, and ranks inventory against user preference profiles with lifecycle-aware scoring.

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
  → Three parallel trend signals:
       • Rule engine (symbolic, over normalized attributes)
       • FashionCLIP + Qdrant k-NN (vector retrieval over labeled refs)
       • Student LoRA (PaliGemma, fine-tuned) — planned
  → Monthly aggregation → Lifecycle classification
  → Ranking (attribute-only baseline vs. lifecycle-aware)
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
| Student LoRA classifier | `src/fashion_forensics/training/lora.py` | scaffold only |
| Time-series aggregation | | planned |
| Lifecycle classifier | | planned |
| Ranking engine (A/B) | | planned |
| Streamlit app | `src/fashion_forensics/app/` | scaffold only |
| Trend curation (scrape + human review) | `src/fashion_forensics/curation/` | shipped |

See the [ARCHITECTURE.md revision log](ARCHITECTURE.md#12-architecture-revision-log) for details on v1–v4.

---

## Quickstart — retrieval pipeline end-to-end

The fashion-CLIP + Qdrant k-NN trend classifier (§3.5) is the most complete stage. To reproduce:

```bash
# 1. Clone and set up a Python 3.13 virtualenv
python3.13 -m venv .venv
.venv/bin/pip install -e '.[training,dev]'

# 2. Configure env (copy and fill in LLM / Langfuse keys if needed)
cp .env.example .env

# 3. Labeled reference images live under:
#    data/02_reference_corpus/labeled/{mob_wife,office_siren,quiet_luxury}/
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

## Trend curation — scrape, review, promote to the reference corpus

Grows `data/02_reference_corpus/labeled/{trend}/` (the fashion-CLIP retrieval
references, §3.5) from live trend-site imagery instead of hand-collecting
files. A human always approves before anything joins the corpus:

```
configs/curation.yaml (trend-site URLs)
  → scraper stages candidate images + source attribution → review queue
  → Streamlit review app: approve (pick a trend) or reject
  → approved images land in data/02_reference_corpus/labeled/{trend}/
    + data/02_reference_corpus/attributions.jsonl (source URL, page, alt text)
```

```bash
# 1. Point configs/curation.yaml at real trend-site pages (editorial
#    roundups / lookbooks — not retailer product listings, those go
#    through configs/mining.yaml). The shipped file has placeholder URLs.

# 2. Scrape into the review queue (data/04_curation/)
.venv/bin/python scripts/scrape_trend_sites.py

# 3. Start the review backend
.venv/bin/uvicorn fashion_forensics.curation.api:app --port 8010

# 4. Start the review app and approve/reject
.venv/bin/streamlit run src/fashion_forensics/curation/review_app.py
```

Every candidate keeps its source name, page URL, and image URL through
approval, so `attributions.jsonl` is a full provenance trail for the
reference corpus. Rejected/duplicate images never touch the labeled corpus.
Content-hash dedup means re-running the scraper is safe — it only queues
images it hasn't seen before.

**Pinterest** isn't wired up yet — Pinterest doesn't offer straightforward
scraping (ToS-gated), so pulling boards would go through their official API
instead. The queue schema already carries a generic `source_name`/`source_url`
per candidate, so adding a Pinterest source later is a new fetcher, not a
redesign.

## Notebooks

| Notebook | What it covers |
|---|---|
| [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Monthly coverage, sample visualization, data QA |
| [02_attributes_and_model.ipynb](notebooks/02_attributes_and_model.ipynb) | Teacher labeling, prompt-iteration lab |
| [03_trend_classification.ipynb](notebooks/03_trend_classification.ipynb) | fashion-CLIP + Qdrant k-NN: LOOCV eval, voting ablation, UMAP, calibration diagrams |

---

## Tech stack

- **Python 3.13**, package layout via hatchling
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
