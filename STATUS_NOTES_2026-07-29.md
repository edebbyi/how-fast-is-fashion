# Repo status notes — 2026-07-29 (updated 2026-08-09)

Snapshot taken by reading the repo in detail (README, ARCHITECTURE.md, source tree, configs, data, git history). This is a point-in-time read, not a living doc — check `git log` and `ARCHITECTURE.md`'s revision log for anything newer.

## Update — 2026-08-09

Two PRs merged on `main` since the 2026-07-29 snapshot (local `main` was 6 commits behind; fast-forwarded to `c742116`):

- **PR #6 `feat/defensibility-bootstrap-vanilla-clip`** (merged 2026-08-04) — added `--bootstrap` flag to `eval_trend_classifier.py` (10k resamples, 95% CIs on accuracy/macro-F1/ECE, logged to MLflow) plus `--embedding-model`/`--collection` flags to run a vanilla-CLIP baseline in a separate Qdrant collection. Result at 74 refs, hybrid α=0.95: fashion-CLIP 0.878 acc (CI 0.797–0.946), ECE 0.074 vs. vanilla-CLIP 0.824 acc (CI 0.730–0.905), ECE 0.111. Fashion-CLIP wins directionally but CIs overlap at n=74; kept fashion-CLIP anyway since the calibration gap (ECE) matters more — similarity feeds the confidence threshold directly.
- **PR #7 `feat/mlflow-runs-export`** — new `scripts/export_mlflow_runs.py`, dumps all MLflow runs to `data/03_shared/runs/trend-classifier-eval_runs.{csv,md}` (30 runs, back to 2026-04-20).

**This resolves open thread #2 below** (accuracy drift as corpus grew 59→74 refs): with CIs in hand, the 67-ref (0.896) and 74-ref (0.878) points are statistically indistinguishable — the "drift" is very likely noise at this sample size, not a real regression. Not worth chasing further; corpus-growth effort should shift to breadth (new trend classes) over more images in existing classes.

Other open threads (#1, #3, #4, #5) are unchanged — untouched since 2026-07-29.

### New design thread — catalog-wide classification (feeds into #1, time-series aggregation)

Confirmed by reading `src/fashion_forensics/retrieval/classifier.py`: `predict_trend()` already supports `open_set_threshold` → `open_set_unknown`, but it is **only ever invoked from `eval_trend_classifier.py`'s LOOCV loop against the 74-image reference corpus** — nothing has run it against the actual monthly catalog yet. The normalized catalog (`data/01_data_audit/normalization/outputs/*.jsonl`) is small: **54 records total across 23 months** (2024-02→2026-01, minus the 2025-10 gap), ~2.3 items/month.

Shape of the work, as scoped in conversation:
1. New catalog-classification step: run every normalized catalog record through `predict_trend` against the existing reference Qdrant store, applying the open-set threshold (0.651 suggested) so sub-threshold items get `trend_pred="unknown"` rather than being forced into the nearest class.
2. **View A — catalog snapshot**: distribution across all trends (incl. `unknown`) over the *entire labeled catalog*, not just the reference set — "what does Zara's page actually look like."
3. **View B — cross-trend comparison over time**: each trend's catalog share (e.g. mob_wife = 33%) tracked by month, using raw **counts, not percentages** — at ~2 items/month, percentages would be near-meaningless noise.
4. **Unknown as first-class, not absence**: zero-count trends/months should log/aggregate as `0`, not a missing key — same "always log the metric" pattern PR #6 established for bootstrap CIs, extended to MLflow logging for the aggregation step.

**Cost-driven scope decision**: start with **image-only** classification (no LLM calls) over the full catalog rather than hybrid. Checked the numbers — `data/01_data_audit/cleaned/` (pre-LLM) has **2,029 records across 23 months**, but `normalization/outputs/` (post-LLM, what hybrid's text tower needs) only has **54 (~2.7%)**. Normalizing the other ~1,975 would be real LLM spend. Image-only eval accuracy (0.865 at 74 refs) is only ~1.3pp below hybrid (0.878), so image-only k-NN over the full 2,029-record cleaned catalog is the cheap first pass; normalizing the rest for hybrid is a later call once the image-only distribution is in hand.

This is the concrete design for §3.7 (time-series aggregation) — next step is implementation.

## What this project is

`how-fast-is-fashion` (package: `fashion-forensics`) is a temporal trend-intelligence pipeline for fast-fashion retail. It ingests monthly Zara product imagery (2024-02 → 2026-02 window), extracts structured garment attributes via a multi-modal LLM, assigns trend labels via three parallel signals (rule engine, FashionCLIP + Qdrant k-NN retrieval, and a planned LoRA classifier), and — eventually — tracks trend lifecycle (rising/persistent/declining) to power lifecycle-aware ranking vs. an attribute-only baseline.

The core hypothesis under test: does explicit trend-lifecycle awareness improve recommendation quality over plain attribute matching?

`ARCHITECTURE.md` is the source of truth — 12 sections mapping module boundaries, data contracts, and a detailed revision log (v1→v7). Treat it as authoritative over this file for design rationale.

## Current pipeline status (per README, cross-checked against code)

| Stage | Module | Status |
|---|---|---|
| Data ingestion | `mining/miner.py` (1118 lines) | shipped |
| Teacher labeling (multi-modal LLM) | `normalization/normalizer.py` (415 lines) | shipped |
| Attribute schema | `configs/trend_rules.yaml` | shipped |
| Rule engine | `nlp/trend_engine.py` (88 lines) | shipped |
| Retrieval: fashion-CLIP + Qdrant k-NN | `retrieval/` (embedder, qdrant_store, classifier) | **shipped, most mature stage (v7)** |
| TF-IDF analytics engine | `nlp/tfidf_engine.py` (258 lines) | shipped, not yet in README's table (added after README last touched) |
| Student LoRA classifier | `training/lora.py` (352 lines) | scaffold only — import guards for optional torch/transformers deps, no trained adapter yet |
| Time-series aggregation (§3.7) | — | planned; `tfidf_engine.py` provides primitives (`monthly_attribute_tfidf`, `trend_signature`, `lifecycle_curve`) that plug in here but the aggregation stage itself isn't built |
| Lifecycle classifier (§3.8) | — | planned |
| Ranking engine A/B (§3.9) | — | planned |
| Streamlit app | `app/app.py` (189 lines) + 6 components (chart/detail/grid/history/performance/query, ~600 lines combined) | more built out than README's "scaffold only" suggests — worth a closer look before trusting the README label |

## Retrieval pipeline — the most complete stage

- Architecture: per-reference k-NN over a local on-disk Qdrant collection (not mean-pooled exemplars) — keeps explainability (returns matched reference filenames).
- Dual-encoder hybrid: fashion-CLIP image tower + text tower on LLM-normalized attribute captions, fused at inference with weight α. Peak accuracy came from α=0.95 (95% image, 5% text).
- Voting: `shepard` (similarity-weighted, default) vs `majority`. Shepard has better-calibrated confidence (lower ECE).
- Reference corpus (`data/02_reference_corpus/labeled/`): **74 images** across `mob_wife` (25), `office_siren` (24), `quiet_luxury` (25); `basics` folder exists but is **empty by design** (it's meant to be populated and classified purely by retrieval, not by rule grammar, since "absence of trend signal" can't be expressed as a rule).
- `data/02_reference_corpus/attributes.jsonl` — 74 records, versioned artifact read by the embedder at ingestion time.
- Accuracy trend across corpus growth (hybrid α=0.95, LOOCV): 35 refs → 0.80 (v4) → 59 refs → 0.915 (v5, peak) → 67 refs → 0.896 → 74 refs → 0.878 (v7, latest). Accuracy has been *drifting down* as the corpus grows past 59, while calibration (ECE) has been improving (0.077 → 0.049) — more images, more honest but slightly less accurate predictions. Worth watching if more references get added.
- Text-only baseline degraded notably as the corpus grew (0.814 → 0.689) — attributed to token overlap (many refs now share similar attribute descriptions like "white wide-leg pants"), so the hybrid lift from text is narrowing (+8.4pp → +1.3pp).
- Open-set threshold auto-tunes from the 5th percentile of correct-prediction max-similarity in LOOCV; currently suggested at 0.651.

## Trend taxonomy (`configs/trend_rules.yaml`, v2)

- `trend_rule_version: v2`. Three sections: `trends` (actively supported, need 15+ ref images), `deferred_trends` (rules written, no images yet), `excluded_trends` (rejected — over-match or no garment-level discriminators).
- Active `trends`: `quiet_luxury`, `mob_wife`, `office_siren`, `basics` — only 4 of the originally-designed 12-trend taxonomy. The other 9 (cottagecore, coquette_balletcore, y2k_revival, utility_cargo, boho_revival, etc.) are deferred pending reference-image curation.
- Rule grammar: `any_of` blocks (OR), implicit AND within a block, `details_any_of` / `details_all_of` for the closed vocabulary.

## Data layer

- `data/01_data_audit/`: monthly records → cleaned → normalized (23-24 files each stage).
- **Coverage gap**: normalization outputs run 2024-02 through 2026-01 but **skip 2025-10** — only 23 files where a full run would have 24. This matches ARCHITECTURE.md §11's documented risk ("per-month coverage gaps... decide whether to downsample, tolerate, or drop sub-threshold months") — but it's not clear from the repo whether this specific gap has been triaged yet. Worth asking about before building time-series aggregation on top of it.
- `data/03_shared/taxonomy/trend_taxonomy_2024_2026.json` — shared taxonomy artifact.
- MLflow tracking is sqlite-backed (`mlflow.db`, gitignored presumably); Langfuse for LLM observability.

## Tests

- 777 lines across 5 test files: `test_benchmark.py` (255), `test_state.py` (124), `test_tfidf_engine.py` (206), `test_trend_engine.py` (138), `test_cleaning.py` (54).
- `test_tfidf_engine.py` (14 tests per the commit message) covers the newest module — token disambiguation, TF-IDF correctness, per-trend discriminative tokens, lifecycle curve smoothing.
- No `.venv` present in the working tree right now — didn't run the suite live, just read it.
- CI (`.github/workflows/ci.yml`): ruff check, ruff format --check, pytest — on every push/PR to main, Python 3.11.

## Git history (most recent 11 commits, main branch)

```
c744dc9 Merge pull request #5 from edebbyi/feat/reference-expansion-67
d0da174 Add 7 mob_wife images to rebalance classes (74 total references)
d3928b9 Expand reference set: 59 -> 67 images, headline accuracy 0.896 at hybrid alpha=0.95
a7aa8a1 Merge pull request #4 from edebbyi/feat/taxonomy-hygiene
04be293 Taxonomy hygiene: align names, add basics class, defer unsupported trends
8f3b937 Merge pull request #3 from edebbyi/feat/tfidf-engine
ca0a723 TF-IDF engine: monthly attribute analytics, trend signatures, lifecycle curves
4ae20cf Merge pull request #2 from edebbyi/feat/hybrid-retrieval-pathb
e3249f0 Section 12: "Wire" -> "Use" wording tweak
d901e72 Notebook 03: plain-English pass, drop em-dashes and ML jargon
5f38c64 Notebook 03: plain-language rewrite, unified green/brown/lime palette, strongest-config example queries
84f2a9c Path-B: hybrid retrieval with LLM-normalized captions, peak accuracy 0.915
e74ea4c Initial commit: temporal trend-intelligence pipeline for fast-fashion retail
```

Working tree is clean on `main` as of this read; no uncommitted changes.

Note: the git log shows a `feat/tfidf-engine` merge (TF-IDF engine, 2026-04-29) that isn't reflected in `ARCHITECTURE.md`'s revision log (which stops narrating new work at v7, dated 2026-04-29 same day as the reference-expansion work). The TF-IDF module may be undocumented in the architecture doc's revision history — worth a follow-up entry if that doc is meant to stay authoritative.

## Open threads / natural next steps (synthesized, not directives)

1. **Time-series aggregation (§3.7) is the next unbuilt stage** — `tfidf_engine.py` already provides the primitives (`monthly_attribute_tfidf`, `trend_signature`, `lifecycle_curve`); the aggregation module itself (turning trend labels into monthly frequency tables) doesn't exist yet.
2. **Reference corpus expansion has hit diminishing/negative returns on raw accuracy** (0.915 → 0.878 as corpus grew 59→74) even as calibration improved — the next expansion round might warrant checking per-class confusion rather than just adding more images blindly. **Update 2026-08-10: confirmed this the hard way** — see below. Adding volume to `basics`/`quiet_luxury` made their mutual confusion worse, not better.
3. **9 of 12 taxonomy trends are still deferred** — no reference images curated for cottagecore, y2k_revival, boho_revival, etc.
4. **The 2025-10 normalization gap** hasn't been visibly addressed and will matter once monthly aggregation/lifecycle work starts. Also now confirmed at the *cleaned* stage, not just normalization — `data/01_data_audit/cleaned/` has no 2025-10 file either (23 of 24 possible months).
5. **Streamlit app (`app/`) looks more built-out in code (≈600+ lines across components) than the README's "scaffold only" label implies** — may be worth verifying its actual functional state before relying on that label.

## Update — 2026-08-10

**`.git` corruption on the original checkout.** `git status` started failing with `fatal: not a git repository` — turned out `.git/HEAD` and several other files from the original Jul 29 checkout (`packed-refs`, `config`, `hooks`, `info`, `logs`, `refs`) had correct sizes in directory listings but read back empty (confirmed via raw Python byte reads, not just `git`/`cat`) — a real filesystem-level fault, not a sandbox artifact (reproduced in the user's own terminal too). Fixed `HEAD` by hand (its content is a fixed, universal string for a repo on `main`), but `packed-refs` isn't recoverable the same way, so rather than keep patching `.git` internals blindly, did a fresh clone instead: **`/Users/esosaimafidon/Documents/GitHub/how-fast-is-fashion-fresh`**, now the working checkout, on branch `feat/catalog-image-classification`. The original directory is left untouched (not deleted) in case Disk Utility → First Aid on the underlying drive is worth running later — recommended but not yet done. The `.env` file (API keys) also turned out to be missing/never-existed in this checkout, possibly related to the same disk issue — regeneration needed later from `.env.example` (`GEMINI_API_KEY`, `LANGFUSE_*`, `MLFLOW_TRACKING_URI`) before any LLM-touching work (normalization) can run again. Not a blocker for retrieval/classification work, which needs no API keys.

**Environment set up in the fresh clone**: Python 3.11.9 (via pyenv, not the system's Homebrew 3.14 — torch wheels don't support 3.14 yet), `.venv` with `pip install -e ".[training,dev]"`. Local MLflow server running on **port 5001**, not the documented default 5000 — macOS's AirPlay Receiver (ControlCenter process) squats on 5000 by default, which produces a confusing `403` rather than connection-refused. Start it with: `.venv/bin/mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db`, then `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` on any script invocation.

**New script: `scripts/classify_catalog.py`** (committed `3170d4a`). Runs image-only k-NN classification over the *entire* cleaned catalog (2,029 records, 23 months) rather than just the 74/135-image reference corpus — answers "what does the catalog actually look like" rather than "how accurate is the classifier." Catalog images aren't stored anywhere (gitignored, never downloaded) — the script downloads+caches them from the Wayback Machine URLs already in the cleaned records, same cache layout `mining/miner.py` uses. Open-set threshold auto-calibrates (5th percentile of correct-prediction max-similarity, image mode specifically) unless passed explicitly; below it, items are labeled `"unknown"` rather than forced into a class. Outputs a catalog-wide snapshot and a month × trend count table, zero-filled across every configured trend + `unknown` (raw counts, not percentages — catalog averages ~2 items/month, so percentages would be noise). Smoke-tested on 2 months (100 records, 97 classified, 72% `unknown` — expected, since the reference corpus only covers 3–4 narrow aesthetics against a generic catalog). Full 23-month run not yet done (~45–60 min, mostly image-download time) — paused to fix the `basics` class first (see below).

**`basics` class populated, then expanded — confusion made worse by more data.** `basics` was defined in `configs/trend_rules.yaml` as an active trend but had zero reference images (empty by design, but never actually populated). Added 40 images, then 43; separately expanded `quiet_luxury` 25→43 to try to fix the resulting confusion. Image-only LOOCV: **74 refs (3-class) 0.865 acc → 114 refs (+basics) 0.763 → 135 refs (+more basics, +more QL) 0.726.** `basics`↔`quiet_luxury` is the dominant confusion pair and it *grew* with more data (12→19 pairwise misclassifications) — looks structural (quiet luxury is itself defined by understated minimalism, so it's inherently close to generic basics in embedding space) rather than fixable by volume. One genuine improvement: ECE dropped 0.110→0.078 (better-calibrated despite being less accurate). Full writeup with confusion matrix and options going forward: `notebooks/03_trend_classification.ipynb` §14 (committed `c60b511`), corpus data committed `0a22c56`. MLflow: experiment `trend-classifier-eval`, both the 114-ref and 135-ref runs logged.

**Not yet decided**: whether to (1) curate `quiet_luxury`/`basics` harder for the taxonomy's actual stated discriminators rather than adding volume, (2) try hybrid mode for just this pair, or (3) accept the overlap as a documented limitation. Also still pending: the full 23-month catalog classification run, and committing/reviewing `data/03_shared/catalog_distribution/` output (currently only a 2-month smoke-test result, not committed).
