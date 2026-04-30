# How Fast is Fashion — Architecture

## Purpose

This document defines the technical architecture for the `how-fast-is-fashion` system. It is intended for implementation support in Claude Code and should be treated as the source-of-truth for module boundaries, data flow, and execution responsibilities.

---

## 1. System Objective

Build a modular pipeline that:

1. collects monthly fashion product imagery over a rolling 24-month window
2. converts images into structured garment attributes
3. maps attributes into trend labels
4. models trend lifecycle behavior over time
5. ranks items against user preference profiles
6. surfaces lifecycle-aware insight before recommendation action

Primary analytical window:

- `2024-02` through `2026-02`

Primary retailer for v1:

- `Zara`

---

## 2. Top-Level Architecture

```text
Raw Retail Snapshots
    ↓
Data Ingestion + Metadata Normalization
    ↓
Teacher Labeling — PERCEPTION (multi-modal LLM: image + text → attributes)
    ↓
Attribute Schema Normalization
    ↓
        ┌────────────────────────────┬────────────────────────────────┐
        ↓                            ↓                                ↓
Trend Rule Engine          FashionCLIP + Qdrant k-NN     Student LoRA (PaliGemma)
(deterministic, over          (per-reference              (fine-tuned trend
 normalized attributes)        embeddings in vector DB;    classifier — eventually)
                               top-k weighted vote)
        ↓                            ↓                                ↓
        └──────── Three parallel trend signals per item ──────────────┘
                                     ↓
                        Monthly Trend Aggregation
                        (signals combined; agreement = strong; disagreement = quality flag)
                                     ↓
                        Lifecycle Classification
                                     ↓
                        Ranking Engine (A/B variants)
                                     ↓
                              Insight Layer
                                     ↓
                  Interface Layer (Notebook demo or Streamlit)
```

**Three parallel trend signals (key design decision)**

Trend assignment is NOT a single brittle pipeline — three independent signals run in parallel and feed the aggregator:

1. **Rule engine** (§3.6) — symbolic, deterministic, transparent. Reads normalized attributes (especially `details` from a closed trend-discriminating vocabulary) and applies rules from `configs/trend_rules.yaml`.
2. **FashionCLIP + Qdrant k-NN** (§3.5) — each labeled reference image is embedded with fashion-CLIP and stored individually in a Qdrant vector DB. Inference: embed the query, retrieve the top-k nearest labeled neighbors, run a similarity-weighted vote. Catches gestalt that attribute extraction misses, and returns the specific matched reference filenames for explainability.
3. **Student LoRA** (§3.4) — fine-tuned on Pinterest trend buckets, cross-domain test on Zara. Higher quality with proper data; FashionCLIP is the baseline it must beat.

The LLM in the perception stage does NOT score trends. Trend interpretation is delegated to these three downstream signals so each can be evaluated, swapped, or improved independently.

---

## 3. Module Graph

### 3.1 `data_ingestion`
**Responsibility:** Construct a temporally indexed image inventory.

**Inputs**
- archive snapshot URLs or live product pages
- monthly sampling targets
- retailer/category filters

**Outputs**
- local image files
- normalized metadata table

**Core functions**
- fetch archived monthly pages
- extract product image URLs
- persist images to `data/images/`
- emit metadata rows to `data/metadata/items.csv` or `items.jsonl`

**Required output schema**
```json
{
  "item_id": "zara_2024_02_0001",
  "source": "zara",
  "date": "2024-02",
  "year": 2024,
  "month": 2,
  "category": "dress",
  "image_path": "data/images/2024-02/zara_2024_02_0001.jpg",
  "product_url": "https://...",
  "snapshot_url": "https://...",
  "status": "ok"
}
```

**Notes**
- This layer must be deterministic and rerunnable.
- Missing months or sparse months should be logged, not silently dropped.

---

### 3.2 `teacher_labeling`
**Responsibility:** Generate high-quality pseudo-ground-truth garment attributes by fusing the product image with retailer-provided text metadata.

**Inputs**
- image paths
- retailer-provided text metadata (product name, description, composition, declared color/category)
- attribute prompt template
- attribute schema specification

**Outputs**
- raw teacher label records (with provenance + per-attribute confidence)
- normalized teacher label records (downstream of `attribute_schema`)

**Core functions**
- batch prompt a vision-language model with **image AND text metadata together**
- treat text composition as authoritative for material; treat image as authoritative for silhouette / sleeve / neckline / pattern; require visual confirmation for color and category
- record per-attribute provenance (`text`, `image`, `both`) and per-attribute confidence
- parse responses into structured JSON
- store raw and normalized variants for audit

**Suggested raw schema**
```json
{
  "item_id": "zara_2024_02_0001",
  "teacher_attributes": {
    "silhouette": {"value": ["flowy"], "source": "image", "confidence": 0.85},
    "material": {"value": ["linen"], "source": "text", "confidence": 0.99},
    "sleeve_style": {"value": ["puff_sleeve"], "source": "image", "confidence": 0.90},
    "neckline": {"value": ["square_neck"], "source": "image", "confidence": 0.80},
    "length": {"value": ["midi"], "source": "image", "confidence": 0.85},
    "color_profile": {"value": ["white"], "source": "both", "confidence": 0.95},
    "details": {"value": ["ruffle"], "source": "image", "confidence": 0.75}
  },
  "label_version": "v2",
  "model": "openai/gpt-4o-mini",
  "prompt_version": "fashion-forensics/teacher_labeling@v2"
}
```

**Why multi-modal (image + text)**
- Material composition: text is authoritative; vision is unreliable for fabric identification.
- Silhouette / sleeve / neckline / fit / pattern: image is authoritative; text often omits these.
- Color and category: both sources together provide a strong consistency signal — disagreement is a quality flag.
- Throwing away retailer text would discard ground truth that is freely available.

**Why per-attribute confidence**
- A single record-level confidence is uninformative. Different attributes have very different reliability profiles.
- Per-attribute confidence enables: (a) downstream filtering of low-confidence labels, (b) curating a clean training set for the student model, (c) routing low-confidence records to human review.

**Notes**
- Store raw LLM output separately from normalized output.
- Schema drift must be tracked by `label_version` AND `prompt_version` (Langfuse-managed).
- Every LLM call is traced through Langfuse with the input image attached for visual QA.

---

### 3.3 `attribute_schema`
**Responsibility:** Define the canonical attribute ontology used across labeling, training, inference, and ranking.

**Inputs**
- raw teacher labels
- manual schema decisions

**Outputs**
- controlled vocabulary
- normalization mappings
- validation rules

**Core artifacts**
- `attribute_schema.yaml`
- `normalization_rules.py`

**Example schema families**
- silhouette
- material
- sleeve_style
- neckline
- length
- color_profile
- garment_details

**Notes**
- This module is a contract layer.
- Downstream modules should never rely on unconstrained strings.

---

### 3.4 `student_model` (`trend_classifier_finetuned`)
**Responsibility:** Train a lightweight vision model to predict per-trend signals from product images. Compared head-to-head against the FashionCLIP zero-shot baseline (§3.5) — this is the experimental claim of the project.

**Inputs**
- training images
- normalized teacher labels
- LoRA/QLoRA config
- train/validation split definition

**Outputs**
- trained adapter weights
- validation metrics
- inference-ready prediction function

**Core functions**
- materialize training examples
- fine-tune PaliGemma with LoRA
- log training/eval runs to MLflow
- export adapter artifact

**Tracked metrics**
- training loss
- validation loss
- exact match
- partial overlap
- micro F1
- macro F1

**MLflow responsibilities**
- log parameters
- log metrics
- log artifact paths
- log dataset version and schema version

**Notes**
- This is the primary experimental ML component in v1.
- This model predicts attributes, not trends.

**Cross-domain risk (elevated from §1 caveat)**
- Pinterest training images are aesthetic/lifestyle shots; Zara test images are e-commerce shots (single garment, neutral background). The domain gap is large enough that naive fine-tuning may fail.
- Mitigations to evaluate, in order of preference:
  1. Add a small Zara-labeled fine-tune set (e.g., 100–200 images) so the student sees the target distribution.
  2. Preprocess Pinterest to isolate garments (background removal / crop).
  3. Apply domain-adversarial regularization during training.
- Track the train/test domain gap explicitly as a metric (e.g., feature-distribution distance between Pinterest and Zara embeddings).
- Watch for data leakage: any image present in BOTH the Pinterest training set and the Zara evaluation set must be removed.

---

### 3.5 `student_inference`
**Responsibility:** Run trained model against inventory to produce predicted structured attributes.

**Inputs**
- trained student adapter
- inventory images
- inference config

**Outputs**
- attribute predictions per item
- confidence data if available

**Example output**
```json
{
  "item_id": "zara_2024_02_0001",
  "student_attributes": {
    "silhouette": ["flowy"],
    "material": ["linen"],
    "sleeve_style": ["puff_sleeve"],
    "neckline": ["square_neck"],
    "length": ["midi"],
    "color_profile": ["white"],
    "details": ["ruffle"]
  },
  "model_version": "paligemma_lora_v3"
}
```

**Notes**
- Keep teacher and student outputs side-by-side for auditability.

---

### 3.5 `retrieval` — FashionCLIP + Qdrant k-NN trend classifier
**Status:** SHIPPED (v4, 2026-04-20).

**Responsibility:** Provide a zero-training, visually-grounded, *explainable* trend signal per item. This is the BASELINE that the fine-tuned LoRA classifier (§3.4) must beat in evaluation.

**Why this architecture (retrieval, not mean-pool cosine)**
- Earlier draft (v3) proposed averaging exemplar embeddings into one vector per trend and scoring items against each trend's mean. That throws away within-class variation and gives no explainability.
- Retrieval-based k-NN keeps every labeled reference as its own point. Inference returns **which specific reference images drove the prediction** — essential for QA, active learning, and human-readable trend tags on retail products.
- With small reference sets (10–15 per trend), k-NN over CLIP features is typically competitive with a LoRA fine-tune but has zero training cost and grows organically: adding more reference images = one embed + upsert.

**Module location:** `src/fashion_forensics/retrieval/`
- `embedder.py` — `FashionClipEmbedder` (wraps `CLIPModel` from transformers; L2-normalizes outputs so cosine = dot product)
- `qdrant_store.py` — `TrendQdrantStore` (local on-disk Qdrant collection, cosine distance, payload-filtered search for LOOCV)
- `classifier.py` — `predict_trend()` entry point (embed → search → vote → `TrendPrediction`)

**Inputs**
- reference corpus: `data/02_reference_corpus/labeled/{trend}/*.{jpg,jpeg,webp,png}`
- query: product image path
- `k` (default 5), `vote_method` (default `shepard`), optional `open_set_threshold`

**Outputs**
- `TrendPrediction{trend_pred, confidence, matched_refs[{filename, trend, score}], open_set_unknown}`

**Reference corpus schema**
- One subfolder per trend under `data/02_reference_corpus/labeled/`
- Folder name IS the trend label (must match `configs/trend_rules.yaml` taxonomy)
- 10+ images per trend recommended; more refs → better neighborhood purity
- v1 ships with 3 reference trends: `mob_wife` (10), `office_siren` (13), `quiet_luxury` (12). Taxonomy will expand toward the 12-trend `configs/trend_rules.yaml` list as references are curated.

**Qdrant schema**
- Collection name: `settings.qdrant_collection` (default `trends_ref_v1`)
- Local on-disk at `settings.qdrant_path` (default `data/02_reference_corpus/qdrant/`)
- Distance: cosine (vectors are L2-normalized at embed time)
- Payload per point: `{trend, filename, image_path, source}`

**Voting methods**
- `majority`: count labels among top-k neighbors; ties broken by insertion order (nearest-first). Confidence = mean cosine similarity of winning-class neighbors.
- `shepard` (default): each neighbor contributes its cosine similarity to its class's score; winner = class with max weighted mass. Confidence = winner's share of total similarity mass (posterior-like). Better-calibrated than majority and eliminates arbitrary tie-breaks.

**Open-set detection**
- When `open_set_threshold` is set: if `max(neighbor.score) < threshold`, flag `open_set_unknown=True`. This protects against force-labeling products that aren't in any known trend.
- Threshold should be tuned on held-out OOD data (deferred; see §11).

**Pipeline integration**
1. Ingest (once, offline): `scripts/embed_reference_corpus.py` walks the labeled/ folder, embeds every image with fashion-CLIP, upserts to Qdrant with trend+filename+path payload.
2. Inference (per retail product): `predict_trend(image_path, embedder, store)` returns the trend prediction + matched references. Append `{trend_pred, confidence, matched_refs, model_version, qdrant_collection}` to the normalized product record.
3. Evaluation: `scripts/eval_trend_classifier.py --k 5 --vote shepard` runs leave-one-out over the reference corpus, logs to MLflow under experiment `trend-classifier-eval`.

**Tracked eval metrics (MLflow)**
- Classification: accuracy, macro-F1, per-class precision/recall/F1
- Calibration: Expected Calibration Error (ECE)
- Retrieval quality: neighbor purity @ k, silhouette score (cosine)
- Artifacts: confusion matrix PNG, UMAP of the embedding space
- v5 baseline (59 refs, k=5, shepard, fashion-CLIP, hybrid α=0.95): accuracy 0.915, macro-F1 0.915, per-class recall {mob_wife 0.89, office_siren 0.95, quiet_luxury 0.90}
- v4 baseline for comparison (35 refs, image-only): accuracy 0.80, macro-F1 0.80, ECE 0.137, purity@5 0.63, silhouette 0.077

**Suggested output schema (appended to normalized product records)**
```json
{
  "record_id": "zara_2024_02_0001",
  "trend_retrieval": {
    "trend_pred": "office_siren",
    "confidence": 0.72,
    "matched_refs": [
      {"filename": "office-13.jpg", "trend": "office_siren", "score": 0.77},
      {"filename": "office-5.jpg",  "trend": "office_siren", "score": 0.66},
      {"filename": "mobwife-2.webp","trend": "mob_wife",       "score": 0.70}
    ],
    "open_set_unknown": false,
    "model": "patrickjohncyh/fashion-clip",
    "embedding_dim": 512,
    "qdrant_collection": "trends_ref_v1",
    "k": 5,
    "vote_method": "shepard"
  }
}
```

**Operating mode**
- Ingest is one-shot per reference corpus revision. Bump `qdrant_collection` (e.g., `trends_ref_v2`) when adding/removing reference images so old predictions remain reproducible against a frozen corpus version.
- Inference is per-product and fast (MPS/CUDA GPU on modern hardware; CPU is acceptable for batches up to ~10k).
- This signal is parallel to §3.6 (rule engine) — both feed §3.7 aggregation.
- Disagreement between rules and retrieval is informative — it identifies items for the active-learning gold set used to train §3.4.

**Interaction with schema versions (see "Deferred to v2" below)**
- v1 flat schema: the `trend_retrieval` block sits at the **record level** (one per SKU image). Matches today's flat multi-value attribute schema naturally — one image, one trend prediction.
- v2 nested schema (`product` + `outfit_components`): `trend_retrieval` belongs on the **`product` field** — retrieval operates on the whole product image and predicts the *overall* trend aesthetic, not the constituent components. A styled outfit photo with e.g. a quiet-luxury blazer and mob-wife accessories will receive one aggregate trend prediction, not per-component predictions. If per-component trend assignment becomes load-bearing later, it requires a segmentation step (YOLO/SAM crop → retrieval per crop) — explicitly out of scope for v4.

---

### 3.6 `trend_rule_engine`
**Responsibility:** Map normalized attributes to interpretable trend labels.

**Inputs**
- normalized attributes
- trend taxonomy
- deterministic mapping rules

**Outputs**
- trend labels per item
- rule trace or rationale

**Trend taxonomy v1**
- cottagecore
- coquette_balletcore
- quiet_luxury
- y2k_revival
- utility_cargo
- boho_revival
- `unmapped` (catch-all for items that match no rule above)

**Why `unmapped`**
- The trend list is editorial, not data-driven. The 2024-02 → 2026-02 Zara inventory may not exhibit all six, and it may exhibit patterns the taxonomy does not name.
- `unmapped` prevents force-fitting items to a label they do not belong to, which would corrupt downstream lifecycle analysis.
- Periodically review `unmapped` items via attribute clustering; if a stable cluster emerges, propose a new trend rule and bump `trend_rule_version`.

**Example rule**
```python
if "linen" in material and "puff_sleeve" in sleeve_style and "flowy" in silhouette:
    trend_labels.add("cottagecore")
```

**Suggested output**
```json
{
  "item_id": "zara_2024_02_0001",
  "trend_labels": ["cottagecore", "boho_revival"],
  "rule_trace": ["rule_cottagecore_v1", "rule_boho_v1"],
  "trend_rule_version": "v1"
}
```

**Notes**
- This is intentionally symbolic and deterministic.
- It functions as the interpretation layer, not the perception layer.

---

### 3.7 `time_series_aggregation`
**Responsibility:** Convert item-level trend labels into month-level frequency statistics.

**Inputs**
- item table with `date` and `trend_labels`
- active threshold configuration

**Outputs**
- monthly frequency table by trend
- active/inactive signals
- persistence windows

**Core computations**
- count items per month
- count trend-labeled items per month
- compute frequency = trend_count / inventory_count
- derive active months from threshold

**Example output**
```json
{
  "date": "2024-07",
  "trend": "boho_revival",
  "inventory_count": 60,
  "trend_count": 14,
  "trend_frequency": 0.2333,
  "is_active": true
}
```

**Notes**
- This module should be implemented as pure data transformation.
- Thresholds must be parameterized, not hard-coded.

---

### 3.8 `lifecycle_classifier`
**Responsibility:** Classify trend temporal behavior from aggregated monthly trend frequencies.

**Inputs**
- monthly trend frequency table
- lifecycle thresholds and heuristics

**Outputs**
- lifecycle state per trend
- lifecycle summary statistics
- recurrence indicators

**Lifecycle states**
- rising
- persistent
- seasonal_recurring
- declining
- inactive

**Candidate features**
- first active month
- last active month
- number of consecutive active months
- number of active periods
- peak month
- peak value
- slope around last N periods
- recurrence after inactivity

**Example output**
```json
{
  "trend": "boho_revival",
  "first_observed": "2024-03",
  "peak_month": "2024-07",
  "last_active": "2025-01",
  "active_month_count": 8,
  "state": "declining",
  "lifecycle_version": "v1"
}
```

**Notes**
- This is a rule-based classifier over derived time-series features.
- Do not oversell it as probabilistic survival analysis unless you actually implement that layer.

---

### 3.9 `ranking_engine`
**Responsibility:** Score and rank inventory items against user preference profiles.

**Inputs**
- item-level attributes
- item-level trend labels
- trend lifecycle states
- user preference profile

**Outputs**
- scored candidate items
- top-K ranked recommendations
- explanation payload per result

**User profile schema**
```json
{
  "user_id": "synthetic_user_01",
  "preferred_attributes": ["linen", "flowy"],
  "preferred_trends": ["boho_revival"],
  "preferred_category": "dress",
  "prefer_rising": true,
  "avoid_declining": true
}
```

**Model A (attribute-only baseline)**
```text
score_A =
    w1 * attribute_match
  + w3 * category_match
```

**Model B (trend + lifecycle-aware)**
```text
score_B =
    w1 * attribute_match
  + w2 * trend_match
  + w3 * category_match
  + w4 * trend_state_weight
```

**Why this contrast (revised from `+trend_state_weight only`)**
- The previous A/B differed only by a single weight, producing a tiny effect that was unlikely to register in evaluation.
- Sharper contrast (A = pure attribute match, B = attribute + trend + lifecycle) directly tests the project's core hypothesis: *does temporal trend awareness improve recommendation quality?*

**Trend state weights example**
- rising: `+1.0`
- persistent: `+0.7`
- seasonal_recurring: `+0.5`
- declining: `-0.3`
- inactive: `-1.0`

**Example ranked record**
```json
{
  "item_id": "zara_2025_06_0031",
  "score": 0.82,
  "rank": 1,
  "matched_attributes": ["linen", "flowy"],
  "matched_trends": ["boho_revival"],
  "trend_state": "rising",
  "category_match": true,
  "ranking_model_version": "model_b_v2"
}
```

**Notes**
- This is a content-based ranking system, not collaborative filtering.
- Use synthetic users for offline evaluation in v1.

---

### 3.10 `recommendation_eval`
**Responsibility:** Evaluate ranking quality offline.

**Inputs**
- ranked outputs
- synthetic user relevance rules
- top-K setting

**Outputs**
- per-user metrics
- aggregate model comparison metrics
- A/B comparison tables

**Metrics**
- Precision@K
- Recall@K
- F1
- trend alignment score
- diversity

**A/B comparison**
- Model A: attribute + trend
- Model B: attribute + trend + lifecycle state

**Notes**
- This module tests whether temporal awareness improves ranking quality.
- Log experiments and results to MLflow.

---

### 3.11 `insight_layer`
**Responsibility:** Generate pre-action contextual summaries for recommended items.

**Inputs**
- selected recommendation
- lifecycle summary for its dominant trend

**Outputs**
- trend insight payload
- chart-ready time-series data

**Example payload**
```json
{
  "trend": "boho_revival",
  "first_observed": "2024-03",
  "peak_month": "2024-07",
  "current_state": "declining",
  "active_month_count": 8,
  "summary_text": "This trend peaked in 2024-07 and is currently declining."
}
```

**Notes**
- This is not an LLM component in v1.
- Summaries should be deterministic and generated from structured fields.

---

### 3.12 `interface_layer`
**Responsibility:** Present ranked items and lifecycle context to the user or reviewer.

**v1 options**
- notebook-driven demo
- Streamlit app

**Primary views**
1. recommendation view
2. trend explorer
3. lifecycle visualization view

**Recommendation view must show**
- top ranked items
- score explanation
- associated trend insight
- mini time-series chart before action

**Notes**
- The interface should expose the ranking system and the critique layer, not just mimic ecommerce UX.

---

## 4. Notebook Allocation

Each stage notebook imports from `src/` and renders the narrative + plots for that stage. Code lives in modules; notebooks tell the story. A final `99_master_narrative.ipynb` stitches headline outputs from each stage into a single portfolio read.

### `01_data_audit.ipynb` (shipped)
- metadata loading
- monthly coverage checks
- sample visualization
- data QA

### `02_attributes_and_model.ipynb` (shipped)
- teacher labeling inspection
- normalization checks
- prompt-iteration lab (perception-only LLM)
- MLflow logging for student model runs

### `03_trend_classification.ipynb` (shipped, v4)
Walks the fashion-CLIP + Qdrant k-NN retrieval pipeline (§3.5):
- reference corpus preview with per-class image renders
- LOOCV eval with **both** voting methods (majority vs Shepard) side-by-side
- sample query renders: top-5 neighbors inline with color-coded correctness + cosine similarities
- confusion matrices side-by-side
- UMAP of the reference corpus in fashion-CLIP space (with silhouette score)
- reliability diagram (calibration) showing why Shepard's ECE is lower

### `04_trend_lifecycle.ipynb` (planned)
- trend rule engine validation
- monthly aggregation (§3.7)
- lifecycle classification (§3.8)
- lifecycle charts

### `05_ranking_and_eval.ipynb` (planned)
- synthetic user definitions
- score functions A/B
- ranking outputs
- Precision@K / Recall@K / F1
- MLflow logging for ranking experiments

### `06_demo_and_insight.ipynb` (planned)
- end-to-end walkthrough
- recommendation examples
- trend insight layer examples
- portfolio/demo outputs

### `99_master_narrative.ipynb` (planned)
- stitches headline outputs from each stage
- single-read portfolio story: mining → perception → trend classification → lifecycle → ranking → insights → generative predictions

---

## 5. Repository Layout

```text
how-fast-is-fashion/
├── data/
│   ├── 01_data_audit/                 # monthly records, normalization outputs
│   ├── 02_reference_corpus/
│   │   ├── labeled/                   # trend-labeled reference images, by class
│   │   │   ├── mob_wife/
│   │   │   ├── office_siren/
│   │   │   └── quiet_luxury/
│   │   └── qdrant/                    # local on-disk Qdrant DB (vector store)
│   ├── 03_shared/
│   └── 04_training/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_attributes_and_model.ipynb
│   └── 03_trend_classification.ipynb  # fashion-CLIP + Qdrant k-NN walkthrough
├── scripts/
│   ├── download_data.py
│   ├── embed_reference_corpus.py      # one-shot ingestion → Qdrant
│   ├── eval_trend_classifier.py       # LOOCV + MLflow logging
│   ├── seed_langfuse_prompts.py
│   └── smoke_normalize.py
├── src/
│   └── fashion_forensics/
│       ├── app/                       # Streamlit interface
│       ├── evaluation/                # benchmark harness
│       ├── mining/                    # data ingestion (§3.1)
│       ├── nlp/                       # trend mapping, trend engine
│       ├── normalization/             # attribute schema normalization (§3.3)
│       ├── retrieval/                 # §3.5: embedder, qdrant_store, classifier
│       ├── training/                  # §3.4: LoRA
│       ├── config.py
│       ├── db.py
│       ├── prompt_lab.py
│       └── tracking.py                # MLflow + Langfuse wrappers
├── configs/
│   ├── mining.yaml
│   └── trend_rules.yaml
├── tests/
├── mlflow.db                          # local MLflow tracking (sqlite)
├── README.md
└── ARCHITECTURE.md
```

---

## 6. MLflow Integration Points

### Experiment: `trend-classifier-eval` (§3.5 retrieval pipeline)
Log per LOOCV run of the fashion-CLIP + Qdrant k-NN classifier:

**Params**
- `embedding_model` (e.g. `patrickjohncyh/fashion-clip`)
- `k` (neighbors retrieved)
- `distance` (cosine)
- `vote_method` (`majority` | `shepard`)
- `n_refs` (size of reference corpus)

**Metrics**
- `accuracy`, `macro_f1`
- `precision_{class}`, `recall_{class}`, `f1_{class}` per trend
- `ece` (Expected Calibration Error)
- `neighbor_purity_at_{k}` (raw retrieval quality, pre-vote)
- `silhouette_cosine` (class separability in embedding space)

**Artifacts**
- `confusion_matrix.png`
- `umap.png` (2D projection of the reference corpus)

**Tags**
- `protocol=leave-one-out`, `classes=<comma-separated trend list>`

### Experiment: `lora-training` (§3.4 student model development)
Log during student model development:
- dataset version
- schema version
- LoRA params
- quantization settings
- learning rate
- epochs
- batch size
- loss
- F1 metrics
- adapter artifact path

### Experiment: ranking (§3.10 A/B)
Log during A/B ranking comparisons:
- scoring model version
- weight configuration
- active threshold version
- Precision@K
- Recall@K
- F1
- diversity
- trend alignment score

---

## 7. Data Contracts

The minimum merged item-level table used by downstream systems should contain:

```json
{
  "item_id": "zara_2025_06_0031",
  "date": "2025-06",
  "category": "dress",
  "image_path": "data/images/2025-06/zara_2025_06_0031.jpg",
  "teacher_attributes": {...},
  "student_attributes": {...},
  "trend_labels": ["boho_revival"],
  "trend_state": "rising"
}
```

This merged table is the core operating table for:
- ranking
- evaluation
- insight generation
- interface rendering

---

## 8. Execution Order

1. build monthly image inventory
2. label sample inventory with teacher model
3. normalize attribute schema
4. fine-tune student model
5. run student inference
6. assign trend labels via rule engine
7. aggregate monthly trend frequencies
8. classify lifecycle states
9. build ranking models A/B
10. evaluate ranking offline
11. expose recommendation + insight outputs in notebook or app

---

## 9. Non-Goals for v1

- direct supervised trend classifier
- collaborative filtering
- real user click or purchase logs
- online A/B testing
- multi-retailer production coverage
- full MLOps deployment stack

---

## 10. Design Principles

### Modularity
Each stage should be replaceable without requiring a full system rewrite.

### Determinism where possible
Use symbolic logic for mapping, scoring, and insight text in v1.

### Auditability
Preserve raw teacher outputs, normalized outputs, student outputs, and rule traces.

### Temporal awareness
Treat time as a first-class feature, not post-hoc decoration.

### Minimal defensible scope
Prefer a complete v1 with constrained coverage over a wider but partial build.

---

## 11. Data Hygiene and Quality

### SKU recycling and dedup
- The same Zara product may appear in multiple monthly snapshots. Counting it as a fresh item in each month inflates trend frequencies.
- Dedup strategy: hash the product image (perceptual hash, e.g. pHash) AND the normalized product name. Same hash across months = same SKU.
- Decide and document whether dedup is global (one record per SKU, dated by first appearance) or per-month (count once per month per SKU). Trend lifecycle interpretation depends on this choice.

### Wayback snapshot quality
- Some snapshots are partial (CSS broken, images missing, prices stripped). Filter at the cleaning stage and log skip reasons.
- Image quality varies by snapshot (CDN compression). Standardize: re-encode to a fixed size and quality before sending to the teacher / student.

### Train/test leakage (Pinterest ↔ Zara)
- Any image present in both the Pinterest training set and the Zara evaluation set must be removed from one side.
- Run a perceptual-hash check across both sets before training begins. Log overlap count to MLflow.

### Per-month coverage gaps
- Coverage is not uniform across the 25-month window (some months may have <60 clean items). Decide whether to: (a) downsample large months for parity, (b) tolerate uneven counts and weight by inventory size, or (c) drop sub-threshold months from lifecycle analysis.
- Document the choice; do not silently let imbalance bias the lifecycle classifier.

### Failure budgets
- Wayback fetch failure: log + retry with exponential backoff; never silently drop.
- VLM returns malformed JSON: log raw output to Langfuse, schedule for retry, do not insert null rows.
- Trend taxonomy evolution: bump `trend_rule_version` and re-label historical items rather than mutating the existing labels in place.

---

## 12. Architecture Revision Log

### v2 (2026-04-19)
- §3.2 reframed teacher labeling as **multi-modal** (image + text) with per-attribute provenance and confidence.
- §3.4 elevated the Pinterest → Zara cross-domain risk from caveat to first-class concern with concrete mitigations.
- §3.6 added an `unmapped` trend bucket to prevent force-fitting items to the editorial taxonomy.
- §3.9 sharpened the A/B contrast: A = attribute-only baseline, B = attribute + trend + lifecycle (was: B = A + trend_state_weight only).
- §11 added: data hygiene (SKU dedup, snapshot quality, leakage, coverage, failure budgets).

### v3 (2026-04-19, later same day)
- §2 reframed the pipeline diagram around **three parallel trend signals** (rule engine + FashionCLIP + LoRA) feeding a single time-series aggregator. Trend assignment is no longer a single brittle path.
- §3.2 LLM is reframed as **PERCEPTION ONLY** — it extracts attributes (including a closed-vocabulary `details` field whose tokens map to trends), but no longer scores trends itself. Trend interpretation is delegated downstream to dedicated components.
- §3.4 renamed `trend_classifier_finetuned`; positioned as the experimental claim that must beat the FashionCLIP baseline.
- §3.5 NEW: `trend_classifier_zeroshot` (FashionCLIP) — visual-similarity baseline using curated Pinterest exemplars per trend.
- §3.6 trend_rule_engine clarified as ONE of three parallel trend signals, not the sole interpretation layer.
- New `configs/trend_rules.yaml` defines the 12-trend taxonomy + closed `details` vocabulary + rule definitions.
- Notebook prompt-iteration lab in `notebooks/02_attributes_and_model.ipynb` uses the perception-only LLM prompt.

### v7 (2026-04-29, later) — Reference set expanded from 59 to 74 images

- Added 15 reference images across two batches: first round added 3 `office_siren` + 5 `quiet_luxury` (total 67). Then 7 `mob_wife` were added to rebalance.
- Counts after this expansion: `mob_wife` 25, `office_siren` 24, `quiet_luxury` 25, `basics` 0 (still empty by design). Total 74. Classes are now roughly balanced.
- Pipeline ran end-to-end after each batch: LLM teacher labeling (idempotent, only the new images each time), re-embedded all references to Qdrant, re-ran the 3-mode eval and α-sensitivity sweep.
- v7 metrics on 74 images at hybrid α=0.95: accuracy **0.878** (was 0.915 on 59), macro-F1 0.878, ECE **0.049** (was 0.077, calibration improved noticeably).
- Per-class recall at hybrid α=0.95: mob_wife **0.88** (was 0.89, held up after the rebalance), office_siren **0.92** (was 0.95), quiet_luxury **0.84** (was 0.90). The classes are now more uniformly balanced rather than office_siren dominating.
- Image-only baseline held steady (0.831 → 0.865). Text-only baseline dropped (0.814 → 0.689) because the larger reference set produces many similar attribute tokens (white wide-leg pants, button-down shirts, black tailored pieces) and text descriptions are less discriminating. The hybrid lift over image-only narrowed from +8.4pp to +1.3pp at this scale, but ECE improved 0.077 → 0.049 — predictions are more honest.
- Suggested open-set threshold settled at 0.651 (auto-tuned from LOOCV similarity distribution).

### v6 (2026-04-29) — Taxonomy hygiene + open-set threshold helper

- **Naming consistency.** `mobwife` (folder) renamed to `mob_wife` to match the `configs/trend_rules.yaml` key. Folder, `attributes.jsonl` records, notebook references, and prose in this doc all use `mob_wife` now. Image filenames inside `mob_wife/` keep their original names (file-level normalization is a separate cleanup); only the folder-as-trend-label changed.
- **`basics` positive class added.** New folder `data/02_reference_corpus/labeled/basics/` (empty for now). New entry in `configs/trend_rules.yaml` under `trends:` with empty `rule.any_of` (intentional — `basics` is assigned by §3.5 retrieval, not the §3.6 rule grammar, since the rule grammar can't express "absence of trend signals"). Lets unbranded staples receive a positive label instead of falling into `unmapped`.
- **Trim active taxonomy to what's supported.** `trends:` block now contains exactly four classes (`mob_wife`, `office_siren`, `quiet_luxury`, `basics`). The other nine were moved to a new top-level `deferred_trends:` block: rules and discriminators are preserved, but they're not loaded as active classes. Promote back to `trends:` once at least 15 reference images per class are curated. `trend_rule_version` bumped to `v2`.
- **Closed `details` vocabulary audit.** Scanned all 24 monthly normalizer outputs (2024-02 through 2026-01). Five of 34 configured tokens fired in real outputs (`gathered`, `lace_trim`, `gold_hardware`, `bow`, `gathered_volume`). Two genuine new tokens added to the vocab: `belted` (in `structural` cluster) and `rhinestone_trim` (in `glamoratti_hardware` cluster). Two LLM bugs found — the LLM occasionally emits cluster names (`sheer_lace_y2k`, `glamoratti_hardware`) as if they were tokens; this is a prompt-side issue tracked separately, not a vocab bug.
- **Open-set threshold helper.** `scripts/eval_trend_classifier.py` now computes a suggested threshold from the 5th percentile of correct-prediction max-similarity in LOOCV. Logged to MLflow as `suggested_open_set_threshold` and printed at the end of every eval run. Re-tunes automatically as new trends are added (the similarity distribution shifts when the class set changes). Below this threshold, `predict_trend` flags the result as `open_set_unknown`. v6 baseline: **0.651** at hybrid α=0.95.
- **Stale `captions.yaml` removed** (path-A leftover; replaced by `attributes.jsonl` in path-B).
- **Pipeline metrics unchanged** post-rename (accuracy 0.915, macro-F1 0.915, ECE 0.077 at hybrid α=0.95). The taxonomy work is structural; predictions and confusions are identical to v5.

### v5 (2026-04-25) — Hybrid retrieval (path-B) shipped, peak accuracy 0.915

- §3.5 retrieval pipeline now uses fashion-CLIP's **dual encoders** end-to-end. Every Qdrant point carries `image_vec` (vision tower on the photo) AND `caption_vec` (text tower on the LLM-normalized attribute description). At inference, query image and query attributes are embedded by the same dual towers; two parallel cosine searches run; scores are fused with weight α before voting.
- New `attributes_to_text(attrs) -> str` in `src/fashion_forensics/normalization/normalizer.py` — deterministic flatten of the §3.2 attribute schema into a fashion-CLIP-compatible caption string. Used on **both** reference and query sides so text-tower comparisons are same-distribution (no surface-form drift).
- New `normalize_image(image_path) -> dict` for ad-hoc single-image labeling decoupled from the monthly folder structure. Used by the reference-corpus normalizer.
- New `scripts/normalize_reference_corpus.py` with **prompt-version-aware idempotency**: each record carries the Langfuse prompt version it was labeled at; re-runs skip up-to-date entries, automatically re-label stale ones (no manual `--force` needed when the prompt evolves).
- New `data/02_reference_corpus/attributes.jsonl` committed as a versioned artifact — 59 records, same schema as the monthly normalization outputs. Source of truth that the embedder reads at ingestion time.
- `scripts/embed_reference_corpus.py` now reads `attributes.jsonl`, flattens via `attributes_to_text()`, embeds with fashion-CLIP's text tower, and writes real `caption_vec` (no longer the image-vec placeholder). Falls back to image_vec only for refs that lack an attributes record (with a warning).
- `scripts/eval_trend_classifier.py` adds `--mode {image, text, hybrid}` and `--alpha`. Each invocation logs a separate MLflow run with `search_mode` and `alpha` as params.
- v5 baseline metrics (59 refs, k=5, Shepard, LOOCV):

  | Mode | Accuracy | Macro F1 | ECE | mob_wife recall | office recall | quiet recall |
  |---|---|---|---|---|---|---|
  | image | 0.831 | 0.832 | 0.098 | 0.83 | 0.86 | 0.80 |
  | text | 0.814 | 0.813 | 0.098 | 0.89 | 0.86 | 0.70 |
  | hybrid α=0.6 | 0.831 | 0.831 | 0.113 | 0.83 | 0.81 | 0.85 |
  | **hybrid α=0.95** | **0.915** | **0.915** | — | 0.89 | 0.95 | 0.90 |

- α sensitivity curve (LOOCV macro-F1): plateau at 0.81–0.83 for α ≤ 0.6, climb from 0.7 onward, peak 0.915 at α=0.95, drop back to 0.83 at α=1.0. About 5% text-weight is the sweet spot — the method is not flat-fragile to the knob; the curve shape is informative.
- Notebook `03_trend_classification.ipynb` extended with sections 9–11: 3-mode LOOCV runner, mode comparison table with per-class recalls, side-by-side confusion matrices (image / text / hybrid), α-sensitivity sweep + plot, findings + caveats.
- `pyproject.toml`: added `openai>=1.0` (was used by normalizer.py but missing from declared deps).

### v4 (2026-04-20) — FashionCLIP baseline SHIPPED via retrieval architecture
- §3.5 architecture changed from **mean-pooled exemplar cosine** to **per-reference k-NN over Qdrant** (vector DB). Every labeled reference image is stored as its own point with `{trend, filename, image_path}` payload. Inference returns the specific matched reference filenames — explainability that mean-pooling couldn't provide.
- New module `src/fashion_forensics/retrieval/` with `embedder.py` (fashion-CLIP wrapper), `qdrant_store.py` (local on-disk Qdrant collection), `classifier.py` (`predict_trend()` entry point with pluggable voting).
- Two voting methods implemented: `majority` (count-based, ties broken by insertion order) and `shepard` (similarity-weighted; default — better-calibrated confidence, no arbitrary tie-breaks).
- New scripts: `scripts/embed_reference_corpus.py` (one-shot ingestion), `scripts/eval_trend_classifier.py` (LOOCV + MLflow logging).
- New MLflow experiment `trend-classifier-eval` with params (embedding_model, k, distance, vote_method, n_refs), metrics (accuracy, macro-F1, per-class P/R/F1, ECE, purity@k, silhouette), and artifacts (confusion matrix, UMAP).
- v1 reference corpus: 3 trends, 35 images (`mob_wife` 10, `office_siren` 13, `quiet_luxury` 12). Taxonomy will expand toward the 12-trend `configs/trend_rules.yaml` list as references are curated.
- New `notebooks/03_trend_classification.ipynb` renders the pipeline with image outputs: per-class reference preview, LOOCV with both voting methods, sample query + top-5 neighbors side-by-side with color-coded correctness, confusion matrices, UMAP, reliability/calibration diagram.
- Open-set detection wired in (`open_set_threshold` parameter); threshold tuning deferred until OOD product data is available.
- Python 3.14 venv rebuilt on 3.13 to unblock `torch` installation; `pyproject.toml` bumped to `torch>=2.5` and added `torchvision`, `qdrant-client`, `umap-learn`, `matplotlib`.
- v4 baseline metrics (35 refs, k=5, shepard): accuracy 0.80, macro-F1 0.80, ECE 0.137, purity@5 0.63, silhouette 0.077. office_siren is the cleanest signal (recall 0.92); mob_wife is the weakest (recall 0.70) and would benefit most from more reference images.

### Deferred to v2: nested `product` + `outfit_components` schema

Considered and deferred. Today's schema is **flat multi-value** — `material: [denim, silk]`, `color_profile: [blue, white, black]`, etc. — covering every garment in the styled outfit in a single list per attribute.

The deferred richer schema would be:

```json
{
  "product": { ...attributes of the SKU itself... },
  "outfit_components": [
    {"role": "top",      "category": "tops",  ...},
    {"role": "footwear", "category": "shoes", ...}
  ]
}
```

**Why deferred:** the rule engine and DuckDB queries get more complex (nested unnest, structured rule grammar) for limited trend-mapping benefit. Flat multi-value is enough for the rule engine; FashionCLIP catches anything visual the flat attributes miss.

**When to revisit (v2 trigger):** if downstream analysis shows the SKU-vs-styling distinction is actionable (e.g. "what colors did Zara *sell* in 2024 vs *style* in 2024"), or if shoes/accessories are systematically driving trend signals that aren't representable in flat attributes.

---

### v3 follow-on plan: FashionCLIP integration — SHIPPED (v4)

Delivered in v4 but with a different architecture than originally planned. Summary of what landed vs. the v3 plan:

| v3 plan | v4 actual |
|---|---|
| `score_item` against per-trend mean exemplar embedding | `predict_trend` against per-reference Qdrant k-NN with similarity-weighted (Shepard) vote |
| Flat parquet embedding cache | Local on-disk Qdrant collection with per-point trend+filename payload |
| Float similarity score per trend | Single predicted trend + confidence + matched-reference filenames (explainability) |
| Module: `trend_classifier_zeroshot.py` | Package: `src/fashion_forensics/retrieval/` (embedder + qdrant_store + classifier) |
| Torch 2.2 / transformers 4.38 | Torch 2.11 / transformers 5.5 (bumped for Python 3.13 wheels) |

**Remaining v4 follow-on work:**
1. **Expand reference corpus** beyond 3 trends toward the 12-trend `configs/trend_rules.yaml` taxonomy. Per trend: aim for 15–25 reference images; mob_wife (currently weakest at 10 refs) is the first priority.
2. **DuckDB integration**: extend `src/fashion_forensics/db.py` with a `trend_retrieval` view and a `merged` view that joins items + teacher_labels + trend_retrieval predictions.
3. **Open-set threshold tuning**: collect a held-out OOD product set, plot the max-similarity distribution for in-distribution vs OOD queries, compute AUROC, pick the threshold that hits a target FPR@TPR=95.
4. **Rule-engine agreement eval**: on the product inventory, compare rule engine (§3.6) predictions with retrieval (§3.5) predictions. Disagreements are the candidate set for hand-labeling the gold set used in §3.4 / §3.10.
5. **Hybrid image+text retrieval**: fashion-CLIP has a text encoder too. Embed the LLM-normalized product description with the text tower, fuse image+text similarity scores at query time. Potential accuracy lift on trends where text cues matter (e.g., material-driven classes like mob_wife).
