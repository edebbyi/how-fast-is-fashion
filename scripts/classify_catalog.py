"""Classify the full cleaned catalog (image-only) to see the actual trend distribution.

Unlike scripts/eval_trend_classifier.py — which LOOCV-evaluates accuracy on the
74-image reference corpus — this classifies every real catalog product against
that same reference corpus, to answer "what does the catalog actually look
like" rather than "how accurate is the classifier."

Image-only by default: the cleaned catalog (2,029 records across 23 months) has
not been LLM-normalized (only the 74 reference images have captions), so hybrid
mode isn't available here without normalizing the rest of the catalog first — a
real LLM cost. Image-only LOOCV accuracy (0.865) is within ~1.3pp of hybrid
(0.878), so it's a cheap first pass; normalizing the rest for hybrid is a later
call once this distribution is in hand.

Catalog images aren't stored in the repo (data/01_data_audit/raw_images/ is
gitignored) and none exist locally yet — only Wayback Machine URLs in the
cleaned records. This script downloads and caches them under
data/01_data_audit/raw_images/<month>/<image_filename>, the same layout
mining/miner.py uses, so the cache is idempotent and reusable — already-cached
images are skipped on re-run.

Below the open-set threshold, an item is labeled "unknown" instead of being
forced into its nearest class (predict_trend's `open_set_unknown` flag). The
threshold is auto-calibrated (5th percentile of correct-prediction
max-similarity, LOOCV over the reference corpus in image mode — mirrors
eval_trend_classifier.py's suggestion, computed fresh since the similarity
distribution is mode-specific) unless passed explicitly.

Outputs (data/03_shared/catalog_distribution/):
  - catalog_classifications.jsonl  one row per catalog item: record_id, month,
    product_code/name, trend_pred ("unknown" if below threshold), confidence
  - monthly_trend_counts.csv       month x trend wide table, raw counts —
    every configured trend + "unknown" is a column in every row, zero-filled,
    never blank
  - catalog_snapshot.md            catalog-wide totals + share, all months
    combined — the "what does the catalog look like right now" view

Also logs to MLflow (experiment "catalog-classification"): catalog-wide totals
per trend and a per-month time series (one metric per trend, stepped by month
index, zero-filled) so counts-over-time is queryable in the MLflow UI too.

Usage:
    .venv/bin/python scripts/classify_catalog.py
    .venv/bin/python scripts/classify_catalog.py --months 2024-02 2024-03  # smoke test
    .venv/bin/python scripts/classify_catalog.py --open-set-threshold 0.651
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from loguru import logger

from fashion_forensics.config import PROJECT_ROOT, load_yaml_config, settings
from fashion_forensics.retrieval.classifier import predict_trend
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import TrendQdrantStore
from fashion_forensics.tracking import track_experiment

CLEANED_DIR = PROJECT_ROOT / "data" / "01_data_audit" / "cleaned"
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "01_data_audit" / "raw_images"
LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
OUTPUT_DIR = PROJECT_ROOT / "data" / "03_shared" / "catalog_distribution"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png"}

REQUEST_TIMEOUT = 20
MIN_IMAGE_BYTES = 1000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_canonical_trends() -> list[str]:
    """Active trend classes from configs/trend_rules.yaml — the fixed column
    set for aggregation, so a trend with zero catalog hits still gets a 0
    column/metric instead of being silently omitted."""
    config = load_yaml_config("trend_rules")
    return sorted(config["trends"].keys())


def load_catalog(months: list[str] | None, limit: int | None) -> list[dict]:
    records: list[dict] = []
    for path in sorted(CLEANED_DIR.glob("*_clean.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    if months:
        wanted = set(months)
        records = [r for r in records if r["month"] in wanted]
    if limit:
        records = records[:limit]
    return records


def download_image(session: requests.Session, url: str) -> bytes | None:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        content = resp.content
        content_type = resp.headers.get("Content-Type", "").lower()
        has_image_ext = any(url.lower().endswith(ext) for ext in IMAGE_EXTS)
        if "image" not in content_type and not has_image_ext:
            return None
        if len(content) < MIN_IMAGE_BYTES:
            return None
        return content
    except requests.exceptions.RequestException:
        return None


def ensure_cached_image(session: requests.Session, record: dict) -> Path | None:
    """Local cache path for a catalog record's image, downloading on first use.

    Same layout as mining/miner.py's RAW_IMAGES_DIR/<month>/<filename> — a
    later miner run or other tooling can reuse this cache directly.
    """
    cache_path = RAW_IMAGES_DIR / record["month"] / record["image_filename"]
    if cache_path.exists():
        return cache_path
    content = download_image(session, record["archive_image_url"])
    if content is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return cache_path


def discover_refs() -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    for trend_dir in sorted(LABELED_ROOT.iterdir()):
        if not trend_dir.is_dir():
            continue
        for p in sorted(trend_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(trend_dir.name)
    return paths, labels


def run_reference_loocv(
    embedder: FashionClipEmbedder, store: TrendQdrantStore, k: int, vote_method: str
) -> list[dict]:
    """LOOCV over the reference corpus in image mode: hold out each labeled
    image, classify it against the rest, and record whether the prediction
    was correct along with its max similarity to its best match.

    This is the only place "correct" is actually knowable (catalog items
    have no ground truth) - used both to auto-calibrate the open-set
    threshold below, and exported to disk so the Threshold Explorer's
    precision/coverage panel can show real accuracy at a given threshold,
    not just how the catalog-wide count shifts.
    """
    paths, labels = discover_refs()
    results: list[dict] = []
    for path, label in zip(paths, labels):
        pred = predict_trend(
            image_path=path,
            embedder=embedder,
            store=store,
            k=k,
            vote_method=vote_method,
            search_mode="image",
            exclude_filenames=[path.name],
        )
        max_sim = max((n["score"] for n in pred.matched_refs), default=0.0)
        results.append(
            {
                "filename": path.name,
                "true_trend": label,
                "predicted_trend": pred.trend_pred,
                "max_sim": round(max_sim, 4),
                "correct": pred.trend_pred == label,
            }
        )
    return results


def calibrate_open_set_threshold(loocv_results: list[dict]) -> float:
    """5th percentile of correct-prediction max-similarity, from a
    reference-corpus LOOCV pass (see run_reference_loocv)."""
    max_sims = np.array([r["max_sim"] for r in loocv_results])
    correct = np.array([r["correct"] for r in loocv_results])
    correct_sims = max_sims[correct]
    return float(np.percentile(correct_sims, 5)) if len(correct_sims) else 0.0


def write_snapshot_md(
    path: Path, snapshot_counts: dict[str, int], total: int, threshold: float, n_failed: int
) -> None:
    lines = [
        "# Catalog trend distribution — full labeled catalog snapshot",
        "",
        f"Image-only k-NN classification, open-set threshold={threshold:.3f}.",
        f"{total} classified, {n_failed} image downloads failed and were skipped.",
        "",
        "| trend | count | share |",
        "|---|---|---|",
    ]
    for trend, count in snapshot_counts.items():
        share = count / total if total else 0.0
        lines.append(f"| {trend} | {count} | {share:.1%} |")
    lines.append(f"| **total** | **{total}** | **100.0%** |")
    path.write_text("\n".join(lines) + "\n")


def write_monthly_csv(
    path: Path,
    monthly_counts: dict[str, dict[str, int]],
    months_sorted: list[str],
    columns: list[str],
) -> None:
    lines = ["month," + ",".join(columns)]
    for month in months_sorted:
        row = [str(monthly_counts[month][c]) for c in columns]
        lines.append(f"{month}," + ",".join(row))
    path.write_text("\n".join(lines) + "\n")


def plot_monthly_counts(
    monthly_counts: dict[str, dict[str, int]],
    months_sorted: list[str],
    columns: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for trend in columns:
        series = [monthly_counts[m][trend] for m in months_sorted]
        ax.plot(months_sorted, series, marker="o", label=trend)
    ax.set_xlabel("Month")
    ax.set_ylabel("Catalog item count")
    ax.set_title("Trend counts over time (full catalog, image-only)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(
    k: int,
    vote_method: str,
    open_set_threshold: float | None,
    embedding_model: str,
    collection: str,
    months: list[str] | None,
    limit: int | None,
) -> None:
    label_columns = [*load_canonical_trends(), "unknown"]

    catalog = load_catalog(months, limit)
    catalog_months = sorted(set(r["month"] for r in catalog))
    logger.info(f"Catalog: {len(catalog)} records across {len(catalog_months)} months")

    embedder = FashionClipEmbedder(model_name=embedding_model)
    store = TrendQdrantStore(
        path=PROJECT_ROOT / settings.qdrant_path,
        collection_name=collection,
        embedding_dim=embedder.embedding_dim,
    )
    if store.count() == 0:
        raise RuntimeError(
            "Qdrant collection is empty. Run scripts/embed_reference_corpus.py first."
        )

    if open_set_threshold is None:
        loocv_results = run_reference_loocv(embedder, store, k, vote_method)
        open_set_threshold = calibrate_open_set_threshold(loocv_results)
        logger.info(f"Auto-calibrated open-set threshold: {open_set_threshold:.3f}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: list[dict] = []
    n_failed = 0
    for i, record in enumerate(catalog, 1):
        image_path = ensure_cached_image(session, record)
        if image_path is None:
            n_failed += 1
            continue

        prediction = predict_trend(
            image_path=image_path,
            embedder=embedder,
            store=store,
            k=k,
            vote_method=vote_method,
            search_mode="image",
            open_set_threshold=open_set_threshold,
        )
        trend_pred = "unknown" if prediction.open_set_unknown else prediction.trend_pred
        max_sim = max((n["score"] for n in prediction.matched_refs), default=0.0)
        results.append(
            {
                "record_id": record["record_id"],
                "month": record["month"],
                "product_code": record.get("product_code"),
                "product_name": record.get("product_name"),
                "trend_pred": trend_pred,
                "confidence": prediction.confidence,
                "open_set_unknown": prediction.open_set_unknown,
                # The trend the classifier actually voted for, before the
                # open-set threshold collapses it to "unknown", plus the raw
                # max similarity that threshold gets compared against - so a
                # different threshold can be explored later without needing
                # to re-run the classifier (see the Lab "Threshold Explorer").
                "winning_trend": prediction.trend_pred,
                "max_sim": round(max_sim, 4),
            }
        )
        if i % 100 == 0:
            logger.info(f"Classified {i}/{len(catalog)} ({n_failed} download failures so far)")

    total = len(results)
    logger.info(f"Classified {total}/{len(catalog)} catalog items ({n_failed} download failures)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classifications_path = OUTPUT_DIR / "catalog_classifications.jsonl"
    with open(classifications_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # View A — catalog-wide snapshot, zero-filled across every configured trend + unknown
    snapshot_counts = {label: 0 for label in label_columns}
    for r in results:
        snapshot_counts[r["trend_pred"]] += 1
    for trend, count in snapshot_counts.items():
        share = count / total if total else 0.0
        logger.info(f"  {trend:15s} {count:5d}  ({share:.1%})")
    snapshot_path = OUTPUT_DIR / "catalog_snapshot.md"
    write_snapshot_md(snapshot_path, snapshot_counts, total, open_set_threshold, n_failed)

    # View B — per-month counts, zero-filled across every configured trend + unknown
    months_sorted = sorted(set(r["month"] for r in results))
    monthly_counts = {m: {label: 0 for label in label_columns} for m in months_sorted}
    for r in results:
        monthly_counts[r["month"]][r["trend_pred"]] += 1
    monthly_csv_path = OUTPUT_DIR / "monthly_trend_counts.csv"
    write_monthly_csv(monthly_csv_path, monthly_counts, months_sorted, label_columns)

    params = {
        "embedding_model": embedding_model,
        "collection": collection,
        "k": k,
        "vote_method": vote_method,
        "search_mode": "image",
        "open_set_threshold": open_set_threshold,
        "n_catalog_items": total,
        "n_months": len(months_sorted),
        "n_download_failed": n_failed,
    }
    tags = {"protocol": "full-catalog-classification", "classes": ",".join(label_columns)}

    with track_experiment(
        experiment_name="catalog-classification",
        run_name=f"catalog-image-k{k}-{vote_method}",
        params=params,
        tags=tags,
    ) as mlflow_tracker:
        for trend, count in snapshot_counts.items():
            mlflow_tracker.log_metric(f"catalog_count_{trend}", count)
            mlflow_tracker.log_metric(f"catalog_share_{trend}", count / total if total else 0.0)
        for step, month in enumerate(months_sorted):
            for trend in label_columns:
                mlflow_tracker.log_metric(
                    f"monthly_count_{trend}", monthly_counts[month][trend], step=step
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "trend_counts_over_time.png"
            plot_monthly_counts(monthly_counts, months_sorted, label_columns, plot_path)
            mlflow_tracker.log_artifact(str(plot_path))
        mlflow_tracker.log_artifact(str(classifications_path))
        mlflow_tracker.log_artifact(str(snapshot_path))
        mlflow_tracker.log_artifact(str(monthly_csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k neighbors for voting")
    parser.add_argument(
        "--vote",
        choices=["majority", "shepard"],
        default="shepard",
        help="Voting method: count-based majority or similarity-weighted (Shepard)",
    )
    parser.add_argument(
        "--open-set-threshold",
        type=float,
        default=None,
        help="Below this max-similarity, label as 'unknown'. Default: auto-calibrate "
        "(5th percentile of correct-prediction max-similarity, LOOCV, image mode).",
    )
    parser.add_argument("--embedding-model", default=settings.fashion_clip_model)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Restrict to specific YYYY-MM months (e.g. --months 2024-02 2024-03) "
        "for a smoke test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total records processed, after --months filtering.",
    )
    args = parser.parse_args()
    main(
        k=args.k,
        vote_method=args.vote,
        open_set_threshold=args.open_set_threshold,
        embedding_model=args.embedding_model,
        collection=args.collection,
        months=args.months,
        limit=args.limit,
    )
