"""LOOCV over the 135-image reference corpus (image-only mode), exported to
disk for the Lab view's Threshold Explorer.

Unlike scripts/classify_catalog.py's catalog-wide classification, this only
touches the reference corpus - the one place "correct" is actually knowable,
since every reference image has a human-verified trend label. Threshold
Explorer uses the per-image (max_sim, correct) output to show real
precision/coverage at a given threshold, alongside the catalog-share panel
that only shows what changes, not whether it's right.

This is deliberately separate from classify_catalog.py's full run: this file
only depends on the reference corpus + embedding model, not the ~2,029-item
catalog, so it doesn't need a ~30-minute full catalog rerun to stay fresh -
just re-run this (135 images, a couple minutes) whenever the reference
corpus or embedding model changes.

Output: data/03_shared/catalog_distribution/reference_loocv.jsonl
  one row per reference image: filename, true_trend, predicted_trend,
  max_sim, correct

Also logs to MLflow (experiment "reference-loocv"): accuracy, macro F1,
per-trend precision/recall/F1, the auto-calibrated threshold this run would
pick, and the full reference_loocv.jsonl as an artifact - one run per call,
so evaluation history over time is browsable in the MLflow tab even though
the local reference_loocv.jsonl file only ever holds the latest run.

Usage:
    .venv/bin/python scripts/reference_loocv.py
"""

from __future__ import annotations

import argparse
import json

from classify_catalog import (
    OUTPUT_DIR,
    calibrate_open_set_threshold,
    run_reference_loocv,
)
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support

from fashion_forensics.config import PROJECT_ROOT, settings
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import TrendQdrantStore
from fashion_forensics.tracking import track_experiment


def main(k: int, vote_method: str, embedding_model: str, collection: str) -> None:
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

    results = run_reference_loocv(embedder, store, k, vote_method)
    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0.0
    threshold = calibrate_open_set_threshold(results)
    logger.info(f"LOOCV accuracy: {accuracy:.3f} over {len(results)} reference images")
    logger.info(f"Auto-calibrated open-set threshold: {threshold:.3f}")

    # Per-trend precision/recall/F1 - logged to MLflow (not just shown in the
    # Curate "Evaluate reference corpus" button) so evaluation history is
    # durable across sessions, not lost when the browser tab closes.
    labels = sorted({r["true_trend"] for r in results})
    precision, recall, f1, _ = precision_recall_fscore_support(
        [r["true_trend"] for r in results],
        [r["predicted_trend"] for r in results],
        labels=labels,
        zero_division=0,
    )
    macro_f1 = f1.mean() if len(f1) else 0.0
    per_trend_metrics = {
        label: {"precision": p, "recall": r, "f1": f}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "reference_loocv.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {out_path}")

    params = {
        "embedding_model": embedding_model,
        "collection": collection,
        "k": k,
        "vote_method": vote_method,
        "search_mode": "image",
        "n_reference_images": len(results),
    }
    with track_experiment(
        experiment_name="reference-loocv",
        run_name=f"reference-loocv-k{k}-{vote_method}",
        params=params,
        tags={"protocol": "reference-corpus-loocv"},
    ) as mlflow_tracker:
        mlflow_tracker.log_metric("accuracy", accuracy)
        mlflow_tracker.log_metric("macro_f1", macro_f1)
        mlflow_tracker.log_metric("calibrated_threshold", threshold)
        for label, m in per_trend_metrics.items():
            mlflow_tracker.log_metric(f"precision_{label}", m["precision"])
            mlflow_tracker.log_metric(f"recall_{label}", m["recall"])
            mlflow_tracker.log_metric(f"f1_{label}", m["f1"])
        mlflow_tracker.log_artifact(str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k neighbors for voting")
    parser.add_argument(
        "--vote",
        choices=["majority", "shepard"],
        default="shepard",
        help="Voting method: count-based majority or similarity-weighted (Shepard)",
    )
    parser.add_argument("--embedding-model", default=settings.fashion_clip_model)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    args = parser.parse_args()
    main(
        k=args.k,
        vote_method=args.vote,
        embedding_model=args.embedding_model,
        collection=args.collection,
    )
