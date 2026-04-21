"""Evaluation benchmark for trend-tagging LoRA + MLflow experiment tracking.

Evaluation framework:
    1. Core metrics (threshold-based): precision, recall, F1 at confidence cutoff
    2. Per-class analysis: per-trend F1, macro F1, confusion matrix
    3. Threshold optimization: precision-recall curves, calibration
    4. Qualitative error analysis: near-misses, top-K failures
    5. Year-over-year performance: temporal decay detection

MLflow is used here — not in mining — because this is where we compare
model/prompt versions against human-labeled ground truth.

Flow:
    evaluation/datasets/benchmark_gold.csv   (human-labeled)
        -> run LoRA/prompt against test images
        -> compare predictions vs gold at confidence threshold
        -> compute full metric suite
        -> log params, metrics, artifacts to MLflow
        -> save to evaluation/runs/run_<name>_<date>/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from fashion_forensics.config import DATA_DIR, init_mlflow

EVAL_DIR = DATA_DIR / "01_data_audit" / "evaluation"
DATASETS_DIR = EVAL_DIR / "datasets"
RUNS_DIR = EVAL_DIR / "runs"

for _d in (DATASETS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIDENCE_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# 1. Core metrics (threshold-based)
# ---------------------------------------------------------------------------


def compute_core_metrics(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Compute precision, recall, F1 at a confidence threshold.

    Only predictions with confidence >= threshold are considered "tagged".
    Below-threshold predictions are treated as abstentions.
    """
    # Apply threshold: only keep predictions above cutoff
    mask = np.array(confidences) >= threshold
    tagged_count = int(mask.sum())
    total = len(y_true)

    if tagged_count == 0:
        logger.warning(f"No predictions above threshold {threshold}")
        return {
            "threshold": threshold,
            "tagged_count": 0,
            "abstained_count": total,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    y_true_filtered = [y_true[i] for i in range(total) if mask[i]]
    y_pred_filtered = [y_pred[i] for i in range(total) if mask[i]]

    precision = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

    return {
        "threshold": threshold,
        "tagged_count": tagged_count,
        "abstained_count": total - tagged_count,
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# 2. Per-class analysis (trend breakdown)
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Per-trend precision, recall, F1 + confusion matrix."""
    mask = np.array(confidences) >= threshold
    y_true_f = [y_true[i] for i in range(len(y_true)) if mask[i]]
    y_pred_f = [y_pred[i] for i in range(len(y_pred)) if mask[i]]

    if not y_true_f:
        return {"per_trend": {}, "confusion_matrix": {}}

    # sklearn classification report as dict
    report = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)

    # Per-trend breakdown
    labels = sorted(set(y_true_f + y_pred_f))
    per_trend = {}
    for label in labels:
        if label in report:
            per_trend[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist(),
    }

    return {"per_trend": per_trend, "confusion_matrix": cm_dict}


# ---------------------------------------------------------------------------
# 3. Threshold optimization (precision-recall curve)
# ---------------------------------------------------------------------------


def compute_threshold_analysis(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    thresholds_to_test: list[float] | None = None,
) -> list[dict[str, float]]:
    """Compute metrics at multiple thresholds to find the sweet spot.

    Returns a list of {threshold, precision, recall, f1, tagged_pct} dicts
    for plotting precision-recall trade-off curves.
    """
    if thresholds_to_test is None:
        thresholds_to_test = [0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.85, 0.9, 0.95]

    results = []
    total = len(y_true)

    for t in thresholds_to_test:
        core = compute_core_metrics(y_true, y_pred, confidences, threshold=t)
        results.append(
            {
                "threshold": t,
                "precision": core["precision_macro"],
                "recall": core["recall_macro"],
                "f1": core["f1_macro"],
                "tagged_pct": round(core["tagged_count"] / total, 4) if total > 0 else 0,
            }
        )

    return results


def compute_calibration(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Check if confidence scores are well-calibrated.

    A confidence of 0.8 should mean the model is correct ~80% of the time.
    Returns bins of {avg_confidence, actual_accuracy, count}.
    """
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    bins = np.linspace(0, 1, n_bins + 1)
    result = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        count = sum(mask)
        if count == 0:
            continue
        avg_conf = np.mean([confidences[j] for j in range(len(mask)) if mask[j]])
        acc = np.mean([correct[j] for j in range(len(mask)) if mask[j]])
        result.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "avg_confidence": round(float(avg_conf), 4),
                "actual_accuracy": round(float(acc), 4),
                "count": count,
            }
        )

    return result


# ---------------------------------------------------------------------------
# 4. Qualitative error analysis (near-misses + top-K failures)
# ---------------------------------------------------------------------------


def compute_error_analysis(
    records: list[dict[str, Any]],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    near_miss_margin: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    """Identify near-misses and hard failures for qualitative review.

    Args:
        records: List of dicts with {record_id, trend_gold, trend_pred, confidence, year}.
        threshold: Confidence cutoff.
        near_miss_margin: How close to threshold counts as a "near miss".

    Returns:
        {
            "near_misses": items just below threshold that were actually correct,
            "hard_failures": high-confidence wrong predictions,
            "confused_pairs": most common (gold, pred) mismatches,
        }
    """
    near_misses = []
    hard_failures = []
    confusion_pairs: dict[tuple[str, str], int] = {}

    for r in records:
        gold = r["trend_gold"]
        pred = r["trend_pred"]
        conf = r["confidence"]

        # Near miss: below threshold but correct
        if conf < threshold and conf >= (threshold - near_miss_margin) and gold == pred:
            near_misses.append(r)

        # Hard failure: above threshold but wrong
        if conf >= threshold and gold != pred:
            hard_failures.append(r)
            pair = (gold, pred)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Sort hard failures by confidence (highest confidence mistakes first)
    hard_failures.sort(key=lambda x: x["confidence"], reverse=True)

    # Top confused pairs
    confused = [
        {"gold": k[0], "predicted": k[1], "count": v}
        for k, v in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "near_misses": near_misses[:20],
        "hard_failures": hard_failures[:20],
        "confused_pairs": confused[:10],
    }


# ---------------------------------------------------------------------------
# 5. Year-over-year performance (temporal decay)
# ---------------------------------------------------------------------------


def compute_temporal_metrics(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    years: list[int],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[int, dict[str, float]]:
    """Compute per-year metrics to detect temporal decay.

    If 2026 metrics are significantly worse than 2024, the training data
    for newer trends is insufficient.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidences,
            "year": years,
        }
    )

    result = {}
    for year, group in df.groupby("year"):
        mask = group["confidence"] >= threshold
        g_true = group.loc[mask, "y_true"].tolist()
        g_pred = group.loc[mask, "y_pred"].tolist()

        if not g_true:
            result[int(year)] = {
                "f1_macro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "count": 0,
            }
            continue

        result[int(year)] = {
            "f1_macro": round(f1_score(g_true, g_pred, average="macro", zero_division=0), 4),
            "precision_macro": round(
                precision_score(g_true, g_pred, average="macro", zero_division=0), 4
            ),
            "recall_macro": round(
                recall_score(g_true, g_pred, average="macro", zero_division=0), 4
            ),
            "count": len(g_true),
        }

    return result


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    predictions: list[dict[str, Any]],
    run_name: str,
    params: dict[str, Any] | None = None,
    gold_path: Path | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run the full evaluation suite against the gold set and log to MLflow.

    Args:
        predictions: List of dicts, each with:
            - record_id: str
            - trend_pred: str (predicted trend label)
            - confidence: float (model confidence score)
        run_name: Name for this experiment run.
        params: Hyperparameters/config to log (model version, prompt, etc.).
        gold_path: Path to gold CSV with columns [record_id, trend_gold, year].
        confidence_threshold: Cutoff for "legitimate tag" (default 0.8).

    Returns:
        Full metrics dict with all five evaluation layers.
    """
    gold_csv = gold_path or DATASETS_DIR / "benchmark_gold.csv"
    if not gold_csv.exists():
        raise FileNotFoundError(
            f"Gold set not found at {gold_csv}. "
            "Create it by hand-labeling 100-200 images stratified by year and trend."
        )

    gold_df = pd.read_csv(gold_csv)
    pred_df = pd.DataFrame(predictions)

    merged = gold_df.merge(pred_df, on="record_id", how="inner")
    if merged.empty:
        logger.error("No matching record_ids between gold and predictions")
        return {}

    y_true = merged["trend_gold"].tolist()
    y_pred = merged["trend_pred"].tolist()
    confidences = merged["confidence"].tolist()
    years = merged["year"].tolist() if "year" in merged.columns else []

    # --- 1. Core metrics ---
    core = compute_core_metrics(y_true, y_pred, confidences, confidence_threshold)

    # --- 2. Per-class analysis ---
    per_class = compute_per_class_metrics(y_true, y_pred, confidences, confidence_threshold)

    # --- 3. Threshold optimization ---
    threshold_curve = compute_threshold_analysis(y_true, y_pred, confidences)
    calibration = compute_calibration(y_true, y_pred, confidences)

    # --- 4. Error analysis ---
    records = merged.to_dict("records")
    errors = compute_error_analysis(records, confidence_threshold)

    # --- 5. Temporal metrics ---
    temporal = {}
    if years:
        temporal = compute_temporal_metrics(
            y_true, y_pred, confidences, years, confidence_threshold
        )

    # --- Assemble full results ---
    full_metrics = {
        "core": core,
        "per_trend": per_class["per_trend"],
        "confusion_matrix": per_class["confusion_matrix"],
        "threshold_curve": threshold_curve,
        "calibration": calibration,
        "error_analysis": errors,
        "temporal": temporal,
    }

    # --- Save artifacts to disk ---
    run_date = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = RUNS_DIR / f"run_{run_name}_{run_date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    pred_path = run_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Full results table
    results_path = run_dir / "results.csv"
    merged.to_csv(results_path, index=False)

    # Summary JSON (everything)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "date": run_date,
                "confidence_threshold": confidence_threshold,
                "params": params or {},
                "metrics": full_metrics,
            },
            indent=2,
            default=str,
        )
    )

    # Error analysis for manual review
    errors_path = run_dir / "error_analysis.json"
    errors_path.write_text(json.dumps(errors, indent=2, default=str))

    # Threshold curve as CSV for plotting
    threshold_path = run_dir / "threshold_curve.csv"
    pd.DataFrame(threshold_curve).to_csv(threshold_path, index=False)

    # Calibration as CSV
    calibration_path = run_dir / "calibration.csv"
    pd.DataFrame(calibration).to_csv(calibration_path, index=False)

    # --- Log to MLflow ---
    init_mlflow("evaluation")
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_param("confidence_threshold", confidence_threshold)

        # Core metrics
        mlflow.log_metrics(
            {
                "f1_macro": core["f1_macro"],
                "precision_macro": core["precision_macro"],
                "recall_macro": core["recall_macro"],
                "tagged_count": core["tagged_count"],
                "abstained_count": core["abstained_count"],
            }
        )

        # Per-trend F1
        for trend, m in per_class["per_trend"].items():
            mlflow.log_metric(f"f1_{trend}", m["f1"])
            mlflow.log_metric(f"precision_{trend}", m["precision"])
            mlflow.log_metric(f"recall_{trend}", m["recall"])

        # Temporal metrics
        for year, m in temporal.items():
            mlflow.log_metric(f"f1_year_{year}", m["f1_macro"])

        # Artifacts
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(errors_path))
        mlflow.log_artifact(str(threshold_path))
        mlflow.log_artifact(str(calibration_path))

    # --- Console summary ---
    logger.info(
        f"Benchmark '{run_name}' @ threshold={confidence_threshold}: "
        f"F1={core['f1_macro']:.3f} P={core['precision_macro']:.3f} R={core['recall_macro']:.3f} "
        f"({core['tagged_count']}/{core['tagged_count'] + core['abstained_count']} tagged)"
    )
    for trend, m in per_class["per_trend"].items():
        logger.info(
            f"  {trend}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} (n={m['support']})"
        )
    if temporal:
        for year, m in sorted(temporal.items()):
            logger.info(f"  {year}: F1={m['f1_macro']:.3f} (n={m['count']})")
    if errors["near_misses"]:
        logger.info(
            f"  Near-misses (correct but below {confidence_threshold}): {len(errors['near_misses'])}"
        )
    if errors["confused_pairs"]:
        top_confused = errors["confused_pairs"][0]
        logger.info(
            f"  Top confusion: {top_confused['gold']} -> {top_confused['predicted']} ({top_confused['count']}x)"
        )

    return full_metrics
