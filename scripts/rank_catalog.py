"""Run the ranking engine (§3.9) over the catalog for each synthetic user
profile, comparing Model A (attribute-only) against Model B (trend +
lifecycle-aware).

Needs these to already exist:
  - data/03_shared/catalog_distribution/catalog_classifications.jsonl
    (made by scripts/classify_catalog.py)
  - data/03_shared/trend_lifecycle/lifecycle_states.jsonl
    (made by scripts/compute_trend_lifecycle.py)
  - data/01_data_audit/normalization/outputs/*_normalized.jsonl
    (item attributes - only 59 catalog items have these so far)

Important caveat, also printed here and in every output file: only items
that have both normalized attributes and a real (not "unknown") trend label
give a fair Model A vs Model B comparison - currently 23 items. The
precision/recall/F1 numbers below decide "relevant" using each profile's own
stated preferred_trends/preferred_category fields, so they only show whether
a scoring formula matches its own inputs, not real recommendation quality
(there's no independent user feedback to check against). Kendall's tau and
top-K overlap don't have that problem and should be trusted more.

Outputs (data/03_shared/ranking/):
  - rankings_<profile>_modelA.jsonl / _modelB.jsonl   full scored+ranked lists
  - eval_summary.csv                                   precision/recall/F1 @ k
  - ab_comparison_summary.md                           top-5 side-by-side, tau, overlap

Logs to MLflow (experiment "ranking-eval").

Usage:
    .venv/bin/python scripts/rank_catalog.py
    .venv/bin/python scripts/rank_catalog.py --profile synthetic_user_rising_basics
"""

from __future__ import annotations

import argparse
import csv
import json

import yaml
from loguru import logger

from fashion_forensics.config import PROJECT_ROOT
from fashion_forensics.nlp.lifecycle_classifier import load_lifecycle_states
from fashion_forensics.nlp.ranking_engine import (
    RANKING_ENGINE_VERSION,
    f1_at_k,
    filter_fair_comparison_subset,
    kendall_tau,
    load_ranking_catalog,
    precision_at_k,
    rank_items,
    recall_at_k,
    top_k,
    top_k_overlap,
)
from fashion_forensics.tracking import track_experiment

OUTPUT_DIR = PROJECT_ROOT / "data" / "03_shared" / "ranking"
PROFILES_PATH = PROJECT_ROOT / "configs" / "user_profiles.yaml"
K_VALUES = (5, 10)

CAVEAT = (
    "Precision/recall/F1 decide what counts as \"relevant\" using each profile's own "
    "stated preferred_trends/preferred_category fields. So these numbers only show "
    "whether a scoring formula matches its own inputs -- they are NOT a measure of "
    "real recommendation quality, since there's no independent user feedback to check "
    "against. Kendall's tau and top-K overlap don't have that problem and should be "
    "trusted more when reading these results."
)


def load_profiles(profile_name: str | None) -> dict[str, dict]:
    all_profiles = yaml.safe_load(PROFILES_PATH.read_text())["profiles"]
    if profile_name:
        if profile_name not in all_profiles:
            raise SystemExit(f"Unknown profile {profile_name!r}. Options: {list(all_profiles)}")
        return {profile_name: all_profiles[profile_name]}
    return all_profiles


def write_ranking_jsonl(path, ranked: list[dict]) -> None:
    with open(path, "w") as f:
        for row in ranked:
            slim = {
                "rank": row["rank"],
                "record_id": row["record_id"],
                "trend_pred": row["trend_pred"],
                "category": row["category"],
                "score": row["score"],
                "components": row["components"],
            }
            f.write(json.dumps(slim) + "\n")


def main(profile_name: str | None) -> None:
    try:
        catalog = load_ranking_catalog()
        lifecycle_by_trend = load_lifecycle_states()
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1) from e

    subset = filter_fair_comparison_subset(catalog)
    if not subset:
        raise SystemExit("Fair-comparison subset is empty -- nothing to rank.")

    profiles = load_profiles(profile_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    summary_sections = [f"# Ranking engine A/B comparison\n\n{CAVEAT}\n"]

    for name, profile in profiles.items():
        ranked_a = rank_items(subset, profile, model="A")
        ranked_b = rank_items(subset, profile, model="B", lifecycle_by_trend=lifecycle_by_trend)

        write_ranking_jsonl(OUTPUT_DIR / f"rankings_{name}_modelA.jsonl", ranked_a)
        write_ranking_jsonl(OUTPUT_DIR / f"rankings_{name}_modelB.jsonl", ranked_b)

        tau = kendall_tau(ranked_a, ranked_b)
        overlap5 = top_k_overlap(ranked_a, ranked_b, k=5)

        section = [f"## {name}\n", f"kendall_tau={tau:.3f}  top5_overlap={overlap5:.3f}\n"]
        section.append("| rank | Model A | Model B |")
        section.append("|---|---|---|")
        top5_a = top_k(ranked_a, 5)
        top5_b = top_k(ranked_b, 5)
        for i in range(5):
            a_desc = f"{top5_a[i]['record_id']} ({top5_a[i]['trend_pred']}, {top5_a[i]['score']:.3f})" if i < len(top5_a) else ""
            b_desc = f"{top5_b[i]['record_id']} ({top5_b[i]['trend_pred']}, {top5_b[i]['score']:.3f})" if i < len(top5_b) else ""
            section.append(f"| {i+1} | {a_desc} | {b_desc} |")

        for model_name, ranked in [("A", ranked_a), ("B", ranked_b)]:
            for k in K_VALUES:
                p = precision_at_k(ranked, profile, k)
                r = recall_at_k(ranked, profile, k)
                f1 = f1_at_k(ranked, profile, k)
                eval_rows.append(
                    {"profile": name, "model": model_name, "k": k, "precision": p, "recall": r, "f1": f1}
                )
                section.append(f"\nModel {model_name} @k={k}: precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

        summary_sections.append("\n".join(section) + "\n")
        logger.info(f"{name}: tau={tau:.3f} top5_overlap={overlap5:.3f}")

    eval_csv_path = OUTPUT_DIR / "eval_summary.csv"
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["profile", "model", "k", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(eval_rows)

    summary_md_path = OUTPUT_DIR / "ab_comparison_summary.md"
    summary_md_path.write_text("\n".join(summary_sections))

    params = {
        "ranking_engine_version": RANKING_ENGINE_VERSION,
        "n_fair_comparison_items": len(subset),
        "n_profiles": len(profiles),
        "k_values": ",".join(str(k) for k in K_VALUES),
    }
    with track_experiment(
        experiment_name="ranking-eval",
        run_name=f"ranking-{RANKING_ENGINE_VERSION}-{profile_name or 'all'}",
        params=params,
        tags={"protocol": "synthetic-profile-ab"},
    ) as mlflow_tracker:
        for row in eval_rows:
            metric_prefix = f"{row['profile']}_model{row['model']}_k{row['k']}"
            mlflow_tracker.log_metric(f"{metric_prefix}_precision", row["precision"])
            mlflow_tracker.log_metric(f"{metric_prefix}_recall", row["recall"])
            mlflow_tracker.log_metric(f"{metric_prefix}_f1", row["f1"])
        mlflow_tracker.log_artifact(str(eval_csv_path))
        mlflow_tracker.log_artifact(str(summary_md_path))
        for name in profiles:
            mlflow_tracker.log_artifact(str(OUTPUT_DIR / f"rankings_{name}_modelA.jsonl"))
            mlflow_tracker.log_artifact(str(OUTPUT_DIR / f"rankings_{name}_modelB.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None, help="Run just one profile by name")
    args = parser.parse_args()
    main(profile_name=args.profile)
