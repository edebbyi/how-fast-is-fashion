"""Run §3.7 time_series_aggregation + §3.8 lifecycle_classifier over the
full catalog classification output.

Requires data/03_shared/catalog_distribution/catalog_classifications.jsonl
to already exist — run scripts/classify_catalog.py first (see its own
docstring for --months smoke-test usage).

Outputs (data/03_shared/trend_lifecycle/):
  - monthly_trend_frequency.jsonl / .csv   §3.7 output, long format
  - lifecycle_states.jsonl / .csv          §3.8 output, one row per trend
  - lifecycle_summary.md                   human-readable state table

Logs to MLflow (experiment "trend-lifecycle"): per-trend active_month_count,
months_since_last_active, recent_slope; per-state trend counts; all five
output files as artifacts.

Usage:
    .venv/bin/python scripts/compute_trend_lifecycle.py
    .venv/bin/python scripts/compute_trend_lifecycle.py --active-threshold 0.08
"""

from __future__ import annotations

import argparse

from loguru import logger

from fashion_forensics.config import PROJECT_ROOT
from fashion_forensics.nlp.catalog_dedup import dedupe_for_counting
from fashion_forensics.nlp.lifecycle_classifier import LIFECYCLE_VERSION, classify_lifecycles
from fashion_forensics.nlp.time_series_aggregation import (
    aggregate_monthly_frequency,
    load_canonical_trends,
    load_classifications,
)
from fashion_forensics.tracking import track_experiment

OUTPUT_DIR = PROJECT_ROOT / "data" / "03_shared" / "trend_lifecycle"


def write_lifecycle_summary_md(path, lifecycle_df) -> None:
    lines = [
        "# Trend lifecycle summary",
        "",
        "| trend | state | active_months | first_observed | last_active | peak_month |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in lifecycle_df.iterrows():
        lines.append(
            f"| {row['trend']} | {row['state']} | {row['active_month_count']} | "
            f"{row['first_observed'] or '—'} | {row['last_active'] or '—'} | "
            f"{row['peak_month'] or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(
    active_threshold: float,
    persistent_min_active_months: int,
    rising_slope_threshold: float,
    declining_slope_threshold: float,
    slope_window: int,
    recency_inactive_months: int,
    recurrence_gap_months: int,
    classifications_path: str | None,
) -> None:
    try:
        records = load_classifications(classifications_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1) from e

    # catalog_classifications.jsonl is photo-level (one row per photo angle,
    # same product often appears several times a month) - deduped here to
    # one row per distinct product so lifecycle counts aren't inflated. See
    # catalog_dedup.py.
    n_raw = len(records)
    records = dedupe_for_counting(records)

    trends = load_canonical_trends()
    n_months = len({r["month"] for r in records})
    logger.info(
        f"Loaded {n_raw} classifications, deduped to {len(records)} distinct products "
        f"across {n_months} months, {len(trends)} trends"
    )

    freq_df = aggregate_monthly_frequency(records, trends=trends, active_threshold=active_threshold)
    lifecycle_df = classify_lifecycles(
        freq_df,
        persistent_min_active_months=persistent_min_active_months,
        rising_slope_threshold=rising_slope_threshold,
        declining_slope_threshold=declining_slope_threshold,
        slope_window=slope_window,
        recency_inactive_months=recency_inactive_months,
        recurrence_gap_months=recurrence_gap_months,
    )

    for _, row in lifecycle_df.iterrows():
        logger.info(
            f"  {row['trend']:15s} {row['state']:20s} active_months={row['active_month_count']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    freq_jsonl = OUTPUT_DIR / "monthly_trend_frequency.jsonl"
    freq_csv = OUTPUT_DIR / "monthly_trend_frequency.csv"
    lifecycle_jsonl = OUTPUT_DIR / "lifecycle_states.jsonl"
    lifecycle_csv = OUTPUT_DIR / "lifecycle_states.csv"
    summary_md = OUTPUT_DIR / "lifecycle_summary.md"

    freq_df.to_json(freq_jsonl, orient="records", lines=True)
    freq_df.to_csv(freq_csv, index=False)
    lifecycle_df.to_json(lifecycle_jsonl, orient="records", lines=True)
    lifecycle_df.to_csv(lifecycle_csv, index=False)
    write_lifecycle_summary_md(summary_md, lifecycle_df)

    state_counts = lifecycle_df["state"].value_counts().to_dict()

    params = {
        "active_threshold": active_threshold,
        "persistent_min_active_months": persistent_min_active_months,
        "rising_slope_threshold": rising_slope_threshold,
        "declining_slope_threshold": declining_slope_threshold,
        "slope_window": slope_window,
        "recency_inactive_months": recency_inactive_months,
        "recurrence_gap_months": recurrence_gap_months,
        "lifecycle_version": LIFECYCLE_VERSION,
        "n_trends": len(trends),
        "n_months": n_months,
        "n_records": len(records),
        "n_raw_records": n_raw,
    }
    tags = {"lifecycle_version": LIFECYCLE_VERSION}

    with track_experiment(
        experiment_name="trend-lifecycle",
        run_name=f"lifecycle-{LIFECYCLE_VERSION}-thresh{active_threshold}",
        params=params,
        tags=tags,
    ) as mlflow_tracker:
        for _, row in lifecycle_df.iterrows():
            trend = row["trend"]
            mlflow_tracker.log_metric(f"active_months_{trend}", row["active_month_count"])
            if row["months_since_last_active"] is not None:
                mlflow_tracker.log_metric(
                    f"months_since_last_active_{trend}", row["months_since_last_active"]
                )
            mlflow_tracker.log_metric(f"recent_slope_{trend}", row["recent_slope"])
        for state in ["rising", "persistent", "seasonal_recurring", "declining", "inactive"]:
            mlflow_tracker.log_metric(f"state_count_{state}", int(state_counts.get(state, 0)))

        for path in [freq_jsonl, freq_csv, lifecycle_jsonl, lifecycle_csv, summary_md]:
            mlflow_tracker.log_artifact(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-threshold", type=float, default=0.05)
    parser.add_argument("--persistent-min-months", type=int, default=4)
    parser.add_argument("--rising-slope", type=float, default=0.03)
    parser.add_argument("--declining-slope", type=float, default=-0.03)
    parser.add_argument("--slope-window", type=int, default=3)
    parser.add_argument("--recency-inactive-months", type=int, default=2)
    parser.add_argument("--recurrence-gap-months", type=int, default=2)
    parser.add_argument("--classifications-path", type=str, default=None)
    args = parser.parse_args()
    main(
        active_threshold=args.active_threshold,
        persistent_min_active_months=args.persistent_min_months,
        rising_slope_threshold=args.rising_slope,
        declining_slope_threshold=args.declining_slope,
        slope_window=args.slope_window,
        recency_inactive_months=args.recency_inactive_months,
        recurrence_gap_months=args.recurrence_gap_months,
        classifications_path=args.classifications_path,
    )
