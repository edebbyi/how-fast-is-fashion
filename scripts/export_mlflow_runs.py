"""Export an MLflow experiment's runs table to CSV under data/03_shared/runs/.

The local MLflow store (sqlite:///mlflow.db) is not visible to reviewers of
this repo. This script serializes the runs table so the numbers behind the
narrative (params, metrics, tags) live in a committable, diff-able artifact
alongside the code that produced them.

Timestamps are converted to ISO strings for readability. `artifact_uri` is
dropped because it embeds the user's absolute home path.

Usage:
    .venv/bin/python scripts/export_mlflow_runs.py
    .venv/bin/python scripts/export_mlflow_runs.py --experiment lora-training
    .venv/bin/python scripts/export_mlflow_runs.py --output somewhere/else.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

from fashion_forensics.config import PROJECT_ROOT, settings

DEFAULT_EXPERIMENT = "trend-classifier-eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "03_shared" / "runs"


def export_runs(experiment_name: str, out_path: Path) -> int:
    """Serialize an experiment's runs table to CSV. Returns row count."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")
    logger.info(f"Experiment: {experiment_name}")

    df = mlflow.search_runs(experiment_names=[experiment_name])
    if df.empty:
        logger.warning(f"No runs found for experiment '{experiment_name}' — nothing to export")
        return 0

    # Convert epoch-ms timestamps to ISO strings so the CSV is human-readable.
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # artifact_uri embeds an absolute local path (leaks the user's home dir).
    df = df.drop(columns=[c for c in ["artifact_uri"] if c in df.columns])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        f"Wrote {len(df)} runs to {out_path.relative_to(PROJECT_ROOT)} "
        f"({out_path.stat().st_size:,} bytes, {len(df.columns)} columns)"
    )
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Output CSV path (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}/"
            "{experiment}_runs.csv)"
        ),
    )
    args = parser.parse_args()

    out = args.output or DEFAULT_OUTPUT_DIR / f"{args.experiment}_runs.csv"
    export_runs(args.experiment, out)
