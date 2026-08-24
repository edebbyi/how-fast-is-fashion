"""Lab view: pointer into MLflow, which is already this project's real
evaluation-history tool - this panel doesn't try to rebuild it, just surface
the most recent runs and a link to the full UI.
"""

from __future__ import annotations

import os

import streamlit as st

from fashion_forensics.config import settings

EXPERIMENTS = [
    "trend-classifier-eval",
    "catalog-classification",
    "trend-lifecycle",
    "ranking-eval",
]


def render_mlflow_panel():
    st.markdown("### MLflow Runs")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    st.link_button("Open full MLflow UI", tracking_uri)
    st.caption(f"Tracking server: {tracking_uri}")
    st.divider()

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(
            experiment_names=EXPERIMENTS,
            order_by=["start_time DESC"],
            max_results=20,
        )
    except Exception as e:
        st.warning(f"Couldn't reach MLflow at {tracking_uri}: {e}")
        return

    if runs.empty:
        st.caption("No runs logged yet.")
        return

    display_cols = [c for c in ("start_time", "experiment_id", "tags.mlflow.runName", "status") if c in runs.columns]
    metric_cols = sorted(c for c in runs.columns if c.startswith("metrics.") and runs[c].notna().any())

    table = runs[display_cols + metric_cols].rename(
        columns={"tags.mlflow.runName": "run", **{c: c.replace("metrics.", "") for c in metric_cols}}
    )
    st.dataframe(table, width="stretch", hide_index=True)
