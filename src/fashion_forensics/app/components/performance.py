"""Lab view: Model Performance — current model metrics + threshold tuning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fashion_forensics.config import DATA_DIR

EVAL_RUNS_DIR = DATA_DIR / "01_data_audit" / "evaluation" / "runs"


def render_performance():
    """Render the model performance dashboard.

    Shows:
    - Current model's core metrics (F1, precision, recall)
    - Per-trend breakdown with bar chart
    - Threshold curve: precision/recall/F1 at different cutoffs
    - Calibration curve: is the model's confidence trustworthy?
    - Year-over-year decay detection
    """
    run_dir = _get_latest_run()
    if not run_dir:
        st.info("No evaluation runs found. Run a benchmark first.")
        return

    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = summary["metrics"]
    core = metrics["core"]
    threshold = summary.get("confidence_threshold", 0.8)

    # --- Header metrics ---
    st.markdown(f"##### Current Run: `{summary['run_name']}`")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F1 (macro)", f"{core['f1_macro']:.3f}")
    with col2:
        st.metric("Precision", f"{core['precision_macro']:.3f}")
    with col3:
        st.metric("Recall", f"{core['recall_macro']:.3f}")
    with col4:
        st.metric("Threshold", f"{threshold}")

    tagged = core["tagged_count"]
    total = tagged + core["abstained_count"]
    st.caption(
        f"{tagged}/{total} images tagged above threshold ({core['abstained_count']} abstained)"
    )

    st.divider()

    # --- Per-trend breakdown ---
    st.markdown("##### Per-Trend Performance")
    per_trend = metrics.get("per_trend", {})
    if per_trend:
        trend_df = pd.DataFrame.from_dict(per_trend, orient="index")
        trend_df = trend_df.sort_values("f1", ascending=True)

        st.bar_chart(trend_df[["f1", "precision", "recall"]])

        # Table view
        st.dataframe(
            trend_df.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    st.divider()

    # --- Threshold curve ---
    st.markdown("##### Threshold Optimization")
    st.caption("Is 0.8 the right cutoff? Find where F1 peaks.")

    threshold_path = run_dir / "threshold_curve.csv"
    if threshold_path.exists():
        tc_df = pd.read_csv(threshold_path)
        st.line_chart(tc_df.set_index("threshold")[["f1", "precision", "recall"]])

        # Highlight optimal threshold
        best_row = tc_df.loc[tc_df["f1"].idxmax()]
        st.caption(
            f"Optimal threshold: **{best_row['threshold']:.2f}** "
            f"(F1={best_row['f1']:.3f}, P={best_row['precision']:.3f}, R={best_row['recall']:.3f})"
        )

    st.divider()

    # --- Calibration ---
    st.markdown("##### Calibration")
    st.caption("Does 0.8 confidence actually mean 80% correct?")

    calibration_path = run_dir / "calibration.csv"
    if calibration_path.exists():
        cal_df = pd.read_csv(calibration_path)
        cal_chart = cal_df[["avg_confidence", "actual_accuracy"]].copy()
        cal_chart = cal_chart.set_index("avg_confidence")
        st.line_chart(cal_chart)

        # Check for over/under confidence
        if not cal_df.empty:
            high_conf = cal_df[cal_df["avg_confidence"] >= 0.75]
            if not high_conf.empty:
                avg_acc = high_conf["actual_accuracy"].mean()
                if avg_acc > 0.85:
                    st.caption(
                        f"Model is **under-confident** at high scores (avg accuracy {avg_acc:.0%} at 0.75+ confidence). Consider lowering threshold."
                    )
                elif avg_acc < 0.7:
                    st.caption(
                        f"Model is **over-confident** at high scores (avg accuracy {avg_acc:.0%} at 0.75+ confidence). Threshold may need raising."
                    )

    st.divider()

    # --- Temporal decay ---
    st.markdown("##### Year-over-Year")
    temporal = metrics.get("temporal", {})
    if temporal:
        years = sorted(temporal.keys())
        temp_df = pd.DataFrame(
            [
                {"year": y, "f1": temporal[y]["f1_macro"], "count": temporal[y]["count"]}
                for y in years
            ]
        ).set_index("year")
        st.bar_chart(temp_df["f1"])

        # Detect decay
        f1_values = [temporal[y]["f1_macro"] for y in years]
        if len(f1_values) >= 2 and f1_values[-1] < f1_values[0] - 0.05:
            st.warning(
                f"Temporal decay detected: F1 dropped from {f1_values[0]:.3f} ({years[0]}) "
                f"to {f1_values[-1]:.3f} ({years[-1]}). Training data for newer trends may be insufficient."
            )

    # --- Confusion highlights ---
    errors_path = run_dir / "error_analysis.json"
    if errors_path.exists():
        errors = json.loads(errors_path.read_text())
        confused = errors.get("confused_pairs", [])
        if confused:
            st.divider()
            st.markdown("##### Top Confusions")
            for pair in confused[:5]:
                st.caption(f"{pair['gold']} mistaken for {pair['predicted']} — {pair['count']}x")


def _get_latest_run() -> Path | None:
    """Return the most recent evaluation run directory."""
    if not EVAL_RUNS_DIR.exists():
        return None
    runs = sorted(EVAL_RUNS_DIR.iterdir(), reverse=True)
    for r in runs:
        if r.is_dir() and (r / "summary.json").exists():
            return r
    return None
