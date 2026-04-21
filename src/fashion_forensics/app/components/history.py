"""Lab view: Run History — compare evaluation runs across model/prompt versions."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from fashion_forensics.config import DATA_DIR

EVAL_RUNS_DIR = DATA_DIR / "01_data_audit" / "evaluation" / "runs"


def render_run_history():
    """Show all evaluation runs with metrics comparison.

    Lets you see how the model improved over time across
    different prompt versions, thresholds, and training checkpoints.
    """
    runs = _load_all_runs()
    if not runs:
        st.info("No evaluation runs found.")
        return

    st.markdown("##### Evaluation History")
    st.caption(f"{len(runs)} runs recorded")

    # --- Comparison table ---
    comparison_rows = []
    for run in runs:
        core = run["metrics"].get("core", {})
        comparison_rows.append(
            {
                "run": run["run_name"],
                "date": run["date"],
                "threshold": run.get("confidence_threshold", 0.8),
                "f1": core.get("f1_macro", 0),
                "precision": core.get("precision_macro", 0),
                "recall": core.get("recall_macro", 0),
                "tagged": core.get("tagged_count", 0),
            }
        )

    comp_df = pd.DataFrame(comparison_rows).sort_values("date", ascending=False)

    # Highlight best F1
    st.dataframe(
        comp_df.style.format(
            {
                "f1": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "threshold": "{:.2f}",
            }
        ).highlight_max(subset=["f1"], color="#2d6a4f"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --- F1 over time chart ---
    st.markdown("##### F1 Over Time")
    if len(comp_df) > 1:
        chart_df = comp_df.sort_values("date").set_index("run")[["f1", "precision", "recall"]]
        st.line_chart(chart_df)

    st.divider()

    # --- Detailed run comparison ---
    st.markdown("##### Compare Two Runs")

    run_names = comp_df["run"].tolist()
    if len(run_names) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            run_a = st.selectbox("Run A", run_names, index=0)
        with col2:
            run_b = st.selectbox("Run B", run_names, index=min(1, len(run_names) - 1))

        if run_a != run_b:
            _render_run_comparison(runs, run_a, run_b)
    elif len(run_names) == 1:
        st.caption("Need at least 2 runs to compare.")


def _render_run_comparison(runs: list[dict], name_a: str, name_b: str):
    """Side-by-side per-trend comparison of two runs."""
    run_a = next((r for r in runs if r["run_name"] == name_a), None)
    run_b = next((r for r in runs if r["run_name"] == name_b), None)

    if not run_a or not run_b:
        return

    trends_a = run_a["metrics"].get("per_trend", {})
    trends_b = run_b["metrics"].get("per_trend", {})
    all_trends = sorted(set(list(trends_a.keys()) + list(trends_b.keys())))

    if not all_trends:
        st.caption("No per-trend data available.")
        return

    rows = []
    for trend in all_trends:
        f1_a = trends_a.get(trend, {}).get("f1", 0)
        f1_b = trends_b.get(trend, {}).get("f1", 0)
        delta = f1_b - f1_a
        rows.append(
            {
                "trend": trend,
                f"F1 ({name_a})": f1_a,
                f"F1 ({name_b})": f1_b,
                "delta": delta,
            }
        )

    delta_df = pd.DataFrame(rows)

    st.dataframe(
        delta_df.style.format(
            {
                f"F1 ({name_a})": "{:.3f}",
                f"F1 ({name_b})": "{:.3f}",
                "delta": "{:+.3f}",
            }
        ).applymap(
            lambda v: (
                "color: #2d6a4f"
                if isinstance(v, float) and v > 0
                else ("color: #c0392b" if isinstance(v, float) and v < 0 else "")
            ),
            subset=["delta"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Temporal comparison
    temp_a = run_a["metrics"].get("temporal", {})
    temp_b = run_b["metrics"].get("temporal", {})
    if temp_a and temp_b:
        st.markdown("##### Year-over-Year Comparison")
        years = sorted(set(list(temp_a.keys()) + list(temp_b.keys())))
        temp_rows = []
        for y in years:
            temp_rows.append(
                {
                    "year": y,
                    f"F1 ({name_a})": temp_a.get(y, {}).get("f1_macro", 0),
                    f"F1 ({name_b})": temp_b.get(y, {}).get("f1_macro", 0),
                }
            )
        st.dataframe(
            pd.DataFrame(temp_rows).style.format(
                {
                    f"F1 ({name_a})": "{:.3f}",
                    f"F1 ({name_b})": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _load_all_runs() -> list[dict]:
    """Load summary.json from all evaluation run directories."""
    if not EVAL_RUNS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(EVAL_RUNS_DIR.iterdir(), reverse=True):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                runs.append(json.loads(summary_path.read_text()))
            except Exception:
                continue

    return runs
