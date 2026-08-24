"""Lab view: interactively explore the open-set confidence threshold.

Items below the threshold get labeled "unknown" instead of their nearest
trend (ARCHITECTURE.md §3.5). The shipped catalog uses one fixed,
auto-calibrated threshold - this view lets you drag it and see how the
"unknown" share and per-trend counts would change, without re-running the
classifier. Ties directly to the deferred threshold-calibration question
(current threshold's sensitivity ranged from ~18% to ~99.6% unknown-share
across different modes - see notebooks/03_trend_classification.ipynb §14.5).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.config import DATA_DIR

CLASSIFICATIONS_PATH = (
    DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_classifications.jsonl"
)
REFERENCE_LOOCV_PATH = DATA_DIR / "03_shared" / "catalog_distribution" / "reference_loocv.jsonl"


def render_threshold_explorer():
    st.markdown("### Threshold Explorer")

    if not CLASSIFICATIONS_PATH.exists():
        st.info("No catalog classifications yet. Run scripts/classify_catalog.py first.")
        return

    df = pd.read_json(CLASSIFICATIONS_PATH, lines=True)
    if "max_sim" not in df.columns or "winning_trend" not in df.columns:
        st.info(
            "This view needs max_sim/winning_trend in catalog_classifications.jsonl - "
            "re-run scripts/classify_catalog.py to regenerate it with those fields."
        )
        return

    current_threshold = df.loc[df["open_set_unknown"], "max_sim"].max()
    shipped_threshold = round(float(current_threshold), 3) if pd.notna(current_threshold) else 0.0

    st.caption(f"Shipped catalog uses threshold ≈ {shipped_threshold:.3f} (auto-calibrated).")
    threshold = st.slider(
        "OPEN-SET THRESHOLD",
        min_value=0.0,
        max_value=1.0,
        value=shipped_threshold,
        step=0.01,
        help="Below this max similarity to any reference image, an item is labeled 'unknown'.",
    )

    simulated = df["winning_trend"].where(df["max_sim"] >= threshold, "unknown")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Shipped")
        st.dataframe(_distribution_table(df["trend_pred"]), width="stretch", hide_index=True)
    with col2:
        st.markdown(f"##### At threshold {threshold:.2f}")
        st.dataframe(_distribution_table(simulated), width="stretch", hide_index=True)

    n_flipped = (df["trend_pred"] != simulated).sum()
    st.caption(f"{n_flipped} of {len(df)} items would change label at this threshold.")

    st.divider()
    st.markdown("##### Precision & coverage vs. threshold, on the 135 reference images")
    st.caption(
        "Catalog items above have no ground truth, so the panels above only show what "
        "changes, not whether it's right. These 135 reference images do have "
        "human-verified labels - the only place \"correct\" is actually knowable. "
        "Precision: of the reference images that clear the threshold, what % are "
        "classified correctly. Coverage: what % of the reference corpus clears the "
        "threshold at all, vs. gets thrown away as \"unknown\"."
    )
    if not REFERENCE_LOOCV_PATH.exists():
        st.info("No reference LOOCV data yet. Run scripts/reference_loocv.py first.")
    else:
        loocv_df = pd.read_json(REFERENCE_LOOCV_PATH, lines=True)
        curve = _precision_coverage_curve(loocv_df)
        st.line_chart(curve.set_index("threshold")[["precision", "coverage"]])

        kept = loocv_df[loocv_df["max_sim"] >= threshold]
        current_coverage = len(kept) / len(loocv_df) if len(loocv_df) else 0.0
        current_precision = kept["correct"].mean() if len(kept) else float("nan")
        precision_text = (
            f"{current_precision:.1%}" if pd.notna(current_precision) else "n/a (nothing clears this threshold)"
        )
        st.caption(
            f"At threshold {threshold:.2f}: precision {precision_text}, coverage "
            f"{current_coverage:.1%} ({len(kept)} of {len(loocv_df)} reference images)."
        )

    st.divider()
    st.markdown("##### Catalog items by best-match similarity score")
    st.caption("Where items actually sit relative to the threshold - helps judge if it's cutting in a sensible place.")
    bins = pd.cut(df["max_sim"], bins=20)
    hist = bins.value_counts().sort_index()
    hist.index = [f"{i.left:.2f}" for i in hist.index]
    st.bar_chart(hist)


def _distribution_table(trend_series: pd.Series) -> pd.DataFrame:
    counts = trend_series.value_counts()
    total = len(trend_series)
    table = counts.reset_index()
    table.columns = ["trend", "count"]
    table["share"] = (table["count"] / total).map(lambda x: f"{x:.1%}")
    return table


def _precision_coverage_curve(loocv_df: pd.DataFrame) -> pd.DataFrame:
    """Sweep threshold values 0-1: coverage is the fraction of the
    reference corpus that clears each threshold, precision is the fraction
    of those that are actually correctly classified."""
    total = len(loocv_df)
    rows = []
    for i in range(0, 101, 2):
        t = i / 100
        kept = loocv_df[loocv_df["max_sim"] >= t]
        coverage = len(kept) / total if total else 0.0
        precision = kept["correct"].mean() if len(kept) else None
        rows.append({"threshold": t, "precision": precision, "coverage": coverage})
    return pd.DataFrame(rows)
