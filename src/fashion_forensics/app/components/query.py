"""Lab view: Image Query — query predictions by trend + confidence to visually validate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fashion_forensics.config import DATA_DIR

EVAL_RUNS_DIR = DATA_DIR / "01_data_audit" / "evaluation" / "runs"
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"


def render_image_query():
    """Query model predictions: 'show me all images tagged boho at 0.82'.

    Lets you visually validate whether the model's confidence
    matches reality for specific trends and score ranges.
    """
    run_dir = _get_latest_run()
    if not run_dir:
        st.info("No evaluation runs found. Run a benchmark first.")
        return

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        st.info("No results.csv found in latest run.")
        return

    df = pd.read_csv(results_path)

    # --- Filters ---
    st.markdown("##### Query Predictions")

    col_trend, col_conf, col_status = st.columns(3)

    with col_trend:
        trends = sorted(df["trend_pred"].dropna().unique())
        selected_trend = st.selectbox("TREND", ["All"] + trends, label_visibility="visible")

    with col_conf:
        conf_range = st.slider(
            "CONFIDENCE RANGE",
            min_value=0.0,
            max_value=1.0,
            value=(0.7, 1.0),
            step=0.05,
        )

    with col_status:
        status_filter = st.selectbox(
            "STATUS",
            ["All", "Correct", "Wrong", "Near-miss"],
        )

    # --- Apply filters ---
    filtered = df.copy()

    if selected_trend != "All":
        filtered = filtered[filtered["trend_pred"] == selected_trend]

    if "confidence" in filtered.columns:
        filtered = filtered[
            (filtered["confidence"] >= conf_range[0]) & (filtered["confidence"] <= conf_range[1])
        ]

    # Compute correctness
    if "trend_gold" in filtered.columns and "trend_pred" in filtered.columns:
        filtered["correct"] = filtered["trend_gold"] == filtered["trend_pred"]

        if status_filter == "Correct":
            filtered = filtered[filtered["correct"]]
        elif status_filter == "Wrong":
            filtered = filtered[~filtered["correct"]]
        elif status_filter == "Near-miss":
            # Correct prediction but below default threshold
            filtered = filtered[
                filtered["correct"]
                & (filtered["confidence"] < 0.8)
                & (filtered["confidence"] >= 0.7)
            ]

    # --- Summary ---
    st.caption(f"Showing {len(filtered)} predictions")
    if len(filtered) > 0 and "confidence" in filtered.columns:
        avg_conf = filtered["confidence"].mean()
        if "correct" in filtered.columns:
            accuracy = filtered["correct"].mean()
            st.caption(f"Avg confidence: {avg_conf:.3f} | Accuracy in selection: {accuracy:.1%}")
        else:
            st.caption(f"Avg confidence: {avg_conf:.3f}")

    st.divider()

    # --- Image grid with metadata ---
    if filtered.empty:
        st.caption("No predictions match your filters.")
        return

    # Sort by confidence descending
    filtered = filtered.sort_values("confidence", ascending=False)

    # Render grid
    cols_per_row = 4
    rows = (len(filtered) + cols_per_row - 1) // cols_per_row

    for row_idx in range(min(rows, 25)):  # Cap at 100 images
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx >= len(filtered):
                break

            item = filtered.iloc[item_idx]
            with cols[col_idx]:
                _render_prediction_card(item)


def _render_prediction_card(item: pd.Series):
    """Render a single prediction card with image + metadata."""
    # Try to find the image file
    img_path = _find_image(item)

    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.markdown(
            '<div style="aspect-ratio:3/4;background:#1a1a1a;display:flex;'
            'align-items:center;justify-content:center;color:#444;font-size:0.7rem;">'
            "Image not found</div>",
            unsafe_allow_html=True,
        )

    # Confidence badge
    conf = item.get("confidence", 0)
    if conf >= 0.8:
        badge_class = "conf-high"
    elif conf >= 0.7:
        badge_class = "conf-mid"
    else:
        badge_class = "conf-low"

    pred = item.get("trend_pred", "?")
    gold = item.get("trend_gold", "?")
    is_correct = item.get("correct", None)

    # Status icon
    if is_correct is True:
        status = "correct"
    elif is_correct is False:
        status = "wrong"
    else:
        status = ""

    st.markdown(
        f'<span class="conf-badge {badge_class}">{pred} {conf:.2f}</span>',
        unsafe_allow_html=True,
    )

    if is_correct is not None:
        if is_correct:
            st.caption(f"Gold: {gold}")
        else:
            st.markdown(
                f'<span style="color:#c0392b;font-size:0.75rem;">Gold: {gold}</span>',
                unsafe_allow_html=True,
            )

    # Year if available
    if "year" in item.index and pd.notna(item["year"]):
        st.caption(f"{int(item['year'])}")


def _find_image(item: pd.Series) -> Path | None:
    """Attempt to locate the raw image file for a prediction record."""
    # Try image_filename + month
    filename = item.get("image_filename")
    month = item.get("month")

    if filename and month:
        month_str = str(month)
        folder_name = f"{month_str[:4]}-{month_str[4:]}" if len(month_str) == 6 else month_str
        path = RAW_IMAGES_DIR / folder_name / filename
        if path.exists():
            return path

    # Try record_id-based lookup
    record_id = item.get("record_id")
    if record_id:
        for month_dir in RAW_IMAGES_DIR.iterdir():
            if month_dir.is_dir():
                for img in month_dir.iterdir():
                    if record_id in img.stem:
                        return img

    return None


def _get_latest_run() -> Path | None:
    if not EVAL_RUNS_DIR.exists():
        return None
    runs = sorted(EVAL_RUNS_DIR.iterdir(), reverse=True)
    for r in runs:
        if r.is_dir() and (r / "results.csv").exists():
            return r
    return None
