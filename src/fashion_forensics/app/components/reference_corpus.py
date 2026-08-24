"""Lab view: browse the labeled reference corpus the classifier votes against.

Every catalog item gets classified by finding its nearest neighbors among
these 135 labeled images (ARCHITECTURE.md §3.5) - so a trend's classification
quality is only as good as this corpus's curation. This view exists to make
that curation visible and inspectable, ahead of the quiet_luxury re-curation
work.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.config import PROJECT_ROOT, DATA_DIR

ATTRIBUTES_PATH = DATA_DIR / "02_reference_corpus" / "attributes.jsonl"


def render_reference_corpus():
    st.markdown("### Reference Images")

    if not ATTRIBUTES_PATH.exists():
        st.info("No reference corpus captions found. Run scripts/caption_reference_corpus.py first.")
        return

    df = pd.read_json(ATTRIBUTES_PATH, lines=True)

    trends = sorted(df["trend"].unique())
    col_trend, col_count = st.columns([3, 1])
    with col_trend:
        selected_trend = st.selectbox("TREND", ["All", *trends])
    with col_count:
        st.caption(f"{len(df)} reference images total")

    filtered = df if selected_trend == "All" else df[df["trend"] == selected_trend]

    counts = df["trend"].value_counts()
    st.caption(" · ".join(f"{t}: {counts.get(t, 0)}" for t in trends))
    st.divider()

    if filtered.empty:
        st.caption("No reference images for this trend.")
        return

    cols_per_row = 5
    rows = (len(filtered) + cols_per_row - 1) // cols_per_row
    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx >= len(filtered):
                break
            item = filtered.iloc[item_idx]
            with cols[col_idx]:
                _render_reference_card(item)


def _render_reference_card(item: pd.Series) -> None:
    img_path = PROJECT_ROOT / item["image_path"]
    if img_path.exists():
        st.image(str(img_path), width="stretch")
    else:
        st.markdown(
            '<div style="aspect-ratio:3/4;background:#1a1a1a;display:flex;'
            'align-items:center;justify-content:center;color:#444;font-size:0.7rem;">'
            "Image not found</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f'<span class="conf-badge conf-mid">{item["trend"]}</span>', unsafe_allow_html=True)

    with st.popover("Details"):
        st.caption(item["record_id"])
        for field in (
            "category",
            "subcategory",
            "silhouette",
            "material",
            "sleeve_style",
            "neckline",
            "length",
            "color_profile",
            "pattern",
            "details",
        ):
            block = item.get(field)
            if block and block.get("value"):
                st.caption(f"{field}: {', '.join(block['value'])}")
