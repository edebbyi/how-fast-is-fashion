"""Drill-down view: filterable image grid of real catalog items."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.app.components._shared import (
    confidence_badge_class,
    render_attributes,
    resolve_image_paths,
)
from fashion_forensics.config import DATA_DIR
from fashion_forensics.nlp.tfidf_engine import load_normalized_records

CLASSIFICATIONS_PATH = (
    DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_classifications.jsonl"
)
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"


def render_drilldown():
    """Show a grid of catalog items, filterable by trend and month.

    Defaults to whatever trend was picked in the Trends tab, if any. Click
    "Details" on any item to see its full info right there, no need to
    switch tabs.
    """
    st.markdown("### Discover")

    if not CLASSIFICATIONS_PATH.exists():
        st.info("No catalog classifications yet. Run scripts/classify_catalog.py first.")
        return

    df = pd.read_json(CLASSIFICATIONS_PATH, lines=True)

    # Loaded once, up front - also used below to filter for "has
    # attributes" and to look up each shown card's attributes without a
    # per-card lookup (a Details popover's content still runs on every
    # rerun even when closed, so 100 individual lookups would be wasteful).
    attrs_by_id = {r["record_id"]: r for r in load_normalized_records()}

    # --- Filters ---
    col_trend, col_unknown, col_month, col_attrs = st.columns(4)

    trends = sorted(t for t in df["trend_pred"].unique() if t != "unknown")

    with col_trend:
        # Uses its own widget key (Streamlit runs every tab's code on every
        # rerun, not just the visible tab, so this can't share a key with
        # the Trends tab's selectbox - that caused a duplicate-key error).
        # Still starts from whatever was picked in Trends, via
        # st.session_state["selected_trend"], so the filter carries over.
        trend_options = ["All", *trends]
        default_trend = st.session_state.get("selected_trend", "All")
        default_index = trend_options.index(default_trend) if default_trend in trend_options else 0
        selected_trend = st.selectbox("TREND", trend_options, index=default_index, key="discover_trend")

    with col_unknown:
        include_unknown = st.checkbox("Include unclassified", value=False)

    with col_month:
        months = sorted(df["month"].unique())
        selected_month = st.selectbox("MONTH", ["All", *months])

    with col_attrs:
        # Only ~3% of the catalog has been through LLM attribute
        # normalization - sorted by confidence, the default view rarely
        # happens to show one of them. This filter makes them findable.
        only_with_attrs = st.checkbox(
            "Only items with attributes", value=True, help=f"{len(attrs_by_id)} items have these"
        )

    filtered = df.copy()
    if selected_trend != "All":
        filtered = filtered[filtered["trend_pred"] == selected_trend]
    if not include_unknown:
        filtered = filtered[filtered["trend_pred"] != "unknown"]
    if selected_month != "All":
        filtered = filtered[filtered["month"] == selected_month]
    if only_with_attrs:
        filtered = filtered[filtered["record_id"].isin(attrs_by_id)]

    st.caption(f"Showing {len(filtered)} items")
    st.divider()

    if filtered.empty:
        st.caption("No items match your filters.")
        return

    filtered = filtered.sort_values("confidence", ascending=False)

    cols_per_row = 4
    max_rows = 25

    # Only resolve image paths for the items we're actually about to show,
    # not the whole filtered set - keeps the lookup query small. We already
    # have "month" from the classification data, so only pull
    # image_filename from the lookup (both sides having a "month" column
    # would make pandas rename them to month_x/month_y on merge).
    shown = filtered.head(cols_per_row * max_rows)
    image_info = resolve_image_paths(shown["record_id"].tolist())[["record_id", "image_filename"]]
    filtered = shown.merge(image_info, on="record_id", how="left")

    rows = min((len(filtered) + cols_per_row - 1) // cols_per_row, max_rows)
    for row_idx in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx >= len(filtered):
                break
            item = filtered.iloc[item_idx]
            with cols[col_idx]:
                _render_card(item, attrs_by_id.get(item["record_id"]))


def _render_card(item: pd.Series, labels: dict | None) -> None:
    """Render one item's image, badge, and a Details popover with its full
    info (product name, trend, and attributes if available)."""
    img_path = None
    if pd.notna(item.get("image_filename")):
        img_path = RAW_IMAGES_DIR / item["month"] / item["image_filename"]

    if img_path and img_path.exists():
        st.image(str(img_path), width="stretch")
    else:
        st.markdown(
            '<div style="aspect-ratio:3/4;background:#1a1a1a;display:flex;'
            'align-items:center;justify-content:center;color:#444;font-size:0.7rem;">'
            "Image not found</div>",
            unsafe_allow_html=True,
        )

    badge_class = confidence_badge_class(item.get("confidence"))
    st.markdown(
        f'<span class="conf-badge {badge_class}">{item["trend_pred"]} {item["confidence"]:.2f}</span>',
        unsafe_allow_html=True,
    )
    st.caption(item["month"])

    with st.popover("Details"):
        st.markdown(f"**{item.get('product_name') or item['record_id']}**")
        if item.get("product_code"):
            st.caption(item["product_code"])
        st.markdown(
            f'<span class="conf-badge {badge_class}">{item["trend_pred"]} {item["confidence"]:.2f}</span>',
            unsafe_allow_html=True,
        )
        st.divider()
        render_attributes(labels)
        st.divider()
        # Copyable so this item can be pulled up again later via the
        # Detail tab's record ID lookup, without re-browsing to find it.
        st.caption("Record ID")
        st.code(item["record_id"], language=None)
