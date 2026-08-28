"""Item detail view: look up one product by record ID.

Discover's "Details" popover now shows an item's full info inline, so this
tab is a standalone lookup tool for when you already know a record ID and
don't want to browse for it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.app.components._shared import confidence_badge_class, render_attributes
from fashion_forensics.config import DATA_DIR
from fashion_forensics.db import get_item

CLASSIFICATIONS_PATH = (
    DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_classifications.jsonl"
)


def render_detail():
    """Look up one item by record ID and show its full image, trend, and
    attributes (if it's been through LLM normalization)."""
    st.markdown("### Item Detail")

    record_id = st.text_input("RECORD ID", placeholder="e.g. 2024-09_002")
    if not record_id:
        st.caption("Enter a record ID to look up an item.")
        return

    item = get_item(record_id=record_id)
    if item is None:
        st.warning(f"No item found for record_id '{record_id}'.")
        return

    st.image(item["image_path"], width=400)
    st.markdown(f"#### {item.get('product_name') or record_id}")
    if item.get("product_code"):
        st.caption(item["product_code"])

    trend_pred, confidence = _lookup_classification(record_id)
    if trend_pred is not None:
        badge_class = confidence_badge_class(confidence)
        st.markdown(
            f'<span class="conf-badge {badge_class}">{trend_pred} {confidence:.2f}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Not yet classified.")

    st.divider()
    st.markdown("##### Attributes")
    render_attributes(item.get("labels"))


def _lookup_classification(record_id: str) -> tuple[str | None, float | None]:
    if not CLASSIFICATIONS_PATH.exists():
        return None, None
    df = pd.read_json(CLASSIFICATIONS_PATH, lines=True)
    match = df[df["record_id"] == record_id]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return row["trend_pred"], row["confidence"]
