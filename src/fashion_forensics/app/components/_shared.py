"""Small helpers shared by more than one tab (grid, detail, recommendations)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.db import connect

ATTRIBUTE_FIELDS = (
    "silhouette",
    "material",
    "sleeve_style",
    "neckline",
    "length",
    "color_profile",
    "pattern",
    "details",
)


def resolve_image_paths(record_ids: list[str]) -> pd.DataFrame:
    """Look up the image filename and month for a list of record_ids, via
    the items table. Returns a DataFrame with columns
    [record_id, image_filename, month] - one row per record_id that was
    found (missing ones just don't get a row).
    """
    if not record_ids:
        return pd.DataFrame(columns=["record_id", "image_filename", "month"])
    con = connect()
    placeholders = ",".join("?" * len(record_ids))
    rows = con.execute(
        f"SELECT record_id, image_filename, month FROM items WHERE record_id IN ({placeholders})",
        record_ids,
    ).fetchall()
    return pd.DataFrame(rows, columns=["record_id", "image_filename", "month"])


def confidence_badge_class(confidence: float | None) -> str:
    """Turn a confidence score into one of the three badge CSS classes
    already defined in app.py (conf-high/conf-mid/conf-low)."""
    if confidence is None:
        return "conf-low"
    if confidence >= 0.8:
        return "conf-high"
    if confidence >= 0.7:
        return "conf-mid"
    return "conf-low"


def render_attributes(labels: dict | None) -> None:
    """Show an item's normalized attributes (silhouette, material, etc.) as
    captions, or a friendly message if this item hasn't been through LLM
    normalization yet - most catalog items haven't, so this is expected,
    not an error."""
    if not labels:
        st.caption("No detailed attributes available for this item.")
        return
    for field in ATTRIBUTE_FIELDS:
        block = labels.get(field)
        if block and block.get("value"):
            st.caption(f"{field}: {', '.join(block['value'])}")
