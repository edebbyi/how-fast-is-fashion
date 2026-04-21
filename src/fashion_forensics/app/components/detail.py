"""Item detail view: individual product inspection."""

from __future__ import annotations

import streamlit as st


def render_detail():
    """Render the item detail panel.

    Shows full image, normalized metadata, trend assignments,
    and confidence scores from the LLM mapper or LoRA.
    """
    st.markdown("### Item Detail")

    # TODO: Display selected item image at full resolution
    # TODO: Show normalized metadata (color, material, silhouette, etc.)
    # TODO: Show trend assignments with confidence scores
    # TODO: Show raw vs normalized metadata comparison

    st.info("Select an item from the Drill-down grid to inspect.")
