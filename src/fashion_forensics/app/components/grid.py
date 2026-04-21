"""Drill-down view: filterable image grid colored by trend characteristics."""

from __future__ import annotations

import streamlit as st


def render_drilldown():
    """Render the image drill-down grid.

    Filters by trend name selected from the tracker chart.
    Grid border/accent colors are influenced by trend characteristics
    (e.g., "plum, burgundy" for a color-driven trend).
    """
    st.markdown("### Drill-down")

    selected = st.session_state.get("selected_trend")
    if selected:
        st.markdown(f"Showing items matching **{selected}**")
    else:
        st.caption("Select a trend from the Tracker to filter.")

    # TODO: Load item_features.parquet filtered by selected trend
    # TODO: Render image grid with dynamic accent colors from trend keywords
    # TODO: Support year/category filters

    st.info("Connect item_features.parquet and trend mappings to activate this view.")
