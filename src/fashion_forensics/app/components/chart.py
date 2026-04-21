"""Tracker view: trend state charts over time."""

from __future__ import annotations

import streamlit as st


def render_tracker():
    """Render the main trend tracker chart.

    Shows trend states (rising/falling/stable) over years.
    Clicking a trend on the chart filters the drill-down grid.
    """
    st.markdown("### Trend States")
    st.caption("Select a trend to filter the image grid in Drill-down.")

    # TODO: Load trend_metrics_yearly.parquet
    # TODO: Render chart with trend state colors
    # TODO: On click, set st.session_state.selected_trend for drill-down filtering

    st.info("Connect trend_metrics_yearly.parquet to activate this view.")
