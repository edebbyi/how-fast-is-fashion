"""Fashion Forensics — dual-audience Streamlit app.

Two modes:
    Fashion view: non-technical, image-forward, editorial aesthetic
    Lab view: technical, model metrics, threshold tuning, image queries
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Fashion Forensics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimalist black/white theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --bg: #0a0a0a;
        --fg: #fafafa;
        --muted: #666;
        --border: #222;
        --accent: #fff;
        --correct: #2d6a4f;
        --wrong: #6b2020;
        --near-miss: #7c6f2a;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--fg);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-weight: 500;
        letter-spacing: -0.02em;
    }

    .trend-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 300;
    }

    /* Image grid */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 2px;
    }

    .image-grid img {
        width: 100%;
        aspect-ratio: 3/4;
        object-fit: cover;
        filter: grayscale(20%);
        transition: filter 0.3s ease;
    }

    .image-grid img:hover {
        filter: grayscale(0%);
    }

    /* Confidence badge */
    .conf-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .conf-high { background: var(--correct); color: #fff; }
    .conf-mid { background: var(--near-miss); color: #fff; }
    .conf-low { background: var(--wrong); color: #fff; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 24px;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Top-level audience switch
    mode = st.radio(
        "",
        ["Fashion", "Lab"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Fashion":
        _render_fashion_view()
    else:
        _render_lab_view()


def _render_fashion_view():
    """Non-technical, image-forward, editorial aesthetic."""
    st.markdown("# Fashion Forensics")
    st.markdown(
        '<p class="trend-label">What\'s moving in fast fashion</p>',
        unsafe_allow_html=True,
    )

    tab_trends, tab_discover, tab_item = st.tabs(["Trends", "Discover", "Detail"])

    with tab_trends:
        from fashion_forensics.app.components.chart import render_tracker

        render_tracker()

    with tab_discover:
        from fashion_forensics.app.components.grid import render_drilldown

        render_drilldown()

    with tab_item:
        from fashion_forensics.app.components.detail import render_detail

        render_detail()


def _render_lab_view():
    """Technical view: model performance, history, and image-level queries."""
    st.markdown("# Fashion Forensics Lab")
    st.markdown(
        '<p class="trend-label">Model evaluation & threshold tuning</p>',
        unsafe_allow_html=True,
    )

    tab_perf, tab_query, tab_history = st.tabs(["Performance", "Image Query", "Run History"])

    with tab_perf:
        from fashion_forensics.app.components.performance import render_performance

        render_performance()

    with tab_query:
        from fashion_forensics.app.components.query import render_image_query

        render_image_query()

    with tab_history:
        from fashion_forensics.app.components.history import render_run_history

        render_run_history()


if __name__ == "__main__":
    main()
