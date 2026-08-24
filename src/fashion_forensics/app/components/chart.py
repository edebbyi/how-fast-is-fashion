"""Tracker view: trend lifecycle states and frequency over time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.config import DATA_DIR

LIFECYCLE_STATES_PATH = DATA_DIR / "03_shared" / "trend_lifecycle" / "lifecycle_states.jsonl"
MONTHLY_FREQUENCY_PATH = DATA_DIR / "03_shared" / "trend_lifecycle" / "monthly_trend_frequency.csv"

TRENDS_PER_PAGE = 4

STATE_BADGE_CLASS = {
    "rising": "conf-high",
    "persistent": "conf-high",
    "seasonal_recurring": "conf-mid",
    "declining": "conf-low",
    "inactive": "conf-low",
}

# Display/priority order for states: currently-active states first (most
# stable to least), then currently-inactive states (most likely to return
# to least). Also doubles as the sort order for which trends show first.
STATE_ORDER = ["persistent", "rising", "declining", "seasonal_recurring", "inactive"]

# What each state MEANS in this system, in general terms - not tied to any
# one trend's specific numbers. Every trend in a given state gets the same
# text; the activity strip below is what shows the trend-specific history.
STATE_DESCRIPTIONS = {
    "persistent": "Been active for a while and holding roughly steady - not clearly growing or shrinking.",
    "rising": "Active and growing in catalog share, or new enough that it's too early to call it steady.",
    "declining": "Still active, but its catalog share has been dropping recently.",
    "seasonal_recurring": "Currently quiet, but has come back before - it comes and goes rather than staying gone for good.",
    "inactive": "Not showing up enough to count as active, with no pattern to suggest it's coming back.",
}


def _explain_state(state: str) -> str:
    """Plain-English definition of what a lifecycle state means in this
    system - the same text for every trend in that state."""
    return STATE_DESCRIPTIONS.get(state, "")


def _activity_strip_html(trend_freq: pd.DataFrame) -> str:
    """A row of small blocks, one per month: lit if the trend was active
    that month, dim if not. Shows the actual active/inactive pattern behind
    a lifecycle label directly, instead of just a summary number."""
    blocks = []
    for _, row in trend_freq.iterrows():
        color = "var(--fg)" if row["is_active"] else "var(--border)"
        label = row["date"].strftime("%b %Y")
        blocks.append(f'<div title="{label}" style="width:7px;height:16px;background:{color};"></div>')
    return '<div style="display:flex;gap:2px;">' + "".join(blocks) + "</div>"


def render_tracker():
    """Show each trend's current lifecycle state (with a plain-English
    reason and an activity timeline), and a chart of how often each trend
    has shown up in the catalog over time.

    The trend/year filters control the badges, activity strips, and chart
    all together. The year filter keeps the activity strip a fixed, readable width no
    matter how much history the catalog eventually accumulates (it grows
    by one block per month otherwise), by only drawing the selected year's
    months instead of the whole history.
    """
    st.markdown("### Trend States")

    if not LIFECYCLE_STATES_PATH.exists() or not MONTHLY_FREQUENCY_PATH.exists():
        st.info(
            "No lifecycle data yet. Run scripts/compute_trend_lifecycle.py "
            "(or scripts/run_pipeline.py) first."
        )
        return

    states_df = pd.read_json(LIFECYCLE_STATES_PATH, lines=True)
    freq_df = pd.read_csv(MONTHLY_FREQUENCY_PATH)
    freq_df = freq_df[freq_df["trend"] != "unknown"].copy()
    freq_df["date"] = pd.to_datetime(freq_df["date"])
    freq_df = freq_df.sort_values("date")

    # Sorted by state priority (persistent/rising first, inactive last), not
    # alphabetically by name - so the most active trends show first as the
    # taxonomy grows past what fits in one row.
    state_by_trend = states_df.set_index("trend")["state"].to_dict()

    def _state_rank(trend: str) -> int:
        state = state_by_trend.get(trend)
        return STATE_ORDER.index(state) if state in STATE_ORDER else len(STATE_ORDER)

    trends = sorted(states_df["trend"].unique(), key=lambda t: (_state_rank(t), t))
    years = sorted(freq_df["date"].dt.year.unique().tolist())

    # --- Filters, control the badges/strips below AND the chart ---
    st.caption("Filter the trend states and chart below.")
    col_trend, col_year = st.columns(2)
    with col_trend:
        selected_trend = st.selectbox("TREND", ["All", *trends], key="selected_trend")
    with col_year:
        selected_year = st.selectbox("YEAR", ["All", *years])

    strip_freq = freq_df if selected_year == "All" else freq_df[freq_df["date"].dt.year == selected_year]

    # --- Pick which trends to show as badges: the one selected in the
    # filter, or a page of "All" trends (paginated once there are more than
    # fit comfortably in one row - today that's 4, but the taxonomy has 8
    # more trends waiting in configs/trend_rules.yaml's deferred_trends). ---
    if selected_trend != "All":
        display_trends = [selected_trend]
    elif len(trends) <= TRENDS_PER_PAGE:
        display_trends = trends
    else:
        total_pages = (len(trends) + TRENDS_PER_PAGE - 1) // TRENDS_PER_PAGE
        page = st.session_state.get("trend_page", 0)
        page = max(0, min(page, total_pages - 1))

        col_prev, col_indicator, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀", disabled=page == 0, key="trend_page_prev"):
                page -= 1
        with col_indicator:
            st.caption(f"Trends {page * TRENDS_PER_PAGE + 1}-{min((page + 1) * TRENDS_PER_PAGE, len(trends))} of {len(trends)}")
        with col_next:
            if st.button("▶", disabled=page >= total_pages - 1, key="trend_page_next"):
                page += 1
        st.session_state.trend_page = page

        start = page * TRENDS_PER_PAGE
        display_trends = trends[start : start + TRENDS_PER_PAGE]

    # --- State badges: one column per shown trend ---
    cols = st.columns(len(display_trends))
    for col, trend in zip(cols, display_trends, strict=True):
        row = states_df[states_df["trend"] == trend].iloc[0]
        badge_class = STATE_BADGE_CLASS.get(row["state"], "conf-low")
        with col:
            st.markdown(f'<p class="trend-label">{trend}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="conf-badge {badge_class}">{row["state"]}</span>',
                unsafe_allow_html=True,
            )
            st.caption(_explain_state(row["state"]))
            # The label above always describes the trend's FULL history
            # (that's what the classifier actually computed) - the strip
            # can be narrowed to one year via the filter, so make that
            # scope difference explicit instead of leaving it to guesswork.
            if selected_year != "All":
                st.caption(f"Strip shows {selected_year} only; the label reflects full history.")
            trend_freq = strip_freq[strip_freq["trend"] == trend]
            st.markdown(_activity_strip_html(trend_freq), unsafe_allow_html=True)

    st.divider()

    # --- Chart, filtered the same way ---
    chart_freq = freq_df.copy()
    if selected_trend != "All":
        chart_freq = chart_freq[chart_freq["trend"] == selected_trend]
    if selected_year != "All":
        chart_freq = chart_freq[chart_freq["date"].dt.year == selected_year]

    # Drilling into one year: month-only labels ("Feb", "Mar", ...) are
    # enough since the year is already fixed by the filter, and keeps the
    # x-axis a lot cleaner than spelling out all ~23 months at once.
    # Viewing "All" years: keep the year in the label ("Feb 2024") since
    # plain month names would otherwise repeat and be ambiguous.
    label_format = "%b" if selected_year != "All" else "%b %Y"
    month_order = chart_freq["date"].dt.strftime(label_format).unique().tolist()
    chart_freq["month_label"] = pd.Categorical(
        chart_freq["date"].dt.strftime(label_format), categories=month_order, ordered=True
    )

    st.markdown("##### Zara Catalog share by month")
    pivoted = chart_freq.pivot(index="month_label", columns="trend", values="trend_frequency")
    st.line_chart(pivoted)
