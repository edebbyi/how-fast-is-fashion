"""Streamlit review app for trend curation.

Shows pending scraped candidates with their source attribution and lets a
reviewer approve (into a trend bucket in the labeled reference corpus) or
reject them. Talks to the FastAPI backend (fashion_forensics.curation.api)
over HTTP.

Run with (backend first):
    .venv/bin/uvicorn fashion_forensics.curation.api:app --port 8010
    .venv/bin/streamlit run src/fashion_forensics/curation/review_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from fashion_forensics.config import settings  # noqa: F401  (triggers .env loading)

API_BASE_URL = os.environ.get("CURATION_API_BASE_URL", "http://localhost:8010")

st.set_page_config(page_title="Trend Curator", layout="wide")
st.title("Trend Curator")
st.caption("Review scraped images before they join the labeled reference corpus.")


def _api_get(path: str, **params):
    resp = requests.get(f"{API_BASE_URL}{path}", params=params or None, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, json: dict | None = None, timeout: int = 15):
    resp = requests.post(f"{API_BASE_URL}{path}", json=json or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


try:
    queue_stats = _api_get("/stats")
    known_trends = _api_get("/trends")["trends"]
except requests.RequestException:
    st.error(
        f"Can't reach the curation API at {API_BASE_URL}. Start it with:\n\n"
        "`.venv/bin/uvicorn fashion_forensics.curation.api:app --port 8010`"
    )
    st.stop()

with st.sidebar:
    st.subheader("Queue")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending", queue_stats.get("pending", 0))
    c2.metric("Approved", queue_stats.get("approved", 0))
    c3.metric("Rejected", queue_stats.get("rejected", 0))

    if queue_stats.get("approved_by_trend"):
        st.caption("Approved by trend")
        st.table(queue_stats["approved_by_trend"])

    st.divider()
    st.subheader("Search the web")
    search_query = st.text_input(
        "Query", placeholder='e.g. "clean luxury fashion outfit"', key="search_query"
    )
    search_trend = st.selectbox("Tag results as", ["Unsorted"] + known_trends, key="search_trend")
    search_max = st.number_input(
        "Max results", min_value=5, max_value=100, value=20, step=5, key="search_max"
    )
    if st.button(
        "Search & queue", use_container_width=True, disabled=not search_query.strip()
    ):
        with st.spinner(f"Searching {search_query!r}..."):
            try:
                summary = _api_post(
                    "/search",
                    {
                        "query": search_query.strip(),
                        "max_results": int(search_max),
                        "trend_hint": None if search_trend == "Unsorted" else search_trend,
                    },
                    timeout=120,
                )
                st.success(f"Queued {summary['added']} new (of {summary['found']} found)")
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
        st.rerun()

    st.divider()
    st.subheader("Scrape a page")
    st.caption("Paste any page URL — a Pinterest board/pin, an editorial roundup, etc.")
    page_url = st.text_input(
        "Page URL", placeholder="https://www.pinterest.com/username/board-name/", key="page_url"
    )
    page_trend = st.selectbox("Tag results as", ["Unsorted"] + known_trends, key="page_trend")
    if st.button("Scrape page", use_container_width=True, disabled=not page_url.strip()):
        with st.spinner(f"Scraping {page_url!r}..."):
            try:
                summary = _api_post(
                    "/scrape-url",
                    {
                        "url": page_url.strip(),
                        "trend_hint": None if page_trend == "Unsorted" else page_trend,
                    },
                    timeout=120,
                )
                st.success(f"Queued {summary['added']} new (of {summary['found']} found)")
            except requests.RequestException as e:
                st.error(f"Scrape failed: {e}")
        st.rerun()

    st.divider()
    if st.button("Scrape configured sources", use_container_width=True):
        with st.spinner("Scraping sources from configs/curation.yaml..."):
            try:
                summary = _api_post("/scrape", timeout=300)
                st.success("Scrape complete")
                st.json(summary)
            except requests.RequestException as e:
                st.error(f"Scrape failed: {e}")
        st.rerun()

    st.divider()
    filter_choice = st.selectbox(
        "Filter by suggested trend", ["All", "Unsorted"] + known_trends, key="filter_trend"
    )

filter_trend_hint: str | None
if filter_choice == "All":
    filter_trend_hint = None
elif filter_choice == "Unsorted":
    filter_trend_hint = ""  # sentinel handled below
else:
    filter_trend_hint = filter_choice

params = {"status": "pending"}
if filter_trend_hint:
    params["trend_hint"] = filter_trend_hint
candidates = _api_get("/candidates", **params)["candidates"]
if filter_choice == "Unsorted":
    candidates = [c for c in candidates if not c.get("trend_hint")]

if not candidates:
    st.info(
        "No pending candidates. Search or scrape a page from the sidebar, or adjust the filter."
    )
else:
    st.caption(f"{len(candidates)} pending review")
    columns = st.columns(3)
    for i, candidate in enumerate(candidates):
        with columns[i % 3]:
            with st.container(border=True):
                image_url = f"{API_BASE_URL}/candidates/{candidate['id']}/image"
                st.image(image_url, use_container_width=True)
                st.markdown(f"**[{candidate['source_name']}]({candidate['source_url']})**")
                if candidate.get("alt_text"):
                    st.caption(candidate["alt_text"])

                trend_options = known_trends or ["quiet_luxury", "mob_wife", "office_siren"]
                default_idx = (
                    trend_options.index(candidate["trend_hint"])
                    if candidate.get("trend_hint") in trend_options
                    else 0
                )
                trend_choice = st.selectbox(
                    "Trend",
                    trend_options,
                    index=default_idx,
                    key=f"trend_{candidate['id']}",
                    label_visibility="collapsed",
                )

                approve_col, reject_col = st.columns(2)
                if approve_col.button(
                    "Approve", key=f"approve_{candidate['id']}", use_container_width=True
                ):
                    _api_post(f"/candidates/{candidate['id']}/approve", {"trend": trend_choice})
                    st.rerun()
                if reject_col.button(
                    "Reject", key=f"reject_{candidate['id']}", use_container_width=True
                ):
                    _api_post(f"/candidates/{candidate['id']}/reject")
                    st.rerun()
