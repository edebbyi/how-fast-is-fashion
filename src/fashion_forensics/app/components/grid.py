"""Drill-down view: filterable image grid of real catalog items."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.app.components._shared import (
    confidence_badge_class,
    render_attributes,
)
from fashion_forensics.catalog_ground_truth import judge_items_batch, load_ground_truth
from fashion_forensics.config import DATA_DIR
from fashion_forensics.db import connect
from fashion_forensics.llm_judge import group_by_trend, judge_items
from fashion_forensics.nlp.tfidf_engine import load_normalized_records
from fashion_forensics.nlp.time_series_aggregation import load_canonical_trends

CLASSIFICATIONS_PATH = (
    DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_classifications.jsonl"
)
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"

COLS_PER_ROW = 4
ITEMS_PER_PAGE = 100  # 25 rows x 4 cols

# Every widget interaction (a filter change, a page click, a thumbs-up)
# triggers a full Streamlit rerun of this whole function - without
# caching, that meant re-reading the ~2,000-item classifications file,
# re-globbing every month's normalized-attributes file, and opening a
# fresh DuckDB connection that rescans the whole raw catalog just to
# resolve ~100 image paths, on every single click. Cached below with a
# 5-minute TTL: long enough that reviewing hundreds of items in a row is
# actually fast, short enough that a reclassify (which already takes
# ~30 minutes and is a deliberate, rare action) shows up again without
# needing to restart the app.
_CACHE_TTL_SECONDS = 300


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_classifications() -> pd.DataFrame:
    return pd.read_json(CLASSIFICATIONS_PATH, lines=True)


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_attrs_by_id() -> dict[str, dict]:
    return {r["record_id"]: r for r in load_normalized_records()}


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_image_index() -> pd.DataFrame:
    """record_id -> image_filename/month for the whole catalog, in one
    query - the per-page lookup below then just filters this in pandas
    instead of hitting DuckDB again for every rerun."""
    con = connect()
    rows = con.execute("SELECT record_id, image_filename, month FROM items").fetchall()
    return pd.DataFrame(rows, columns=["record_id", "image_filename", "month"])


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

    df = _load_classifications()

    # Loaded once, up front - also used below to filter for "has
    # attributes" and to look up each shown card's attributes without a
    # per-card lookup (a Details popover's content still runs on every
    # rerun even when closed, so 100 individual lookups would be wasteful).
    attrs_by_id = _load_attrs_by_id()

    # Also loaded once, up front, same reasoning as attrs_by_id above - lets
    # each card show "already judged" without a per-card file read.
    ground_truth_by_id = load_ground_truth()

    # --- Filters ---
    col_trend, col_unknown, col_month, col_attrs, col_judgment = st.columns(5)

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
        selected_trend = st.selectbox(
            "TREND", trend_options, index=default_index, key="discover_trend"
        )

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

    with col_judgment:
        # Defaults to "All" (no filter) on purpose - opening Discover
        # shouldn't already be narrowed to a judgment subset. Useful once
        # you're actually collecting ground truth: "Unjudged" to find fresh
        # items, or 👍/👎 to review what's already been judged.
        judgment_filter = st.selectbox(
            "JUDGMENT", ["All", "👍 Correct", "👎 Incorrect", "Unjudged"], key="discover_judgment"
        )

    filtered = df.copy()
    if selected_trend != "All":
        filtered = filtered[filtered["trend_pred"] == selected_trend]
    if not include_unknown:
        filtered = filtered[filtered["trend_pred"] != "unknown"]
    if selected_month != "All":
        filtered = filtered[filtered["month"] == selected_month]
    if judgment_filter == "👍 Correct":
        judged_ids = {rid for rid, r in ground_truth_by_id.items() if r["judged_correct"]}
        filtered = filtered[filtered["record_id"].isin(judged_ids)]
    elif judgment_filter == "👎 Incorrect":
        judged_ids = {rid for rid, r in ground_truth_by_id.items() if not r["judged_correct"]}
        filtered = filtered[filtered["record_id"].isin(judged_ids)]
    elif judgment_filter == "Unjudged":
        filtered = filtered[~filtered["record_id"].isin(ground_truth_by_id)]
    if only_with_attrs:
        filtered = filtered[filtered["record_id"].isin(attrs_by_id)]

    st.caption(
        f"Showing {len(filtered)} items - {len(ground_truth_by_id)} judged so far "
        "(select 👍/👎 on any cards, then Submit at the bottom to save them all at once)"
    )
    st.divider()

    if filtered.empty:
        st.caption("No items match your filters.")
        return

    filtered = filtered.sort_values("confidence", ascending=False)

    if len(filtered) <= ITEMS_PER_PAGE:
        display_items = filtered
    else:
        total_pages = (len(filtered) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        page = st.session_state.get("discover_page", 0)
        page = max(0, min(page, total_pages - 1))

        col_prev, col_indicator, col_next = st.columns([1, 4, 1])
        with col_prev:
            if st.button("◀", disabled=page == 0, key="discover_page_prev"):
                page -= 1
        with col_indicator:
            start_n = page * ITEMS_PER_PAGE + 1
            end_n = min((page + 1) * ITEMS_PER_PAGE, len(filtered))
            st.caption(f"Items {start_n}-{end_n} of {len(filtered)}")
        with col_next:
            if st.button("▶", disabled=page >= total_pages - 1, key="discover_page_next"):
                page += 1
        st.session_state.discover_page = page

        start = page * ITEMS_PER_PAGE
        display_items = filtered.iloc[start : start + ITEMS_PER_PAGE]

    # _load_image_index() is cached for the whole catalog - filtering it
    # down to this page's record_ids here is a pandas lookup, not a fresh
    # DuckDB query. We already have "month" from the classification data,
    # so only pull image_filename from the lookup (both sides having a
    # "month" column would make pandas rename them to month_x/month_y on
    # merge).
    image_info = _load_image_index()
    image_info = image_info[image_info["record_id"].isin(display_items["record_id"])][
        ["record_id", "image_filename"]
    ]
    display_items = display_items.merge(image_info, on="record_id", how="left")

    # AI-assisted first pass, not a replacement for review - reviewing
    # 1000+ items by hand is genuinely slow (RETROSPECTIVE.md section 9).
    # A real LLM call per item, so this is a real (small) cost per click -
    # shown up front, triggered explicitly, never automatic. Outside the
    # form on purpose: this is a fetch action, not something to batch into
    # the Submit click below.
    unjudged_ids = [rid for rid in display_items["record_id"] if rid not in ground_truth_by_id]
    st.caption(
        f"🤖 {len(unjudged_ids)} unjudged item(s) on this page - AI suggestions call the LLM "
        "once per item (a real, small cost), pre-filling verdict/reason/corrected-trend below "
        "as an editable starting point. You still confirm before submitting."
    )
    if st.button(
        f"🤖 Get AI suggestions for {len(unjudged_ids)} unjudged item(s) on this page",
        disabled=not unjudged_ids,
    ):
        trend_pred_by_id = dict(zip(display_items["record_id"], display_items["trend_pred"]))
        with st.spinner(f"Asking the LLM about {len(unjudged_ids)} item(s)..."):
            suggestions = judge_items(group_by_trend(unjudged_ids, trend_pred_by_id))
        for record_id, suggestion in suggestions.items():
            st.session_state[f"judge_verdict_{record_id}"] = "👍" if suggestion["correct"] else "👎"
            if not suggestion["correct"]:
                st.session_state[f"judge_correction_{record_id}"] = (
                    suggestion["corrected_trend"] or "(not specified)"
                )
                st.session_state[f"judge_reason_{record_id}"] = suggestion["reasoning"]
        st.success(f"Got {len(suggestions)}/{len(unjudged_ids)} AI suggestion(s) - review below.")
        st.rerun()

    # Everything below lives inside one form: selecting a verdict (or a
    # reason/corrected-trend in a card's Details popover) causes no server
    # round-trip at all - only the Submit button does, and it saves every
    # selection on the page in one batch. This replaced a per-click
    # instant-save model that made reviewing hundreds of items slow (every
    # click was a full round-trip) and, per real usage, occasionally
    # unreliable. See RETROSPECTIVE.md section 9.
    pending: list[dict] = []
    with st.form("discover_judgments"):
        submitted_top = st.form_submit_button(
            "Submit judgments on this page", key="discover_submit_top"
        )
        rows = (len(display_items) + COLS_PER_ROW - 1) // COLS_PER_ROW
        for row_idx in range(rows):
            cols = st.columns(COLS_PER_ROW)
            for col_idx in range(COLS_PER_ROW):
                item_idx = row_idx * COLS_PER_ROW + col_idx
                if item_idx >= len(display_items):
                    break
                item = display_items.iloc[item_idx]
                with cols[col_idx]:
                    pending.append(
                        _render_card(
                            item,
                            attrs_by_id.get(item["record_id"]),
                            ground_truth_by_id.get(item["record_id"]),
                        )
                    )
        submitted_bottom = st.form_submit_button(
            "Submit judgments on this page", key="discover_submit_bottom"
        )

    if submitted_top or submitted_bottom:
        to_save = []
        for entry in pending:
            if entry["verdict"] is None:
                continue
            judged_correct = entry["verdict"] == "👍"
            # reason/corrected_trend only mean anything for a 👎 - dropped
            # here rather than saved unused if someone filled them in on a
            # card they ultimately marked 👍.
            # .strip() or None normalizes the text_input's default "" the
            # same way judge_items_batch() would when actually saving -
            # without this, an untouched empty reason box (widget value
            # "") never compares equal to an already-saved empty reason
            # (None), so the "did this actually change" check below would
            # never match and every unwritten card would look "changed".
            reason = (entry["reason"].strip() or None) if entry["verdict"] == "👎" else None
            corrected_trend = entry["corrected_trend"] if entry["verdict"] == "👎" else None

            # Already-judged cards default to their saved verdict (so
            # leaving one alone preserves it on resubmit) - but that means
            # every already-judged card on the page is "selected" on every
            # submit, even ones nobody touched. Skip anything that matches
            # what's already saved, so judged_at only updates for cards
            # that actually changed, not the whole page every time.
            original = entry["original_judgment"]
            if (
                original is not None
                and original["judged_correct"] == judged_correct
                and (original.get("reason") or None) == reason
                and (original.get("corrected_trend") or None) == corrected_trend
            ):
                continue

            to_save.append(
                {
                    "record_id": entry["record_id"],
                    "trend_pred": entry["trend_pred"],
                    "confidence": entry["confidence"],
                    "max_sim": entry["max_sim"],
                    "judged_correct": judged_correct,
                    "reason": reason,
                    "corrected_trend": corrected_trend,
                }
            )

        if to_save:
            judge_items_batch(to_save)
            st.success(f"Saved {len(to_save)} judgment(s).")
            st.rerun()
        else:
            st.info("No verdicts selected - pick 👍/👎 on at least one card before submitting.")


def _render_card(item: pd.Series, labels: dict | None, judgment: dict | None) -> dict:
    """Render one item's image, badge, verdict selector, and a Details
    popover (product info, attributes, and - only meaningful for a 👎 -
    an optional reason and corrected-trend picker). Returns the card's
    current form values; nothing is saved here. render_drilldown() collects
    one of these per card and writes them all in a single batch when the
    form's Submit button is clicked - see RETROSPECTIVE.md section 9 for
    why (this used to save-and-rerun on every single click, which was both
    slow across hundreds of items and, per real usage, occasionally
    unreliable)."""
    record_id = item["record_id"]
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
        f'<span class="conf-badge {badge_class}">'
        f"{item['trend_pred']} {item['confidence']:.2f}</span>",
        unsafe_allow_html=True,
    )
    st.caption(item["month"])

    # Pre-selected to the currently-saved verdict, if any - leaving it
    # alone and resubmitting keeps that judgment rather than clearing it.
    # None (nothing highlighted) means "skip this card in this submission".
    default_verdict = None
    if judgment is not None:
        default_verdict = "👍" if judgment["judged_correct"] else "👎"
    verdict = st.segmented_control(
        "Verdict",
        ["👍", "👎"],
        default=default_verdict,
        key=f"judge_verdict_{record_id}",
        label_visibility="collapsed",
    )

    with st.popover("Details"):
        st.markdown(f"**{item.get('product_name') or record_id}**")
        if item.get("product_code"):
            st.caption(item["product_code"])
        st.markdown(
            f'<span class="conf-badge {badge_class}">'
            f"{item['trend_pred']} {item['confidence']:.2f}</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        render_attributes(labels)
        st.divider()
        # Copyable so this item can be pulled up again later via the
        # Detail tab's record ID lookup, without re-browsing to find it.
        st.caption("Record ID")
        st.code(record_id, language=None)
        st.divider()

        # Only meaningful for a 👎, but rendered regardless of the verdict
        # selector above - a form can't react to another widget's
        # in-progress selection without a rerun, which is exactly what
        # this form is avoiding. Dropped at submit time (see
        # render_drilldown()) unless this card actually ends up marked 👎.
        correction_options = ["(not specified)", *load_canonical_trends(), "unknown"]
        default_correction = (judgment.get("corrected_trend") if judgment else None) or (
            "(not specified)"
        )
        corrected_trend = st.selectbox(
            "If wrong, what should it be?",
            correction_options,
            index=correction_options.index(default_correction)
            if default_correction in correction_options
            else 0,
            key=f"judge_correction_{record_id}",
        )
        reason = st.text_input(
            "Why? (optional)",
            value=(judgment.get("reason") if judgment else None) or "",
            key=f"judge_reason_{record_id}",
            placeholder="e.g. wrong material",
        )

    return {
        "record_id": record_id,
        "trend_pred": item.get("trend_pred"),
        "confidence": item.get("confidence"),
        "max_sim": item.get("max_sim"),
        "verdict": verdict,
        "reason": reason,
        "corrected_trend": None if corrected_trend == "(not specified)" else corrected_trend,
        # Handed back so render_drilldown() can tell "unchanged, saved
        # already" apart from "actually new/different this submission" -
        # see the comment there on why that distinction matters.
        "original_judgment": judgment,
    }
