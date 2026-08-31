"""Curate mode: review scraped candidate trend-reference images.

The full flow, five manual steps (each cheap enough to run separately, and
each a real gate before paying for the next one):
  0. Search & scrape - find fresh article URLs for a trend via Claude's web
     search (a small real API cost) and pull their images in as candidates.
  1. Approve/reject/relabel candidates below - moves a file into labeled/{trend}/
     instantly, no LLM call, no re-embed.
  2. Sync to classifier - normalizes + re-embeds the corpus into Qdrant.
  3. Evaluate reference images - fast (~30s) LOOCV check that the sync actually
     helped before spending real time on step 4.
  4. Reclassify catalog (~30 min) - re-runs classification/lifecycle/ranking
     against the updated corpus, so Trends/Discover/Model Comparison catch up.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics import curation
from fashion_forensics.config import DATA_DIR, PROJECT_ROOT, load_yaml_config

IMAGES_PER_PAGE = 25  # 5 rows x 5 cols

STATUS_OPTIONS = ["Pending", "All", "Approved", "Rejected", "Duplicate"]
REJECT_REASONS = ["off-trend", "low quality", "not fashion editorial", "ambiguous"]

REFERENCE_LOOCV_PATH = DATA_DIR / "03_shared" / "catalog_distribution" / "reference_loocv.jsonl"

# Re-picked from 0.62 (see ARCHITECTURE.md v12) once real catalog ground truth
# existed to check it against - RETROSPECTIVE.md section 9 found 0.62 only
# gave 0.509 precision on 161 real human judgments (well below the
# reference-corpus LOOCV's 0.865), while 0.70 gave 0.647, at the cost of
# ~2/3 of the catalog falling to "unknown". Reclassifying must reuse this
# value, not fall back to open_set_threshold=None (auto-calibrate), or it
# would silently undo that decision.
CATALOG_OPEN_SET_THRESHOLD = 0.70


def render_review_queue():
    st.markdown("### Review Queue")
    st.caption(
        "Approve, reject, or relabel scraped candidate images before they join "
        "the classifier's reference images. Approving moves the file instantly; "
        "the classifier itself only updates when you sync."
    )

    _render_search_button()
    st.divider()

    candidates = curation.load_candidates()
    if not candidates:
        st.info("No scraped candidates yet. Run scripts/scrape_trend_images.py first.")
        return

    df = pd.DataFrame(candidates.values())

    trend_config = load_yaml_config("trend_rules")
    active_trends = sorted(trend_config["trends"].keys())
    deferred_trends = sorted(trend_config["deferred_trends"].keys())
    trend_options = active_trends + deferred_trends
    trend_defs = {**trend_config["trends"], **trend_config["deferred_trends"]}

    col_status, col_trend = st.columns(2)
    with col_status:
        selected_status = st.selectbox("STATUS", STATUS_OPTIONS)
    with col_trend:
        trends_present = sorted(t for t in df["suggested_trend"].dropna().unique())
        selected_trend = st.selectbox("TREND", ["All", *trends_present])

    filtered = df
    if selected_status != "All":
        filtered = filtered[filtered["review_status"] == selected_status.lower()]
    if selected_trend != "All":
        filtered = filtered[filtered["suggested_trend"] == selected_trend]

    st.caption(f"Showing {len(filtered)} of {len(df)} candidates")
    st.divider()

    if filtered.empty:
        st.caption("No candidates match this filter.")
    else:
        filtered = filtered.sort_values("scraped_at", ascending=False)

        if len(filtered) <= IMAGES_PER_PAGE:
            display_items = filtered
        else:
            total_pages = (len(filtered) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
            page = st.session_state.get("review_queue_page", 0)
            page = max(0, min(page, total_pages - 1))

            col_prev, col_indicator, col_next = st.columns([1, 4, 1])
            with col_prev:
                if st.button("◀", disabled=page == 0, key="review_queue_page_prev"):
                    page -= 1
            with col_indicator:
                start_n = page * IMAGES_PER_PAGE + 1
                end_n = min((page + 1) * IMAGES_PER_PAGE, len(filtered))
                st.caption(f"Candidates {start_n}-{end_n} of {len(filtered)}")
            with col_next:
                if st.button("▶", disabled=page >= total_pages - 1, key="review_queue_page_next"):
                    page += 1
            st.session_state.review_queue_page = page

            start = page * IMAGES_PER_PAGE
            display_items = filtered.iloc[start : start + IMAGES_PER_PAGE]

        cols_per_row = 5
        rows = (len(display_items) + cols_per_row - 1) // cols_per_row
        for row_idx in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                item_idx = row_idx * cols_per_row + col_idx
                if item_idx >= len(display_items):
                    break
                item = display_items.iloc[item_idx]
                with cols[col_idx]:
                    _render_candidate_card(item, trend_options, trend_defs)

    st.divider()
    _render_sync_button()
    _render_evaluate_button()
    _render_reclassify_button()


def _site_domain(site: str) -> str:
    """Bare domain (e.g. "whowhatwear.com") from configs/scraping.yaml's
    base_url, for the web_search tool's allowed_domains param."""
    base_url = load_yaml_config("scraping")["sites"][site]["base_url"]
    return base_url.split("//", 1)[-1].removeprefix("www.").rstrip("/")


def _render_search_button() -> None:
    from fashion_forensics.mining.site_parsers import SITE_PARSERS

    sites = sorted(SITE_PARSERS.keys())
    st.markdown("##### Search & scrape")
    st.caption(
        f"Uses Claude's web search (a small real API cost per search) to find fresh "
        f"articles for a trend across every registered site ({', '.join(sites)}), then "
        f"scrapes them the same way as the curated list in configs/scraping.yaml. Not "
        f"saved back to that file - each search is live, so a later search can surface "
        f"newer articles."
    )

    trend_config = load_yaml_config("trend_rules")
    search_trend_options = sorted(trend_config["trends"].keys()) + sorted(
        trend_config["deferred_trends"].keys()
    )
    col_trend, col_button = st.columns([2, 1])
    with col_trend:
        search_trend = st.selectbox(
            "Trend", search_trend_options, key="search_trend", label_visibility="collapsed"
        )
    with col_button:
        do_search = st.button("Search & scrape")

    if not do_search:
        return

    totals = [0, 0, 0]
    found_any = False
    for site in sites:
        try:
            with st.spinner(f"Searching {_site_domain(site)} for {search_trend}..."):
                urls = _run_search(search_trend, site)
        except Exception as e:
            st.error(f"Search failed on {site}: {e}")
            continue

        if not urls:
            st.caption(f"{site}: no articles found.")
            continue

        found_any = True
        with st.spinner(f"Scraping {len(urls)} article(s) from {site}..."):
            result = _run_scrape_urls(site, search_trend, urls)
        totals = [a + b for a, b in zip(totals, result)]
        st.caption(
            f"{site}: found {len(urls)} article(s) - {result[0]} new, "
            f"{result[1]} duplicate, {result[2]} skipped."
        )

    if found_any:
        st.success(
            f"Total: {totals[0]} new candidates, {totals[1]} auto-flagged duplicates, "
            f"{totals[2]} skipped."
        )
        st.rerun()


def _run_search(trend: str, site: str) -> list[str]:
    from fashion_forensics.mining.trend_search import search_articles_for_trend

    return search_articles_for_trend(trend, site, _site_domain(site), max_urls=3)


def _run_scrape_urls(site: str, trend: str, urls: list[str]) -> tuple[int, int, int]:
    _ensure_scripts_on_path()
    from scrape_trend_images import scrape_articles

    return scrape_articles(site, trend, urls, max_images=15, dry_run=False)


def _render_candidate_card(item: pd.Series, trend_options: list[str], trend_defs: dict) -> None:
    candidate_id = item["candidate_id"]
    img_path = PROJECT_ROOT / item["local_path"]

    with st.container(border=True):
        if img_path.exists():
            st.image(str(img_path), width="stretch")
        else:
            st.caption("Image not found (already moved or deleted).")

        scraped_date = item["scraped_at"][:10] if item.get("scraped_at") else "?"
        st.caption(f"{item['source_site']} · {scraped_date}")
        st.caption(f"[source]({item['source_url']})")

        if item.get("dup_of"):
            st.markdown(
                '<span class="conf-badge conf-low">Possible duplicate</span>',
                unsafe_allow_html=True,
            )

        if item["review_status"] != "pending":
            st.caption(f"status: {item['review_status']}")
            return

        default_trend = item.get("suggested_trend")
        default_index = trend_options.index(default_trend) if default_trend in trend_options else 0
        chosen_trend = st.selectbox(
            "Trend",
            trend_options,
            index=default_index,
            key=f"trend_{candidate_id}",
            label_visibility="collapsed",
        )
        st.caption(_discriminating_detail_guidance(chosen_trend, trend_defs))

        col_a, col_r, col_d = st.columns(3)
        with col_a:
            if st.button("Approve", key=f"approve_{candidate_id}"):
                curation.approve_candidate(candidate_id, chosen_trend)
                st.rerun()
        with col_r:
            if st.button("Reject", key=f"reject_{candidate_id}"):
                st.session_state[f"show_reject_{candidate_id}"] = True
        with col_d:
            if st.button("Duplicate", key=f"duplicate_{candidate_id}"):
                curation.reject_candidate(candidate_id, "duplicate")
                st.rerun()

        if st.session_state.get(f"show_reject_{candidate_id}"):
            reason = st.selectbox(
                "Reason",
                REJECT_REASONS,
                key=f"reject_reason_{candidate_id}",
                label_visibility="collapsed",
            )
            if st.button("Confirm reject", key=f"confirm_reject_{candidate_id}"):
                curation.reject_candidate(candidate_id, "rejected", reason=reason)
                del st.session_state[f"show_reject_{candidate_id}"]
                st.rerun()


def _discriminating_detail_guidance(trend: str, trend_defs: dict) -> str:
    """What to actually check for before approving into this trend - reject
    plain/generic images that don't show it, even if the source article was
    genuinely about this trend. See notebook 03 section 14.2/14.6: the
    curator's "real, on-topic article" bar isn't strict enough on its own
    for trends defined more by absence than by a concrete visual signature.
    """
    entry = trend_defs.get(trend, {})
    details = entry.get("discriminating_details") or []
    if details:
        detail_list = ", ".join(d.replace("_", " ") for d in details)
        return f"Look for: {detail_list}. Reject plain pieces without any of these."
    return (
        "Defined by absence, not presence - reject anything with a distinctive "
        "feature that belongs to another trend instead."
    )


def _ensure_scripts_on_path() -> None:
    """scripts/ isn't a package - this lets `from <script> import main` work
    from inside src/, the same way scripts/run_pipeline.py's own sibling
    imports work by virtue of living in scripts/ itself."""
    import sys

    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _render_sync_button() -> None:
    n_pending_sync = curation.pending_sync_count()
    label = (
        f"Sync {n_pending_sync} approved image{'s' if n_pending_sync != 1 else ''} to classifier"
    )
    if st.button(label, disabled=n_pending_sync == 0):
        with st.spinner("Syncing... this re-embeds the full set of reference images"):
            _run_sync()
        st.success("Synced.")
        st.rerun()
    if n_pending_sync == 0:
        st.caption("Approvals are staged instantly; the classifier isn't updated until you sync.")


def _run_sync() -> None:
    from fashion_forensics.config import settings

    _ensure_scripts_on_path()
    from embed_reference_corpus import main as embed_main
    from normalize_reference_corpus import main as normalize_main

    normalize_main(force=False, max_records=None)
    embed_main(embedding_model=settings.fashion_clip_model, collection=settings.qdrant_collection)
    curation.mark_synced()


def _load_loocv_summary() -> dict | None:
    """Overall accuracy/macro-F1 + per-trend precision/recall/F1 from the last
    reference_loocv.py run. true_trend/predicted_trend are both already in
    reference_loocv.jsonl - this is a display-only computation, no need to
    change what that script writes."""
    if not REFERENCE_LOOCV_PATH.exists():
        return None
    df = pd.read_json(REFERENCE_LOOCV_PATH, lines=True)
    if df.empty:
        return None

    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(df["true_trend"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        df["true_trend"], df["predicted_trend"], labels=labels, zero_division=0
    )
    per_trend = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=labels
    )
    return {
        "accuracy": df["correct"].mean(),
        "macro_f1": f1.mean(),
        "per_trend": per_trend,
    }


def _render_evaluate_button() -> None:
    st.caption(
        "After syncing: check the updated reference images are actually good "
        "(fast, ~30s) before spending ~30 minutes reclassifying the catalog. The "
        "before/after view below only lasts this browser session, but every click also "
        "logs accuracy, macro F1, and per-trend precision/recall/F1 to MLflow (experiment "
        '"reference-loocv") - so real history across sessions lives in the MLflow tab, '
        "not here. The fuller evaluation harness (scripts/eval_trend_classifier.py, adds "
        "ECE + confusion matrix + UMAP) logs there too."
    )
    if st.button("Evaluate reference images"):
        st.session_state["loocv_before"] = _load_loocv_summary()
        with st.spinner("Running LOOCV on the reference images..."):
            _run_reference_loocv()
        st.session_state["loocv_after"] = _load_loocv_summary()

    after = st.session_state.get("loocv_after")
    if after is not None:
        before = st.session_state.get("loocv_before")
        col_before, col_after = st.columns(2)
        with col_before:
            st.caption("Before this evaluation")
            st.metric("Accuracy", f"{before['accuracy']:.1%}" if before else "n/a")
            st.metric("Macro F1", f"{before['macro_f1']:.3f}" if before else "n/a")
        with col_after:
            st.caption("After this evaluation")
            acc_delta = f"{after['accuracy'] - before['accuracy']:+.1%}" if before else None
            f1_delta = f"{after['macro_f1'] - before['macro_f1']:+.3f}" if before else None
            st.metric("Accuracy", f"{after['accuracy']:.1%}", delta=acc_delta)
            st.metric("Macro F1", f"{after['macro_f1']:.3f}", delta=f1_delta)
        st.caption("Per-trend precision / recall / F1 (after)")
        st.dataframe(
            after["per_trend"].style.format(
                {"precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.3f}", "support": "{:.0f}"}
            ),
            width="stretch",
        )


def _run_reference_loocv() -> None:
    from fashion_forensics.config import settings

    _ensure_scripts_on_path()
    from reference_loocv import main as loocv_main

    loocv_main(
        k=5,
        vote_method="shepard",
        embedding_model=settings.fashion_clip_model,
        collection=settings.qdrant_collection,
    )


def _render_reclassify_button() -> None:
    st.caption(
        f"Re-runs classification, lifecycle, and ranking against the updated corpus, "
        f"reusing the threshold already chosen for this catalog ({CATALOG_OPEN_SET_THRESHOLD}, "
        f"not auto-calibrated - see RETROSPECTIVE.md section 9). Takes about 30 minutes and "
        f"blocks this page while running."
    )
    if st.button("Reclassify catalog (~30 min)"):
        with st.spinner("Reclassifying... this takes about 30 minutes"):
            _run_reclassify()
        st.success("Reclassified. Trends, Discover, and Model Comparison now reflect the update.")


def _run_reclassify() -> None:
    from fashion_forensics.config import settings

    _ensure_scripts_on_path()
    from classify_catalog import main as classify_main
    from compute_trend_lifecycle import main as lifecycle_main
    from rank_catalog import main as rank_main

    classify_main(
        k=5,
        vote_method="shepard",
        open_set_threshold=CATALOG_OPEN_SET_THRESHOLD,
        embedding_model=settings.fashion_clip_model,
        collection=settings.qdrant_collection,
        months=None,
        limit=None,
    )
    lifecycle_main(
        active_threshold=0.05,
        persistent_min_active_months=4,
        rising_slope_threshold=0.03,
        declining_slope_threshold=-0.03,
        slope_window=3,
        recency_inactive_months=2,
        recurrence_gap_months=2,
        classifications_path=None,
    )
    rank_main(profile_name=None)
