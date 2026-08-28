"""Model comparison view: Model A vs Model B side by side, for each synthetic
test profile (ARCHITECTURE.md §3.9).

This isn't a real recommendations feature - there's no logged-in user. It's a
validation tool: pick a made-up shopper profile and see whether the
trend-aware ranking model (B) actually picks different items than the
attribute-only baseline (A) for that profile.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fashion_forensics.app.components._shared import resolve_image_paths
from fashion_forensics.config import DATA_DIR, load_yaml_config

RANKING_DIR = DATA_DIR / "03_shared" / "ranking"
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"
COMPARISON_CSV_PATH = RANKING_DIR / "model_comparison_summary.csv"


def render_recommendations():
    """Pick a synthetic test profile, and compare what Model A
    (attribute-only) and Model B (trend + lifecycle aware) would each rank
    highest for it - a way to check whether the trend-aware model is
    actually doing something different, not a real recommendations feature."""
    st.markdown("### Model Comparison")
    st.caption(
        "Each option below is a made-up shopper persona, not a real customer - "
        "defined by style preferences like 'likes relaxed cotton basics.' Picking "
        "one shows what two ranking approaches would each recommend to that "
        "persona: Model A matches only the stated preferences; Model B also "
        "factors in what's currently trending and how established that trend is."
    )

    try:
        profiles = load_yaml_config("user_profiles")["profiles"]
    except FileNotFoundError:
        st.info("No user profiles configured yet. See configs/user_profiles.yaml.")
        return

    profile_name = st.selectbox(
        "SHOPPER PERSONA",
        list(profiles.keys()),
        format_func=lambda p: p.replace("synthetic_user_", "").replace("_", " ").title(),
    )
    profile = profiles[profile_name]
    st.caption(
        f"Prefers: {', '.join(profile['preferred_attributes'])} · "
        f"{', '.join(profile['preferred_trends'])} · {profile['preferred_category']}"
    )

    path_a = RANKING_DIR / f"rankings_{profile_name}_modelA.jsonl"
    path_b = RANKING_DIR / f"rankings_{profile_name}_modelB.jsonl"
    if not path_a.exists() or not path_b.exists():
        st.info(f"No ranking results yet for {profile_name}. Run scripts/rank_catalog.py first.")
        return

    top_a = pd.read_json(path_a, lines=True).head(5)
    top_b = pd.read_json(path_b, lines=True).head(5)

    all_ids = list(top_a["record_id"]) + list(top_b["record_id"])
    image_info = resolve_image_paths(all_ids)
    top_a = top_a.merge(image_info, on="record_id", how="left")
    top_b = top_b.merge(image_info, on="record_id", how="left")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Model A")
        st.caption("attribute + category only")
        for _, row in top_a.iterrows():
            _render_recommendation_card(row)
    with col_b:
        st.markdown("#### Model B")
        st.caption("attribute + category, also trend + lifecycle state")
        for _, row in top_b.iterrows():
            _render_recommendation_card(row)

    st.divider()
    tau, overlap = _load_comparison_stats(profile_name)
    if tau is not None:
        shared_ids = set(top_a["record_id"]) & set(top_b["record_id"])
        union_ids = set(top_a["record_id"]) | set(top_b["record_id"])
        st.caption(
            f"Kendall's tau: {tau:.3f} ({_describe_tau(tau)} rank agreement between Model A "
            f"and B, where 1.0 is identical order and 0 is no relationship). "
            f"Top-5 overlap: {overlap:.3f} ({len(shared_ids)} of {len(union_ids)} distinct "
            f"items across both top-5 lists are shared by both models)."
        )


def _describe_tau(tau: float) -> str:
    """Rough qualitative bucket for Kendall's tau, just to make the raw
    number easier to scan - not a statistical claim."""
    magnitude = abs(tau)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    return "weak"


def _render_recommendation_card(row: pd.Series) -> None:
    img_path = None
    if pd.notna(row.get("image_filename")) and pd.notna(row.get("month")):
        img_path = RAW_IMAGES_DIR / row["month"] / row["image_filename"]

    if img_path and img_path.exists():
        st.image(str(img_path), width=150)

    # Not using the confidence-style badge here on purpose: `score` is a
    # weighted sum of several signals, not a 0-1 confidence value, so the
    # confidence color tiers wouldn't mean anything for it.
    st.markdown(
        f'<span class="conf-badge conf-mid">{row["trend_pred"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"score {row['score']:.2f}")
    with st.expander("details"):
        for key, value in row["components"].items():
            st.caption(f"{key}: {value}")
    st.divider()


def _load_comparison_stats(profile_name: str) -> tuple[float | None, float | None]:
    """Kendall's tau and top-5 overlap for one profile - see ab_comparison_summary.md
    for why these are trusted over precision/recall/F1 (not shown in this UI)."""
    if not COMPARISON_CSV_PATH.exists():
        return None, None
    df = pd.read_csv(COMPARISON_CSV_PATH)
    match = df[df["profile"] == profile_name]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return row["kendall_tau"], row["top5_overlap"]
