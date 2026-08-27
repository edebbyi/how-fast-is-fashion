"""Score and rank catalog items against a user preference profile.

Implements ARCHITECTURE.md §3.9. This compares two ways of scoring items, to
test the project's main question (see README): does knowing a trend and how
it's trending actually make recommendations better, compared to just
matching on attributes?

  Model A (attribute-only baseline): score_A = w1*attribute_match + w3*category_match
  Model B (trend+lifecycle-aware):   score_B = w1*attribute_match + w2*trend_match
                                                + w3*category_match + w4*trend_state_weight

Important limit on the data: only 59 of about 2025 catalog items have
LLM-generated attributes (needed for attribute_match), and all 59 already
have a trend label. The scoring functions here still work on any item —
attribute_match just comes back as 0.0 if an item has no attributes — so
score_a can technically run over the whole catalog. But a fair comparison
between Model A and Model B only makes sense on items where both have real
data to work with, not defaults. Use filter_fair_comparison_subset() for
that. Comparing Model A over the full catalog against Model B over just the
59-item subset would confuse "this item really has no trend signal" with
"this item was just never given attributes in the first place."

A note on what the evaluation numbers below actually mean: is_relevant(),
precision_at_k(), recall_at_k(), and f1_at_k() decide whether an item counts
as "relevant" using the same profile fields the scoring formulas use
(preferred_trends, preferred_category). So these numbers only show whether
each formula finds items that match its own stated inputs — they are NOT a
measure of real recommendation quality, since there's no independent user
feedback to check against. Kendall's tau and top-K overlap (how much Model A
and Model B's rankings agree) don't have this problem, and should be given
at least as much weight when reading the results.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from loguru import logger
from scipy.stats import kendalltau

from fashion_forensics.db import connect
from fashion_forensics.nlp.tfidf_engine import load_normalized_records
from fashion_forensics.nlp.time_series_aggregation import load_classifications

RANKING_ENGINE_VERSION = "v1"

ATTRIBUTE_MATCH_FIELDS: tuple[str, ...] = (
    "silhouette",
    "material",
    "sleeve_style",
    "neckline",
    "length",
    "color_profile",
    "pattern",
    "details",
)
# category and subcategory are left out on purpose - matching category is
# already category_match's job (w3). Including it here too would count the
# same signal twice.

DEFAULT_WEIGHTS_A = {"w1": 0.6, "w3": 0.4}
DEFAULT_WEIGHTS_B = {"w1": 0.4, "w2": 0.25, "w3": 0.2, "w4": 0.15}
# These are first-guess defaults, not tuned - same approach as the threshold
# defaults in lifecycle_classifier.py. Each model's own weights add up to
# 1.0, but A and B are never compared by their raw score, only by the order
# they rank items in, so it's fine that they don't share one weight budget.

# Each lifecycle state maps to (weight when the matching flag is on, weight
# otherwise). "inactive" gets a milder penalty than "declining" even when
# avoid_declining is on - a trend that's already dead is a different signal
# than one that's actively dying.
STATE_WEIGHTS: dict[str, tuple[float, float]] = {
    "rising": (1.0, 0.3),
    "persistent": (0.5, 0.5),
    "seasonal_recurring": (0.3, 0.3),
    "declining": (-1.0, -0.3),
    "inactive": (-0.5, -0.2),
}


def _normalize_category(value: str | None) -> str | None:
    """Lowercase a category and strip a simple plural 's', so "dress" and
    "dresses" count as the same thing. This is not a real plural handler,
    just enough to fix the one collision we actually have in the data (the
    LLM-generated category field spells the same category both ways across
    different records).

    Words that already end in a double "s" ("dress", "class") are left as
    they are - blindly stripping one "s" would turn "dress" into "dres".
    Their plural ("dresses") ends in "es" instead, so we strip "es" for
    those, which brings both forms back to the same word.
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v.endswith("sses"):
        return v[:-2]
    if v.endswith("ss"):
        return v
    return v[:-1] if v.endswith("s") and len(v) > 1 else v


def attribute_match(
    normalized: dict | None,
    preferred_attributes: list[str],
    fields: Iterable[str] = ATTRIBUTE_MATCH_FIELDS,
) -> float:
    """Fraction of preferred_attributes present among the item's flattened
    attribute VALUES (not field:value tokens — user preferences are plain
    value strings like "linen", not field-qualified).

    Returns 0.0 if normalized is None or preferred_attributes is empty.
    """
    if not normalized or not preferred_attributes:
        return 0.0
    item_values: set[str] = set()
    for field in fields:
        block = normalized.get(field)
        if block and block.get("value"):
            item_values.update(v.lower() for v in block["value"] if v)
    preferred = {p.lower() for p in preferred_attributes}
    matched = preferred & item_values
    return len(matched) / len(preferred)


def category_match(item_category: str | None, preferred_category: str | None) -> float:
    """1.0 if normalized categories match, else 0.0."""
    a, b = _normalize_category(item_category), _normalize_category(preferred_category)
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def trend_match(trend_pred: str | None, preferred_trends: list[str]) -> float:
    """1.0 if trend_pred is in preferred_trends, else 0.0. A trend outside the
    taxonomy (or 'unknown') simply never matches — no special-case error."""
    if not trend_pred or not preferred_trends:
        return 0.0
    return 1.0 if trend_pred in preferred_trends else 0.0


def trend_state_weight(state: str | None, prefer_rising: bool, avoid_declining: bool) -> float:
    """Turn an item's trend lifecycle state into a number, based on whether
    the user wants rising trends and/or wants to avoid declining ones.

    Returns 0.0 (neutral) if there's no state to look at. If state is set
    but isn't one of the 5 known lifecycle states, raises ValueError - this
    should never actually happen since LifecycleState only allows those 5,
    but it's a cheap safety check.
    """
    if state is None:
        return 0.0
    if state not in STATE_WEIGHTS:
        raise ValueError(f"Unrecognized lifecycle state: {state!r}")
    engaged_weight, default_weight = STATE_WEIGHTS[state]
    # avoid_declining applies to both "declining" and "inactive" states -
    # inactive just gets a milder penalty (see STATE_WEIGHTS above), since a
    # dead trend and a dying trend aren't quite the same thing.
    flag_engaged = (state == "rising" and prefer_rising) or (
        state in ("declining", "inactive") and avoid_declining
    )
    return engaged_weight if flag_engaged else default_weight


def score_a(item: dict, profile: dict, weights: dict | None = None) -> dict:
    """Model A (attribute-only baseline): w1*attribute_match + w3*category_match."""
    w = weights or DEFAULT_WEIGHTS_A
    am = attribute_match(item.get("normalized"), profile.get("preferred_attributes", []))
    cm = category_match(item.get("category"), profile.get("preferred_category"))
    score = w["w1"] * am + w["w3"] * cm
    return {"score": score, "components": {"attribute_match": am, "category_match": cm}}


def score_b(
    item: dict,
    profile: dict,
    lifecycle_by_trend: dict[str, dict],
    weights: dict | None = None,
) -> dict:
    """Model B (trend+lifecycle-aware): adds w2*trend_match + w4*(trend_state_weight * trend_match).

    trend_state_weight is still looked up from the item's own trend even
    when that trend isn't one the user asked for - useful to see in the
    components breakdown - but it only counts toward the score when the
    item's trend actually matches the profile's preferred trends. Without
    that gate, an off-trend item in a "hot" lifecycle state (e.g. rising,
    weight 1.0) could outscore a genuinely on-trend item sitting in a
    calmer state (e.g. seasonal_recurring, weight 0.3) - the lifecycle
    bonus has to be about the trend the user actually wants, not just
    whichever trend happens to be currently hot. See ARCHITECTURE.md v10.
    """
    w = weights or DEFAULT_WEIGHTS_B
    am = attribute_match(item.get("normalized"), profile.get("preferred_attributes", []))
    cm = category_match(item.get("category"), profile.get("preferred_category"))
    tm = trend_match(item.get("trend_pred"), profile.get("preferred_trends", []))

    trend_pred = item.get("trend_pred")
    lifecycle_row = lifecycle_by_trend.get(trend_pred) if trend_pred else None
    state = lifecycle_row["state"] if lifecycle_row else None
    tsw = trend_state_weight(
        state, profile.get("prefer_rising", False), profile.get("avoid_declining", False)
    )

    score = w["w1"] * am + w["w2"] * tm + w["w3"] * cm + w["w4"] * (tsw * tm)
    return {
        "score": score,
        "components": {
            "attribute_match": am,
            "trend_match": tm,
            "category_match": cm,
            "trend_state_weight": tsw,
            "lifecycle_state": state,
        },
    }


def rank_items(
    items: list[dict],
    profile: dict,
    model: Literal["A", "B"],
    lifecycle_by_trend: dict[str, dict] | None = None,
    weights: dict | None = None,
) -> list[dict]:
    """Score every item, sort descending by score (ties broken by record_id
    ascending, for determinism), attach rank 1..N."""
    if model == "B" and lifecycle_by_trend is None:
        raise ValueError("model='B' requires lifecycle_by_trend")

    scored = []
    for item in items:
        result = (
            score_a(item, profile, weights)
            if model == "A"
            else score_b(item, profile, lifecycle_by_trend, weights)
        )
        scored.append({**item, **result})

    scored.sort(key=lambda r: (-r["score"], r["record_id"]))
    for i, row in enumerate(scored, 1):
        row["rank"] = i
    return scored


def top_k(ranked: list[dict], k: int) -> list[dict]:
    return ranked[:k]


def is_relevant(item: dict, profile: dict) -> bool:
    """Decide whether an item counts as "relevant" for evaluation, using
    only the profile's trend and category preferences (see the module
    docstring for why this isn't a real quality measure). We leave out
    preferred_attributes on purpose - that's Model A's only signal, so
    using it here would make Model A look better just by definition."""
    return (
        trend_match(item.get("trend_pred"), profile.get("preferred_trends", [])) == 1.0
        and category_match(item.get("category"), profile.get("preferred_category")) == 1.0
    )


def precision_at_k(ranked: list[dict], profile: dict, k: int) -> float:
    top = top_k(ranked, k)
    if not top:
        return 0.0
    relevant = sum(1 for item in top if is_relevant(item, profile))
    return relevant / len(top)


def recall_at_k(ranked: list[dict], profile: dict, k: int) -> float:
    total_relevant = sum(1 for item in ranked if is_relevant(item, profile))
    if total_relevant == 0:
        return 0.0
    top = top_k(ranked, k)
    relevant_in_top = sum(1 for item in top if is_relevant(item, profile))
    return relevant_in_top / total_relevant


def f1_at_k(ranked: list[dict], profile: dict, k: int) -> float:
    p = precision_at_k(ranked, profile, k)
    r = recall_at_k(ranked, profile, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def kendall_tau(ranked_a: list[dict], ranked_b: list[dict]) -> float:
    """Kendall's tau (rank agreement) between two rankings of the same set
    of items. Raises ValueError if the two rankings don't contain the same
    record_ids - this catches the mistake of comparing a full-catalog
    ranking against a subset ranking by accident."""
    ids_a = {r["record_id"] for r in ranked_a}
    ids_b = {r["record_id"] for r in ranked_b}
    if ids_a != ids_b:
        raise ValueError("ranked_a and ranked_b must cover the same record_id set")

    rank_a_by_id = {r["record_id"]: r["rank"] for r in ranked_a}
    rank_b_by_id = {r["record_id"]: r["rank"] for r in ranked_b}
    ids = sorted(ids_a)
    a_ranks = [rank_a_by_id[i] for i in ids]
    b_ranks = [rank_b_by_id[i] for i in ids]
    tau, _ = kendalltau(a_ranks, b_ranks)
    return float(tau)


def top_k_overlap(ranked_a: list[dict], ranked_b: list[dict], k: int = 5) -> float:
    """Jaccard similarity of the two rankings' top-k record_id sets."""
    top_a = {r["record_id"] for r in top_k(ranked_a, k)}
    top_b = {r["record_id"] for r in top_k(ranked_b, k)}
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def load_ranking_catalog(
    normalized_records: list[dict] | None = None,
    classifications: list[dict] | None = None,
) -> list[dict]:
    """Build one item list by combining category, trend label, and (when
    available) normalized attributes for each catalog record.

    Category comes from the raw retailer data ("dress", clean and singular),
    not from the LLM-normalized attributes, which spell the same category
    both ways ("dress" and "dresses") across different records.

    `classifications` is treated as the full list of items that exist.
    Returns one dict per record: {record_id, month, category, trend_pred,
    confidence, normalized}. `normalized` is None when an item hasn't been
    through LLM normalization yet - callers should treat that as "no
    attribute data available," not as an error.
    """
    if classifications is None:
        classifications = load_classifications()
    if normalized_records is None:
        normalized_records = load_normalized_records()

    normalized_by_id = {r["record_id"]: r for r in normalized_records}

    con = connect()
    category_by_id = {
        row[0]: row[1] for row in con.execute("SELECT record_id, category FROM items").fetchall()
    }

    catalog = []
    for rec in classifications:
        record_id = rec["record_id"]
        catalog.append(
            {
                "record_id": record_id,
                "month": rec["month"],
                "category": category_by_id.get(record_id),
                "trend_pred": rec["trend_pred"],
                "confidence": rec.get("confidence"),
                "normalized": normalized_by_id.get(record_id),
            }
        )
    return catalog


def filter_fair_comparison_subset(catalog: list[dict]) -> list[dict]:
    """Keep only items that have both normalized attributes and a real
    trend label (not "unknown"). These are the only items where Model A and
    Model B are working from equally complete data, so they're the only
    ones a fair comparison should use. Logs how many items that ends up being."""
    subset = [
        item
        for item in catalog
        if item.get("normalized") is not None and item.get("trend_pred") != "unknown"
    ]
    logger.info(
        f"Fair-comparison subset: {len(subset)}/{len(catalog)} items "
        "(has both normalized attributes and a real trend label)"
    )
    return subset
