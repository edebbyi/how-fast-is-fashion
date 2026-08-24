"""Tests for the ranking engine's scoring, ranking, and synthetic evaluation
functions (ARCHITECTURE.md §3.9)."""

from __future__ import annotations

import pytest

from fashion_forensics.nlp.ranking_engine import (
    attribute_match,
    category_match,
    f1_at_k,
    is_relevant,
    kendall_tau,
    precision_at_k,
    rank_items,
    recall_at_k,
    score_a,
    score_b,
    top_k_overlap,
    trend_match,
    trend_state_weight,
)


def _normalized(**field_values: list[str]) -> dict:
    """Build a minimal normalized-attribute record: field -> list of values."""
    return {field: {"value": values, "source": "image", "confidence": 0.8} for field, values in field_values.items()}


def _item(record_id: str, trend_pred: str | None, category: str | None, normalized: dict | None = None) -> dict:
    return {"record_id": record_id, "trend_pred": trend_pred, "category": category, "normalized": normalized}


def _profile(**kwargs) -> dict:
    base = {
        "preferred_attributes": [],
        "preferred_trends": [],
        "preferred_category": None,
        "prefer_rising": False,
        "avoid_declining": False,
    }
    base.update(kwargs)
    return base


class TestAttributeMatch:
    def test_full_overlap(self):
        normalized = _normalized(material=["linen"], silhouette=["flowy"])
        assert attribute_match(normalized, ["linen", "flowy"]) == 1.0

    def test_partial_overlap(self):
        normalized = _normalized(material=["linen"], silhouette=["fitted"])
        assert attribute_match(normalized, ["linen", "flowy"]) == pytest.approx(0.5)

    def test_no_overlap(self):
        normalized = _normalized(material=["cotton"])
        assert attribute_match(normalized, ["sequined"]) == 0.0

    def test_normalized_none_degrades_to_zero(self):
        assert attribute_match(None, ["linen"]) == 0.0


class TestCategoryMatch:
    def test_singular_plural_collision_matches(self):
        assert category_match("dresses", "dress") == 1.0

    def test_mismatch(self):
        assert category_match("tops", "dress") == 0.0

    def test_item_category_none(self):
        assert category_match(None, "dress") == 0.0


class TestTrendMatch:
    def test_match(self):
        assert trend_match("basics", ["basics", "quiet_luxury"]) == 1.0

    def test_trend_outside_taxonomy_degrades_safely(self):
        assert trend_match("cottagecore", ["basics"]) == 0.0


class TestTrendStateWeight:
    def test_rising_flag_engaged(self):
        assert trend_state_weight("rising", prefer_rising=True, avoid_declining=False) == 1.0

    def test_rising_flag_not_engaged(self):
        assert trend_state_weight("rising", prefer_rising=False, avoid_declining=False) == 0.3

    def test_declining_flag_engaged(self):
        assert trend_state_weight("declining", prefer_rising=False, avoid_declining=True) == -1.0

    def test_declining_flag_not_engaged(self):
        assert trend_state_weight("declining", prefer_rising=False, avoid_declining=False) == -0.3

    def test_inactive_distinct_from_declining(self):
        inactive = trend_state_weight("inactive", prefer_rising=False, avoid_declining=True)
        declining = trend_state_weight("declining", prefer_rising=False, avoid_declining=True)
        assert inactive == -0.5
        assert inactive != declining

    def test_persistent_flag_independent(self):
        assert trend_state_weight("persistent", True, True) == 0.5
        assert trend_state_weight("persistent", False, False) == 0.5

    def test_seasonal_recurring_flag_independent(self):
        assert trend_state_weight("seasonal_recurring", True, True) == 0.3
        assert trend_state_weight("seasonal_recurring", False, False) == 0.3

    def test_none_state_is_neutral(self):
        assert trend_state_weight(None, True, True) == 0.0

    def test_unrecognized_state_raises(self):
        with pytest.raises(ValueError):
            trend_state_weight("bogus_state", True, True)


class TestScoreComposition:
    def test_score_a_works_with_no_normalized_attributes(self):
        item = _item("r1", trend_pred="basics", category="shirt", normalized=None)
        profile = _profile(preferred_category="shirt")
        result = score_a(item, profile)
        assert isinstance(result["score"], float)
        assert result["components"]["attribute_match"] == 0.0
        assert result["components"]["category_match"] == 1.0

    def test_score_a_vs_score_b_produce_different_rankings(self):
        # Item 1: full attribute+category match, off-trend (quiet_luxury/persistent).
        # score_a = 0.6*1.0 + 0.4*1.0 = 1.0
        # score_b = 0.4*1.0 + 0.25*0.0 + 0.2*1.0 + 0.15*(0.5*0.0 gated off-trend) = 0.6
        item_off_trend = _item(
            "r_off_trend", trend_pred="quiet_luxury", category="shirt",
            normalized=_normalized(material=["cotton"], silhouette=["relaxed"]),
        )
        # Item 2: partial attribute match, but on-trend, category matches, and rising.
        # score_a = 0.6*0.5 + 0.4*1.0 = 0.7  (loses to item 1 under A)
        # score_b = 0.4*0.5 + 0.25*1.0 + 0.2*1.0 + 0.15*1.0(rising, engaged) = 0.8  (wins under B)
        item_on_trend = _item(
            "r_on_trend", trend_pred="basics", category="shirt",
            normalized=_normalized(material=["cotton"]),
        )
        items = [item_off_trend, item_on_trend]
        profile = _profile(
            preferred_attributes=["cotton", "relaxed"],
            preferred_trends=["basics"],
            preferred_category="shirt",
            prefer_rising=True,
        )
        lifecycle_by_trend = {"basics": {"state": "rising"}, "quiet_luxury": {"state": "persistent"}}

        ranked_a = rank_items(items, profile, model="A")
        ranked_b = rank_items(items, profile, model="B", lifecycle_by_trend=lifecycle_by_trend)

        assert ranked_a[0]["record_id"] == "r_off_trend"
        assert ranked_b[0]["record_id"] == "r_on_trend"

    def test_rank_items_model_b_requires_lifecycle(self):
        items = [_item("r1", "basics", "shirt", None)]
        profile = _profile()
        with pytest.raises(ValueError):
            rank_items(items, profile, model="B")


class TestSyntheticEvaluation:
    def test_is_relevant_false_for_never_matching_trend(self):
        items = [
            _item("r1", "quiet_luxury", "dress", None),
            _item("r2", "basics", "dress", None),
        ]
        profile = _profile(preferred_trends=["office_siren"], preferred_category="dress")
        assert all(not is_relevant(item, profile) for item in items)

    def test_precision_and_f1_are_zero_not_exception_when_nothing_relevant(self):
        items = [_item("r1", "quiet_luxury", "dress", None), _item("r2", "basics", "shirt", None)]
        profile = _profile(preferred_trends=["office_siren"], preferred_category="dress")
        ranked = rank_items(items, profile, model="A")
        assert precision_at_k(ranked, profile, 5) == 0.0
        assert recall_at_k(ranked, profile, 5) == 0.0
        assert f1_at_k(ranked, profile, 5) == 0.0

    def test_precision_recall_f1_nonzero_when_relevant_items_present(self):
        items = [
            _item("r1", "basics", "shirt", None),
            _item("r2", "quiet_luxury", "dress", None),
        ]
        profile = _profile(preferred_trends=["basics"], preferred_category="shirt")
        ranked = rank_items(items, profile, model="A")
        assert precision_at_k(ranked, profile, 1) == 1.0
        assert recall_at_k(ranked, profile, 1) == 1.0
        assert f1_at_k(ranked, profile, 1) == pytest.approx(1.0)


class TestRankingComparisons:
    def test_kendall_tau_raises_on_mismatched_item_sets(self):
        items_a = [_item("r1", "basics", "shirt", None), _item("r2", "quiet_luxury", "dress", None)]
        items_b = [_item("r1", "basics", "shirt", None)]
        profile = _profile()
        ranked_a = rank_items(items_a, profile, model="A")
        ranked_b = rank_items(items_b, profile, model="A")
        with pytest.raises(ValueError):
            kendall_tau(ranked_a, ranked_b)

    def test_kendall_tau_identical_rankings_is_one(self):
        items = [_item("r1", "basics", "shirt", None), _item("r2", "quiet_luxury", "dress", None)]
        profile = _profile()
        ranked = rank_items(items, profile, model="A")
        assert kendall_tau(ranked, ranked) == pytest.approx(1.0)

    def test_top_k_overlap_identical_rankings_is_one(self):
        items = [_item("r1", "basics", "shirt", None), _item("r2", "quiet_luxury", "dress", None)]
        profile = _profile()
        ranked = rank_items(items, profile, model="A")
        assert top_k_overlap(ranked, ranked, k=5) == 1.0
