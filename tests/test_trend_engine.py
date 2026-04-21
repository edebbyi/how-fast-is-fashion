"""Tests for trend state classification logic and get_top_features."""

from __future__ import annotations

import pandas as pd

from fashion_forensics.nlp.trend_engine import compute_trend_states, get_top_features


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["feature", "year", "count", "share"])


class TestComputeTrendStates:
    """Test _classify_row via compute_trend_states."""

    def test_rising_feature(self):
        df = _make_df(
            [
                {"feature": "cargo_pants", "year": 2024, "count": 50, "share": 0.10},
                {"feature": "cargo_pants", "year": 2025, "count": 120, "share": 0.30},
            ]
        )
        result = compute_trend_states(df)
        states = result[result["feature"] == "cargo_pants"]["state"].tolist()
        # First year is emerging (NaN delta), second year has delta +0.20 -> rising
        assert states[0] == "emerging"
        assert states[1] == "rising"

    def test_falling_feature(self):
        df = _make_df(
            [
                {"feature": "skinny_jeans", "year": 2024, "count": 100, "share": 0.40},
                {"feature": "skinny_jeans", "year": 2025, "count": 30, "share": 0.10},
            ]
        )
        result = compute_trend_states(df)
        states = result[result["feature"] == "skinny_jeans"]["state"].tolist()
        assert states[0] == "emerging"
        assert states[1] == "falling"

    def test_stable_feature(self):
        df = _make_df(
            [
                {"feature": "denim", "year": 2024, "count": 80, "share": 0.30},
                {"feature": "denim", "year": 2025, "count": 85, "share": 0.32},
            ]
        )
        result = compute_trend_states(df)
        states = result[result["feature"] == "denim"]["state"].tolist()
        # delta = 0.02, within [-0.15, 0.15], share > 0.01 -> stable
        assert states[1] == "stable"

    def test_first_year_is_emerging(self):
        df = _make_df(
            [
                {"feature": "new_trend", "year": 2025, "count": 10, "share": 0.05},
            ]
        )
        result = compute_trend_states(df)
        assert result["state"].iloc[0] == "emerging"

    def test_near_zero_share_is_archived(self):
        df = _make_df(
            [
                {"feature": "old_trend", "year": 2024, "count": 5, "share": 0.02},
                {"feature": "old_trend", "year": 2025, "count": 1, "share": 0.005},
            ]
        )
        result = compute_trend_states(df)
        states = result[result["feature"] == "old_trend"]["state"].tolist()
        # delta = 0.005 - 0.02 = -0.015, within thresholds, but share < 0.01 -> archived
        assert states[1] == "archived"

    def test_custom_thresholds(self):
        df = _make_df(
            [
                {"feature": "x", "year": 2024, "count": 50, "share": 0.20},
                {"feature": "x", "year": 2025, "count": 60, "share": 0.30},
            ]
        )
        # With a tight threshold of 0.05, delta 0.10 should be rising
        result = compute_trend_states(df, rising_threshold=0.05)
        assert result.iloc[1]["state"] == "rising"

    def test_multiple_features(self):
        df = _make_df(
            [
                {"feature": "a", "year": 2024, "count": 10, "share": 0.10},
                {"feature": "a", "year": 2025, "count": 30, "share": 0.40},
                {"feature": "b", "year": 2024, "count": 50, "share": 0.50},
                {"feature": "b", "year": 2025, "count": 20, "share": 0.15},
            ]
        )
        result = compute_trend_states(df)
        a_states = result[result["feature"] == "a"]["state"].tolist()
        b_states = result[result["feature"] == "b"]["state"].tolist()
        assert a_states[1] == "rising"
        assert b_states[1] == "falling"


class TestGetTopFeatures:
    """get_top_features returns correct N items for a year."""

    def test_returns_top_n(self):
        df = pd.DataFrame(
            [
                {"feature": f"f{i}", "year": 2025, "count": i * 10, "share": i * 0.05}
                for i in range(1, 11)
            ]
        )
        top3 = get_top_features(df, year=2025, n=3)
        assert len(top3) == 3
        # Highest shares should come first (f10, f9, f8)
        assert top3.iloc[0]["feature"] == "f10"
        assert top3.iloc[1]["feature"] == "f9"
        assert top3.iloc[2]["feature"] == "f8"

    def test_filters_by_year(self):
        df = pd.DataFrame(
            [
                {"feature": "a", "year": 2024, "count": 100, "share": 0.80},
                {"feature": "b", "year": 2025, "count": 50, "share": 0.40},
                {"feature": "c", "year": 2025, "count": 30, "share": 0.20},
            ]
        )
        result = get_top_features(df, year=2025, n=10)
        assert len(result) == 2
        assert set(result["feature"]) == {"b", "c"}

    def test_returns_fewer_than_n_if_not_enough(self):
        df = pd.DataFrame(
            [
                {"feature": "only_one", "year": 2025, "count": 10, "share": 0.10},
            ]
        )
        result = get_top_features(df, year=2025, n=5)
        assert len(result) == 1
