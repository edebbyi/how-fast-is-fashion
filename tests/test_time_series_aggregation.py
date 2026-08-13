"""Tests for aggregate_monthly_frequency (ARCHITECTURE.md §3.7)."""

from __future__ import annotations

import pytest

from fashion_forensics.nlp.time_series_aggregation import aggregate_monthly_frequency


def _records(rows: list[tuple[str, str]]) -> list[dict]:
    """rows: list of (month, trend_pred) tuples."""
    return [{"month": m, "trend_pred": t} for m, t in rows]


class TestAggregateMonthlyFrequency:
    def test_basic_counts(self):
        # Replicates ARCHITECTURE.md's own worked example: 60 inventory,
        # 14 boho_revival -> 0.2333 frequency.
        rows = [("2024-07", "boho_revival")] * 14 + [("2024-07", "other")] * 46
        records = _records(rows)
        df = aggregate_monthly_frequency(records, trends=["boho_revival", "other"])
        row = df[(df["date"] == "2024-07") & (df["trend"] == "boho_revival")].iloc[0]
        assert row["inventory_count"] == 60
        assert row["trend_count"] == 14
        assert row["trend_frequency"] == pytest.approx(14 / 60, abs=1e-4)

    def test_zero_fill_missing_trend_month(self):
        records = _records([("2024-01", "a"), ("2024-02", "a")])
        df = aggregate_monthly_frequency(records, trends=["a", "b"])
        row = df[(df["date"] == "2024-01") & (df["trend"] == "b")].iloc[0]
        assert row["trend_count"] == 0
        assert not row["is_active"]

    def test_is_active_threshold_boundary(self):
        records = _records([("2024-01", "a")] * 5 + [("2024-01", "other")] * 95)
        df = aggregate_monthly_frequency(records, trends=["a"], active_threshold=0.05)
        row = df[(df["date"] == "2024-01") & (df["trend"] == "a")].iloc[0]
        assert row["trend_frequency"] == pytest.approx(0.05)
        assert row["is_active"]  # inclusive >=

        df2 = aggregate_monthly_frequency(records, trends=["a"], active_threshold=0.051)
        row2 = df2[(df2["date"] == "2024-01") & (df2["trend"] == "a")].iloc[0]
        assert not row2["is_active"]

    def test_custom_active_threshold(self):
        records = _records([("2024-01", "a")] * 10 + [("2024-01", "other")] * 90)
        low = aggregate_monthly_frequency(records, trends=["a"], active_threshold=0.05)
        high = aggregate_monthly_frequency(records, trends=["a"], active_threshold=0.5)
        assert low[low["trend"] == "a"].iloc[0]["is_active"]
        assert not high[high["trend"] == "a"].iloc[0]["is_active"]

    def test_unknown_counted_in_inventory_and_gets_own_row(self):
        records = _records([("2024-01", "a")] * 3 + [("2024-01", "unknown")] * 2)
        df = aggregate_monthly_frequency(records, trends=["a"])
        unk_row = df[(df["date"] == "2024-01") & (df["trend"] == "unknown")]
        assert len(unk_row) == 1
        assert unk_row.iloc[0]["trend_count"] == 2
        a_row = df[(df["date"] == "2024-01") & (df["trend"] == "a")].iloc[0]
        assert a_row["inventory_count"] == 5

    def test_unknown_row_absent_when_zero(self):
        records = _records([("2024-01", "a")] * 3)
        df = aggregate_monthly_frequency(records, trends=["a"])
        unk_row = df[(df["date"] == "2024-01") & (df["trend"] == "unknown")]
        assert len(unk_row) == 0

    def test_include_unknown_false_still_counts_inventory(self):
        records = _records([("2024-01", "a")] * 3 + [("2024-01", "unknown")] * 2)
        df = aggregate_monthly_frequency(records, trends=["a"], include_unknown=False)
        assert len(df[df["trend"] == "unknown"]) == 0
        a_row = df[(df["date"] == "2024-01") & (df["trend"] == "a")].iloc[0]
        assert a_row["inventory_count"] == 5

    def test_custom_field_names(self):
        records = [
            {"date": "2024-01", "trend_labels": "a"},
            {"date": "2024-01", "trend_labels": "a"},
        ]
        df = aggregate_monthly_frequency(
            records, trends=["a"], date_field="date", trend_field="trend_labels"
        )
        row = df[(df["date"] == "2024-01") & (df["trend"] == "a")].iloc[0]
        assert row["trend_count"] == 2

    def test_trend_counts_sum_to_inventory_count(self):
        rows = [("2024-01", "a")] * 4 + [("2024-01", "b")] * 3 + [("2024-01", "unknown")] * 2
        records = _records(rows)
        df = aggregate_monthly_frequency(records, trends=["a", "b"])
        month_rows = df[df["date"] == "2024-01"]
        assert month_rows["trend_count"].sum() == 9
        assert month_rows.iloc[0]["inventory_count"] == 9

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            aggregate_monthly_frequency([], trends=["a"])
