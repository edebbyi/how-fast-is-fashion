"""Tests for lifecycle feature extraction and state classification (ARCHITECTURE.md §3.8)."""

from __future__ import annotations

import pandas as pd

from fashion_forensics.nlp.lifecycle_classifier import (
    LIFECYCLE_VERSION,
    classify_lifecycles,
    compute_lifecycle_features,
)


def _freq_df(rows: list[tuple[str, str, float, bool]]) -> pd.DataFrame:
    """rows: list of (date, trend, trend_frequency, is_active) tuples."""
    return pd.DataFrame(rows, columns=["date", "trend", "trend_frequency", "is_active"])


def _months(n: int, start_year: int = 2024, start_month: int = 1) -> list[str]:
    out = []
    y, m = start_year, start_month
    for _ in range(n):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


class TestComputeLifecycleFeatures:
    def test_first_last_active_and_peak_month(self):
        months = _months(4)
        rows = [
            (months[0], "a", 0.0, False),
            (months[1], "a", 0.10, True),
            (months[2], "a", 0.25, True),
            (months[3], "a", 0.15, True),
        ]
        features = compute_lifecycle_features(_freq_df(rows))
        row = features[features["trend"] == "a"].iloc[0]
        assert row["first_observed"] == months[1]
        assert row["last_active"] == months[3]
        assert row["peak_month"] == months[2]
        assert row["active_month_count"] == 3

    def test_active_run_length_merges_short_gap(self):
        months = _months(5)
        rows = [
            (months[0], "a", 0.10, True),
            (months[1], "a", 0.10, True),
            (months[2], "a", 0.0, False),  # 1-month gap, default recurrence_gap_months=2
            (months[3], "a", 0.10, True),
            (months[4], "a", 0.10, True),
        ]
        features = compute_lifecycle_features(_freq_df(rows))
        row = features[features["trend"] == "a"].iloc[0]
        assert row["num_active_periods"] == 1
        assert row["active_run_length"] == 4

    def test_num_active_periods_counts_real_gaps(self):
        months = _months(5)
        rows = [
            (months[0], "a", 0.10, True),
            (months[1], "a", 0.0, False),
            (months[2], "a", 0.0, False),
            (months[3], "a", 0.0, False),  # 3-month gap > recurrence_gap_months=2
            (months[4], "a", 0.10, True),
        ]
        features = compute_lifecycle_features(_freq_df(rows))
        row = features[features["trend"] == "a"].iloc[0]
        assert row["num_active_periods"] == 2

    def test_excludes_unknown_by_default(self):
        months = _months(2)
        rows = [
            (months[0], "a", 0.10, True),
            (months[1], "a", 0.10, True),
            (months[0], "unknown", 0.50, True),
            (months[1], "unknown", 0.50, True),
        ]
        features = compute_lifecycle_features(_freq_df(rows))
        assert "unknown" not in features["trend"].tolist()
        assert list(features["trend"]) == ["a"]


class TestClassifyLifecycles:
    def test_persistent_trend(self):
        months = _months(8)
        rows = [(m, "a", 0.20, True) for m in months]
        result = classify_lifecycles(_freq_df(rows))
        assert result[result["trend"] == "a"].iloc[0]["state"] == "persistent"

    def test_rising_trend(self):
        months = _months(8)
        rows = [(m, "a", 0.0, False) for m in months[:5]]
        rows += [
            (months[5], "a", 0.10, True),
            (months[6], "a", 0.20, True),
            (months[7], "a", 0.35, True),
        ]
        result = classify_lifecycles(_freq_df(rows))
        assert result[result["trend"] == "a"].iloc[0]["state"] == "rising"

    def test_declining_trend(self):
        months = _months(10)
        freqs = [0.30, 0.32, 0.35, 0.40, 0.38, 0.30, 0.20, 0.15, 0.10, 0.08]
        rows = [(m, "a", f, True) for m, f in zip(months, freqs, strict=True)]
        result = classify_lifecycles(_freq_df(rows))
        row = result[result["trend"] == "a"].iloc[0]
        assert row["state"] == "declining"
        assert row["peak_month"] == months[3]  # 0.40 is the peak

    def test_seasonal_recurring_trend(self):
        # Active Jun-Aug 2024, quiet through spring, active Jun-Aug 2025, quiet since.
        months = _months(24)  # 2024-01 through 2025-12
        active_months = {"2024-06", "2024-07", "2024-08", "2025-06", "2025-07", "2025-08"}
        rows = [(m, "a", 0.20 if m in active_months else 0.0, m in active_months) for m in months]
        result = classify_lifecycles(_freq_df(rows))
        row = result[result["trend"] == "a"].iloc[0]
        assert row["num_active_periods"] == 2
        assert row["state"] == "seasonal_recurring"

    def test_inactive_never_active(self):
        months = _months(6)
        rows = [(m, "a", 0.0, False) for m in months]
        result = classify_lifecycles(_freq_df(rows))
        row = result[result["trend"] == "a"].iloc[0]
        assert row["state"] == "inactive"
        assert row["first_observed"] is None
        assert row["last_active"] is None
        assert row["peak_month"] is None

    def test_inactive_single_period_then_quiet(self):
        months = _months(8)
        active_months = {months[0], months[1], months[2]}
        rows = [(m, "a", 0.20 if m in active_months else 0.0, m in active_months) for m in months]
        result = classify_lifecycles(_freq_df(rows))
        row = result[result["trend"] == "a"].iloc[0]
        assert row["num_active_periods"] == 1
        assert row["state"] == "inactive"

    def test_lifecycle_version_stamped(self):
        months = _months(4)
        rows = [(m, "a", 0.20, True) for m in months]
        default_result = classify_lifecycles(_freq_df(rows))
        assert default_result.iloc[0]["lifecycle_version"] == LIFECYCLE_VERSION

        custom_result = classify_lifecycles(_freq_df(rows), lifecycle_version="v2-test")
        assert custom_result.iloc[0]["lifecycle_version"] == "v2-test"

    def test_custom_thresholds_change_classification(self):
        months = _months(8)
        freqs = [0.0] * 5 + [0.20, 0.21, 0.22]
        active = [False] * 5 + [True, True, True]
        rows = [(m, "a", f, a) for m, f, a in zip(months, freqs, active, strict=True)]
        # slope over last 3 points = (0.22-0.20)/2 = 0.01, active_run_length=3
        default_result = classify_lifecycles(_freq_df(rows))
        assert default_result.iloc[0]["state"] == "rising"  # run length 3 < persistent_min 4

        # Lengthen the run so persistent_min is met, and use a tight rising threshold
        months2 = _months(10)
        freqs2 = [0.20] * 7 + [0.20, 0.21, 0.22]
        active2 = [True] * 10
        rows2 = [(m, "a", f, a) for m, f, a in zip(months2, freqs2, active2, strict=True)]
        loose = classify_lifecycles(_freq_df(rows2), rising_slope_threshold=0.03)
        tight = classify_lifecycles(_freq_df(rows2), rising_slope_threshold=0.005)
        assert loose.iloc[0]["state"] == "persistent"
        assert tight.iloc[0]["state"] == "rising"
