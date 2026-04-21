"""Tests for evaluation benchmark metric functions."""

from __future__ import annotations

from fashion_forensics.evaluation.benchmark import (
    compute_calibration,
    compute_core_metrics,
    compute_error_analysis,
    compute_per_class_metrics,
    compute_threshold_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Perfect predictions for a simple three-class problem, all above threshold.
Y_TRUE = ["rising", "rising", "falling", "stable", "stable", "falling"]
Y_PRED = ["rising", "rising", "falling", "stable", "stable", "falling"]
CONFS = [0.95, 0.85, 0.90, 0.82, 0.88, 0.92]

# Imperfect predictions: two wrong, one below threshold.
Y_TRUE_NOISY = ["rising", "rising", "falling", "stable", "stable", "falling"]
Y_PRED_NOISY = ["rising", "stable", "falling", "stable", "rising", "falling"]
CONFS_NOISY = [0.95, 0.85, 0.70, 0.82, 0.88, 0.92]


# ---------------------------------------------------------------------------
# 1. Core metrics
# ---------------------------------------------------------------------------


class TestCoreMetrics:
    def test_perfect_predictions_at_threshold(self):
        result = compute_core_metrics(Y_TRUE, Y_PRED, CONFS, threshold=0.8)
        assert result["f1_macro"] == 1.0
        assert result["precision_macro"] == 1.0
        assert result["recall_macro"] == 1.0
        assert result["tagged_count"] == 6
        assert result["abstained_count"] == 0

    def test_below_threshold_excluded(self):
        result = compute_core_metrics(Y_TRUE_NOISY, Y_PRED_NOISY, CONFS_NOISY, threshold=0.8)
        # conf=0.70 is below 0.8, so that prediction (index 2) is excluded
        assert result["tagged_count"] == 5
        assert result["abstained_count"] == 1

    def test_all_below_threshold_returns_zeros(self):
        confs_low = [0.1, 0.2, 0.3]
        result = compute_core_metrics(["a", "b", "c"], ["a", "b", "c"], confs_low, threshold=0.8)
        assert result["tagged_count"] == 0
        assert result["f1"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


# ---------------------------------------------------------------------------
# 2. Per-class metrics
# ---------------------------------------------------------------------------


class TestPerClassMetrics:
    def test_perfect_per_class_f1(self):
        result = compute_per_class_metrics(Y_TRUE, Y_PRED, CONFS, threshold=0.8)
        per_trend = result["per_trend"]
        for label in ("rising", "falling", "stable"):
            assert label in per_trend
            assert per_trend[label]["f1"] == 1.0

    def test_imperfect_per_class(self):
        result = compute_per_class_metrics(Y_TRUE_NOISY, Y_PRED_NOISY, CONFS_NOISY, threshold=0.8)
        per_trend = result["per_trend"]
        # "rising" has one correct and one mislabeled as "stable" in gold,
        # plus "stable" predicted where "rising" was true -> precision/recall < 1
        assert "rising" in per_trend
        assert per_trend["rising"]["f1"] < 1.0

    def test_empty_after_threshold(self):
        result = compute_per_class_metrics(["a"], ["a"], [0.1], threshold=0.8)
        assert result["per_trend"] == {}


# ---------------------------------------------------------------------------
# 3. Threshold analysis
# ---------------------------------------------------------------------------


class TestThresholdAnalysis:
    def test_returns_result_per_threshold(self):
        thresholds = [0.5, 0.7, 0.9]
        results = compute_threshold_analysis(Y_TRUE, Y_PRED, CONFS, thresholds_to_test=thresholds)
        assert len(results) == 3
        assert [r["threshold"] for r in results] == thresholds

    def test_higher_threshold_fewer_tagged(self):
        thresholds = [0.5, 0.9]
        results = compute_threshold_analysis(Y_TRUE, Y_PRED, CONFS, thresholds_to_test=thresholds)
        assert results[0]["tagged_pct"] >= results[1]["tagged_pct"]

    def test_default_thresholds_used(self):
        results = compute_threshold_analysis(Y_TRUE, Y_PRED, CONFS)
        # Default list has 12 thresholds
        assert len(results) == 12


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_bins_computed(self):
        bins = compute_calibration(Y_TRUE, Y_PRED, CONFS, n_bins=10)
        assert isinstance(bins, list)
        assert len(bins) > 0
        for b in bins:
            assert "avg_confidence" in b
            assert "actual_accuracy" in b
            assert "count" in b
            assert b["count"] > 0

    def test_perfect_calibration(self):
        # All correct, all in the 0.9-1.0 bin
        bins = compute_calibration(["a", "a", "a"], ["a", "a", "a"], [0.95, 0.95, 0.95], n_bins=10)
        high_bin = [b for b in bins if b["avg_confidence"] > 0.9]
        assert len(high_bin) == 1
        assert high_bin[0]["actual_accuracy"] == 1.0

    def test_empty_bins_skipped(self):
        # All confidences in one narrow range -> most bins empty
        bins = compute_calibration(["a"], ["a"], [0.55], n_bins=10)
        assert len(bins) == 1


# ---------------------------------------------------------------------------
# 5. Error analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_near_miss_detected(self):
        records = [
            {
                "record_id": "1",
                "trend_gold": "rising",
                "trend_pred": "rising",
                "confidence": 0.77,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8, near_miss_margin=0.05)
        # conf 0.77 is in [0.75, 0.80) -> near miss, and gold == pred
        assert len(result["near_misses"]) == 1
        assert result["near_misses"][0]["record_id"] == "1"

    def test_hard_failure_detected(self):
        records = [
            {
                "record_id": "2",
                "trend_gold": "rising",
                "trend_pred": "falling",
                "confidence": 0.95,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8)
        assert len(result["hard_failures"]) == 1
        assert result["hard_failures"][0]["record_id"] == "2"

    def test_correct_above_threshold_not_flagged(self):
        records = [
            {
                "record_id": "3",
                "trend_gold": "stable",
                "trend_pred": "stable",
                "confidence": 0.90,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8)
        assert len(result["near_misses"]) == 0
        assert len(result["hard_failures"]) == 0

    def test_confused_pairs_counted(self):
        records = [
            {
                "record_id": "a",
                "trend_gold": "rising",
                "trend_pred": "stable",
                "confidence": 0.85,
                "year": 2025,
            },
            {
                "record_id": "b",
                "trend_gold": "rising",
                "trend_pred": "stable",
                "confidence": 0.90,
                "year": 2025,
            },
            {
                "record_id": "c",
                "trend_gold": "falling",
                "trend_pred": "stable",
                "confidence": 0.82,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8)
        pairs = result["confused_pairs"]
        assert len(pairs) == 2
        # Most common pair first
        assert pairs[0]["gold"] == "rising"
        assert pairs[0]["predicted"] == "stable"
        assert pairs[0]["count"] == 2

    def test_hard_failures_sorted_by_confidence_desc(self):
        records = [
            {
                "record_id": "x",
                "trend_gold": "a",
                "trend_pred": "b",
                "confidence": 0.85,
                "year": 2025,
            },
            {
                "record_id": "y",
                "trend_gold": "a",
                "trend_pred": "b",
                "confidence": 0.99,
                "year": 2025,
            },
            {
                "record_id": "z",
                "trend_gold": "a",
                "trend_pred": "b",
                "confidence": 0.90,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8)
        confs = [r["confidence"] for r in result["hard_failures"]]
        assert confs == [0.99, 0.90, 0.85]

    def test_below_threshold_wrong_not_hard_failure(self):
        records = [
            {
                "record_id": "w",
                "trend_gold": "rising",
                "trend_pred": "falling",
                "confidence": 0.50,
                "year": 2025,
            },
        ]
        result = compute_error_analysis(records, threshold=0.8)
        assert len(result["hard_failures"]) == 0
