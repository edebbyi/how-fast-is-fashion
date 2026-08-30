"""Tests for src/fashion_forensics/catalog_ground_truth.py - the Discover
tab's form-based ground-truth judgments. Filesystem paths are redirected
into tmp_path via monkeypatch so these never touch the real
data/03_shared/catalog_distribution/.
"""

from __future__ import annotations

import json

import pytest

from fashion_forensics import catalog_ground_truth


@pytest.fixture
def isolated_ground_truth(tmp_path, monkeypatch):
    path = tmp_path / "catalog_ground_truth.jsonl"
    monkeypatch.setattr(catalog_ground_truth, "GROUND_TRUTH_PATH", path)
    return path


def _judgment(record_id: str, judged_correct: bool, **overrides) -> dict:
    base = {
        "record_id": record_id,
        "trend_pred": "quiet_luxury",
        "confidence": 0.87,
        "max_sim": 0.71,
        "judged_correct": judged_correct,
    }
    base.update(overrides)
    return base


def test_load_ground_truth_missing_file_returns_empty(isolated_ground_truth):
    assert catalog_ground_truth.load_ground_truth() == {}


def test_judge_items_batch_writes_and_reads_back(isolated_ground_truth):
    catalog_ground_truth.judge_items_batch([_judgment("2024-02_001", True)])

    loaded = catalog_ground_truth.load_ground_truth()
    assert set(loaded) == {"2024-02_001"}
    record = loaded["2024-02_001"]
    assert record["trend_pred"] == "quiet_luxury"
    assert record["confidence"] == 0.87
    assert record["max_sim"] == 0.71
    assert record["judged_correct"] is True
    assert record["reason"] is None
    assert record["corrected_trend"] is None
    assert "judged_at" in record


def test_judge_items_batch_casts_non_native_numeric_types(isolated_ground_truth):
    """confidence/max_sim are typically a pandas row's values (numpy
    float64), not plain floats - json.dumps can't serialize those
    directly, so this must cast them before writing."""

    class FakeNumpyFloat:
        """Behaves like numpy.float64 for this purpose: not a plain float,
        but float() works on it. Avoids a hard numpy test dependency."""

        def __init__(self, value: float) -> None:
            self._value = value

        def __float__(self) -> float:
            return self._value

    saved = catalog_ground_truth.judge_items_batch(
        [
            _judgment(
                "2024-02_002",
                False,
                confidence=FakeNumpyFloat(0.5),
                max_sim=FakeNumpyFloat(0.6),
            )
        ]
    )
    assert isinstance(saved[0]["confidence"], float)
    assert isinstance(saved[0]["max_sim"], float)

    # Confirms it's actually JSON-serializable now, not just the right type.
    raw = isolated_ground_truth.read_text().strip()
    parsed = json.loads(raw)
    assert parsed["confidence"] == 0.5
    assert parsed["max_sim"] == 0.6


def test_judge_items_batch_handles_missing_confidence_and_max_sim(isolated_ground_truth):
    saved = catalog_ground_truth.judge_items_batch(
        [_judgment("2024-02_003", False, trend_pred="unknown", confidence=None, max_sim=None)]
    )
    assert saved[0]["confidence"] is None
    assert saved[0]["max_sim"] is None


def test_rejudging_same_record_overwrites_not_duplicates(isolated_ground_truth):
    catalog_ground_truth.judge_items_batch([_judgment("2024-02_001", True)])
    catalog_ground_truth.judge_items_batch([_judgment("2024-02_001", False)])

    loaded = catalog_ground_truth.load_ground_truth()
    assert len(loaded) == 1
    assert loaded["2024-02_001"]["judged_correct"] is False


def test_batch_persists_multiple_records_and_preserves_earlier_ones(isolated_ground_truth):
    catalog_ground_truth.judge_items_batch([_judgment("a_record", True)])
    catalog_ground_truth.judge_items_batch(
        [_judgment("b_record", True), _judgment("c_record", False)]
    )

    loaded = catalog_ground_truth.load_ground_truth()
    assert set(loaded) == {"a_record", "b_record", "c_record"}


def test_reason_and_corrected_trend_saved_when_provided(isolated_ground_truth):
    saved = catalog_ground_truth.judge_items_batch(
        [
            _judgment(
                "2024-02_004",
                False,
                reason="wrong material",
                corrected_trend="mob_wife",
            )
        ]
    )
    assert saved[0]["reason"] == "wrong material"
    assert saved[0]["corrected_trend"] == "mob_wife"


def test_reason_whitespace_stripped_and_blank_becomes_none(isolated_ground_truth):
    saved = catalog_ground_truth.judge_items_batch(
        [_judgment("2024-02_005", False, reason="  too plain  ")]
    )
    assert saved[0]["reason"] == "too plain"

    saved = catalog_ground_truth.judge_items_batch([_judgment("2024-02_005", False, reason="   ")])
    assert saved[0]["reason"] is None


def test_corrected_trend_accepts_unknown_as_a_valid_value(isolated_ground_truth):
    """'unknown' means the item doesn't belong to any of the 4 active
    trends at all - a real, meaningful answer, not a placeholder."""
    saved = catalog_ground_truth.judge_items_batch(
        [_judgment("2024-02_006", False, corrected_trend="unknown")]
    )
    assert saved[0]["corrected_trend"] == "unknown"


def test_omitted_reason_and_corrected_trend_default_to_none(isolated_ground_truth):
    saved = catalog_ground_truth.judge_items_batch([_judgment("2024-02_007", True)])
    assert saved[0]["reason"] is None
    assert saved[0]["corrected_trend"] is None
