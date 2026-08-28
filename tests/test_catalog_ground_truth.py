"""Tests for src/fashion_forensics/catalog_ground_truth.py - the Discover
tab's thumbs-up/down ground-truth judgments. Filesystem paths are redirected
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


def test_load_ground_truth_missing_file_returns_empty(isolated_ground_truth):
    assert catalog_ground_truth.load_ground_truth() == {}


def test_judge_item_writes_and_reads_back(isolated_ground_truth):
    catalog_ground_truth.judge_item(
        record_id="2024-02_001",
        trend_pred="quiet_luxury",
        confidence=0.87,
        max_sim=0.71,
        judged_correct=True,
    )

    loaded = catalog_ground_truth.load_ground_truth()
    assert set(loaded) == {"2024-02_001"}
    record = loaded["2024-02_001"]
    assert record["trend_pred"] == "quiet_luxury"
    assert record["confidence"] == 0.87
    assert record["max_sim"] == 0.71
    assert record["judged_correct"] is True
    assert "judged_at" in record


def test_judge_item_casts_non_native_numeric_types(isolated_ground_truth):
    """confidence/max_sim are typically a pandas row's values (numpy
    float64), not plain floats - json.dumps can't serialize those directly,
    so judge_item must cast them before writing."""

    class FakeNumpyFloat:
        """Behaves like numpy.float64 for this purpose: not a plain float,
        but float() works on it. Avoids a hard numpy test dependency."""

        def __init__(self, value: float) -> None:
            self._value = value

        def __float__(self) -> float:
            return self._value

    record = catalog_ground_truth.judge_item(
        record_id="2024-02_002",
        trend_pred="basics",
        confidence=FakeNumpyFloat(0.5),
        max_sim=FakeNumpyFloat(0.6),
        judged_correct=False,
    )
    assert isinstance(record["confidence"], float)
    assert isinstance(record["max_sim"], float)

    # Confirms it's actually JSON-serializable now, not just the right type.
    raw = isolated_ground_truth.read_text().strip()
    parsed = json.loads(raw)
    assert parsed["confidence"] == 0.5
    assert parsed["max_sim"] == 0.6


def test_judge_item_handles_missing_confidence_and_max_sim(isolated_ground_truth):
    record = catalog_ground_truth.judge_item(
        record_id="2024-02_003",
        trend_pred="unknown",
        confidence=None,
        max_sim=None,
        judged_correct=False,
    )
    assert record["confidence"] is None
    assert record["max_sim"] is None


def test_rejudging_same_record_overwrites_not_duplicates(isolated_ground_truth):
    catalog_ground_truth.judge_item(
        record_id="2024-02_001",
        trend_pred="quiet_luxury",
        confidence=0.87,
        max_sim=0.71,
        judged_correct=True,
    )
    catalog_ground_truth.judge_item(
        record_id="2024-02_001",
        trend_pred="quiet_luxury",
        confidence=0.87,
        max_sim=0.71,
        judged_correct=False,
    )

    loaded = catalog_ground_truth.load_ground_truth()
    assert len(loaded) == 1
    assert loaded["2024-02_001"]["judged_correct"] is False


def test_multiple_records_all_persisted(isolated_ground_truth):
    catalog_ground_truth.judge_item("b_record", "mob_wife", 0.9, 0.8, True)
    catalog_ground_truth.judge_item("a_record", "office_siren", 0.6, 0.5, False)

    loaded = catalog_ground_truth.load_ground_truth()
    assert set(loaded) == {"a_record", "b_record"}
