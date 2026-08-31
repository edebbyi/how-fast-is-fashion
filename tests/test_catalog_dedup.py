"""Tests for src/fashion_forensics/nlp/catalog_dedup.py."""

from __future__ import annotations

from fashion_forensics.nlp.catalog_dedup import dedupe_for_counting


def _row(record_id, month, product_code, confidence, **overrides):
    row = {
        "record_id": record_id,
        "month": month,
        "product_code": product_code,
        "trend_pred": "basics",
        "confidence": confidence,
    }
    row.update(overrides)
    return row


def test_keeps_single_occurrence_products_untouched():
    records = [_row("2024-02_001", "2024-02", "111", 0.7)]
    assert dedupe_for_counting(records) == records


def test_collapses_same_month_product_to_highest_confidence():
    records = [
        _row("2025-05_062", "2025-05", "01758532700", 0.60),
        _row("2025-05_079", "2025-05", "01758532700", 0.41),
        _row("2025-05_087", "2025-05", "01758532700", 0.59),
    ]
    result = dedupe_for_counting(records)
    assert len(result) == 1
    assert result[0]["record_id"] == "2025-05_062"


def test_ties_broken_by_lowest_record_id():
    records = [
        _row("2025-05_088", "2025-05", "01758532700", 0.50),
        _row("2025-05_062", "2025-05", "01758532700", 0.50),
    ]
    result = dedupe_for_counting(records)
    assert len(result) == 1
    assert result[0]["record_id"] == "2025-05_062"


def test_same_product_code_different_months_stay_separate():
    records = [
        _row("2024-02_001", "2024-02", "111", 0.7),
        _row("2024-03_001", "2024-03", "111", 0.7),
    ]
    result = dedupe_for_counting(records)
    assert len(result) == 2


def test_records_without_product_code_are_kept_individually():
    records = [
        _row("2024-02_001", "2024-02", None, 0.7),
        _row("2024-02_002", "2024-02", None, 0.6),
    ]
    result = dedupe_for_counting(records)
    assert len(result) == 2


def test_mixed_grouped_and_ungrouped_records():
    records = [
        _row("2025-05_062", "2025-05", "01758532700", 0.60),
        _row("2025-05_079", "2025-05", "01758532700", 0.41),
        _row("2025-05_010", "2025-05", None, 0.55),
    ]
    result = dedupe_for_counting(records)
    ids = {r["record_id"] for r in result}
    assert ids == {"2025-05_062", "2025-05_010"}
