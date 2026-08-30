"""Tests for src/fashion_forensics/nlp/material_override.py - the cheap
text-keyword override for basics predictions that are actually
quiet_luxury per the retailer's own product name/description.
"""

from __future__ import annotations

from fashion_forensics.nlp.material_override import (
    load_fine_material_keywords,
    match_fine_material_keyword,
)


def test_load_fine_material_keywords_matches_shipped_taxonomy():
    """Not hardcoded against a snapshot - reads the real
    configs/trend_rules.yaml so this test breaks (loudly) if that rule
    block's material list ever changes, rather than silently drifting."""
    keywords = load_fine_material_keywords()
    assert set(keywords) == {"wool", "cashmere", "tweed", "silk"}


def test_match_finds_keyword_in_product_name():
    record = {"product_name": "WOOL AND CASHMERE BLEND KNIT SWEATER", "description": None}
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) == "wool"


def test_match_finds_keyword_in_description_when_not_in_name():
    record = {
        "product_name": "OVERSIZED SWEATER",
        "description": "Relaxed-fit sweater in a soft cashmere blend.",
    }
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) == "cashmere"


def test_match_returns_none_when_no_keyword_present():
    record = {"product_name": "COTTON T-SHIRT", "description": "Plain crew-neck tee."}
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) is None


def test_match_is_case_insensitive():
    record = {"product_name": "Silk Camisole Top", "description": None}
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) == "silk"


def test_match_handles_missing_product_name_and_description():
    record = {"product_name": None, "description": None}
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) is None


def test_match_returns_first_matching_keyword_in_list_order():
    record = {"product_name": "WOOL AND SILK BLEND SCARF", "description": None}
    # "wool" comes before "silk" in the keyword list, so it wins - this
    # just documents the (arbitrary but deterministic) tie-break, not a
    # claim that one keyword matters more than another.
    assert match_fine_material_keyword(record, ["wool", "cashmere", "tweed", "silk"]) == "wool"
