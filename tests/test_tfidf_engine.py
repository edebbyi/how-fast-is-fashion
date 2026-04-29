"""Tests for fashion_forensics.nlp.tfidf_engine."""

from __future__ import annotations

import pytest

from fashion_forensics.nlp.tfidf_engine import (
    flatten_attributes,
    lifecycle_curve,
    monthly_attribute_tfidf,
    trend_signature,
)


class TestFlattenAttributes:
    def test_basic(self):
        rec = {
            "material": {"value": ["linen"], "source": "text", "confidence": 0.9},
            "color_profile": {"value": ["white", "blue"], "source": "image", "confidence": 0.8},
        }
        tokens = flatten_attributes(rec, fields=("material", "color_profile"))
        assert tokens == ["material:linen", "color_profile:white", "color_profile:blue"]

    def test_skips_empty_values(self):
        rec = {
            "material": {"value": [], "source": "text", "confidence": 0.0},
            "color_profile": {"value": ["white"]},
        }
        tokens = flatten_attributes(rec, fields=("material", "color_profile"))
        assert tokens == ["color_profile:white"]

    def test_skips_missing_field(self):
        rec = {"material": {"value": ["linen"]}}
        tokens = flatten_attributes(rec, fields=("material", "pattern"))
        assert tokens == ["material:linen"]

    def test_field_prefix_disambiguates(self):
        # If 'beige' could appear as both color and material, the field prefix
        # keeps them as distinct tokens.
        rec = {
            "color_profile": {"value": ["beige"]},
            "material": {"value": ["beige_linen"]},
        }
        tokens = flatten_attributes(rec, fields=("color_profile", "material"))
        assert "color_profile:beige" in tokens
        assert "material:beige_linen" in tokens
        assert tokens != ["beige", "beige_linen"]


class TestMonthlyAttributeTfidf:
    def test_two_months_one_distinguishing_attribute(self):
        # 'leopard_print' appears only in month 2024-03 -> should have high TF-IDF there.
        # 'crew' appears in both months -> lower TF-IDF (lower IDF).
        records = [
            {
                "record_id": "2024-02_001",
                "month": "2024-02",
                "neckline": {"value": ["crew"]},
                "material": {"value": ["cotton"]},
            },
            {
                "record_id": "2024-03_001",
                "month": "2024-03",
                "neckline": {"value": ["crew"]},
                "pattern": {"value": ["leopard_print"]},
            },
        ]
        df = monthly_attribute_tfidf(records)

        assert "neckline:crew" in df.index
        assert "pattern:leopard_print" in df.index
        assert list(df.columns) == ["2024-02", "2024-03"]

        # leopard_print has zero in 2024-02, positive in 2024-03
        assert df.loc["pattern:leopard_print", "2024-02"] == 0.0
        assert df.loc["pattern:leopard_print", "2024-03"] > 0.0

        # crew appears in both months and should have lower IDF -> lower max TF-IDF
        # than leopard_print which is concentrated in one month.
        assert df.loc["pattern:leopard_print", "2024-03"] > df.loc["neckline:crew", "2024-02"]

    def test_falls_back_to_record_id_for_month(self):
        # No `month` field, but record_id encodes it.
        records = [
            {"record_id": "2024-02_001", "material": {"value": ["linen"]}},
            {"record_id": "2024-03_001", "material": {"value": ["silk"]}},
        ]
        df = monthly_attribute_tfidf(records)
        assert set(df.columns) == {"2024-02", "2024-03"}

    def test_empty_records_returns_empty_dataframe(self):
        df = monthly_attribute_tfidf([])
        assert df.empty


class TestTrendSignature:
    def test_per_trend_top_tokens(self):
        # mob_wife products always have leopard_print + fur, never quiet_luxury.
        # quiet_luxury products always have cashmere + neutral, never mob_wife.
        records = [
            {
                "record_id": f"2024-02_{i:03d}",
                "trend_pred": "mob_wife",
                "pattern": {"value": ["leopard_print"]},
                "material": {"value": ["fur"]},
            }
            for i in range(3)
        ] + [
            {
                "record_id": f"2024-03_{i:03d}",
                "trend_pred": "quiet_luxury",
                "material": {"value": ["cashmere"]},
                "color_profile": {"value": ["neutral"]},
            }
            for i in range(3)
        ]

        sig = trend_signature(records, top_k=5)

        assert set(sig.keys()) == {"mob_wife", "quiet_luxury"}

        mob_tokens = {tok for tok, _ in sig["mob_wife"]}
        quiet_tokens = {tok for tok, _ in sig["quiet_luxury"]}

        assert "pattern:leopard_print" in mob_tokens
        assert "material:fur" in mob_tokens
        assert "material:cashmere" in quiet_tokens
        assert "color_profile:neutral" in quiet_tokens

        # Cross-class tokens should not appear in the other trend's signature
        assert "pattern:leopard_print" not in quiet_tokens
        assert "material:cashmere" not in mob_tokens

    def test_top_k_respected(self):
        records = [
            {
                "record_id": f"2024-02_{i:03d}",
                "trend_pred": "mob_wife",
                "material": {"value": [f"mat_{i}"]},
                "pattern": {"value": [f"pat_{i}"]},
            }
            for i in range(10)
        ]
        sig = trend_signature(records, top_k=3)
        assert len(sig["mob_wife"]) <= 3

    def test_skips_records_without_trend(self):
        records = [
            {
                "record_id": "2024-02_001",
                "trend_pred": "mob_wife",
                "material": {"value": ["fur"]},
            },
            {
                "record_id": "2024-02_002",
                # no trend_pred set -> should be skipped
                "material": {"value": ["cotton"]},
            },
        ]
        sig = trend_signature(records)
        assert set(sig.keys()) == {"mob_wife"}
        assert "material:cotton" not in {tok for tok, _ in sig["mob_wife"]}


class TestLifecycleCurve:
    def test_curve_has_one_row_per_month(self):
        records = [
            {"record_id": f"2024-{m:02d}_001", "material": {"value": ["linen"]}}
            for m in range(1, 7)
        ]
        df = lifecycle_curve(records, "material:linen", smooth_window=3)
        assert len(df) == 6
        assert list(df.columns) == ["month", "tfidf", "tfidf_smooth"]

    def test_smooth_window_one_disables_smoothing_column(self):
        records = [
            {"record_id": f"2024-{m:02d}_001", "material": {"value": ["linen"]}}
            for m in range(1, 4)
        ]
        df = lifecycle_curve(records, "material:linen", smooth_window=1)
        assert "tfidf_smooth" not in df.columns

    def test_unknown_token_raises(self):
        records = [{"record_id": "2024-02_001", "material": {"value": ["linen"]}}]
        with pytest.raises(ValueError, match="not found"):
            lifecycle_curve(records, "material:silk_organza")

    def test_smoothed_curve_smooths(self):
        # Build a spike: zero, zero, high, zero, zero. Smoothed should reduce it.
        records = []
        for m in range(1, 6):
            n = 10 if m == 3 else 1
            for i in range(n):
                rec = {"record_id": f"2024-{m:02d}_{i:03d}"}
                if m == 3:
                    rec["pattern"] = {"value": ["spike"]}
                else:
                    rec["pattern"] = {"value": ["baseline"]}
                records.append(rec)

        df = lifecycle_curve(records, "pattern:spike", smooth_window=3)
        # Raw: spike concentrated in month 3
        spike_raw = df.loc[df["month"] == "2024-03", "tfidf"].iloc[0]
        spike_smooth = df.loc[df["month"] == "2024-03", "tfidf_smooth"].iloc[0]
        # Smoothed value at the peak averages with neighbors -> lower than raw peak
        assert spike_smooth < spike_raw
