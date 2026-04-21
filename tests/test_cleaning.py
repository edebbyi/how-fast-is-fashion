"""Tests for the is_valid_product_record cleaning function."""

from __future__ import annotations

from fashion_forensics.mining.miner import is_valid_product_record


class TestIsValidProductRecord:
    """Test domain-specific cleaning rules."""

    def test_valid_record_passes(self):
        record = {
            "product_name": "GATHERED COTTON SHIRT",
            "product_code": "07521800",
            "price": "$29.90",
            "category": "shirt",
            "color": "white",
        }
        assert is_valid_product_record(record) is True

    def test_look_name_rejected(self):
        record = {"product_name": "LOOK", "product_code": "123456"}
        assert is_valid_product_record(record) is False

    def test_no_product_name_rejected(self):
        record = {"product_code": "123456", "price": "$29.90"}
        assert is_valid_product_record(record) is False

    def test_t_prefix_product_code_rejected(self):
        record = {"product_name": "Shirt", "product_code": "T0252104177"}
        assert is_valid_product_record(record) is False

    def test_bundle_products_rejected(self):
        record = {
            "product_name": "Bundle Set",
            "raw_card_text": '{"bundleProducts": [1, 2, 3]}',
        }
        assert is_valid_product_record(record) is False

    def test_numeric_color_rejected(self):
        record = {"product_name": "Dress", "color": "12345", "price": "$49.90"}
        assert is_valid_product_record(record) is False

    def test_no_signal_rejected(self):
        """Record with no product_name, price, or archive_product_url."""
        record = {"product_code": "07521800", "color": "red"}
        assert is_valid_product_record(record) is False

    def test_minimal_valid_with_url(self):
        record = {
            "product_name": "Jacket",
            "archive_product_url": "https://zara.com/jacket",
        }
        assert is_valid_product_record(record) is True
