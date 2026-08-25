"""Tests for Google Custom Search JSON response parsing (pure, no network)."""

from __future__ import annotations

from fashion_forensics.curation.image_search import _parse_cse_items


class TestParseCseItems:
    def test_extracts_image_url_context_link_and_title(self):
        payload = {
            "items": [
                {
                    "title": "Camel trench coat street style",
                    "link": "https://cdn.example.com/images/coat.jpg",
                    "image": {
                        "contextLink": "https://blog.example.com/quiet-luxury",
                        "thumbnailLink": "https://cdn.example.com/thumbs/coat.jpg",
                    },
                }
            ]
        }
        results = _parse_cse_items(payload)
        assert results == [
            {
                "image_url": "https://cdn.example.com/images/coat.jpg",
                "source_page_url": "https://blog.example.com/quiet-luxury",
                "title": "Camel trench coat street style",
            }
        ]

    def test_skips_items_missing_link(self):
        payload = {"items": [{"title": "No image URL", "image": {}}]}
        assert _parse_cse_items(payload) == []

    def test_falls_back_to_image_url_when_context_link_missing(self):
        payload = {"items": [{"title": "t", "link": "https://x.com/a.jpg", "image": {}}]}
        results = _parse_cse_items(payload)
        assert results[0]["source_page_url"] == "https://x.com/a.jpg"

    def test_empty_response_returns_empty_list(self):
        assert _parse_cse_items({}) == []
