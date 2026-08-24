"""Tests for trend-site candidate-image extraction (pure HTML parsing, no network)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from fashion_forensics.curation.scraper import extract_candidate_images

BAD_SIGNALS = ["logo", "icon", "sprite", "favicon", ".svg", "pixel", "tracking"]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestExtractCandidateImages:
    def test_extracts_plain_img_src_with_alt_text(self):
        html = '<img src="/images/coat.jpg" alt="camel trench coat">'
        candidates = extract_candidate_images(_soup(html), "https://example.com/trends/x", [])

        assert len(candidates) == 1
        assert candidates[0]["image_url"] == "https://example.com/images/coat.jpg"
        assert candidates[0]["alt_text"] == "camel trench coat"

    def test_resolves_protocol_relative_urls(self):
        html = '<img src="//cdn.example.com/a.png">'
        candidates = extract_candidate_images(_soup(html), "https://example.com/page", [])
        assert candidates[0]["image_url"] == "https://cdn.example.com/a.png"

    def test_prefers_lazy_load_data_attrs_over_placeholder_src(self):
        html = '<img src="/loading.gif" data-src="/images/real.webp" alt="look">'
        candidates = extract_candidate_images(_soup(html), "https://example.com", [])
        urls = {c["image_url"] for c in candidates}
        assert "https://example.com/images/real.webp" in urls
        # the .gif placeholder isn't an accepted extension, so it's dropped
        assert not any(u.endswith(".gif") for u in urls)

    def test_takes_first_url_from_srcset(self):
        html = (
            '<img srcset="/images/small.jpg 480w, /images/large.jpg 1200w" '
            'alt="mob wife look">'
        )
        candidates = extract_candidate_images(_soup(html), "https://example.com", [])
        assert candidates[0]["image_url"] == "https://example.com/images/small.jpg"

    def test_filters_bad_url_signals(self):
        html = """
        <img src="/assets/logo.png" alt="logo">
        <img src="/assets/icons/icon.svg" alt="icon">
        <img src="/editorial/office-siren.jpg" alt="office siren look">
        """
        candidates = extract_candidate_images(_soup(html), "https://example.com", BAD_SIGNALS)
        urls = [c["image_url"] for c in candidates]
        assert urls == ["https://example.com/editorial/office-siren.jpg"]

    def test_ignores_data_uris(self):
        html = '<img src="data:image/png;base64,iVBORw0KGgo=" alt="inline">'
        candidates = extract_candidate_images(_soup(html), "https://example.com", [])
        assert candidates == []

    def test_picks_up_og_image_meta_tag(self):
        html = '<meta property="og:image" content="https://example.com/social/share.jpg">'
        candidates = extract_candidate_images(_soup(html), "https://example.com", [])
        assert candidates[0]["image_url"] == "https://example.com/social/share.jpg"

    def test_dedupes_repeated_urls(self):
        html = """
        <img src="/a.jpg" alt="first mention">
        <img data-src="/a.jpg" alt="second mention">
        """
        candidates = extract_candidate_images(_soup(html), "https://example.com", [])
        assert len(candidates) == 1
        assert candidates[0]["alt_text"] == "first mention"
