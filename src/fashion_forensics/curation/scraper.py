"""Generic trend-site image scraper.

Unlike `fashion_forensics.mining.miner` (which recovers historical retailer
product pages via the Wayback Machine), this scrapes the *live* web for
editorial/trend pages — "what quiet luxury looks like right now" roundups,
lookbooks, trend reports — and stages candidate images with source
attribution for human review. Nothing here writes to the labeled reference
corpus directly; see `fashion_forensics.curation.store.approve`.

Sites are configured in configs/curation.yaml. Each source page is scraped
with a single generic strategy (img tags, lazy-load attributes, og:image)
since trend blogs vary too much in markup to warrant per-site parsers.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from fashion_forensics.config import load_yaml_config
from fashion_forensics.curation import store

IMAGE_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp)(?:\?|$)", re.IGNORECASE)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class CurationConfig:
    """Parsed curation configuration from configs/curation.yaml."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.sources: list[dict[str, Any]] = raw.get("sources", [])
        self.min_image_bytes: int = raw.get("min_image_bytes", 8000)
        self.max_images_per_source: int = raw.get("max_images_per_source", 40)
        self.request_timeout: int = raw.get("request_timeout_seconds", 10)
        self.retry_count: int = raw.get("retry_count", 2)
        self.delay_between_sources: int = raw.get("delay_between_sources_seconds", 2)
        self.bad_url_signals: list[str] = raw.get("bad_url_signals", [])


def extract_candidate_images(
    soup: BeautifulSoup, page_url: str, bad_url_signals: list[str]
) -> list[dict[str, str | None]]:
    """Pull plausible content-image URLs (+ alt text) out of a trend-site page.

    Pure function of the parsed HTML — no network calls — so it's unit
    testable against fixture HTML.
    """
    candidates: dict[str, str | None] = {}

    def add(raw_url: str | None, alt: str | None = None) -> None:
        if not raw_url:
            return
        raw_url = raw_url.strip()
        if not raw_url or raw_url.startswith("data:"):
            return
        if "," in raw_url:  # srcset: "url 1x, url2 2x" -> take the first candidate
            raw_url = raw_url.split(",")[0].strip()
        raw_url = raw_url.split(" ")[0].strip()
        if not raw_url:
            return

        if raw_url.startswith("//"):
            url = "https:" + raw_url
        else:
            url = urljoin(page_url, raw_url)

        lower = url.lower()
        if not IMAGE_EXT_RE.search(lower):
            return
        if any(bad in lower for bad in bad_url_signals):
            return

        if url not in candidates or (alt and not candidates[url]):
            candidates[url] = alt

    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        add(og_image["content"])

    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip() or None
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-lazy-src", "srcset"):
            add(img.get(attr), alt)

    for source_tag in soup.find_all("source"):
        for attr in ("srcset", "data-srcset", "src"):
            add(source_tag.get(attr))

    return [{"image_url": url, "alt_text": alt} for url, alt in candidates.items()]


class TrendSiteScraper:
    """Scrapes configured trend-site pages into the curation review queue."""

    def __init__(self, config: CurationConfig | None = None) -> None:
        if config is None:
            config = CurationConfig(load_yaml_config("curation"))
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _fetch_soup(self, url: str) -> BeautifulSoup | None:
        for attempt in range(self.config.retry_count):
            try:
                resp = self.session.get(url, timeout=self.config.request_timeout)
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "html.parser")
            except Exception as e:
                logger.warning(f"fetch {url} attempt {attempt + 1}: {e}")
                time.sleep(1)
        return None

    def _download_image(self, image_url: str) -> bytes | None:
        try:
            resp = self.session.get(image_url, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("Content-Type", "").lower()
            if "image" not in content_type and not IMAGE_EXT_RE.search(image_url.lower()):
                return None
            if len(resp.content) < self.config.min_image_bytes:
                return None
            return resp.content
        except Exception as e:
            logger.debug(f"download {image_url} failed: {e}")
            return None

    def _detect_ext(self, url: str) -> str:
        match = IMAGE_EXT_RE.search(url.lower())
        if match:
            return f".{match.group(1)}"
        return ".jpg"

    def scrape_source(self, source: dict[str, Any]) -> dict[str, int]:
        name = source["name"]
        url = source["url"]
        trend_hint = source.get("trend_hint")
        counts = {"found": 0, "added": 0, "skipped_duplicate": 0, "skipped_download": 0}

        soup = self._fetch_soup(url)
        if soup is None:
            logger.warning(f"{name}: could not fetch {url}")
            return counts

        candidates = extract_candidate_images(soup, url, self.config.bad_url_signals)
        candidates = candidates[: self.config.max_images_per_source]
        counts["found"] = len(candidates)
        logger.info(f"{name}: {len(candidates)} candidate images")

        for candidate in candidates:
            image_url = candidate["image_url"]
            content = self._download_image(image_url)
            if content is None:
                counts["skipped_download"] += 1
                continue

            record = store.add_candidate(
                content,
                self._detect_ext(image_url),
                source_name=name,
                source_url=url,
                image_url=image_url,
                alt_text=candidate.get("alt_text"),
                trend_hint=trend_hint,
            )
            if record is None:
                counts["skipped_duplicate"] += 1
            else:
                counts["added"] += 1

        return counts

    def run(self) -> dict[str, dict[str, int]]:
        """Scrape every configured source. Returns per-source counts."""
        summary: dict[str, dict[str, int]] = {}
        for source in self.config.sources:
            summary[source["name"]] = self.scrape_source(source)
            time.sleep(self.config.delay_between_sources)

        total_added = sum(s["added"] for s in summary.values())
        logger.info(f"Curation scrape complete: +{total_added} new candidates queued for review")
        return summary
