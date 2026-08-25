"""Page scraper for trend curation: configured trend-site pages, or any
single page URL pasted on demand (e.g. a Pinterest board/pin link).

Unlike `fashion_forensics.mining.miner` (which recovers historical retailer
product pages via the Wayback Machine), this scrapes the *live* web —
editorial roundups, lookbooks, trend reports, or a specific board/pin a
reviewer points it at — and stages candidate images with source attribution
for human review. Nothing here writes to the labeled reference corpus
directly; see `fashion_forensics.curation.store.approve`.

Two extraction passes run on every fetched page:
  - `extract_candidate_images`  — generic img tags / lazy-load attrs / og:image.
    Works for most editorial pages.
  - `extract_pinterest_images`  — regex-scans the raw HTML/JSON for
    i.pinimg.com URLs. Pinterest hydrates board/pin pages from a JSON
    payload embedded in <script> tags for SEO, so a DOM-only scan misses
    most pins; this only runs when the URL looks like Pinterest.
    Note: this only sees what Pinterest server-renders on first load, not
    everything that loads as you scroll a board — expect a first batch,
    not the whole board.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger

from fashion_forensics.config import load_yaml_config
from fashion_forensics.curation import store
from fashion_forensics.curation.net import IMAGE_EXT_RE, detect_ext, download_image, new_session

PINIMG_URL_RE = re.compile(
    r"https://i\.pinimg\.com/[^\s\"'\\]+?\.(?:jpg|jpeg|png|webp)", re.IGNORECASE
)
_PINIMG_SIZE_RE = re.compile(r"/(\d+x\d*|originals)/")


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
    """Pull plausible content-image URLs (+ alt text) out of a page's DOM.

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


def _upsize_pinimg_url(url: str, target_size: str = "736x") -> str:
    """Swap a small pinimg size segment (e.g. /236x/) for a larger one.

    Leaves /originals/ URLs alone (already max quality) and leaves the URL
    unchanged if no recognizable size segment is found.
    """
    match = _PINIMG_SIZE_RE.search(url)
    if not match or match.group(1) == "originals":
        return url
    return url[: match.start(1)] + target_size + url[match.end(1) :]


def extract_pinterest_images(html_text: str) -> list[dict[str, str | None]]:
    """Regex-scan raw page text for i.pinimg.com image URLs.

    Pinterest board/pin pages embed pin data as JSON inside <script> tags
    rather than plain <img src>, so this scans the full response body the
    way fashion_forensics.mining.miner scans retailer script JSON, rather
    than relying on a parsed-DOM walk.
    """
    seen: dict[str, None] = {}
    for match in PINIMG_URL_RE.finditer(html_text):
        seen.setdefault(_upsize_pinimg_url(match.group(0)), None)
    return [{"image_url": url, "alt_text": None} for url in seen]


def _looks_like_pinterest(url: str) -> bool:
    lower = url.lower()
    return "pinterest." in lower or "pinimg.com" in lower


class TrendSiteScraper:
    """Scrapes configured trend-site pages, or a single ad-hoc page, into the
    curation review queue."""

    def __init__(self, config: CurationConfig | None = None) -> None:
        if config is None:
            config = CurationConfig(load_yaml_config("curation"))
        self.config = config
        self.session = new_session()

    def _fetch(self, url: str) -> tuple[BeautifulSoup, str] | None:
        for attempt in range(self.config.retry_count):
            try:
                resp = self.session.get(url, timeout=self.config.request_timeout)
                resp.raise_for_status()
                return BeautifulSoup(resp.text, "html.parser"), resp.text
            except Exception as e:
                logger.warning(f"fetch {url} attempt {attempt + 1}: {e}")
                time.sleep(1)
        return None

    def scrape_source(self, source: dict[str, Any]) -> dict[str, int]:
        name = source["name"]
        url = source["url"]
        trend_hint = source.get("trend_hint")
        counts = {"found": 0, "added": 0, "skipped_duplicate": 0, "skipped_download": 0}

        fetched = self._fetch(url)
        if fetched is None:
            logger.warning(f"{name}: could not fetch {url}")
            return counts
        soup, html_text = fetched

        candidates = extract_candidate_images(soup, url, self.config.bad_url_signals)
        if _looks_like_pinterest(url):
            seen_urls = {c["image_url"] for c in candidates}
            for extra in extract_pinterest_images(html_text):
                if extra["image_url"] not in seen_urls:
                    candidates.append(extra)
                    seen_urls.add(extra["image_url"])

        candidates = candidates[: self.config.max_images_per_source]
        counts["found"] = len(candidates)
        logger.info(f"{name}: {len(candidates)} candidate images")

        for candidate in candidates:
            image_url = candidate["image_url"]
            content = download_image(
                self.session,
                image_url,
                timeout=self.config.request_timeout,
                min_bytes=self.config.min_image_bytes,
            )
            if content is None:
                counts["skipped_download"] += 1
                continue

            record = store.add_candidate(
                content,
                detect_ext(image_url),
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

    def scrape_url(
        self, url: str, *, name: str | None = None, trend_hint: str | None = None
    ) -> dict[str, int]:
        """Scrape a single page on demand — e.g. a Pinterest board/pin link
        pasted into the review app — without adding it to configs/curation.yaml.
        """
        return self.scrape_source({"name": name or url, "url": url, "trend_hint": trend_hint})

    def run(self) -> dict[str, dict[str, int]]:
        """Scrape every configured source. Returns per-source counts."""
        summary: dict[str, dict[str, int]] = {}
        for source in self.config.sources:
            summary[source["name"]] = self.scrape_source(source)
            time.sleep(self.config.delay_between_sources)

        total_added = sum(s["added"] for s in summary.values())
        logger.info(f"Curation scrape complete: +{total_added} new candidates queued for review")
        return summary
