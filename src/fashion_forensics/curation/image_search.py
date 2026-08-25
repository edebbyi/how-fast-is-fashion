"""Query-driven image search: free-text queries like "clean luxury fashion
outfit" against Google's Custom Search JSON API (image mode), queued for
human review the same way as page scraping (fashion_forensics.curation.scraper).

Requires two env vars (see .env.example):
    GOOGLE_CSE_API_KEY  — Google Cloud API key with the Custom Search API enabled
                          (console.cloud.google.com -> APIs & Services -> enable
                          "Custom Search API" -> Credentials -> Create API key)
    GOOGLE_CSE_ID       — a Programmable Search Engine ID (the "cx" param), created
                          at programmablesearchengine.google.com with "Image search"
                          and "Search the entire web" both turned on

Free tier: 100 queries/day, 10 results per query (this module paginates up
to 100 results per call, Google's hard per-query cap).
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from fashion_forensics.config import load_yaml_config, settings
from fashion_forensics.curation import store
from fashion_forensics.curation.net import detect_ext, download_image, new_session

CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
RESULTS_PER_PAGE = 10
MAX_TOTAL_RESULTS = 100  # Google CSE hard limit: start + num - 1 <= 100


class ImageSearchConfig:
    """Parsed image-search settings from configs/curation.yaml."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.min_image_bytes: int = raw.get("min_image_bytes", 8000)
        self.request_timeout: int = raw.get("request_timeout_seconds", 10)
        self.retry_count: int = raw.get("retry_count", 2)
        self.default_max_results: int = raw.get("search_max_results_default", 20)
        self.safe_mode: str = raw.get("search_safe_mode", "active")


def _parse_cse_items(payload: dict[str, Any]) -> list[dict[str, str | None]]:
    """Pull {image_url, source_page_url, title} out of a Google CSE JSON response.

    Pure function of the parsed JSON — no network — so it's unit testable
    against a fixture payload.
    """
    results: list[dict[str, str | None]] = []
    for item in payload.get("items", []):
        image_url = item.get("link")
        if not image_url:
            continue
        image_meta = item.get("image") or {}
        results.append(
            {
                "image_url": image_url,
                "source_page_url": image_meta.get("contextLink") or image_url,
                "title": item.get("title"),
            }
        )
    return results


class GoogleImageSearchProvider:
    """Runs a free-text image search and queues results for review."""

    def __init__(self, config: ImageSearchConfig | None = None) -> None:
        if config is None:
            config = ImageSearchConfig(load_yaml_config("curation"))
        self.config = config
        self.session = new_session()

        if not settings.google_cse_api_key or not settings.google_cse_id:
            raise RuntimeError(
                "GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID must be set (see .env.example) "
                "to use image search."
            )

    def search(self, query: str, max_results: int) -> list[dict[str, str | None]]:
        """Query Google CSE, paginating in pages of 10 up to max_results (capped at 100)."""
        max_results = min(max_results, MAX_TOTAL_RESULTS)
        hits: list[dict[str, str | None]] = []
        start = 1

        while len(hits) < max_results and start <= MAX_TOTAL_RESULTS - RESULTS_PER_PAGE + 1:
            num = min(RESULTS_PER_PAGE, max_results - len(hits))
            params = {
                "key": settings.google_cse_api_key,
                "cx": settings.google_cse_id,
                "q": query,
                "searchType": "image",
                "num": num,
                "start": start,
                "safe": self.config.safe_mode,
            }

            payload = None
            for attempt in range(self.config.retry_count):
                try:
                    resp = self.session.get(
                        CSE_ENDPOINT, params=params, timeout=self.config.request_timeout
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception as e:
                    logger.warning(f"CSE query {query!r} attempt {attempt + 1}: {e}")
                    time.sleep(1)

            if payload is None:
                break

            page_hits = _parse_cse_items(payload)
            if not page_hits:
                break

            hits.extend(page_hits)
            start += num

        return hits[:max_results]

    def run(
        self, query: str, max_results: int | None = None, trend_hint: str | None = None
    ) -> dict[str, int]:
        """Search + download + queue. Returns the same counts shape as scraper.run()."""
        max_results = max_results or self.config.default_max_results
        hits = self.search(query, max_results)
        counts = {"found": len(hits), "added": 0, "skipped_duplicate": 0, "skipped_download": 0}
        logger.info(f"Image search {query!r}: {len(hits)} results")

        for hit in hits:
            image_url = hit["image_url"]
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
                source_name=f'Google Image Search: "{query}"',
                source_url=hit.get("source_page_url") or image_url,
                image_url=image_url,
                alt_text=hit.get("title"),
                trend_hint=trend_hint,
            )
            if record is None:
                counts["skipped_duplicate"] += 1
            else:
                counts["added"] += 1

        logger.info(f"Image search {query!r} complete: +{counts['added']} new candidates queued")
        return counts
