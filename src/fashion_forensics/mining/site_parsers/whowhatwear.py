"""Who What Wear site parser for the trend-image scraper.

Content images sit inside `<figure class="van-image-figure ...">` blocks; the
plain `<img src=...>` inside each one is a clean, unsuffixed CDN URL (the
sibling `<picture>`/`<source>` elements are responsive variants of the same
photo and aren't needed). Verified against real article HTML on 2026-08-24.

robots.txt disallows scraping the site's own search (`*searchTerm=*`), and no
discoverable tag-index page exists for trend names like "quiet luxury" - so
this parser takes a specific article URL rather than crawling an index page
itself. The curated article URLs per trend live in configs/scraping.yaml.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

SITE_NAME = "whowhatwear"


@dataclass
class ScrapeCandidate:
    source_url: str
    image_url: str
    suggested_trend: str | None
    suggestion_basis: str | None


def parse_whowhatwear(
    session: requests.Session,
    article_url: str,
    trend_hint: str | None,
    max_images: int,
    timeout: int,
) -> Iterator[ScrapeCandidate]:
    """Yield candidate images from one Who What Wear article."""
    resp = session.get(article_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    count = 0
    for figure in soup.find_all("figure", class_="van-image-figure"):
        if count >= max_images:
            break
        img = figure.find("img")
        image_url = img.get("src") if img else None
        if not image_url:
            continue
        yield ScrapeCandidate(
            source_url=article_url,
            image_url=image_url,
            suggested_trend=trend_hint,
            suggestion_basis="curated_article",
        )
        count += 1
