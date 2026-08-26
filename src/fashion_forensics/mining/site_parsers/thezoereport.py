"""The Zoe Report site parser for the trend-image scraper.

Unlike whowhatwear.com, this site (Bustle Digital Group) uses auto-generated,
non-semantic CSS class names on its <figure> wrappers (e.g. "Qf5 E1M") - those
are build artifacts likely to change on the next frontend deploy, too fragile
to select on. Content images are identified instead by a stable signal: they
load from the imgix.bustle.com CDN and carry a real (non-empty) alt attribute,
which cleanly excludes tracking pixels (quantserve.com, scorecardresearch.com,
facebook.com/tr, etc. - no alt text, different domains).

Real content mixes genuine editorial/outfit photos with shoppable
product-grid images (plain-background product shots) - this parser doesn't
try to tell them apart; that's left to human review in the Curate UI.

robots.txt is open for general crawling here, and thezoereport.com declares
an RSL license (rslcollective.org/attribution.xml) explicitly permitting
AI training/input use with an attribution requirement - satisfied by
candidates.jsonl already storing source_url per image. Verified against
real article HTML on 2026-08-25.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

SITE_NAME = "thezoereport"
CONTENT_IMAGE_DOMAIN = "imgix.bustle.com"


@dataclass
class ScrapeCandidate:
    source_url: str
    image_url: str
    suggested_trend: str | None
    suggestion_basis: str | None


def parse_thezoereport(
    session: requests.Session,
    article_url: str,
    trend_hint: str | None,
    max_images: int,
    timeout: int,
) -> Iterator[ScrapeCandidate]:
    """Yield candidate images from one The Zoe Report article."""
    resp = session.get(article_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    count = 0
    for img in soup.find_all("img"):
        if count >= max_images:
            break
        src = img.get("src")
        alt = img.get("alt", "").strip()
        if not src or CONTENT_IMAGE_DOMAIN not in src or not alt:
            continue
        yield ScrapeCandidate(
            source_url=article_url,
            image_url=src,
            suggested_trend=trend_hint,
            suggestion_basis="curated_article",
        )
        count += 1
