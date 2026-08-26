"""Registry of per-site trend-image parsers.

Adding a new site is: write a parse_<site>(session, article_url, trend_hint,
max_images, timeout) -> Iterator[ScrapeCandidate] function in its own module,
then register it here.
"""

from __future__ import annotations

from .thezoereport import parse_thezoereport
from .whowhatwear import ScrapeCandidate, parse_whowhatwear

SITE_PARSERS = {
    "whowhatwear": parse_whowhatwear,
    "thezoereport": parse_thezoereport,
}

__all__ = ["SITE_PARSERS", "ScrapeCandidate"]
