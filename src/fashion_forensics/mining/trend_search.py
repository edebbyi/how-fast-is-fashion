"""Find fresh article URLs for a trend using Claude's web search tool, so
curating a trend doesn't depend on a one-off manual search every time.

Constrained to a single site domain per call (`allowed_domains`), so results
are guaranteed compatible with that site's existing parser in
src/fashion_forensics/mining/site_parsers/ - this doesn't build a generic
"scrape any site" fallback, it just makes finding more URLs for a site we
can already parse faster than a manual search.

Needs ANTHROPIC_API_KEY in .env. Each call costs a small, real Claude API fee.
"""

from __future__ import annotations

import json
import re

import anthropic

import fashion_forensics.config  # noqa: F401 - side effect: loads .env (ANTHROPIC_API_KEY)
from fashion_forensics.mining.site_parsers import SITE_PARSERS

SEARCH_MODEL = "claude-opus-5"


def search_articles_for_trend(
    trend: str, site: str, site_domain: str, max_urls: int = 3
) -> list[str]:
    """Real article URLs on `site_domain` about `trend`, found via Claude's
    web search tool. Raises ValueError if `site` has no registered parser -
    a URL this project can't parse isn't useful here."""
    if site not in SITE_PARSERS:
        raise ValueError(f"No parser for site {site!r}. Options: {list(SITE_PARSERS)}")

    client = anthropic.Anthropic()
    trend_phrase = trend.replace("_", " ")
    search_budget = max(max_urls * 3, 6)
    prompt = (
        f"Find {max_urls} DIFFERENT real articles on {site_domain} with real outfit photos "
        f"(not shopping roundups) that fit the '{trend_phrase}' fashion trend or aesthetic.\n\n"
        f"Do not stop after one search, and do not limit yourself to articles that literally "
        f"say '{trend_phrase}' in the title - that exact phrase may only have one dedicated "
        f"feature on this site. Also search for the trend's stylistic COMPONENTS and related "
        f"angles: think about what specific garments, silhouettes, materials, or styling "
        f"details define '{trend_phrase}' (e.g. if it's about corporate-but-sexy dressing, "
        f"also try pencil skirts, sheer blouses, workwear; if it's about heritage minimalism, "
        f"also try tailoring, neutral palettes, quiet designer pieces), and search those terms "
        f"too. A same-aesthetic article that doesn't use the exact trend name is just as good "
        f"as one that does. Keep trying different queries until you have {max_urls} distinct "
        f"article URLs or you've genuinely exhausted what's findable on this site.\n\n"
        f"Respond with ONLY a JSON array of the distinct article URLs you found, nothing else "
        f'- e.g. ["https://...", "https://..."]. No markdown, no explanation, no extra text.'
    )

    response = client.messages.create(
        model=SEARCH_MODEL,
        max_tokens=1024,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "allowed_domains": [site_domain],
                "max_uses": search_budget,
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    urls = _extract_urls(text)
    return [u for u in urls if site_domain in u][:max_urls]


def _extract_urls(text: str) -> list[str]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [u for u in parsed if isinstance(u, str)]
    except json.JSONDecodeError:
        pass
    # Fall back to pulling anything URL-shaped out of the text, in case
    # Claude didn't return clean JSON.
    return re.findall(r'https?://[^\s"\'\]]+', text)
