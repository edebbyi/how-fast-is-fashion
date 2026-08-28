"""Scrape candidate trend-reference images from fashion editorial sites for
human review, ahead of adding them to the classifier's reference corpus.

Reads a curated list of article URLs per trend from configs/scraping.yaml
(not a live search or a guessed tag-page pattern - see that file's header
for why), downloads content images found in each article via a per-site
parser in src/fashion_forensics/mining/site_parsers/, dedups against
everything already scraped (candidates.jsonl, by exact MD5 and near-duplicate
perceptual hash) and everything already in the reference corpus (labeled/,
by perceptual hash), and writes the rest to
data/02_reference_corpus/raw/{trend}/ for human review in the Streamlit
"Curate" mode.

No LLM calls here - attribute normalization only happens after a human
approves a candidate, via the existing scripts/normalize_reference_corpus.py.

Usage:
    .venv/bin/python scripts/scrape_trend_images.py --site whowhatwear
    .venv/bin/python scripts/scrape_trend_images.py --site whowhatwear --trend quiet_luxury
    .venv/bin/python scripts/scrape_trend_images.py --site whowhatwear --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import UTC, datetime

import requests
from loguru import logger

from fashion_forensics.config import DATA_DIR, load_yaml_config
from fashion_forensics.curation import (
    compute_phash,
    find_duplicate,
    load_candidates,
    load_reference_phashes,
    save_candidates,
)
from fashion_forensics.mining.site_parsers import SITE_PARSERS

RAW_ROOT = DATA_DIR / "02_reference_corpus" / "raw"
REPO_ROOT = DATA_DIR.parent


def _detect_ext(url: str) -> str:
    lower = url.lower()
    if ".png" in lower:
        return ".png"
    if ".webp" in lower:
        return ".webp"
    return ".jpg"


def _download_image(
    session: requests.Session, url: str, timeout: int, min_bytes: int
) -> tuple[bytes | None, str]:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None, "http_error"
        content = resp.content
        content_type = resp.headers.get("Content-Type", "").lower()
        if "image" not in content_type:
            return None, "not_image"
        if len(content) < min_bytes:
            return None, "too_small"
        return content, "ok"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException:
        return None, "exception"


def scrape_article(
    session: requests.Session,
    config: dict,
    parser_fn,
    site: str,
    article_url: str,
    trend: str | None,
    max_images: int,
    candidates: dict[str, dict],
    existing_md5s: set[str],
    reference_phashes: dict[str, str],
    dry_run: bool,
) -> tuple[int, int, int]:
    """Fetch one article, download+dedup+write its candidate images.
    Mutates `candidates`/`existing_md5s` in place. Returns (n_new, n_duplicate,
    n_skipped) for this article. Shared by main() (config-driven articles) and
    the Claude-search flow (freshly-discovered articles) so both go through
    the same download/dedup/write path.
    """
    logger.info(f"Fetching {article_url}")
    try:
        scrape_candidates = list(
            parser_fn(session, article_url, trend, max_images, config["request_timeout_seconds"])
        )
    except requests.exceptions.RequestException as e:
        logger.warning(f"  failed to fetch article: {e}")
        return 0, 0, 0

    logger.info(f"  found {len(scrape_candidates)} candidate images")
    n_new, n_duplicate, n_skipped = 0, 0, 0

    for sc in scrape_candidates:
        if dry_run:
            logger.info(f"  [dry-run] {sc.image_url}")
            continue

        content = None
        reason = "unknown"
        for _ in range(config["retry_count"]):
            content, reason = _download_image(
                session, sc.image_url, config["request_timeout_seconds"], config["min_image_bytes"]
            )
            if content is not None:
                break
            time.sleep(1)
        if content is None:
            logger.warning(f"  skip {sc.image_url} ({reason})")
            n_skipped += 1
            continue

        md5 = hashlib.md5(content).hexdigest()
        if md5 in existing_md5s:
            logger.info(f"  skip (already scraped, identical bytes): {sc.image_url}")
            n_skipped += 1
            continue

        phash = compute_phash(content)
        dup_of = find_duplicate(phash, candidates, reference_phashes)

        ext = _detect_ext(sc.image_url)
        candidate_id = f"{site}_{datetime.now(UTC):%Y%m%d}_{phash[:10]}"
        dest_trend = sc.suggested_trend or "_unassigned"
        dest_dir = RAW_ROOT / dest_trend
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{candidate_id}{ext}"
        if dest_path.exists() or candidate_id in candidates:
            candidate_id = f"{candidate_id}-{md5[:6]}"
            dest_path = dest_dir / f"{candidate_id}{ext}"
        dest_path.write_bytes(content)

        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "source_site": site,
            "source_url": sc.source_url,
            "source_image_url": sc.image_url,
            "scraped_at": datetime.now(UTC).isoformat(),
            "suggested_trend": sc.suggested_trend,
            "suggestion_basis": sc.suggestion_basis,
            "phash": phash,
            "md5": md5,
            "local_path": str(dest_path.relative_to(REPO_ROOT)),
            "review_status": "duplicate" if dup_of else "pending",
            "reviewed_trend": None,
            "reject_reason": None,
            "promoted_filename": None,
            "promoted_at": None,
            "dup_of": dup_of,
        }
        existing_md5s.add(md5)
        if dup_of:
            n_duplicate += 1
        else:
            n_new += 1

        time.sleep(config["delay_between_requests_seconds"])

    return n_new, n_duplicate, n_skipped


def scrape_articles(
    site: str, trend: str | None, article_urls: list[str], max_images: int, dry_run: bool = False
) -> tuple[int, int, int]:
    """Scrape a specific list of article URLs (not config-driven) - used by
    the Claude-search flow for freshly-discovered articles. Returns
    (n_new, n_duplicate, n_skipped) totals across all given URLs."""
    config = load_yaml_config("scraping")
    parser_fn = SITE_PARSERS.get(site)
    if parser_fn is None:
        raise ValueError(f"No parser registered for site {site!r}. Options: {list(SITE_PARSERS)}")

    session = requests.Session()
    session.headers.update({"User-Agent": config["user_agent"]})

    candidates = load_candidates()
    reference_phashes = {} if dry_run else load_reference_phashes()
    existing_md5s = {r["md5"] for r in candidates.values() if r.get("md5")}

    totals = [0, 0, 0]
    for article_url in article_urls:
        result = scrape_article(
            session,
            config,
            parser_fn,
            site,
            article_url,
            trend,
            max_images,
            candidates,
            existing_md5s,
            reference_phashes,
            dry_run,
        )
        totals = [a + b for a, b in zip(totals, result)]

    if not dry_run:
        save_candidates(candidates)
    return tuple(totals)


def main(site: str, trend: str | None, max_images: int, dry_run: bool) -> None:
    config = load_yaml_config("scraping")
    site_config = config["sites"].get(site)
    if site_config is None:
        raise SystemExit(f"Unknown site {site!r}. Options: {list(config['sites'])}")

    parser_fn = SITE_PARSERS.get(site)
    if parser_fn is None:
        raise SystemExit(f"No parser registered for site {site!r}. Options: {list(SITE_PARSERS)}")

    articles_by_trend = site_config["articles"]
    trends_to_scrape = [trend] if trend else list(articles_by_trend.keys())

    session = requests.Session()
    session.headers.update({"User-Agent": config["user_agent"]})

    candidates = load_candidates()
    reference_phashes = {} if dry_run else load_reference_phashes()
    existing_md5s = {r["md5"] for r in candidates.values() if r.get("md5")}

    n_new, n_duplicate, n_skipped = 0, 0, 0

    for t in trends_to_scrape:
        for article_url in articles_by_trend.get(t, []):
            result = scrape_article(
                session,
                config,
                parser_fn,
                site,
                article_url,
                t,
                max_images,
                candidates,
                existing_md5s,
                reference_phashes,
                dry_run,
            )
            n_new += result[0]
            n_duplicate += result[1]
            n_skipped += result[2]

    if not dry_run:
        save_candidates(candidates)

    logger.info(
        f"Done. {n_new} new candidates, {n_duplicate} auto-flagged duplicates, "
        f"{n_skipped} skipped (download failures)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="Site to scrape (see configs/scraping.yaml)")
    parser.add_argument("--trend", default=None, help="Restrict to one trend's curated articles")
    parser.add_argument("--max-images", type=int, default=30, help="Max images per article")
    parser.add_argument("--dry-run", action="store_true", help="Log candidate URLs, don't download")
    args = parser.parse_args()
    main(site=args.site, trend=args.trend, max_images=args.max_images, dry_run=args.dry_run)
