"""Shared HTTP helpers for curation sourcing (page scraping + image search)."""

from __future__ import annotations

import re

import requests
from loguru import logger

IMAGE_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp)(?:\?|$)", re.IGNORECASE)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def download_image(
    session: requests.Session, url: str, *, timeout: int, min_bytes: int
) -> bytes | None:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "").lower()
        if "image" not in content_type and not IMAGE_EXT_RE.search(url.lower()):
            return None
        if len(resp.content) < min_bytes:
            return None
        return resp.content
    except Exception as e:
        logger.debug(f"download {url} failed: {e}")
        return None


def detect_ext(url: str) -> str:
    match = IMAGE_EXT_RE.search(url.lower())
    if match:
        return f".{match.group(1)}"
    return ".jpg"
