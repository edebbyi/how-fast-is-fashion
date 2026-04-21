"""Wayback Machine miner with rich metadata extraction, stateful resumability,
and adaptive clean-target mining.

Ported from the working Colab notebook — preserves all metadata recovery logic
(script/JSON extraction, anchor-linked images, product code parsing) while
adapting to the project's config/logging structure.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from fashion_forensics.config import DATA_DIR, load_yaml_config

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
AUDIT_DIR = DATA_DIR / "01_data_audit"
RAW_IMAGES_DIR = AUDIT_DIR / "raw_images"
RECORDS_DIR = AUDIT_DIR / "records"
STATE_DIR = AUDIT_DIR / "state"
LOGS_DIR = AUDIT_DIR / "logs"
CLEANED_DIR = AUDIT_DIR / "cleaned"

for _d in (RAW_IMAGES_DIR, RECORDS_DIR, STATE_DIR, LOGS_DIR, CLEANED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_configured = False


def _configure_logging() -> Path:
    global _log_configured
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_{run_ts}.log"

    if not _log_configured:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}",
            level="INFO",
        )
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
        )
        _log_configured = True

    return log_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MiningConfig:
    """Parsed mining configuration from YAML."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.target_urls: list[str] = raw.get("target_urls", [raw.get("target_url", "")])
        self.start_date: str = raw["start_date"]
        self.end_date: str = raw["end_date"]
        self.target_clean_per_month: int = raw.get("target_clean_per_month", 60)
        self.min_image_bytes: int = raw.get("min_image_bytes", 12000)
        self.sampling_strategy: str = raw.get("sampling_strategy", "every_month")
        self.random_month_count: int = raw.get("random_month_count", 6)
        self.request_timeout: int = raw.get("request_timeout_seconds", 8)
        self.retry_count: int = raw.get("retry_count", 3)
        self.delay_between_months: int = raw.get("delay_between_months_seconds", 2)
        self.max_snapshots_per_url: int = raw.get("max_snapshots_per_url", 100)
        self.good_url_signals: list[str] = raw.get("good_url_signals", [".jpg", ".jpeg", ".png"])
        self.bad_url_signals: list[str] = raw.get("bad_url_signals", ["logo", "icon", ".svg"])


# ---------------------------------------------------------------------------
# Month state
# ---------------------------------------------------------------------------


class MonthState:
    """Persistent per-month state for resumable mining."""

    def __init__(self, month_str: str) -> None:
        self.month_str = month_str
        self.path = STATE_DIR / f"{month_str[:4]}-{month_str[4:]}_state.json"
        self.attempted_urls: set[str] = set()
        self.saved_hashes: set[str] = set()
        self.used_snapshots: set[str] = set()
        self.saved_records: int = 0
        self.clean_records: int = 0
        self.estimated_keep_rate: float = 0.5
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.attempted_urls = set(data.get("attempted_candidate_urls", []))
            self.saved_hashes = set(data.get("saved_hashes", []))
            self.used_snapshots = set(data.get("used_snapshots", []))
            self.saved_records = data.get("saved_records", 0)
            self.clean_records = data.get("clean_records", 0)
            self.estimated_keep_rate = data.get("estimated_keep_rate", 0.5)

    def save(self) -> None:
        data = {
            "month": self.month_str,
            "attempted_candidate_urls": sorted(self.attempted_urls),
            "saved_hashes": sorted(self.saved_hashes),
            "used_snapshots": sorted(self.used_snapshots),
            "saved_records": self.saved_records,
            "clean_records": self.clean_records,
            "estimated_keep_rate": self.estimated_keep_rate,
            "updated_at": datetime.now().isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def update_keep_rate(self, raw_added: int, clean_added: int) -> None:
        if raw_added == 0:
            return
        observed = clean_added / raw_added
        self.estimated_keep_rate = 0.7 * self.estimated_keep_rate + 0.3 * observed
        self.estimated_keep_rate = max(0.15, min(0.9, self.estimated_keep_rate))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def is_valid_product_record(record: dict[str, Any]) -> bool:
    """Domain-specific cleaning rules for Zara product records."""
    product_code = record.get("product_code")
    product_name = record.get("product_name")
    color = record.get("color")
    raw_text = record.get("raw_card_text") or ""

    # Bundle product codes
    if product_code and str(product_code).startswith("T"):
        return False

    if not product_name:
        return False

    if str(product_name).strip().upper() == "LOOK":
        return False

    if "bundleProducts" in raw_text:
        return False

    # Numeric color junk
    if isinstance(color, str) and color.isdigit():
        return False

    # No real product signal
    if not any(
        [
            record.get("product_name"),
            record.get("price"),
            record.get("archive_product_url"),
        ]
    ):
        return False

    return True


def clean_new_raw_records(month_str: str) -> dict[str, int]:
    """Evaluate only new raw records and append valid ones to clean dataset."""
    folder = f"{month_str[:4]}-{month_str[4:]}"
    raw_path = RECORDS_DIR / f"{folder}_records.jsonl"
    clean_path = CLEANED_DIR / f"{folder}_clean.jsonl"

    if not raw_path.exists():
        return {
            "raw_total": 0,
            "clean_existing": 0,
            "newly_evaluated": 0,
            "newly_kept": 0,
            "removed": 0,
        }

    raw_rows = [json.loads(line) for line in raw_path.read_text().strip().split("\n") if line]

    existing_clean_ids: set[str] = set()
    if clean_path.exists():
        for line in clean_path.read_text().strip().split("\n"):
            if line:
                existing_clean_ids.add(json.loads(line)["record_id"])

    new_rows = [r for r in raw_rows if r["record_id"] not in existing_clean_ids]

    kept: list[dict[str, Any]] = []
    removed = 0

    for record in new_rows:
        if is_valid_product_record(record):
            record["metadata_quality"] = "clean"
            kept.append(record)
        else:
            removed += 1

    if kept:
        with open(clean_path, "a") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "raw_total": len(raw_rows),
        "clean_existing": len(existing_clean_ids),
        "newly_evaluated": len(new_rows),
        "newly_kept": len(kept),
        "removed": removed,
    }


def get_clean_count(month_str: str) -> int:
    folder = f"{month_str[:4]}-{month_str[4:]}"
    clean_path = CLEANED_DIR / f"{folder}_clean.jsonl"
    if not clean_path.exists():
        return 0
    return sum(1 for line in clean_path.read_text().strip().split("\n") if line)


# ---------------------------------------------------------------------------
# Main miner
# ---------------------------------------------------------------------------


class FashionMiner:
    """Mines archived retail pages for product images and metadata.

    Architecture:
        raw_images/   — archived image files
        records/      — raw JSONL metadata, append-only
        state/        — per-month machine memory
        logs/         — run logs
        cleaned/      — filtered JSONL, only valid product records
    """

    CDX_API = "https://web.archive.org/cdx/search/cdx"

    def __init__(self, config: MiningConfig | None = None) -> None:
        if config is None:
            config = MiningConfig(load_yaml_config("mining"))
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        self.log_path = _configure_logging()
        logger.info(
            f"Miner initialized | urls={len(config.target_urls)} | "
            f"range={config.start_date}-{config.end_date} | "
            f"strategy={config.sampling_strategy} | "
            f"target_clean={config.target_clean_per_month}/month"
        )

    # ------------------------------------------------------------------
    # Month generation
    # ------------------------------------------------------------------

    def generate_months(self) -> list[str]:
        start = datetime.strptime(self.config.start_date, "%Y%m")
        end = datetime.strptime(self.config.end_date, "%Y%m")
        current = start
        all_months: list[str] = []

        while current <= end:
            all_months.append(current.strftime("%Y%m"))
            current += timedelta(days=32)
            current = current.replace(day=1)

        if self.config.sampling_strategy == "quarterly":
            return [m for m in all_months if int(m[4:6]) in (1, 4, 7, 10)]
        if self.config.sampling_strategy == "random_months":
            k = min(self.config.random_month_count, len(all_months))
            return sorted(random.sample(all_months, k))
        return all_months

    # ------------------------------------------------------------------
    # Snapshot lookup (returns ALL snapshots for a month, not just one)
    # ------------------------------------------------------------------

    def get_month_snapshots(self, target_url: str, month_str: str) -> list[str]:
        """Return all available snapshots for a URL in a given month."""
        last_day = monthrange(int(month_str[:4]), int(month_str[4:6]))[1]
        params = {
            "url": target_url,
            "from": f"{month_str}01",
            "to": f"{month_str}{last_day}",
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype",
            "filter": "statuscode:200",
            "collapse": "digest",
            "limit": self.config.max_snapshots_per_url,
        }

        for attempt in range(self.config.retry_count):
            try:
                resp = self.session.get(self.CDX_API, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                if len(data) <= 1:
                    return []

                return [f"https://web.archive.org/web/{row[0]}/{row[1]}" for row in data[1:]]
            except Exception as e:
                logger.warning(f"CDX {month_str} attempt {attempt + 1}: {e}")
                time.sleep(2)

        return []

    # ------------------------------------------------------------------
    # HTML parsing and candidate collection
    # ------------------------------------------------------------------

    def _fetch_soup(self, archive_url: str) -> BeautifulSoup | None:
        try:
            resp = self.session.get(archive_url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return None

    def collect_candidate_records(
        self, soup: BeautifulSoup, archive_url: str, target_url: str
    ) -> list[dict[str, Any]]:
        """Extract candidate product records from HTML + script content."""
        candidates_by_url: dict[str, dict[str, Any]] = {}

        def add(raw_url: str, **meta: Any) -> None:
            img_url = self._normalize_url(raw_url, archive_url, target_url)
            if not img_url or not self._looks_like_product(img_url):
                return

            candidate = {
                "archive_image_url": img_url,
                "product_code": meta.get("product_code") or self._extract_product_code(img_url),
                **{
                    k: meta.get(k)
                    for k in (
                        "archive_product_url",
                        "product_name",
                        "price",
                        "currency",
                        "raw_card_text",
                        "description",
                        "composition",
                        "category",
                        "color",
                        "metadata_source",
                    )
                },
            }

            existing = candidates_by_url.get(img_url)
            if existing is None or self._metadata_score(candidate) > self._metadata_score(existing):
                candidates_by_url[img_url] = candidate

        # --- img tags ---
        for img in soup.find_all("img"):
            parent = img.parent
            parent_text = " ".join(parent.stripped_strings) if parent else ""
            alt = img.get("alt", "").strip() or None

            for attr in (
                "src",
                "data-src",
                "data-original",
                "data-image",
                "data-lazy",
                "data-srcset",
                "srcset",
            ):
                val = img.get(attr)
                if val:
                    add(
                        val,
                        product_name=alt,
                        raw_card_text=parent_text or None,
                        category=self._derive_category(parent_text),
                        color=self._derive_color(parent_text),
                        metadata_source=f"img_{attr}",
                    )

        # --- source tags ---
        for source in soup.find_all("source"):
            parent_text = " ".join(source.parent.stripped_strings) if source.parent else ""
            for attr in ("srcset", "data-srcset", "src"):
                val = source.get(attr)
                if val:
                    add(
                        val,
                        raw_card_text=parent_text or None,
                        category=self._derive_category(parent_text),
                        color=self._derive_color(parent_text),
                        metadata_source=f"source_{attr}",
                    )

        # --- anchor-linked images ---
        for a in soup.find_all("a", href=True):
            link_text = " ".join(a.stripped_strings).strip()
            product_url = urljoin(target_url, a["href"]) if a.get("href") else None

            for img in a.find_all("img"):
                alt = img.get("alt", "").strip() or None
                for attr in (
                    "src",
                    "data-src",
                    "data-original",
                    "data-image",
                    "data-lazy",
                    "data-srcset",
                    "srcset",
                ):
                    val = img.get(attr)
                    if val:
                        add(
                            val,
                            product_name=alt or (link_text or None),
                            raw_card_text=link_text or None,
                            archive_product_url=product_url,
                            category=self._derive_category(link_text),
                            color=self._derive_color(link_text),
                            metadata_source=f"anchor_img_{attr}",
                        )

        # --- generic lazy-load attributes ---
        for tag in soup.find_all(True):
            tag_text = " ".join(tag.stripped_strings)
            for attr in ("data-src", "data-srcset", "data-background", "data-image"):
                val = tag.get(attr)
                if val:
                    add(
                        val,
                        raw_card_text=tag_text or None,
                        category=self._derive_category(tag_text),
                        color=self._derive_color(tag_text),
                        metadata_source=f"generic_{attr}",
                    )

        # --- script/JSON extraction ---
        for c in self._extract_json_candidates(soup, archive_url, target_url):
            img_url = c["archive_image_url"]
            existing = candidates_by_url.get(img_url)
            if existing is None or self._metadata_score(c) > self._metadata_score(existing):
                candidates_by_url[img_url] = c

        return list(candidates_by_url.values())

    # ------------------------------------------------------------------
    # Script / JSON metadata extraction
    # ------------------------------------------------------------------

    def _extract_json_candidates(
        self, soup: BeautifulSoup, archive_url: str, target_url: str
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for script in soup.find_all("script"):
            text = script.string or script.get_text()
            if not text:
                continue

            # Find image URLs in script text
            img_matches = re.findall(
                r'https?://[^"\']*zara[^"\']+\.(?:jpg|jpeg|png|webp)'
                r"|//[^\"']*zara[^\"']+\.(?:jpg|jpeg|png|webp)",
                text,
                flags=re.IGNORECASE,
            )

            # Parse any embedded JSON
            parsed_objects: list[Any] = []
            for pattern in (
                r"window\.__INITIAL_STATE__\s*=\s*({.*?});",
                r"window\.__NEXT_DATA__\s*=\s*({.*?});",
                r"=\s*({.*?});",
            ):
                for m in re.finditer(pattern, text, flags=re.DOTALL):
                    blob = m.group(1).strip()
                    if len(blob) > 20:
                        try:
                            parsed_objects.append(json.loads(blob))
                        except Exception:
                            pass

            for raw_url in img_matches:
                img_url = self._normalize_url(raw_url, archive_url, target_url)
                if not img_url or not self._looks_like_product(img_url):
                    continue

                product_code = self._extract_product_code(img_url)
                meta = self._extract_meta_from_json(parsed_objects, product_code, target_url)
                if meta is None:
                    meta = self._extract_meta_from_text(text, product_code, target_url)

                source = (
                    "script_json_exact"
                    if any(meta.get(k) for k in ("product_name", "archive_product_url", "price"))
                    else "script_json_fallback"
                )

                candidates.append({"archive_image_url": img_url, "metadata_source": source, **meta})

        return candidates

    def _extract_meta_from_json(
        self, parsed_objects: list[Any], product_code: str | None, target_url: str
    ) -> dict[str, Any] | None:
        """Walk parsed JSON objects looking for product metadata matching product_code."""
        if not product_code or not parsed_objects:
            return None

        best_obj: dict[str, Any] | None = None
        best_score = -1

        for parsed in parsed_objects:
            for obj in self._walk_objects(parsed):
                if not isinstance(obj, dict):
                    continue
                if not self._obj_contains_code(obj, product_code):
                    continue

                score = 0
                if self._deep_get(obj, "name", "seoKeyword", "title"):
                    score += 3
                if self._deep_get(obj, "url", "link"):
                    score += 2
                if self._extract_price(obj):
                    score += 2
                if self._extract_color_from_obj(obj):
                    score += 1

                if score > best_score:
                    best_obj = obj
                    best_score = score

        if best_obj is None:
            return None

        name = self._deep_get(best_obj, "name", "seoKeyword", "title")
        description = self._deep_get(best_obj, "description", "detail")
        raw_url = self._deep_get(best_obj, "url", "link")

        combined = " ".join(filter(None, [str(name or ""), str(description or "")]))

        if isinstance(name, str) and name.strip().upper() == "LOOK":
            name = None

        return {
            "product_code": product_code,
            "product_name": name,
            "price": self._extract_price(best_obj),
            "currency": "USD" if self._extract_price(best_obj) else None,
            "archive_product_url": self._normalize_product_url(raw_url, target_url),
            "raw_card_text": json.dumps(best_obj, ensure_ascii=False)[:1000],
            "description": description if isinstance(description, str) else None,
            "composition": None,
            "category": self._derive_category(combined),
            "color": self._extract_color_from_obj(best_obj) or self._derive_color(combined),
        }

    def _extract_meta_from_text(
        self, script_text: str, product_code: str | None, target_url: str
    ) -> dict[str, Any]:
        """Regex-based fallback metadata extraction from script text."""
        meta: dict[str, Any] = {
            "product_code": product_code,
            "product_name": None,
            "price": None,
            "currency": None,
            "archive_product_url": None,
            "raw_card_text": None,
            "description": None,
            "composition": None,
            "category": None,
            "color": None,
        }

        search_text = script_text
        if product_code:
            idx = script_text.find(product_code)
            if idx != -1:
                search_text = script_text[max(0, idx - 1200) : min(len(script_text), idx + 1800)]

        compact = re.sub(r"\s+", " ", search_text).strip()
        meta["raw_card_text"] = compact[:1000] if compact else None

        for pattern in (
            r'"name"\s*:\s*"([^"]+)"',
            r'"seoKeyword"\s*:\s*"([^"]+)"',
            r'"title"\s*:\s*"([^"]+)"',
        ):
            matches = re.findall(pattern, search_text, flags=re.IGNORECASE)
            cleaned = [
                m.strip()
                for m in matches
                if m.strip()
                and m.strip().upper() != "LOOK"
                and "http" not in m.lower()
                and not any(m.lower().endswith(ext) for ext in (".jpg", ".png", ".webp"))
            ]
            if cleaned:
                meta["product_name"] = cleaned[0]
                break

        desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', search_text, flags=re.IGNORECASE)
        if desc_match:
            meta["description"] = desc_match.group(1).strip()

        url_match = re.search(r'"url"\s*:\s*"([^"]+)"', search_text, flags=re.IGNORECASE)
        if url_match:
            meta["archive_product_url"] = self._normalize_product_url(
                url_match.group(1), target_url
            )

        combined = " ".join(filter(None, [meta["product_name"], meta["description"]]))
        meta["category"] = self._derive_category(combined)
        meta["color"] = self._derive_color(combined)

        return meta

    # ------------------------------------------------------------------
    # Raw mining batch (one pass per month)
    # ------------------------------------------------------------------

    def mine_raw_batch(self, month_str: str, raw_batch_target: int) -> int:
        """Recover up to raw_batch_target new raw records for a month."""
        folder = f"{month_str[:4]}-{month_str[4:]}"
        image_dir = RAW_IMAGES_DIR / folder
        image_dir.mkdir(parents=True, exist_ok=True)

        state = MonthState(month_str)
        saved_hashes = self._rebuild_hashes(image_dir)
        state.saved_hashes = saved_hashes

        file_index = sum(
            1 for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        raw_added = 0

        logger.info(f"{folder}: requesting {raw_batch_target} raw records")

        for target_url in self.config.target_urls:
            if raw_added >= raw_batch_target:
                break

            snapshots = self.get_month_snapshots(target_url, month_str)
            if not snapshots:
                logger.debug(f"{folder}: no snapshots for {target_url}")
                continue

            logger.info(f"{folder}: {len(snapshots)} snapshots for {target_url}")

            for archive_url in snapshots:
                if raw_added >= raw_batch_target:
                    break
                if archive_url in state.used_snapshots:
                    continue

                soup = self._fetch_soup(archive_url)
                if soup is None:
                    continue

                candidates = self.collect_candidate_records(soup, archive_url, target_url)
                fresh = [
                    c for c in candidates if c["archive_image_url"] not in state.attempted_urls
                ]

                logger.info(f"{folder}: {len(fresh)} untried candidates from {archive_url[:80]}")

                saved_from_snapshot = 0
                for candidate in fresh:
                    if raw_added >= raw_batch_target:
                        break

                    img_url = candidate["archive_image_url"]
                    state.attempted_urls.add(img_url)

                    content, reason = self._download_image(img_url)
                    if content is None:
                        continue

                    img_hash = hashlib.md5(content).hexdigest()
                    if img_hash in state.saved_hashes:
                        continue

                    ext = self._detect_ext(img_url)
                    filename = f"item_{file_index:03d}{ext}"
                    (image_dir / filename).write_bytes(content)

                    record = {
                        "record_id": f"{folder}_{file_index:03d}",
                        "month": folder,
                        "image_filename": filename,
                        "source_target_url": target_url,
                        "archive_snapshot_url": archive_url,
                        "archive_image_url": img_url,
                        "archive_product_url": candidate.get("archive_product_url"),
                        "product_code": candidate.get("product_code"),
                        "product_name": candidate.get("product_name"),
                        "price": candidate.get("price"),
                        "currency": candidate.get("currency"),
                        "raw_card_text": candidate.get("raw_card_text"),
                        "description": candidate.get("description"),
                        "composition": candidate.get("composition"),
                        "category": candidate.get("category"),
                        "color": candidate.get("color"),
                        "metadata_source": candidate.get("metadata_source"),
                        "metadata_quality": "raw",
                    }

                    self._append_record(month_str, record)
                    state.saved_hashes.add(img_hash)
                    file_index += 1
                    raw_added += 1
                    saved_from_snapshot += 1

                if saved_from_snapshot > 0:
                    state.used_snapshots.add(archive_url)

                # Save state after each snapshot
                state.saved_records += saved_from_snapshot
                state.save()

        logger.info(f"{folder}: raw batch complete, +{raw_added} records")
        return raw_added

    # ------------------------------------------------------------------
    # Adaptive build-to-clean-target
    # ------------------------------------------------------------------

    def build_month_to_target(self, month_str: str) -> None:
        """Mine and clean incrementally until clean target is reached or supply exhausted."""
        folder = f"{month_str[:4]}-{month_str[4:]}"
        state = MonthState(month_str)
        target = self.config.target_clean_per_month

        while True:
            clean_count = get_clean_count(month_str)

            if clean_count >= target:
                logger.info(f"{folder}: target achieved ({clean_count}/{target} clean)")
                state.clean_records = clean_count
                state.save()
                break

            clean_gap = target - clean_count
            raw_batch_target = math.ceil(clean_gap / state.estimated_keep_rate)

            logger.info(
                f"{folder}: clean={clean_count}/{target} gap={clean_gap} "
                f"keep_rate={state.estimated_keep_rate:.2f} -> raw_batch={raw_batch_target}"
            )

            raw_added = self.mine_raw_batch(month_str, raw_batch_target)
            clean_result = clean_new_raw_records(month_str)
            clean_added = clean_result["newly_kept"]

            logger.info(
                f"{folder}: +{raw_added} raw, +{clean_added} clean, "
                f"{clean_result['removed']} removed during cleaning"
            )

            if raw_added == 0:
                logger.info(f"{folder}: no more raw records available, stopping")
                break

            state.update_keep_rate(raw_added, clean_added)
            state.clean_records = get_clean_count(month_str)
            state.save()

            if clean_added == 0:
                logger.info(f"{folder}: batch produced no clean records, stopping")
                break

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, int]:
        """Execute the full mining pipeline. Returns {month: clean_count}."""
        months = self.generate_months()
        logger.info(f"Mining {len(months)} months from {len(self.config.target_urls)} target URLs")

        run_log: dict[str, Any] = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "config": {
                "target_urls": self.config.target_urls,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "target_clean_per_month": self.config.target_clean_per_month,
                "sampling_strategy": self.config.sampling_strategy,
            },
            "months": {},
        }

        results: dict[str, int] = {}
        for i, m in enumerate(months, 1):
            logger.info(f"[{i}/{len(months)}] Processing {m}...")
            self.build_month_to_target(m)

            state = MonthState(m)
            clean = get_clean_count(m)
            results[m] = clean

            run_log["months"][m] = {
                "raw_total": state.saved_records,
                "clean_total": clean,
                "target": self.config.target_clean_per_month,
                "keep_rate": round(state.estimated_keep_rate, 3),
            }

            time.sleep(self.config.delay_between_months)

        # Write run log
        run_log["finished_at"] = datetime.now().isoformat()
        total_clean = sum(results.values())
        run_log["summary"] = {
            "total_months": len(months),
            "months_at_target": sum(
                1 for v in results.values() if v >= self.config.target_clean_per_month
            ),
            "total_clean_records": total_clean,
        }

        log_path = LOGS_DIR / f"run_{run_log['run_id']}.json"
        log_path.write_text(json.dumps(run_log, indent=2))
        logger.info(f"Run log saved: {log_path}")
        logger.info(f"Run complete: {total_clean} clean records across {len(months)} months")

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _download_image(self, img_url: str) -> tuple[bytes | None, str]:
        try:
            resp = self.session.get(img_url, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return None, "http_error"

            content = resp.content
            content_type = resp.headers.get("Content-Type", "").lower()

            if "image" not in content_type and not self._has_image_ext(img_url):
                return None, "not_image"
            if len(content) < self.config.min_image_bytes:
                return None, "too_small"

            return content, "ok"
        except requests.exceptions.Timeout:
            return None, "timeout"
        except Exception:
            return None, "exception"

    def _append_record(self, month_str: str, record: dict[str, Any]) -> None:
        folder = f"{month_str[:4]}-{month_str[4:]}"
        path = RECORDS_DIR / f"{folder}_records.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rebuild_hashes(self, folder: Path) -> set[str]:
        hashes: set[str] = set()
        if not folder.exists():
            return hashes
        for f in folder.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
        return hashes

    def _normalize_url(self, raw: str, archive_url: str, target_url: str) -> str | None:
        raw = raw.strip()
        if not raw or raw.startswith("data:"):
            return None
        if "," in raw:
            raw = raw.split(",")[0].strip()
        raw = raw.split(" ")[0].strip()
        if not raw:
            return None
        if "web.archive.org/web/" in raw:
            return raw
        if raw.startswith("//"):
            raw = "https:" + raw
        elif raw.startswith("/"):
            raw = urljoin(target_url, raw)
        elif not raw.startswith("http"):
            raw = urljoin(target_url, raw)

        ts = self._extract_timestamp(archive_url)
        if not ts:
            return None
        return f"https://web.archive.org/web/{ts}im_/{raw}"

    def _extract_timestamp(self, archive_url: str) -> str | None:
        match = re.search(r"/web/(\d{14})", archive_url)
        return match.group(1) if match else None

    def _looks_like_product(self, url: str) -> bool:
        lower = url.lower()
        has_good = any(s in lower for s in self.config.good_url_signals)
        has_bad = any(s in lower for s in self.config.bad_url_signals)
        return has_good and not has_bad

    def _has_image_ext(self, url: str) -> bool:
        return any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp"))

    def _detect_ext(self, url: str) -> str:
        lower = url.lower()
        if ".png" in lower:
            return ".png"
        if ".webp" in lower:
            return ".webp"
        return ".jpg"

    def _extract_product_code(self, img_url: str) -> str | None:
        if not img_url:
            return None
        fname = img_url.split("/")[-1].split("?")[0]
        for pattern in (
            r"([A-Z]?\d{9,12})-\d{2,3}-p\.(?:jpg|jpeg|png|webp)$",
            r"([A-Z]?\d{9,12})-p\.(?:jpg|jpeg|png|webp)$",
            r"([A-Z]?\d{9,12})",
        ):
            m = re.search(pattern, fname, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _metadata_score(self, candidate: dict[str, Any]) -> int:
        score = 0
        weights = {
            "product_name": 4,
            "description": 3,
            "price": 3,
            "archive_product_url": 2,
            "category": 2,
            "color": 2,
            "product_code": 1,
            "raw_card_text": 1,
        }
        for field, weight in weights.items():
            if candidate.get(field):
                score += weight
        return score

    def _normalize_product_url(self, url: str | None, target_url: str) -> str | None:
        if not url or not isinstance(url, str):
            return None
        if url.startswith("http"):
            return url
        if url.startswith("/"):
            return urljoin(target_url, url)
        return None

    def _derive_category(self, text: str) -> str | None:
        if not text:
            return None
        lower = text.lower()
        categories = [
            "dress",
            "jacket",
            "coat",
            "blazer",
            "shirt",
            "top",
            "t-shirt",
            "skirt",
            "pants",
            "trousers",
            "jeans",
            "shorts",
            "sweater",
            "cardigan",
            "hoodie",
            "bodysuit",
            "vest",
            "bag",
            "shoe",
            "sandals",
            "tote",
            "boots",
        ]
        for cat in categories:
            if cat in lower:
                return cat
        return None

    def _derive_color(self, text: str) -> str | None:
        if not text:
            return None
        lower = text.lower()
        colors = [
            "black",
            "white",
            "ecru",
            "beige",
            "brown",
            "blue",
            "navy",
            "red",
            "green",
            "khaki",
            "grey",
            "gray",
            "pink",
            "burgundy",
            "cream",
            "ivory",
            "yellow",
            "orange",
            "silver",
            "gold",
        ]
        for color in colors:
            if color in lower:
                return color
        return None

    def _extract_price(self, obj: dict[str, Any]) -> str | None:
        candidates = [obj.get("price"), obj.get("priceAmount"), obj.get("amount")]
        detail = obj.get("detail")
        if isinstance(detail, dict):
            candidates.extend(
                [detail.get("price"), detail.get("priceAmount"), detail.get("amount")]
            )

        for val in candidates:
            if isinstance(val, (int, float)):
                return f"${val / 100:.2f}" if val > 999 else f"${val:.2f}"
            if isinstance(val, str):
                stripped = val.strip()
                if re.fullmatch(r"\d{4,6}", stripped):
                    return f"${int(stripped) / 100:.2f}"
                if re.fullmatch(r"\$?\d+(?:\.\d{2})?", stripped):
                    return stripped if stripped.startswith("$") else f"${float(stripped):.2f}"
        return None

    def _extract_color_from_obj(self, obj: dict[str, Any]) -> str | None:
        for key in ("colors",):
            colors_obj = obj.get(key)
            if isinstance(colors_obj, list):
                for item in colors_obj:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("colorName")
                        if isinstance(name, str):
                            return name
        detail = obj.get("detail")
        if isinstance(detail, dict):
            return self._extract_color_from_obj(detail)
        return None

    def _deep_get(self, obj: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in obj and obj[key] not in (None, "", [], {}):
                return obj[key]
        for value in obj.values():
            if isinstance(value, dict):
                result = self._deep_get(value, *keys)
                if result not in (None, "", [], {}):
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = self._deep_get(item, *keys)
                        if result not in (None, "", [], {}):
                            return result
        return None

    def _obj_contains_code(self, obj: dict[str, Any], code: str) -> bool:
        for field in (
            "reference",
            "productReference",
            "displayReference",
            "id",
            "seoProductId",
            "productId",
        ):
            val = obj.get(field)
            if isinstance(val, str) and code == val:
                return True
            if isinstance(val, int) and code == str(val):
                return True
        return code in json.dumps(obj, ensure_ascii=False)

    def _walk_objects(self, obj: Any):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from self._walk_objects(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from self._walk_objects(item)
