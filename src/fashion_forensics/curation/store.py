"""Review-queue persistence and approve/reject decisions for trend curation.

Layout:
    data/04_curation/staging/{candidate_id}.{ext}   — downloaded, unreviewed images
    data/04_curation/queue.json                     — {candidate_id: record}, rewritten atomically
    data/02_reference_corpus/labeled/{trend}/        — approved images land here (existing corpus)
    data/02_reference_corpus/attributions.jsonl      — append-only provenance log for approvals

A candidate's `id` is the first 16 hex chars of the md5 of its image bytes,
which doubles as content-based dedup: the same image scraped twice (even
from different pages) collides on `id` and is skipped.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fashion_forensics.config import DATA_DIR, PROJECT_ROOT

CURATION_DIR = DATA_DIR / "04_curation"
STAGING_DIR = CURATION_DIR / "staging"
QUEUE_PATH = CURATION_DIR / "queue.json"
LABELED_DIR = DATA_DIR / "02_reference_corpus" / "labeled"
ATTRIBUTIONS_PATH = DATA_DIR / "02_reference_corpus" / "attributions.jsonl"
TREND_RULES_PATH = PROJECT_ROOT / "configs" / "trend_rules.yaml"

STAGING_DIR.mkdir(parents=True, exist_ok=True)


def image_id(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()[:16]


def load_queue() -> dict[str, dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return {}
    return json.loads(QUEUE_PATH.read_text())


def _save_queue(queue: dict[str, dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=QUEUE_PATH.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(queue, f, indent=2, sort_keys=True)
        Path(tmp_name).replace(QUEUE_PATH)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def known_hashes() -> set[str]:
    """Candidate ids already seen (any status) — used to skip re-downloading duplicates."""
    return set(load_queue().keys())


def add_candidate(
    image_bytes: bytes,
    ext: str,
    *,
    source_name: str,
    source_url: str,
    image_url: str,
    alt_text: str | None = None,
    trend_hint: str | None = None,
) -> dict[str, Any] | None:
    """Stage a scraped image and add it to the review queue.

    Returns the new record, or None if this exact image is already known
    (pending, approved, or rejected).
    """
    candidate_id = image_id(image_bytes)
    queue = load_queue()
    if candidate_id in queue:
        return None

    filename = f"{candidate_id}{ext}"
    (STAGING_DIR / filename).write_bytes(image_bytes)

    record = {
        "id": candidate_id,
        "filename": filename,
        "source_name": source_name,
        "source_url": source_url,
        "image_url": image_url,
        "alt_text": alt_text,
        "trend_hint": trend_hint,
        "scraped_at": datetime.now().isoformat(),
        "status": "pending",
        "decided_at": None,
        "decided_trend": None,
        "reviewer": None,
        "reject_reason": None,
    }
    queue[candidate_id] = record
    _save_queue(queue)
    return record


def list_candidates(
    status: str | None = "pending", trend_hint: str | None = None
) -> list[dict[str, Any]]:
    queue = load_queue()
    records = list(queue.values())
    if status is not None:
        records = [r for r in records if r["status"] == status]
    if trend_hint is not None:
        records = [r for r in records if r.get("trend_hint") == trend_hint]
    return sorted(records, key=lambda r: r["scraped_at"], reverse=True)


def get_candidate(candidate_id: str) -> dict[str, Any] | None:
    return load_queue().get(candidate_id)


def image_path(candidate_id: str) -> Path | None:
    """Resolve a candidate's image file on disk, wherever its current status put it."""
    record = get_candidate(candidate_id)
    if record is None:
        return None
    if record["status"] == "approved" and record.get("decided_trend"):
        path = LABELED_DIR / record["decided_trend"] / record["filename"]
    else:
        path = STAGING_DIR / record["filename"]
    return path if path.exists() else None


def approve(candidate_id: str, trend: str, *, reviewer: str | None = None) -> dict[str, Any]:
    queue = load_queue()
    record = queue.get(candidate_id)
    if record is None:
        raise KeyError(candidate_id)
    if record["status"] != "pending":
        raise ValueError(f"candidate {candidate_id} is already {record['status']}")

    src_path = STAGING_DIR / record["filename"]
    if not src_path.exists():
        raise FileNotFoundError(f"staged image missing: {src_path}")

    dest_dir = LABELED_DIR / trend
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / record["filename"]
    shutil.move(str(src_path), str(dest_path))

    now = datetime.now().isoformat()
    record.update(
        status="approved",
        decided_at=now,
        decided_trend=trend,
        reviewer=reviewer,
    )
    queue[candidate_id] = record
    _save_queue(queue)

    try:
        image_path_str = str(dest_path.relative_to(PROJECT_ROOT))
    except ValueError:
        image_path_str = str(dest_path)

    ATTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTRIBUTIONS_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "filename": record["filename"],
                    "trend": trend,
                    "image_path": image_path_str,
                    "source_name": record["source_name"],
                    "source_url": record["source_url"],
                    "image_url": record["image_url"],
                    "alt_text": record.get("alt_text"),
                    "trend_hint": record.get("trend_hint"),
                    "scraped_at": record["scraped_at"],
                    "approved_at": now,
                    "reviewer": reviewer,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    return record


def reject(
    candidate_id: str, *, reviewer: str | None = None, reason: str | None = None
) -> dict[str, Any]:
    queue = load_queue()
    record = queue.get(candidate_id)
    if record is None:
        raise KeyError(candidate_id)
    if record["status"] != "pending":
        raise ValueError(f"candidate {candidate_id} is already {record['status']}")

    src_path = STAGING_DIR / record["filename"]
    src_path.unlink(missing_ok=True)

    record.update(
        status="rejected",
        decided_at=datetime.now().isoformat(),
        reviewer=reviewer,
        reject_reason=reason,
    )
    queue[candidate_id] = record
    _save_queue(queue)
    return record


def stats() -> dict[str, Any]:
    queue = load_queue()
    by_status: dict[str, int] = {}
    by_trend: dict[str, int] = {}
    for record in queue.values():
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
        if record["status"] == "approved" and record.get("decided_trend"):
            by_trend[record["decided_trend"]] = by_trend.get(record["decided_trend"], 0) + 1
    return {
        "pending": by_status.get("pending", 0),
        "approved": by_status.get("approved", 0),
        "rejected": by_status.get("rejected", 0),
        "approved_by_trend": by_trend,
    }


def list_known_trends() -> list[str]:
    """Trend names for the review UI: taxonomy (active + deferred) union existing labeled dirs."""
    names: set[str] = set()

    if TREND_RULES_PATH.exists():
        rules = yaml.safe_load(TREND_RULES_PATH.read_text()) or {}
        for section in ("trends", "deferred_trends"):
            names.update((rules.get(section) or {}).keys())

    if LABELED_DIR.exists():
        names.update(p.name for p in LABELED_DIR.iterdir() if p.is_dir())

    return sorted(names)
