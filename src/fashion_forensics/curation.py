"""Shared logic for the trend-image curator.

Used by both scripts/scrape_trend_images.py (writes candidates) and the
Streamlit review_queue component (reviews/approves/rejects them), so neither
one reimplements candidate storage, dedup, or promotion.
"""

from __future__ import annotations

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

from fashion_forensics.config import DATA_DIR, PROJECT_ROOT

RAW_ROOT = DATA_DIR / "02_reference_corpus" / "raw"
LABELED_ROOT = DATA_DIR / "02_reference_corpus" / "labeled"
CANDIDATES_PATH = DATA_DIR / "02_reference_corpus" / "candidates.jsonl"
PHASHES_PATH = DATA_DIR / "02_reference_corpus" / "phashes.json"
LAST_SYNC_PATH = DATA_DIR / "02_reference_corpus" / ".last_sync"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png"}

# Hamming distance below which two images are treated as near-duplicates.
# 8 is imagehash's commonly-used default for "likely the same photo."
DUPLICATE_HAMMING_THRESHOLD = 8


def load_candidates() -> dict[str, dict]:
    """Read candidates.jsonl into {candidate_id: record}."""
    if not CANDIDATES_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CANDIDATES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["candidate_id"]] = rec
    return out


def save_candidates(records: dict[str, dict]) -> None:
    """Rewrite candidates.jsonl from a {candidate_id: record} dict, sorted by id."""
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w") as f:
        for candidate_id in sorted(records):
            f.write(json.dumps(records[candidate_id]) + "\n")


def compute_phash(image_bytes: bytes) -> str:
    """Perceptual hash of image bytes, as a hex string."""
    return str(imagehash.phash(Image.open(io.BytesIO(image_bytes))))


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def load_reference_phashes() -> dict[str, str]:
    """{relative_image_path: phash} for every already-labeled reference image.

    Cached in phashes.json, only computing entries missing from the cache -
    cheap on repeat calls since just the new images since the last run need
    hashing.
    """
    cached: dict[str, str] = {}
    if PHASHES_PATH.exists():
        cached = json.loads(PHASHES_PATH.read_text())

    changed = False
    if LABELED_ROOT.exists():
        for trend_dir in sorted(LABELED_ROOT.iterdir()):
            if not trend_dir.is_dir():
                continue
            for img_path in sorted(trend_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = str(img_path.relative_to(PROJECT_ROOT))
                if rel not in cached:
                    cached[rel] = compute_phash(img_path.read_bytes())
                    changed = True

    if changed:
        PHASHES_PATH.write_text(json.dumps(cached, indent=2, sort_keys=True))
    return cached


def find_duplicate(
    phash: str, candidates: dict[str, dict], reference_phashes: dict[str, str]
) -> str | None:
    """Id of a near-duplicate (a candidate_id, or a reference image's relative
    path) within the Hamming-distance threshold, else None. Checks already-seen
    candidates first, then the promoted reference corpus.
    """
    for other_id, record in candidates.items():
        other_phash = record.get("phash")
        if other_phash and _hamming_distance(phash, other_phash) <= DUPLICATE_HAMMING_THRESHOLD:
            return other_id
    for ref_path, ref_phash in reference_phashes.items():
        if _hamming_distance(phash, ref_phash) <= DUPLICATE_HAMMING_THRESHOLD:
            return ref_path
    return None


def unique_filename(directory: Path, stem: str, suffix: str) -> str:
    """A filename guaranteed not to collide with anything already in directory."""
    candidate = f"{stem}{suffix}"
    if not (directory / candidate).exists():
        return candidate
    n = 2
    while (directory / f"{stem}-{n}{suffix}").exists():
        n += 1
    return f"{stem}-{n}{suffix}"


def approve_candidate(candidate_id: str, chosen_trend: str) -> dict:
    """Move a candidate's file from raw/ into labeled/{chosen_trend}/, prefixed
    "scraped_" - that prefix is what lets .gitignore keep scraped images out of
    git (they're someone else's copyrighted editorial photos, unlike the
    hand-picked reference set) without touching the classifier pipeline at all.
    Updates and returns the candidates.jsonl row.
    """
    candidates = load_candidates()
    record = candidates[candidate_id]

    src = PROJECT_ROOT / record["local_path"]
    dest_dir = LABELED_ROOT / chosen_trend
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_name = unique_filename(dest_dir, f"scraped_{candidate_id}", src.suffix)
    dest = dest_dir / final_name
    shutil.move(str(src), str(dest))

    record["review_status"] = "approved"
    record["reviewed_trend"] = chosen_trend
    record["promoted_filename"] = final_name
    record["promoted_at"] = datetime.now(timezone.utc).isoformat()
    candidates[candidate_id] = record
    save_candidates(candidates)
    return record


def reject_candidate(candidate_id: str, status: str, reason: str | None = None) -> dict:
    """Mark a candidate "rejected" or "duplicate". The file stays on disk (not
    deleted) so a reviewer can later see what was already considered and why -
    matches this project's stated auditability principle (ARCHITECTURE.md §10).
    """
    candidates = load_candidates()
    record = candidates[candidate_id]
    record["review_status"] = status
    record["reject_reason"] = reason
    candidates[candidate_id] = record
    save_candidates(candidates)
    return record


def pending_sync_count() -> int:
    """How many approved candidates haven't been synced to the classifier yet."""
    last_sync = _read_last_sync()
    count = 0
    for record in load_candidates().values():
        if record.get("review_status") != "approved":
            continue
        promoted_at = record.get("promoted_at")
        if promoted_at and (last_sync is None or promoted_at > last_sync):
            count += 1
    return count


def _read_last_sync() -> str | None:
    if LAST_SYNC_PATH.exists():
        return LAST_SYNC_PATH.read_text().strip()
    return None


def mark_synced() -> None:
    LAST_SYNC_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_SYNC_PATH.write_text(datetime.now(timezone.utc).isoformat())
