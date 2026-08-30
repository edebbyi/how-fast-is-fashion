"""Shared logic for collecting real catalog ground truth.

Used by the Discover tab (src/fashion_forensics/app/components/grid.py) to
let someone thumbs-up/down a catalog item's predicted trend while browsing.
This is the real-catalog-data counterpart to scripts/reference_loocv.py,
which only measures precision on the reference corpus, not on actual
catalog images - see RETROSPECTIVE.md section 9 for why this exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fashion_forensics.config import DATA_DIR

GROUND_TRUTH_PATH = DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_ground_truth.jsonl"


def load_ground_truth() -> dict[str, dict]:
    """Read catalog_ground_truth.jsonl into {record_id: record}."""
    if not GROUND_TRUTH_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in GROUND_TRUTH_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["record_id"]] = rec
    return out


def save_ground_truth(records: dict[str, dict]) -> None:
    """Rewrite catalog_ground_truth.jsonl from a {record_id: record} dict, sorted by id."""
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w") as f:
        for record_id in sorted(records):
            f.write(json.dumps(records[record_id]) + "\n")


def judge_items_batch(judgments: list[dict]) -> list[dict]:
    """Record many judgments in one file read + write, instead of one
    read-modify-write cycle per item. Built for Discover's form-based
    review flow (RETROSPECTIVE.md section 9): selecting verdicts (and,
    for a 👎, an optional reason/corrected-trend) on a whole page of
    cards causes no server round-trip at all until a single Submit
    click, at which point every selection lands here together.

    Each dict needs record_id/trend_pred/confidence/max_sim/judged_correct,
    plus the optional reason/corrected_trend (both default to None - only
    meaningful when judged_correct is False, but this function doesn't
    enforce that itself; the caller decides what's worth saving).
    confidence/max_sim are cast to plain float (or None) since callers
    often pass pandas/numpy values, which json.dumps can't serialize
    as-is."""
    records = load_ground_truth()
    now = datetime.now(UTC).isoformat()
    saved = []
    for j in judgments:
        confidence = j.get("confidence")
        max_sim = j.get("max_sim")
        record = {
            "record_id": j["record_id"],
            "trend_pred": j.get("trend_pred"),
            "confidence": float(confidence) if confidence is not None else None,
            "max_sim": float(max_sim) if max_sim is not None else None,
            "judged_correct": bool(j["judged_correct"]),
            "reason": (j.get("reason") or "").strip() or None,
            "corrected_trend": j.get("corrected_trend") or None,
            "judged_at": now,
        }
        records[record["record_id"]] = record
        saved.append(record)
    save_ground_truth(records)
    return saved
