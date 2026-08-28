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


def judge_item(
    record_id: str,
    trend_pred: str | None,
    confidence: float | None,
    max_sim: float | None,
    judged_correct: bool,
) -> dict:
    """Record a thumbs-up/down judgment for one catalog item's predicted
    trend. Overwrites any earlier judgment for the same record_id - the
    most recent click wins.

    confidence/max_sim are cast to plain float (or None) since callers
    often pass pandas/numpy values (e.g. a DataFrame row's .get() result),
    which json.dumps can't serialize as-is.

    `reason` always starts empty here - it's a separate, optional step
    (see add_reason) so a thumbs-down stays a single fast click, not
    something that requires typing an explanation to register at all."""
    records = load_ground_truth()
    record = {
        "record_id": record_id,
        "trend_pred": trend_pred,
        "confidence": float(confidence) if confidence is not None else None,
        "max_sim": float(max_sim) if max_sim is not None else None,
        "judged_correct": bool(judged_correct),
        "reason": None,
        "judged_at": datetime.now(UTC).isoformat(),
    }
    records[record_id] = record
    save_ground_truth(records)
    return record


def add_reason(record_id: str, reason: str) -> dict:
    """Attach (or replace) a brief free-text reason on an already-judged
    item - e.g. why a 👎 prediction was wrong. Raises KeyError if
    record_id hasn't been judged yet; a reason without a judgment doesn't
    mean anything here."""
    records = load_ground_truth()
    if record_id not in records:
        raise KeyError(f"{record_id!r} has not been judged yet - judge it before adding a reason")
    records[record_id]["reason"] = reason.strip() or None
    save_ground_truth(records)
    return records[record_id]
