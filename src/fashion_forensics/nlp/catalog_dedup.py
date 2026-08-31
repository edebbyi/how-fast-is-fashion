"""Collapse photo-level catalog rows down to one row per real product,
for counting purposes only.

The scraper captures one row per product photo (front, alternate angles,
detail shots, ...), not one row per product - the same physical item can
appear several times a month under the same product_code, and each photo
gets classified independently (sometimes into different trends, since
different angles embed differently). Left as-is, every catalog-wide count
(catalog_snapshot.md, monthly_trend_counts.csv, trend lifecycle states)
over- and double-counts real inventory.

Deliberately scoped to counting only - Discover (grid.py) and the ranking
engine keep reading catalog_classifications.jsonl at the full photo level
untouched, since a photo-level view is useful there (reviewing a product's
different angles, or ranking on whichever photo scores best). Only
scripts/classify_catalog.py's snapshot/monthly-csv step and
scripts/compute_trend_lifecycle.py call this.
"""

from __future__ import annotations


def dedupe_for_counting(records: list[dict]) -> list[dict]:
    """One record per (month, product_code), keeping the highest-confidence
    photo (ties broken by lowest record_id, for determinism) as that
    product's representative classification.

    Records with no product_code (a real gap in some scraped metadata, not
    a bug to paper over) are kept as-is, one per record_id - there's no
    product_code to group them by, so nothing to dedupe.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    ungrouped: list[dict] = []
    for r in records:
        code = r.get("product_code")
        if not code:
            ungrouped.append(r)
            continue
        key = (r["month"], code)
        groups.setdefault(key, []).append(r)

    deduped = list(ungrouped)
    for group in groups.values():
        best = max(group, key=lambda r: (r["confidence"], -_record_id_sort_key(r)))
        deduped.append(best)
    return deduped


def _record_id_sort_key(record: dict) -> int:
    """record_id is "{month}_{NNN}" - the numeric suffix, for a stable tie-break."""
    return int(record["record_id"].rsplit("_", 1)[-1])
