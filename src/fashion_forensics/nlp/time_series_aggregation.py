"""Convert item-level trend predictions into month-level frequency statistics.

Implements ARCHITECTURE.md §3.7. Consumes the retrieval pipeline's output
(scripts/classify_catalog.py -> catalog_classifications.jsonl), which uses
field names `month`/`trend_pred` rather than ARCHITECTURE.md's illustrative
`date`/`trend_labels` — this module accepts the real field names directly
as parameters (no adapter layer) and documents the mapping here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

from fashion_forensics.config import PROJECT_ROOT, load_yaml_config

CLASSIFICATIONS_PATH = (
    PROJECT_ROOT / "data" / "03_shared" / "catalog_distribution" / "catalog_classifications.jsonl"
)
UNKNOWN_LABEL = "unknown"


def load_canonical_trends() -> list[str]:
    """Sorted trend keys from configs/trend_rules.yaml.

    Duplicated from scripts/classify_catalog.py.load_canonical_trends()
    rather than imported, to keep this module free of scripts/'s heavier
    deps (retrieval/, requests, matplotlib).
    """
    config = load_yaml_config("trend_rules")
    return sorted(config["trends"].keys())


def load_classifications(path: str | None = None) -> list[dict]:
    """Load catalog_classifications.jsonl into a list of dicts.

    Raises FileNotFoundError with a message pointing at
    scripts/classify_catalog.py if the file doesn't exist yet.
    """
    target = CLASSIFICATIONS_PATH if path is None else path
    from pathlib import Path

    target = Path(target)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run scripts/classify_catalog.py first "
            "to produce it (see that script's docstring for --months smoke-test usage)."
        )
    records = []
    for line in target.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def aggregate_monthly_frequency(
    records: Iterable[dict],
    trends: list[str] | None = None,
    date_field: str = "month",
    trend_field: str = "trend_pred",
    active_threshold: float = 0.05,
    include_unknown: bool = True,
) -> pd.DataFrame:
    """Long-format month x trend frequency table (ARCHITECTURE.md §3.7).

    One row per (month, trend), zero-filled across every canonical trend
    (mirrors classify_catalog.py's write_monthly_csv zero-fill pattern) for
    every month present in `records`. `unknown` gets a row only for months
    it actually occurred in (not zero-filled — it isn't a taxonomy trend),
    but always counts toward that month's inventory_count.

    Args:
        records: item-level records, each with a date field and a trend
            field (e.g. catalog_classifications.jsonl rows).
        trends: canonical trend list to zero-fill against. Defaults to
            load_canonical_trends().
        date_field: name of the date/month field on each record.
        trend_field: name of the trend-label field on each record.
        active_threshold: trend_frequency >= this value marks is_active
            True (inclusive). Must be parameterized, not hard-coded, per
            ARCHITECTURE.md §3.7's note.
        include_unknown: whether to include UNKNOWN_LABEL rows at all.
            When False, unknown-labeled items still count toward
            inventory_count (they're real catalog items) but no `unknown`
            row is emitted.

    Returns:
        DataFrame with columns [date, trend, inventory_count, trend_count,
        trend_frequency, is_active], sorted by [date, trend].

    Raises:
        ValueError if `records` is empty.
    """
    records = list(records)
    if not records:
        raise ValueError("records is empty — nothing to aggregate")

    if trends is None:
        trends = load_canonical_trends()

    inventory_count: dict[str, int] = {}
    trend_count: dict[tuple[str, str], int] = {}
    for rec in records:
        month = rec[date_field]
        label = rec[trend_field]
        inventory_count[month] = inventory_count.get(month, 0) + 1
        trend_count[(month, label)] = trend_count.get((month, label), 0) + 1

    months = sorted(inventory_count.keys())

    rows: list[dict] = []
    for month in months:
        inv = inventory_count[month]
        for trend in trends:
            count = trend_count.get((month, trend), 0)
            freq = count / inv if inv else 0.0
            rows.append(
                {
                    "date": month,
                    "trend": trend,
                    "inventory_count": inv,
                    "trend_count": count,
                    "trend_frequency": freq,
                    "is_active": freq >= active_threshold,
                }
            )
        if include_unknown:
            unk_count = trend_count.get((month, UNKNOWN_LABEL), 0)
            if unk_count > 0:
                freq = unk_count / inv if inv else 0.0
                rows.append(
                    {
                        "date": month,
                        "trend": UNKNOWN_LABEL,
                        "inventory_count": inv,
                        "trend_count": unk_count,
                        "trend_frequency": freq,
                        "is_active": freq >= active_threshold,
                    }
                )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "trend"]).reset_index(drop=True)
