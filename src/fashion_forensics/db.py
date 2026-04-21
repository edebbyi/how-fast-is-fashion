"""DuckDB views over the JSONL pipeline outputs.

DuckDB reads the JSONL files in-place — no ingestion step. New labels written
to disk are immediately queryable.

Views registered:
  - items           : raw cleaned records from the miner
  - teacher_labels  : multi-modal teacher labels (per-attribute structs)
  - merged          : items LEFT JOIN teacher_labels ON record_id

Usage:
    from fashion_forensics.db import connect

    con = connect()
    df = con.execute("SELECT month, COUNT(*) FROM items GROUP BY month").df()

    # Inspect labeling coverage
    con.execute('''
        SELECT i.month,
               COUNT(*) AS items,
               COUNT(t.record_id) AS labeled,
               ROUND(100.0 * COUNT(t.record_id) / COUNT(*), 1) AS pct
        FROM items i LEFT JOIN teacher_labels t USING (record_id)
        GROUP BY i.month ORDER BY i.month
    ''').df()
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from fashion_forensics.config import DATA_DIR

_CLEANED_DIR = DATA_DIR / "01_data_audit" / "cleaned"
_LABELS_DIR = DATA_DIR / "01_data_audit" / "normalization" / "outputs"
_CLEANED_GLOB = str(_CLEANED_DIR / "*_clean.jsonl")
_LABELS_GLOB = str(_LABELS_DIR / "*_normalized.jsonl")


def _glob_has_matches(directory: Path, pattern: str) -> bool:
    return any(directory.glob(pattern))


def connect() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with views over the pipeline outputs.

    Views read the JSONL files lazily — re-running a query after new labels are
    written picks them up automatically (no refresh needed).
    """
    con = duckdb.connect(":memory:")

    if _glob_has_matches(_CLEANED_DIR, "*_clean.jsonl"):
        con.execute(
            f"CREATE OR REPLACE VIEW items AS "
            f"SELECT * FROM read_json_auto('{_CLEANED_GLOB}', union_by_name=true)"
        )
    else:
        con.execute("CREATE OR REPLACE VIEW items AS SELECT NULL::VARCHAR AS record_id WHERE FALSE")

    # union_by_name handles month-to-month schema drift in label outputs
    # (e.g. early months may not have all attribute keys).
    if _glob_has_matches(_LABELS_DIR, "*_normalized.jsonl"):
        con.execute(
            f"CREATE OR REPLACE VIEW teacher_labels AS "
            f"SELECT * FROM read_json_auto('{_LABELS_GLOB}', union_by_name=true)"
        )
    else:
        con.execute(
            "CREATE OR REPLACE VIEW teacher_labels AS SELECT NULL::VARCHAR AS record_id WHERE FALSE"
        )

    con.execute(
        """
        CREATE OR REPLACE VIEW merged AS
        SELECT i.*, t.* EXCLUDE (record_id)
        FROM items i
        LEFT JOIN teacher_labels t USING (record_id)
        """
    )

    return con


def get_item(record_id: str | None = None, image_filename: str | None = None) -> dict | None:
    """Fetch a single item with its retailer fields and (if any) labeled attributes.

    Returned dict shape:
      {
        ...all retailer columns from `items`...,
        "image_path": "/abs/path/to/item.jpg",
        "labels": { ...labeled attribute dicts, or None if unlabeled... }
      }

    Labeled attributes are nested under "labels" to avoid name collisions with
    retailer columns (both tables have `category`, `color`, etc.).
    """
    if not record_id and not image_filename:
        raise ValueError("Provide record_id or image_filename")

    con = connect()
    where = "record_id = ?" if record_id else "image_filename = ?"
    params = [record_id or image_filename]

    item_row = con.execute(f"SELECT * FROM items WHERE {where} LIMIT 1", params).fetchone()
    if item_row is None:
        return None
    item_cols = [d[0] for d in con.description]
    item = dict(zip(item_cols, item_row))

    label_row = con.execute(
        "SELECT * FROM teacher_labels WHERE record_id = ?", [item["record_id"]]
    ).fetchone()
    if label_row is None:
        item["labels"] = None
    else:
        label_cols = [d[0] for d in con.description]
        item["labels"] = dict(zip(label_cols, label_row))

    item["image_path"] = str(_image_full_path(item["month"], item["image_filename"]))
    return item


def _image_full_path(month: str, image_filename: str):
    """Resolve an item's image path on disk."""
    return DATA_DIR / "01_data_audit" / "raw_images" / month / image_filename


def coverage_report() -> str:
    """Return a small text summary of items vs labels per month."""
    con = connect()
    rows = con.execute(
        """
        SELECT
            i.month,
            COUNT(*) AS items,
            COUNT(t.record_id) AS labeled,
            ROUND(100.0 * COUNT(t.record_id) / COUNT(*), 1) AS pct_labeled
        FROM items i
        LEFT JOIN teacher_labels t USING (record_id)
        GROUP BY i.month
        ORDER BY i.month
        """
    ).fetchall()

    header = f"{'month':<10} {'items':>6} {'labeled':>8} {'pct':>6}"
    sep = "-" * len(header)
    body = "\n".join(f"{m:<10} {n:>6} {lab:>8} {pct or 0:>5}%" for m, n, lab, pct in rows)
    return f"{header}\n{sep}\n{body}"
