"""TF-IDF over normalized attribute records, by month and by trend.

Implements the analytical layer that ARCHITECTURE.md sections 3.7 and 3.8
will consume. Operates over the structured attribute records produced by
the section 3.2 normalizer (one JSONL per month under
data/01_data_audit/normalization/outputs/).

Three things this module does:

1. monthly_attribute_tfidf(records)
   Computes TF-IDF per (attribute_token, month). Each month's products
   form a "document"; each attribute value is a "term". TF-IDF separates
   real spikes from year-round baseline frequency. Used by section 3.7
   time-series aggregation to turn raw counts into rarity-weighted
   signals.

2. trend_signature(records, trend_field)
   For each trend, the top-k discriminative attribute tokens by TF-IDF.
   Each trend's record bag is one "document"; attribute tokens are the
   "terms". This is a data-driven complement to the hand-written rules
   in configs/trend_rules.yaml. Useful for spot-checking rules and
   auditing whether the closed details vocabulary is missing
   high-signal tokens.

3. lifecycle_curve(records, attribute_token)
   Monthly TF-IDF curve for one attribute token, with optional rolling
   smoothing for noisy small-month data. Feeds section 3.8 lifecycle
   classifier (rising / persistent / declining shapes follow directly
   from the curve).

Token format: "{field}:{value}", e.g. "material:linen", "pattern:leopard_print".
The field prefix avoids ambiguity when the same value can appear under
different fields.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fashion_forensics.config import PROJECT_ROOT

DEFAULT_FIELDS: tuple[str, ...] = (
    "silhouette",
    "material",
    "sleeve_style",
    "neckline",
    "length",
    "color_profile",
    "pattern",
    "details",
    "subcategory",
    "category",
)

NORMALIZED_OUTPUTS_DIR = PROJECT_ROOT / "data" / "01_data_audit" / "normalization" / "outputs"


def _month_from_record_id(record_id: str) -> str | None:
    """Extract YYYY-MM from a record_id of the form 'YYYY-MM_NNN' or 'YYYYMM_NNN'."""
    if not record_id:
        return None
    head = record_id.split("_", 1)[0]
    if len(head) == 7 and head[4] == "-":
        return head
    if len(head) == 6 and head.isdigit():
        return f"{head[:4]}-{head[4:]}"
    return None


def flatten_attributes(record: dict, fields: Iterable[str] | None = None) -> list[str]:
    """Flatten a normalized record's attribute fields into `field:value` tokens.

    Skips empty fields and empty value lists. Returns tokens like
    'material:linen', 'pattern:leopard_print'. Multi-value fields produce
    multiple tokens.
    """
    fields = fields if fields is not None else DEFAULT_FIELDS
    tokens: list[str] = []
    for field in fields:
        block = record.get(field)
        if not block:
            continue
        values = block.get("value") if isinstance(block, dict) else None
        if not values:
            continue
        for v in values:
            if v:
                tokens.append(f"{field}:{v}")
    return tokens


def load_normalized_records(months_glob: str = "*") -> list[dict]:
    """Load normalized attribute records from the monthly JSONL outputs.

    Each record gets a `month` field added based on the source filename
    (YYYY-MM). Pass `months_glob` to filter, e.g. '2024-*' to load only
    2024 records.
    """
    records: list[dict] = []
    if not NORMALIZED_OUTPUTS_DIR.exists():
        return records
    for path in sorted(NORMALIZED_OUTPUTS_DIR.glob(f"{months_glob}_normalized.jsonl")):
        month = path.stem.replace("_normalized", "")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["month"] = month
            records.append(rec)
    return records


def _records_by_key(
    records: Iterable[dict],
    key_fn,
    fields: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Group records by key_fn, accumulating attribute tokens per group.

    Returns {key: [tokens...]} suitable for feeding to TfidfVectorizer
    as a corpus where each entry is one "document".
    """
    fields = fields if fields is not None else DEFAULT_FIELDS
    by_key: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        k = key_fn(rec)
        if k is None:
            continue
        by_key[k].extend(flatten_attributes(rec, fields))
    return dict(by_key)


def monthly_attribute_tfidf(
    records: Iterable[dict],
    fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute TF-IDF per (attribute_token, month).

    Each month's products are concatenated into one "document" of attribute
    tokens. TfidfVectorizer fits across all months, so IDF is computed over
    the full window. Returns a DataFrame indexed by attribute_token with
    one column per month, values = TF-IDF for that token in that month.

    Records must have a `month` field (added by load_normalized_records or
    parsed from `record_id`).
    """

    def month_key(rec):
        return rec.get("month") or _month_from_record_id(rec.get("record_id", ""))

    by_month = _records_by_key(records, month_key, fields)
    if not by_month:
        return pd.DataFrame()

    months = sorted(by_month.keys())
    docs = [by_month[m] for m in months]

    vectorizer = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    df = pd.DataFrame(matrix.T.toarray(), index=features, columns=months)
    df.index.name = "attribute"
    return df


def trend_signature(
    records: Iterable[dict],
    trend_field: str = "trend_pred",
    top_k: int = 10,
    fields: Iterable[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Top-k discriminative attribute tokens per trend, by TF-IDF.

    Each trend's record bag is one "document"; attribute tokens are the
    "terms". TfidfVectorizer fits across trends, so IDF is computed over
    the trend axis (an attribute that appears in every trend has lower
    weight than one concentrated in one trend).

    Args:
        records: iterable of normalized records, each with `trend_field` set
        trend_field: name of the field carrying the trend label (default
            'trend_pred' for retrieval-pipeline output; use 'trend' for
            ground-truth reference data)
        top_k: number of top-scoring tokens to return per trend
        fields: which attribute fields to include (defaults to DEFAULT_FIELDS)

    Returns:
        {trend_label: [(token, tfidf_score), ...]} sorted by score descending,
        with zero-score tokens omitted.
    """

    def trend_key(rec):
        return rec.get(trend_field)

    by_trend = _records_by_key(records, trend_key, fields)
    if not by_trend:
        return {}

    trends = sorted(by_trend.keys())
    docs = [by_trend[t] for t in trends]

    vectorizer = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    out: dict[str, list[tuple[str, float]]] = {}
    for i, trend in enumerate(trends):
        row = matrix[i].toarray()[0]
        ranked = sorted(
            ((features[j], float(row[j])) for j in range(len(features)) if row[j] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        out[trend] = ranked[:top_k]
    return out


def lifecycle_curve(
    records: Iterable[dict],
    attribute_token: str,
    smooth_window: int = 3,
    fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Monthly TF-IDF curve for one attribute token.

    Args:
        records: normalized records (must carry `month`)
        attribute_token: token in 'field:value' form, e.g. 'material:linen'
        smooth_window: rolling-mean window size for the smoothed curve.
            Pass 1 to disable smoothing. The smoothed column is centered
            with min_periods=1, so the ends are not dropped.
        fields: which attribute fields to include

    Returns:
        DataFrame with columns 'month', 'tfidf', and (if smooth_window > 1)
        'tfidf_smooth'. One row per month, sorted chronologically.

    Raises:
        ValueError if `attribute_token` does not appear in any record.
    """
    matrix = monthly_attribute_tfidf(records, fields)
    if matrix.empty or attribute_token not in matrix.index:
        raise ValueError(
            f"Attribute token {attribute_token!r} not found in any record. "
            f"Available tokens: {len(matrix.index)} (try one of: {list(matrix.index[:5])})"
        )

    series = matrix.loc[attribute_token]
    df = pd.DataFrame({"month": series.index.tolist(), "tfidf": series.values})
    if smooth_window > 1:
        df["tfidf_smooth"] = df["tfidf"].rolling(smooth_window, center=True, min_periods=1).mean()
    return df
