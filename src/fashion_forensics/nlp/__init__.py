"""NLP layer over normalized attribute records.

Modules:
    - tfidf_engine: TF-IDF analytics for time-series and trend-signature
      discovery (ARCHITECTURE.md sections 3.7 and 3.8 will consume this)
    - trend_engine: rule-based trend mapping (section 3.6)
    - mapper: attribute-to-trend mapping helpers
"""

from fashion_forensics.nlp.tfidf_engine import (
    DEFAULT_FIELDS,
    flatten_attributes,
    lifecycle_curve,
    load_normalized_records,
    monthly_attribute_tfidf,
    trend_signature,
)

__all__ = [
    "DEFAULT_FIELDS",
    "flatten_attributes",
    "lifecycle_curve",
    "load_normalized_records",
    "monthly_attribute_tfidf",
    "trend_signature",
]
