"""Multi-modal teacher labeling (ARCHITECTURE.md §3.2) and retrieval helpers (§3.5).

Exports:
    - normalize_month: monthly batch labeling for the mining pipeline (§3.2)
    - normalize_image: ad-hoc single-image labeling for the reference corpus (§3.5)
    - attributes_to_text: deterministic flatten of attribute JSON to a
      fashion-CLIP-compatible caption string; used on both reference and
      query sides for hybrid retrieval (§3.5).
"""

from fashion_forensics.normalization.normalizer import (
    attributes_to_text,
    normalize_image,
    normalize_month,
)

__all__ = ["attributes_to_text", "normalize_image", "normalize_month"]
