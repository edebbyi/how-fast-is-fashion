"""Implements ARCHITECTURE.md §3.5 — retrieval-based trend classification.

Fashion-CLIP embeds every labeled reference image into a 512-dim space,
Qdrant stores them with trend-label payloads, and inference returns
top-k labeled neighbors for majority or Shepard voting. One of three
parallel trend signals (§2); serves as the baseline the LoRA classifier
(§3.4) must beat in evaluation.
"""

from fashion_forensics.retrieval.classifier import TrendPrediction, predict_trend
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import TrendQdrantStore

__all__ = [
    "FashionClipEmbedder",
    "TrendPrediction",
    "TrendQdrantStore",
    "predict_trend",
]
