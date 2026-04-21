"""Implements ARCHITECTURE.md §3.5 — k-NN trend classifier over the Qdrant reference store.

Search modes:
  - image:  query image → image_vec search only (pure visual k-NN)
  - text:   query text  → caption_vec search only (pure text-to-text)
  - hybrid: run BOTH searches, fuse with alpha · image + (1−alpha) · text

Hybrid exploits CLIP's text-to-text tightness (same-modal comparisons
avoid the image↔text modality gap) so LLM-normalized product descriptions
(from §3.2 perception) can contribute at full weight alongside the raw
pixels — see §3.5 "Interaction with schema versions" for how this output
slots into the v1 flat schema vs the deferred v2 nested schema.

Voting over the (fused) top-k labeled neighbors yields the trend
prediction and confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import (
    CAPTION_VECTOR_NAME,
    IMAGE_VECTOR_NAME,
    Neighbor,
    TrendQdrantStore,
)

VoteMethod = Literal["majority", "shepard"]
SearchMode = Literal["image", "text", "hybrid"]


@dataclass
class TrendPrediction:
    trend_pred: str
    confidence: float
    matched_refs: list[dict]
    open_set_unknown: bool
    search_mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def _vote_majority(neighbors: list[Neighbor]) -> tuple[str, float]:
    votes = Counter(n.trend for n in neighbors)
    winner, _ = votes.most_common(1)[0]
    winning_scores = [n.score for n in neighbors if n.trend == winner]
    confidence = sum(winning_scores) / len(winning_scores)
    return winner, confidence


def _vote_shepard(neighbors: list[Neighbor]) -> tuple[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for n in neighbors:
        scores[n.trend] += n.score
    winner = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[winner] / total if total > 0 else 0.0
    return winner, confidence


def _fuse_neighbors(
    image_hits: list[Neighbor],
    text_hits: list[Neighbor],
    alpha: float,
    k: int,
) -> list[Neighbor]:
    """Weighted-sum fusion of two ranked lists keyed by filename.

    A reference may appear in only one list; missing scores contribute 0.
    Returns the top-k by fused score, re-expressed as Neighbor records so
    downstream voting is identical to the single-mode path.
    """
    by_filename: dict[str, dict] = {}
    for n in image_hits:
        by_filename.setdefault(n.filename, {"neighbor": n, "img": 0.0, "txt": 0.0})
        by_filename[n.filename]["img"] = n.score
    for n in text_hits:
        by_filename.setdefault(n.filename, {"neighbor": n, "img": 0.0, "txt": 0.0})
        by_filename[n.filename]["txt"] = n.score

    fused = []
    for entry in by_filename.values():
        n = entry["neighbor"]
        score = alpha * entry["img"] + (1 - alpha) * entry["txt"]
        fused.append(
            Neighbor(
                trend=n.trend,
                filename=n.filename,
                image_path=n.image_path,
                score=score,
            )
        )
    fused.sort(key=lambda n: n.score, reverse=True)
    return fused[:k]


def predict_trend(
    image_path: str | Path,
    embedder: FashionClipEmbedder,
    store: TrendQdrantStore,
    query_text: str | None = None,
    k: int = 5,
    vote_method: VoteMethod = "shepard",
    search_mode: SearchMode = "hybrid",
    alpha: float = 0.6,
    open_set_threshold: float | None = None,
    exclude_filenames: list[str] | None = None,
) -> TrendPrediction:
    """Classify an image (optionally with a text description) via Qdrant k-NN.

    Args:
        image_path: Query image.
        embedder: Loaded fashion-CLIP embedder.
        store: Qdrant store with named-vector reference corpus.
        query_text: Text description of the query product (from the LLM
            normalization step, §3.2). Required when search_mode is "text"
            or "hybrid".
        k: Number of neighbors to retrieve (from the fused list for hybrid).
        vote_method: "majority" or "shepard" (default).
        search_mode: "image" | "text" | "hybrid" (default).
        alpha: Weight on image score in hybrid fusion; text gets (1 − alpha).
            Only used when search_mode == "hybrid".
        open_set_threshold: If max fused score < threshold, set
            open_set_unknown=True.
        exclude_filenames: Used by LOOCV to omit the held-out reference.

    Returns:
        TrendPrediction with the predicted trend, confidence, matched
        references, open-set flag, and which search_mode was used.
    """
    if search_mode in {"text", "hybrid"} and not query_text:
        raise ValueError(f"search_mode={search_mode!r} requires query_text")

    # Retrieve from each relevant tower. Pull more than k per tower so
    # fusion has enough overlap to rank confidently.
    per_tower_limit = max(k * 3, 10)
    image_hits: list[Neighbor] = []
    text_hits: list[Neighbor] = []

    if search_mode in {"image", "hybrid"}:
        q_img = embedder.embed_images([image_path])[0]
        image_hits = store.search(
            q_img,
            limit=per_tower_limit,
            using=IMAGE_VECTOR_NAME,
            exclude_filenames=exclude_filenames,
        )

    if search_mode in {"text", "hybrid"}:
        q_txt = embedder.embed_texts([query_text])[0]
        text_hits = store.search(
            q_txt,
            limit=per_tower_limit,
            using=CAPTION_VECTOR_NAME,
            exclude_filenames=exclude_filenames,
        )

    if search_mode == "image":
        neighbors = image_hits[:k]
    elif search_mode == "text":
        neighbors = text_hits[:k]
    else:
        neighbors = _fuse_neighbors(image_hits, text_hits, alpha=alpha, k=k)

    if not neighbors:
        return TrendPrediction(
            trend_pred="unknown",
            confidence=0.0,
            matched_refs=[],
            open_set_unknown=True,
            search_mode=search_mode,
        )

    if vote_method == "shepard":
        winner, confidence = _vote_shepard(neighbors)
    else:
        winner, confidence = _vote_majority(neighbors)

    max_sim = max(n.score for n in neighbors)
    open_set_unknown = open_set_threshold is not None and max_sim < open_set_threshold

    return TrendPrediction(
        trend_pred=winner,
        confidence=round(confidence, 4),
        matched_refs=[
            {"filename": n.filename, "trend": n.trend, "score": round(n.score, 4)}
            for n in neighbors
        ],
        open_set_unknown=open_set_unknown,
        search_mode=search_mode,
    )
