"""Implements ARCHITECTURE.md §3.5 — Qdrant store for labeled reference embeddings.

Local on-disk Qdrant collection (cosine distance). Each reference point
carries two named vectors in v2+:
  - image_vec:   fashion-CLIP image embedding of the reference photo
  - caption_vec: fashion-CLIP text embedding of the LLM-normalized
                 attribute description (path-B; image_vec is reused as a
                 placeholder until path-B ships)

Plus payload {trend, filename, image_path, source}. This enables hybrid
search at query time: image-to-image on image_vec, text-to-text on
caption_vec, fused score drives the trend prediction. See §3.5
"Voting methods" and "Operating mode" for the schema-versioning policy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    PointStruct,
    VectorParams,
)

IMAGE_VECTOR_NAME = "image_vec"
CAPTION_VECTOR_NAME = "caption_vec"


@dataclass
class ReferencePoint:
    trend: str
    filename: str
    image_path: str
    image_vector: np.ndarray
    caption_vector: np.ndarray


@dataclass
class Neighbor:
    trend: str
    filename: str
    image_path: str
    score: float


class TrendQdrantStore:
    """Wrapper around a local Qdrant collection with named-vector support."""

    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        embedding_dim: int = 512,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(path=str(self.path))

    def reset_collection(self) -> None:
        """Drop and recreate with the named-vector schema (image_vec + caption_vec)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                IMAGE_VECTOR_NAME: VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                CAPTION_VECTOR_NAME: VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            },
        )
        logger.info(
            f"Created collection: {self.collection_name} "
            f"(dim={self.embedding_dim}, named vectors: {IMAGE_VECTOR_NAME}, {CAPTION_VECTOR_NAME})"
        )

    def upsert_references(self, points: list[ReferencePoint]) -> None:
        """Insert reference embeddings (both named vectors) + payloads."""
        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    IMAGE_VECTOR_NAME: p.image_vector.tolist(),
                    CAPTION_VECTOR_NAME: p.caption_vector.tolist(),
                },
                payload={
                    "trend": p.trend,
                    "filename": p.filename,
                    "image_path": p.image_path,
                    "source": "reference_corpus",
                },
            )
            for p in points
        ]
        self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
        logger.info(f"Upserted {len(qdrant_points)} reference points (named vectors)")

    def search(
        self,
        vector: np.ndarray,
        limit: int = 5,
        using: str = IMAGE_VECTOR_NAME,
        exclude_filenames: list[str] | None = None,
    ) -> list[Neighbor]:
        """Search a single named vector field. `exclude_filenames` supports LOOCV."""
        query_filter = None
        if exclude_filenames:
            query_filter = Filter(
                must_not=[
                    {"key": "filename", "match": {"value": fname}} for fname in exclude_filenames
                ]
            )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            using=using,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

        return [
            Neighbor(
                trend=hit.payload["trend"],
                filename=hit.payload["filename"],
                image_path=hit.payload["image_path"],
                score=float(hit.score),
            )
            for hit in hits
        ]

    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count
