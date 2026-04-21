"""Embed labeled reference images with fashion-CLIP and upsert to local Qdrant.

Reads `data/02_reference_corpus/labeled/{trend}/*.{jpg,jpeg,webp,png}`,
embeds with fashion-CLIP, and writes to a local on-disk Qdrant collection
configured via `Settings.qdrant_*`.

Usage:
    .venv/bin/python scripts/embed_reference_corpus.py
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from fashion_forensics.config import PROJECT_ROOT, settings
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import ReferencePoint, TrendQdrantStore

LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png"}


def discover_reference_images() -> dict[str, list[Path]]:
    """Return {trend_name: [image_paths]} for every subfolder under labeled/."""
    found: dict[str, list[Path]] = {}
    for trend_dir in sorted(LABELED_ROOT.iterdir()):
        if not trend_dir.is_dir():
            continue
        images = sorted(p for p in trend_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if images:
            found[trend_dir.name] = images
    return found


def main() -> None:
    refs_by_trend = discover_reference_images()
    if not refs_by_trend:
        raise RuntimeError(f"No labeled images found under {LABELED_ROOT}")

    for trend, paths in refs_by_trend.items():
        logger.info(f"  {trend}: {len(paths)} images")

    embedder = FashionClipEmbedder(model_name=settings.fashion_clip_model)

    all_paths: list[Path] = []
    all_trends: list[str] = []
    for trend, paths in refs_by_trend.items():
        all_paths.extend(paths)
        all_trends.extend([trend] * len(paths))

    logger.info(f"Embedding {len(all_paths)} images")
    vectors = embedder.embed_images(all_paths)

    store = TrendQdrantStore(
        path=PROJECT_ROOT / settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        embedding_dim=embedder.embedding_dim,
    )
    store.reset_collection()
    # INTERIM: reuse image vector as caption_vec placeholder until path-B ships
    # (scripts/normalize_reference_corpus.py + attributes_to_text). Once path-B
    # lands, caption_vector will be the fashion-CLIP text embedding of the
    # LLM-normalized attributes flattened via attributes_to_text().
    store.upsert_references(
        [
            ReferencePoint(
                trend=trend,
                filename=path.name,
                image_path=str(path),
                image_vector=vec,
                caption_vector=vec,
            )
            for trend, path, vec in zip(all_trends, all_paths, vectors)
        ]
    )

    logger.info(f"Done. Collection '{settings.qdrant_collection}' now has {store.count()} points")


if __name__ == "__main__":
    main()
