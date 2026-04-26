"""Embed labeled reference images with fashion-CLIP and upsert to local Qdrant.

For each image under `data/02_reference_corpus/labeled/{trend}/`:
  - image_vec:   fashion-CLIP image embedding of the photo
  - caption_vec: fashion-CLIP text embedding of attributes_to_text(attrs),
                 where `attrs` comes from data/02_reference_corpus/attributes.jsonl
                 (produced by scripts/normalize_reference_corpus.py)

Both named vectors are upserted into the Qdrant collection configured by
`Settings.qdrant_collection`. Images that don't yet have an attributes record
fall back to using image_vec for caption_vec (warned), which keeps the
collection valid for image-only mode evaluation but disables text/hybrid
modes for those points.

Usage:
    .venv/bin/python scripts/embed_reference_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from fashion_forensics.config import PROJECT_ROOT, settings
from fashion_forensics.normalization import attributes_to_text
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import ReferencePoint, TrendQdrantStore

LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
ATTRIBUTES_PATH = PROJECT_ROOT / "data" / "02_reference_corpus" / "attributes.jsonl"
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


def load_attributes_by_filename() -> dict[str, dict]:
    """Read attributes.jsonl into {filename: full_record}."""
    if not ATTRIBUTES_PATH.exists():
        logger.warning(
            f"No attributes file at {ATTRIBUTES_PATH}. caption_vec will fall back to "
            f"image_vec for all points. Run scripts/normalize_reference_corpus.py first."
        )
        return {}
    out: dict[str, dict] = {}
    for line in ATTRIBUTES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["filename"]] = rec
    return out


def main() -> None:
    refs_by_trend = discover_reference_images()
    if not refs_by_trend:
        raise RuntimeError(f"No labeled images found under {LABELED_ROOT}")

    for trend, paths in refs_by_trend.items():
        logger.info(f"  {trend}: {len(paths)} images")

    attrs_by_filename = load_attributes_by_filename()
    logger.info(f"Loaded {len(attrs_by_filename)} attribute records from {ATTRIBUTES_PATH.name}")

    embedder = FashionClipEmbedder(model_name=settings.fashion_clip_model)

    all_paths: list[Path] = []
    all_trends: list[str] = []
    for trend, paths in refs_by_trend.items():
        all_paths.extend(paths)
        all_trends.extend([trend] * len(paths))

    # --- Image embeddings ---
    logger.info(f"Embedding {len(all_paths)} images with fashion-CLIP vision tower")
    image_vectors = embedder.embed_images(all_paths)

    # --- Caption embeddings (parallel arrays, same order as all_paths) ---
    captions: list[str] = []
    missing_attrs: list[str] = []
    for path in all_paths:
        attrs = attrs_by_filename.get(path.name)
        if attrs is None:
            missing_attrs.append(path.name)
            captions.append("")  # placeholder; fall back to image_vec below
        else:
            captions.append(attributes_to_text(attrs))

    if missing_attrs:
        logger.warning(
            f"{len(missing_attrs)} images have no attributes record (caption_vec "
            f"will reuse image_vec): {missing_attrs[:5]}{'...' if len(missing_attrs) > 5 else ''}"
        )

    # Embed only the non-empty captions; map back to all_paths order
    nonempty_indices = [i for i, c in enumerate(captions) if c]
    if nonempty_indices:
        logger.info(
            f"Embedding {len(nonempty_indices)} captions with fashion-CLIP text tower "
            f"(sample: {captions[nonempty_indices[0]]!r})"
        )
        text_vectors_dense = embedder.embed_texts([captions[i] for i in nonempty_indices])
    else:
        text_vectors_dense = None

    caption_vectors = []
    j = 0
    for i in range(len(all_paths)):
        if captions[i]:
            caption_vectors.append(text_vectors_dense[j])
            j += 1
        else:
            caption_vectors.append(image_vectors[i])  # fallback

    # --- Qdrant upsert ---
    store = TrendQdrantStore(
        path=PROJECT_ROOT / settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        embedding_dim=embedder.embedding_dim,
    )
    store.reset_collection()
    store.upsert_references(
        [
            ReferencePoint(
                trend=trend,
                filename=path.name,
                image_path=str(path),
                image_vector=img_vec,
                caption_vector=cap_vec,
            )
            for trend, path, img_vec, cap_vec in zip(
                all_trends, all_paths, image_vectors, caption_vectors
            )
        ]
    )

    logger.info(f"Done. Collection '{settings.qdrant_collection}' now has {store.count()} points")


if __name__ == "__main__":
    main()
