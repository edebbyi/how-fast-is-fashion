"""Run the whole trend pipeline in one go: embed references -> classify the
catalog -> compute lifecycle states -> rank against user profiles.

This is the "I updated the reference images, now update everything" entry
point. Use --skip-embed if the reference images haven't changed and you
only need to re-run classification/lifecycle/ranking (this is the common
case, since re-embedding + re-classifying the full catalog takes a while).

Each stage calls the same script's own main() with the same defaults that
script's own command-line flags already use, so this reproduces "run each
script by itself with no flags," just chained together.

Usage:
    .venv/bin/python scripts/run_pipeline.py
    .venv/bin/python scripts/run_pipeline.py --skip-embed
"""

from __future__ import annotations

import argparse

from loguru import logger

from fashion_forensics.config import settings


def main(skip_embed: bool) -> None:
    if not skip_embed:
        logger.info("=== Stage 1: embed reference corpus ===")
        from embed_reference_corpus import main as embed_main

        embed_main(
            embedding_model=settings.fashion_clip_model,
            collection=settings.qdrant_collection,
        )
    else:
        logger.info("=== Stage 1: embed reference corpus (skipped) ===")

    logger.info("=== Stage 2: classify catalog ===")
    from classify_catalog import main as classify_main

    classify_main(
        k=5,
        vote_method="shepard",
        open_set_threshold=None,
        embedding_model=settings.fashion_clip_model,
        collection=settings.qdrant_collection,
        months=None,
        limit=None,
    )

    logger.info("=== Stage 3: compute trend lifecycle ===")
    from compute_trend_lifecycle import main as lifecycle_main

    lifecycle_main(
        active_threshold=0.05,
        persistent_min_active_months=4,
        rising_slope_threshold=0.03,
        declining_slope_threshold=-0.03,
        slope_window=3,
        recency_inactive_months=2,
        recurrence_gap_months=2,
        classifications_path=None,
    )

    logger.info("=== Stage 4: rank catalog ===")
    from rank_catalog import main as rank_main

    rank_main(profile_name=None)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip re-embedding the reference corpus (use when reference images haven't changed)",
    )
    args = parser.parse_args()
    main(skip_embed=args.skip_embed)
