"""Run the §3.2 teacher-labeling LLM on every labeled reference image.

Walks `data/02_reference_corpus/labeled/{trend}/*` and runs `normalize_image`
on each. Produces `data/02_reference_corpus/attributes.jsonl` with one record
per image. Each record carries a `prompt_version` stamp from Langfuse, which
drives prompt-version-aware idempotency.

Idempotency rules (mode a3, see ARCHITECTURE.md §3.5):
    - New image (no entry in attributes.jsonl) -> labeled
    - Existing entry whose prompt_version matches the CURRENT Langfuse
      production version -> SKIPPED (no LLM call, no cost)
    - Existing entry whose prompt_version is OUTDATED -> automatically
      re-labeled (the new entry replaces the old one)
    - --force re-labels everything regardless of version (debugging only)

Re-runs at the same prompt version do no LLM work.

Usage:
    .venv/bin/python scripts/normalize_reference_corpus.py
    .venv/bin/python scripts/normalize_reference_corpus.py --force
    .venv/bin/python scripts/normalize_reference_corpus.py --max 5    # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from fashion_forensics.config import PROJECT_ROOT, settings
from fashion_forensics.normalization import normalize_image
from fashion_forensics.tracking import LLMTracer

LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
ATTRIBUTES_PATH = PROJECT_ROOT / "data" / "02_reference_corpus" / "attributes.jsonl"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png"}


def discover_reference_images() -> list[tuple[Path, str]]:
    """Return [(image_path, trend_label)] for every labeled image."""
    items: list[tuple[Path, str]] = []
    for trend_dir in sorted(LABELED_ROOT.iterdir()):
        if not trend_dir.is_dir():
            continue
        for p in sorted(trend_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((p, trend_dir.name))
    return items


def load_existing_attributes() -> dict[str, dict]:
    """Read attributes.jsonl into {filename: record} (latest wins)."""
    if not ATTRIBUTES_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in ATTRIBUTES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["filename"]] = rec
    return out


def save_attributes(records: dict[str, dict]) -> None:
    """Rewrite attributes.jsonl from a {filename: record} dict (sorted by filename)."""
    ATTRIBUTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTRIBUTES_PATH, "w") as f:
        for filename in sorted(records):
            f.write(json.dumps(records[filename]) + "\n")


def get_current_prompt_version() -> str:
    """Fetch the current production prompt version from Langfuse, formatted as the stamp."""
    tracer = LLMTracer()
    prompt_obj = tracer.get_prompt(settings.langfuse_prompt_teacher_labeling, label="production")
    return f"{settings.langfuse_prompt_teacher_labeling}@v{prompt_obj.version}"


def main(force: bool, max_records: int | None) -> None:
    refs = discover_reference_images()
    if not refs:
        raise RuntimeError(f"No reference images found under {LABELED_ROOT}")
    logger.info(f"Discovered {len(refs)} reference images across {LABELED_ROOT}")

    current_version = get_current_prompt_version()
    logger.info(f"Current prompt version: {current_version}")

    existing = load_existing_attributes()
    logger.info(f"Existing attributes records: {len(existing)} in {ATTRIBUTES_PATH.name}")

    # Decide which images need labeling
    pending: list[tuple[Path, str]] = []
    for img_path, trend in refs:
        if force:
            pending.append((img_path, trend))
            continue
        prior = existing.get(img_path.name)
        if prior and prior.get("prompt_version") == current_version:
            continue  # already labeled with current prompt -> skip
        pending.append((img_path, trend))

    if max_records is not None:
        pending = pending[:max_records]

    if not pending:
        logger.info("All references are up-to-date. Nothing to do.")
        return

    skipped = len(refs) - len(pending)
    logger.info(f"Labeling {len(pending)} images ({skipped} already up-to-date)")

    # Run labeling, append to in-memory dict, persist after each successful call
    # (so partial failures don't lose work).
    for i, (img_path, trend) in enumerate(pending, start=1):
        logger.info(f"  [{i}/{len(pending)}] {trend}/{img_path.name}")
        record = normalize_image(image_path=img_path, record_id=img_path.stem)
        if record is None:
            logger.warning(f"  -> failed, skipping {img_path.name}")
            continue
        record["filename"] = img_path.name
        record["trend"] = trend
        record["image_path"] = str(img_path.relative_to(PROJECT_ROOT))
        existing[img_path.name] = record
        save_attributes(existing)

    logger.info(
        f"Done. {ATTRIBUTES_PATH.name} now has {len(existing)} records "
        f"({len(pending)} labeled this run)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-label every reference, even if up-to-date (debugging only)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap the number of images to label (smoke testing)",
    )
    args = parser.parse_args()
    main(force=args.force, max_records=args.max)
