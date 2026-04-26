"""Multi-modal teacher labeling: fuse product image + retailer text metadata into structured attributes.

Aligns with ARCHITECTURE.md §3.2 v2: image is authoritative for visual attributes
(silhouette, sleeve, neckline, pattern, length, details); text is authoritative for
material composition; both must agree on color and category.

Flow:
    cleaned/YYYY-MM_clean.jsonl + raw_images/YYYY-MM/*.jpg
        -> Langfuse prompt template (compiled with text metadata)
        -> Vision LLM (OpenRouter, traced; image attached to Langfuse trace)
        -> normalization/outputs/YYYY-MM_normalized.jsonl
        -> normalization/review/YYYY-MM_review.csv

Also exports for the retrieval pipeline (§3.5):
    - attributes_to_text(attrs): deterministic flatten of an attribute JSON
      record into a fashion-CLIP-compatible caption string. Used on BOTH the
      reference corpus AND query products so text-tower comparisons are
      same-distribution on both sides (no modality + no format gap).
    - normalize_image(image_path): single-image variant of label_record that
      doesn't require the monthly folder structure -- used to label reference
      images at any path on disk.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from fashion_forensics.config import DATA_DIR, settings
from fashion_forensics.tracking import LLMTracer

CLEANED_DIR = DATA_DIR / "01_data_audit" / "cleaned"
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"
NORM_DIR = DATA_DIR / "01_data_audit" / "normalization"
OUTPUTS_DIR = NORM_DIR / "outputs"
REVIEW_DIR = NORM_DIR / "review"

for _d in (OUTPUTS_DIR, REVIEW_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _image_path(record: dict) -> Path:
    """Resolve the on-disk image path for a record."""
    return RAW_IMAGES_DIR / record["month"] / record["image_filename"]


def _image_data_url(path: Path) -> str:
    """Encode an image as a data URL for OpenAI/OpenRouter vision messages."""
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _build_messages(prompt_text: str, image_data_url: str) -> list[dict]:
    """Construct the OpenAI-format multi-modal user message."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]


def label_record(
    record: dict,
    *,
    client,
    prompt_obj,
    tracer: LLMTracer,
    model: str,
    month_str: str,
) -> dict | None:
    """Label one record: fuse its image + text metadata via the vision LLM.

    Returns the parsed attribute JSON (with record_id added) or None on failure.
    """
    img_path = _image_path(record)
    if not img_path.exists():
        logger.warning(f"{record['record_id']}: image missing at {img_path}, skipping")
        return None

    compiled_prompt = prompt_obj.compile(
        product_name=record.get("product_name") or "",
        description=record.get("description") or "",
        declared_category=record.get("category") or "",
        declared_color=record.get("color") or "",
        composition=record.get("composition") or "",
    )

    image_url = _image_data_url(img_path)
    messages = _build_messages(compiled_prompt, image_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"{record['record_id']}: LLM call failed: {e}")
        return None

    raw_output = response.choices[0].message.content
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"{record['record_id']}: JSON parse failed: {e} -- raw: {raw_output[:200]}")
        return None

    # Pass the OpenAI messages list as trace input. Langfuse auto-detects
    # `image_url` parts in chat messages and renders the image in the dashboard
    # (custom dict shapes do not get auto-rendered).
    tracer.trace_llm_call(
        name="teacher_labeling",
        prompt_input=messages,
        completion=raw_output,
        model=model,
        prompt=prompt_obj,
        metadata={
            "record_id": record["record_id"],
            "month": month_str,
            "image_filename": record["image_filename"],
            "product_name": record.get("product_name"),
        },
        input_tokens=getattr(response.usage, "prompt_tokens", None),
        output_tokens=getattr(response.usage, "completion_tokens", None),
    )

    parsed["record_id"] = record["record_id"]
    parsed["model"] = model
    parsed["prompt_version"] = f"{settings.langfuse_prompt_teacher_labeling}@v{prompt_obj.version}"
    return parsed


def normalize_month(
    month_str: str,
    model: str | None = None,
    prompt_label: str = "production",
    max_records: int | None = None,
) -> int:
    """Run multi-modal teacher labeling for all un-labeled records of a month.

    Args:
        month_str: YYYY-MM or YYYYMM.
        model: OpenRouter model id. Defaults to env OPEN_ROUTER_MODEL.
        prompt_label: Langfuse label of the prompt version to fetch.
        max_records: Cap pending records (useful for smoke tests).

    Returns:
        Number of records successfully labeled.
    """
    from openai import OpenAI

    if model is None:
        model = settings.open_router_model

    if "-" not in month_str and len(month_str) == 6:
        month_str = f"{month_str[:4]}-{month_str[4:]}"

    clean_path = CLEANED_DIR / f"{month_str}_clean.jsonl"
    output_path = OUTPUTS_DIR / f"{month_str}_normalized.jsonl"

    if not clean_path.exists():
        logger.warning(f"No clean data for {month_str}")
        return 0

    clean_records = [
        json.loads(line) for line in clean_path.read_text().strip().split("\n") if line
    ]

    existing_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().strip().split("\n"):
            if line:
                existing_ids.add(json.loads(line)["record_id"])

    pending = [r for r in clean_records if r["record_id"] not in existing_ids]
    if max_records is not None:
        pending = pending[:max_records]

    if not pending:
        logger.info(f"{month_str}: all {len(clean_records)} records already labeled")
        return 0

    logger.info(
        f"{month_str}: labeling {len(pending)} records (model={model}, prompt_label={prompt_label})"
    )

    tracer = LLMTracer()
    prompt_obj = tracer.get_prompt(settings.langfuse_prompt_teacher_labeling, label=prompt_label)
    client = OpenAI(
        api_key=settings.open_router_api_key,
        base_url=settings.open_router_base_url,
    )

    labeled_count = 0
    with open(output_path, "a") as f:
        for record in pending:
            result = label_record(
                record,
                client=client,
                prompt_obj=prompt_obj,
                tracer=tracer,
                model=model,
                month_str=month_str,
            )
            if result is not None:
                f.write(json.dumps(result) + "\n")
                f.flush()
                labeled_count += 1

    tracer.flush()
    _generate_review_csv(month_str)

    logger.info(f"{month_str}: labeled {labeled_count} records")
    return labeled_count


def _generate_review_csv(month_str: str) -> None:
    """Create a flat review CSV for human inspection."""
    output_path = OUTPUTS_DIR / f"{month_str}_normalized.jsonl"
    review_path = REVIEW_DIR / f"{month_str}_review.csv"

    if not output_path.exists():
        return

    records = [json.loads(line) for line in output_path.read_text().strip().split("\n") if line]

    if records:
        df = pd.json_normalize(records)
        df.to_csv(review_path, index=False)
        logger.info(f"{month_str}: review CSV saved to {review_path}")


# ---------------------------------------------------------------------------
# Retrieval-pipeline helpers (§3.5)
# ---------------------------------------------------------------------------


def _values(attrs: dict, field: str) -> list[str]:
    """Pull values list from a normalized attribute field, defaulting to [].

    Each field in the normalized JSON is shaped {value: [...], source, confidence}.
    """
    field_obj = attrs.get(field) or {}
    values = field_obj.get("value") or []
    return [str(v).replace("_", " ") for v in values if v]


def attributes_to_text(attrs: dict) -> str:
    """Deterministic flatten: attribute JSON -> fashion-CLIP-compatible caption.

    Same function MUST be used on both reference and query sides so text-tower
    embeddings live in the same surface-form distribution. See ARCHITECTURE.md
    §3.5 "Hybrid search" for the rationale.

    Output shape: "{adjectives} {length}-length {sleeve} {head_noun}, {neckline} neckline, {details}"

    Example:
        {"category": {"value": ["outerwear"], ...},
         "subcategory": {"value": ["sweater"], ...},
         "color_profile": {"value": ["yellow"], ...},
         "silhouette": {"value": ["oversized", "flowy"], ...},
         "sleeve_style": {"value": ["long_sleeve"], ...},
         "neckline": {"value": ["crew"], ...},
         "length": {"value": ["hip"], ...}}
        -> "yellow oversized flowy hip-length long sleeve sweater, crew neckline"

    Empty fields are skipped. Stop-word patterns (e.g., "solid" pattern) are
    kept since the same value will appear on both sides.
    """
    # Adjective stack: surface-level visual descriptors
    adjectives: list[str] = []
    for field in ("color_profile", "pattern", "silhouette", "material"):
        adjectives.extend(_values(attrs, field))

    # Length as "X-length" so "hip" -> "hip-length"
    lengths = [f"{v}-length" for v in _values(attrs, "length")]

    # Sleeve style values are self-contained (e.g. "long sleeve", "sleeveless")
    sleeves = _values(attrs, "sleeve_style")

    # Head noun: prefer specific subcategory, fall back to category
    head_candidates = _values(attrs, "subcategory") or _values(attrs, "category")
    head = head_candidates[0] if head_candidates else "garment"

    # Suffix clauses: details that read more naturally after the head noun
    suffix_parts: list[str] = []
    necklines = _values(attrs, "neckline")
    if necklines:
        suffix_parts.append(f"{' '.join(necklines)} neckline")
    details = _values(attrs, "details")
    if details:
        suffix_parts.append(", ".join(details))

    main = " ".join(filter(None, adjectives + lengths + sleeves + [head])).strip()
    if suffix_parts:
        return f"{main}, {', '.join(suffix_parts)}"
    return main


def normalize_image(
    image_path: str | Path,
    *,
    record_id: str | None = None,
    product_name: str = "",
    description: str = "",
    declared_category: str = "",
    declared_color: str = "",
    composition: str = "",
    model: str | None = None,
    prompt_label: str = "production",
    trace: bool = True,
) -> dict | None:
    """Run the §3.2 teacher-labeling prompt on an arbitrary image path.

    Decoupled from the monthly folder structure used by `normalize_month`.
    Used by the retrieval pipeline to label reference images that live under
    `data/02_reference_corpus/labeled/{trend}/`.

    Args:
        image_path: Path to any image file on disk.
        record_id: Identifier for the resulting record (defaults to filename stem).
        product_name / description / etc.: Optional retailer-text fields. For
            reference images these are typically empty since refs come from
            mood boards, not retail listings -- the LLM falls back to vision.
        model: OpenRouter model id; defaults to env OPEN_ROUTER_MODEL.
        prompt_label: Langfuse label of the prompt version to fetch.
        trace: Whether to log the call to Langfuse (default True).

    Returns:
        Parsed attribute JSON with `record_id`, `model`, `prompt_version`
        stamped on. Same schema as `normalize_month` outputs. Returns None on
        LLM failure (logged + skipped, not raised).
    """
    from openai import OpenAI

    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning(f"normalize_image: {image_path} does not exist, skipping")
        return None

    if model is None:
        model = settings.open_router_model

    if record_id is None:
        record_id = image_path.stem

    tracer = LLMTracer() if trace else None
    prompt_obj = (tracer or LLMTracer()).get_prompt(
        settings.langfuse_prompt_teacher_labeling, label=prompt_label
    )
    client = OpenAI(
        api_key=settings.open_router_api_key,
        base_url=settings.open_router_base_url,
    )

    compiled_prompt = prompt_obj.compile(
        product_name=product_name,
        description=description,
        declared_category=declared_category,
        declared_color=declared_color,
        composition=composition,
    )
    image_url = _image_data_url(image_path)
    messages = _build_messages(compiled_prompt, image_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"{record_id}: LLM call failed: {e}")
        return None

    raw_output = response.choices[0].message.content
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"{record_id}: JSON parse failed: {e} -- raw: {raw_output[:200]}")
        return None

    if tracer is not None:
        tracer.trace_llm_call(
            name="reference_labeling",
            prompt_input=messages,
            completion=raw_output,
            model=model,
            prompt=prompt_obj,
            metadata={
                "record_id": record_id,
                "image_path": str(image_path),
                "source": "reference_corpus",
            },
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
        )
        tracer.flush()

    parsed["record_id"] = record_id
    parsed["model"] = model
    parsed["prompt_version"] = f"{settings.langfuse_prompt_teacher_labeling}@v{prompt_obj.version}"
    return parsed
