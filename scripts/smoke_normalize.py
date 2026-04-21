"""Smoke test for the normalization pipeline.

Bootstraps the Langfuse-managed prompt (creates it if missing) and normalizes
3 records from the smallest pending month so you can verify traces show up
in your Langfuse dashboard before running the full pipeline.

Usage:
    python scripts/smoke_normalize.py
"""

from __future__ import annotations

from loguru import logger

from fashion_forensics.config import settings
from fashion_forensics.mining.miner import get_clean_count
from fashion_forensics.normalization.normalizer import normalize_month
from fashion_forensics.tracking import LLMTracer

NORMALIZATION_PROMPT_V1 = """You are a fashion product metadata standardizer.
Given raw product metadata, normalize it into structured JSON objects.

Output format (one object per input record):
{
    "record_id": "<copy from input>",
    "category": "one of: tops, bottoms, dresses, outerwear, shoes, accessories, bags",
    "subcategory": "e.g. blazer, midi skirt, ankle boot",
    "color_primary": "standardized color name",
    "color_secondary": "if applicable, else null",
    "material_primary": "e.g. cotton, polyester, linen, wool",
    "material_secondary": "if applicable, else null",
    "silhouette": "e.g. oversized, fitted, a-line, relaxed",
    "pattern": "e.g. solid, striped, floral, animal print, plaid",
    "details": ["list of notable details: bow, ruffle, slit, hardware, etc."],
    "confidence": 0.0-1.0
}

Rules:
- Use lowercase for all values
- If a field cannot be determined, use null
- Return a valid JSON array with one normalized object per input record
- Each object MUST include the original record_id

Normalize these product records:

{{records}}
"""


def pick_smallest_month_with_data() -> str:
    """Pick a month that has clean records (so we have inputs to normalize)."""
    months = [
        "202402",
        "202403",
        "202404",
        "202405",
        "202406",
        "202407",
        "202408",
        "202409",
        "202410",
        "202411",
        "202412",
        "202501",
        "202502",
        "202503",
        "202504",
        "202505",
        "202506",
        "202507",
        "202508",
        "202509",
        "202511",
        "202512",
        "202601",
    ]
    candidates = [(m, get_clean_count(m)) for m in months]
    candidates = [(m, c) for m, c in candidates if c > 0]
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def main():
    tracer = LLMTracer()
    logger.info(f"Bootstrapping Langfuse prompt: {settings.langfuse_prompt_normalization}")
    prompt_obj = tracer.ensure_prompt(
        name=settings.langfuse_prompt_normalization,
        default_text=NORMALIZATION_PROMPT_V1,
        label="production",
        prompt_type="text",
        config={"model": "gemini-2.0-flash", "temperature": 0.1},
    )
    logger.info(f"Prompt ready: v{prompt_obj.version}")

    month = pick_smallest_month_with_data()
    logger.info(f"Smoke-normalizing 3 records from {month}")

    count = normalize_month(
        month_str=month,
        model="gemini-2.0-flash",
        batch_size=3,
        max_records=3,
        prompt_label="production",
    )
    logger.info(f"Done. Normalized {count} records. Check Langfuse for traces.")


if __name__ == "__main__":
    main()
