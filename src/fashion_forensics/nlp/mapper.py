"""LLM trend mapper: bridges raw TF-IDF features to named trends via taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from fashion_forensics.config import DATA_DIR, settings
from fashion_forensics.tracking import LLMTracer

TAXONOMY_PATH = DATA_DIR / "03_shared" / "taxonomy" / "trend_taxonomy_2024_2026.json"

SYSTEM_PROMPT = """You are a fashion trend analyst. Given a list of product features
that are distinctive to a specific year, map them to named fashion trends from the
provided taxonomy.

Rules:
- Only use trend names from the taxonomy. Never invent new ones.
- A feature can map to multiple trends or none.
- Return valid JSON only. No markdown, no commentary.
- Format: {"trend_name": ["feature1", "feature2"], ...}
"""


def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    """Load the trend taxonomy JSON file."""
    p = path or TAXONOMY_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Taxonomy not found at {p}. Create it manually or run the editorial enrichment script."
        )
    with open(p) as f:
        return json.load(f)


def map_features_to_trends(
    features: list[str],
    year: int,
    taxonomy: dict[str, Any] | None = None,
    model: str = "gemini-2.0-flash",
    tracer: LLMTracer | None = None,
) -> dict[str, list[str]]:
    """Map a list of raw features to named trends using Gemini.

    Args:
        features: List of distinctive features for the year (from TF-IDF).
        year: The year these features belong to.
        taxonomy: Pre-loaded taxonomy dict. Loaded from disk if None.
        model: Gemini model to use.
        tracer: Optional Langfuse tracer for observability.

    Returns:
        Dict mapping trend names to lists of matched features.
    """
    from google import genai

    if taxonomy is None:
        taxonomy = load_taxonomy()

    taxonomy_summary = _format_taxonomy_for_prompt(taxonomy)

    user_prompt = (
        f"Year: {year}\n\n"
        f"Distinctive features this year:\n{json.dumps(features, indent=2)}\n\n"
        f"Taxonomy of known trends:\n{taxonomy_summary}\n\n"
        "Map features to trends. Return JSON only."
    )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    raw_output = response.text
    result = json.loads(raw_output)

    input_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
    output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

    if tracer:
        tracer.trace_llm_call(
            name="trend_mapping",
            prompt=user_prompt,
            completion=raw_output,
            model=model,
            metadata={"year": year, "feature_count": len(features)},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    logger.info(f"Mapped {len(features)} features to {len(result)} trends for {year}")
    return result


def _format_taxonomy_for_prompt(taxonomy: dict[str, Any]) -> str:
    lines = []
    for name, info in taxonomy.items():
        desc = info.get("description", "")
        keywords = ", ".join(info.get("keywords", []))
        lines.append(f"- {name}: {desc} (keywords: {keywords})")
    return "\n".join(lines)
