"""Prompt iteration helpers for notebook-based testing.

These helpers let you run a candidate prompt (raw Python string) against any
subset of records WITHOUT writing the results to disk or fetching the prompt
from Langfuse. Use this for iterating on prompt text before committing a new
version.

Calls are still traced to Langfuse under name="prompt_test" with a tag so
they can be filtered out of production analytics.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from loguru import logger

from fashion_forensics.config import DATA_DIR, settings
from fashion_forensics.tracking import LLMTracer

CLEANED_DIR = DATA_DIR / "01_data_audit" / "cleaned"
RAW_IMAGES_DIR = DATA_DIR / "01_data_audit" / "raw_images"


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _fetch_record(record_id: str) -> dict | None:
    """Read one record from the cleaned JSONL files (no DuckDB dependency)."""
    month = record_id.split("_")[0]  # e.g. "2024-02_005" -> "2024-02"
    path = CLEANED_DIR / f"{month}_clean.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        if rec["record_id"] == record_id:
            return rec
    return None


def _compile_prompt(template: str, record: dict) -> str:
    """Substitute Langfuse-style {{var}} placeholders from the record."""
    out = template
    substitutions = {
        "product_name": record.get("product_name") or "",
        "description": record.get("description") or "",
        "declared_category": record.get("category") or "",
        "declared_color": record.get("color") or "",
        "composition": record.get("composition") or "",
    }
    for key, val in substitutions.items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


def run_prompt(
    prompt_text: str,
    record_ids: list[str],
    model: str = "openai/gpt-4o-mini",
    *,
    label: str = "experiment",
    trace: bool = True,
) -> list[dict]:
    """Run a candidate prompt against a list of records. Does not persist results.

    Args:
        prompt_text: Raw prompt template (uses {{var}} placeholders that match
            the production schema: product_name, description, declared_category,
            declared_color, composition).
        record_ids: Record IDs to test (e.g. ["2024-02_002", "2024-02_005"]).
        model: OpenRouter model id.
        label: Tag used in the Langfuse trace metadata so test runs can be
            filtered out of production analytics.
        trace: If False, skip Langfuse tracing entirely.

    Returns:
        List of dicts, one per record_id. Each contains: record_id, image_path,
        product_name, raw_output, parsed (or error).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.open_router_api_key,
        base_url=settings.open_router_base_url,
    )
    tracer = LLMTracer() if trace else None

    results = []
    for record_id in record_ids:
        record = _fetch_record(record_id)
        if record is None:
            results.append({"record_id": record_id, "error": "record not found"})
            continue

        img_path = RAW_IMAGES_DIR / record["month"] / record["image_filename"]
        if not img_path.exists():
            results.append({"record_id": record_id, "error": "image missing on disk"})
            continue

        compiled = _compile_prompt(prompt_text, record)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": compiled},
                    {"type": "image_url", "image_url": {"url": _image_data_url(img_path)}},
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"{record_id}: LLM call failed: {e}")
            results.append({"record_id": record_id, "error": str(e)})
            continue

        raw_output = response.choices[0].message.content
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            parsed = {"_parse_error": str(e), "_raw": raw_output}

        if tracer is not None:
            tracer.trace_llm_call(
                name="prompt_test",
                prompt_input=messages,
                completion=raw_output,
                model=model,
                metadata={
                    "record_id": record_id,
                    "month": record["month"],
                    "label": label,
                    "image_filename": record["image_filename"],
                },
                input_tokens=getattr(response.usage, "prompt_tokens", None),
                output_tokens=getattr(response.usage, "completion_tokens", None),
            )

        results.append(
            {
                "record_id": record_id,
                "image_path": str(img_path),
                "product_name": record.get("product_name"),
                "month": record["month"],
                "retailer": {
                    "color": record.get("color"),
                    "category": record.get("category"),
                    "composition": record.get("composition"),
                    "description": record.get("description"),
                },
                "raw_output": raw_output,
                "parsed": parsed,
            }
        )

    if tracer is not None:
        tracer.flush()

    return results


def random_record_ids(n: int = 5, month: str | None = None) -> list[str]:
    """Sample N random record_ids from the cleaned data (optionally one month)."""
    import random

    if month:
        files = [CLEANED_DIR / f"{month}_clean.jsonl"]
    else:
        files = sorted(CLEANED_DIR.glob("*_clean.jsonl"))

    ids: list[str] = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().strip().split("\n"):
            if line:
                ids.append(json.loads(line)["record_id"])

    return random.sample(ids, min(n, len(ids)))
