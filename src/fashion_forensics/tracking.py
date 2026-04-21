"""MLflow (model training/eval) and Langfuse (LLM observability) wrappers.

MLflow is used only for LoRA training and gold set evaluation — not for mining.
Langfuse is used for all LLM calls (normalization, trend mapping, teacher eval).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow
from loguru import logger

from fashion_forensics.config import get_langfuse_client, init_mlflow


@contextmanager
def track_experiment(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager for MLflow experiment runs."""
    init_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"MLflow run started: {experiment_name}/{run_name} ({run.info.run_id})")
        yield mlflow
    logger.info(f"MLflow run ended: {run.info.run_id}")


class LLMTracer:
    """Langfuse v4 wrapper for LLM observability with managed prompt linkage."""

    def __init__(self):
        self.client = get_langfuse_client()

    def get_prompt(self, name: str, label: str = "production"):
        """Fetch a managed prompt from Langfuse. Returns the prompt client object."""
        return self.client.get_prompt(name, label=label)

    def ensure_prompt(
        self,
        name: str,
        default_text: str,
        label: str = "production",
        prompt_type: str = "text",
        config: dict[str, Any] | None = None,
    ):
        """Fetch a managed prompt from Langfuse; create it from `default_text` if missing.

        Returns the Langfuse prompt object — pass to `trace_llm_call(prompt=...)` so
        every generation is linked to the prompt version for per-version analytics.
        """
        try:
            prompt = self.client.get_prompt(name, label=label)
            logger.info(f"Langfuse prompt '{name}' v{prompt.version} loaded (label={label})")
            return prompt
        except Exception:
            logger.info(f"Langfuse prompt '{name}' not found — creating initial version")
            self.client.create_prompt(
                name=name,
                type=prompt_type,
                prompt=default_text,
                labels=[label],
                config=config or {},
            )
            return self.client.get_prompt(name, label=label)

    def trace_llm_call(
        self,
        name: str,
        prompt_input: str,
        completion: str,
        model: str,
        prompt=None,
        metadata: dict[str, Any] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Log a single LLM generation to Langfuse, linked to a managed prompt version."""
        usage = None
        if input_tokens is not None or output_tokens is not None:
            usage = {"input": input_tokens, "output": output_tokens}

        gen = self.client.start_observation(
            name=name,
            as_type="generation",
            input=prompt_input,
            output=completion,
            model=model,
            metadata=metadata or {},
            usage_details=usage,
            prompt=prompt,
        )
        gen.end()

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        self.client.flush()
