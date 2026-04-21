"""Centralized configuration: env vars, YAML configs, MLflow + Langfuse init."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    gemini_api_key: str = Field(default="")
    open_router_api_key: str = Field(default="")
    open_router_model: str = Field(default="google/gemini-2.0-flash-001")
    open_router_base_url: str = Field(default="https://openrouter.ai/api/v1")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    langfuse_prompt_teacher_labeling: str = Field(default="fashion-forensics/teacher_labeling")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="fashion-forensics")

    # Retrieval / vector DB
    fashion_clip_model: str = Field(default="patrickjohncyh/fashion-clip")
    qdrant_path: str = Field(default="data/02_reference_corpus/qdrant")
    qdrant_collection: str = Field(
        default="trends_ref_v2"
    )  # v2 adds named vectors (image + caption)
    reference_captions_path: str = Field(default="data/02_reference_corpus/captions.yaml")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from the configs/ directory."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def init_mlflow(experiment_name: str | None = None) -> None:
    """Initialize MLflow tracking against the running server."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)


def get_langfuse_client():
    """Return an initialized Langfuse client for LLM observability."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
