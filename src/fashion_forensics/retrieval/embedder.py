"""Implements ARCHITECTURE.md §3.5 — fashion-CLIP image/text embedder.

Wraps `transformers.CLIPModel` with a fashion-domain checkpoint
(`patrickjohncyh/fashion-clip` by default) and exposes a numpy-friendly
API for batch image AND text embedding. Both towers output 512-dim
vectors in the same shared space (CLIP's contrastive training objective),
which is what makes named-vector hybrid search possible downstream.

Embeddings are L2-normalized so cosine similarity collapses to a dot
product — simpler to fuse and easier for Qdrant's cosine-distance index.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from pathlib import Path

import numpy as np
import torch
import transformers
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

transformers.logging.set_verbosity_error()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FashionClipEmbedder:
    """Embed images into a unit-norm 512-dim fashion-CLIP space."""

    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: torch.device | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else _select_device()
        logger.info(f"Loading {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        # transformers >=5 returns BaseModelOutputWithPooling from get_*_features;
        # the projected embedding is in .pooler_output. <5 returned a tensor directly.
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        return out

    @torch.no_grad()
    def embed_images(self, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        """Embed a list of image paths. Returns (N, embedding_dim) float32 array, L2-normalized."""
        vectors: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self._as_tensor(self.model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text prompts into the same space (for hybrid search later)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        features = self._as_tensor(self.model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
