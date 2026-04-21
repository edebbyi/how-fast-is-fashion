"""LoRA fine-tuning module for trend-tagging vision models.

Applies LoRA adapters to a pretrained vision transformer and trains on
trend-labeled fashion images. Integrates with MLflow via tracking.py
for experiment tracking, and outputs predictions in the format expected
by evaluation/benchmark.py.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Guard optional training dependencies
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError(
        "Training dependencies not installed. Install them with: pip install -e '.[training]'"
    )

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    raise ImportError(
        "transformers is required for LoRA training. Install with: pip install -e '.[training]'"
    )

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    raise ImportError(
        "peft is required for LoRA training. Install with: pip install -e '.[training]'"
    )

from PIL import Image
from torchvision import transforms

from fashion_forensics.tracking import track_experiment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning."""

    model_name: str = "google/vit-base-patch16-224"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 16
    output_dir: str = "models/lora_adapters"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a flat dict for MLflow param logging."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TrendDataset(Dataset):
    """Torch Dataset that loads fashion images with trend labels.

    Args:
        image_paths: List of paths to image files.
        labels: List of integer trend-label indices.
        transform: Optional torchvision transform override.
    """

    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        image_paths: list[str | Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and labels ({len(labels)}) must have same length"
            )
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform or self.DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_lora(
    config: LoRATrainingConfig,
    train_image_paths: list[str | Path],
    train_labels: list[int],
    label_names: list[str],
    val_image_paths: list[str | Path] | None = None,
    val_labels: list[int] | None = None,
    run_name: str = "lora-trend-tagger",
) -> Path:
    """Fine-tune a vision model with LoRA adapters on trend-labeled images.

    Args:
        config: LoRA training hyperparameters.
        train_image_paths: Paths to training images.
        train_labels: Integer labels for each training image.
        label_names: Human-readable trend names (index-aligned with label ints).
        val_image_paths: Optional validation image paths.
        val_labels: Optional validation labels.
        run_name: MLflow run name.

    Returns:
        Path to saved LoRA adapter weights directory.
    """
    num_labels = len(label_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # --- Load pretrained model ---
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # --- Datasets & dataloaders ---
    train_dataset = TrendDataset(train_image_paths, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = None
    if val_image_paths is not None and val_labels is not None:
        val_dataset = TrendDataset(val_image_paths, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # --- Training loop with MLflow tracking ---
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with track_experiment(
        experiment_name="lora-training",
        run_name=run_name,
        params=config.to_dict(),
        tags={"label_names": ",".join(label_names)},
    ) as mlflow_tracker:
        mlflow_tracker.log_param("num_train_samples", len(train_dataset))
        mlflow_tracker.log_param("num_labels", num_labels)
        if val_loader is not None:
            mlflow_tracker.log_param("num_val_samples", len(val_loader.dataset))

        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            mlflow_tracker.log_metric("train_loss", avg_train_loss, step=epoch)
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - train_loss: {avg_train_loss:.4f}")

            # --- Validation ---
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        pixel_values = batch["pixel_values"].to(device)
                        labels = batch["labels"].to(device)

                        outputs = model(pixel_values=pixel_values, labels=labels)
                        val_loss += outputs.loss.item()

                        preds = outputs.logits.argmax(dim=-1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                avg_val_loss = val_loss / max(len(val_loader), 1)
                val_accuracy = correct / max(total, 1)

                mlflow_tracker.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow_tracker.log_metric("val_accuracy", val_accuracy, step=epoch)
                logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"val_loss: {avg_val_loss:.4f}, val_accuracy: {val_accuracy:.4f}"
                )

        # --- Save LoRA adapter weights ---
        adapter_path = output_path / run_name
        model.save_pretrained(str(adapter_path))
        logger.info(f"LoRA adapter saved to {adapter_path}")
        mlflow_tracker.log_artifact(str(adapter_path))

    return adapter_path


# ---------------------------------------------------------------------------
# Evaluation / inference
# ---------------------------------------------------------------------------


def evaluate_lora(
    adapter_path: str | Path,
    image_paths: list[str | Path],
    record_ids: list[str],
    label_names: list[str],
    model_name: str = "google/vit-base-patch16-224",
) -> list[dict[str, Any]]:
    """Load a trained LoRA adapter and run inference on test images.

    Args:
        adapter_path: Path to saved LoRA adapter directory.
        image_paths: Paths to test images.
        record_ids: Unique identifiers for each test image (aligned with image_paths).
        label_names: Human-readable trend names (index-aligned with model outputs).
        model_name: Base model name (must match training).

    Returns:
        List of prediction dicts compatible with benchmark.run_benchmark():
            [{"record_id": str, "trend_pred": str, "confidence": float}, ...]
    """
    if len(image_paths) != len(record_ids):
        raise ValueError(
            f"image_paths ({len(image_paths)}) and record_ids ({len(record_ids)}) "
            "must have same length"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device: {device}")

    # --- Load base model + LoRA adapter ---
    num_labels = len(label_names)
    base_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)
    model.eval()

    # --- Prepare images ---
    transform = TrendDataset.DEFAULT_TRANSFORM
    predictions: list[dict[str, Any]] = []

    logger.info(f"Running inference on {len(image_paths)} images")

    with torch.no_grad():
        for img_path, record_id in zip(image_paths, record_ids):
            try:
                image = Image.open(img_path).convert("RGB")
                pixel_values = transform(image).unsqueeze(0).to(device)

                outputs = model(pixel_values=pixel_values)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
                confidence, pred_idx = probs.max(dim=0)

                predictions.append(
                    {
                        "record_id": record_id,
                        "trend_pred": label_names[pred_idx.item()],
                        "confidence": round(confidence.item(), 4),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                predictions.append(
                    {
                        "record_id": record_id,
                        "trend_pred": "unknown",
                        "confidence": 0.0,
                    }
                )

    logger.info(f"Inference complete: {len(predictions)} predictions generated")
    return predictions
