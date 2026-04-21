"""Leave-one-out evaluation of the fashion-CLIP + Qdrant trend classifier.

For each of the 35 reference images: query Qdrant with that image's
embedding *excluding itself*, get top-k neighbors, vote, compare to the
ground-truth label. Logs to MLflow:
  - confusion matrix PNG
  - per-class precision / recall / F1
  - macro-F1, accuracy
  - Expected Calibration Error (ECE)
  - silhouette score on the raw embeddings
  - neighbor purity @ k
  - 2D UMAP scatter of the embeddings

Usage:
    .venv/bin/python scripts/eval_trend_classifier.py --k 5
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)

from fashion_forensics.config import PROJECT_ROOT, settings
from fashion_forensics.retrieval.classifier import predict_trend
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import TrendQdrantStore
from fashion_forensics.tracking import track_experiment

LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".png"}


def discover_refs() -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    for trend_dir in sorted(LABELED_ROOT.iterdir()):
        if not trend_dir.is_dir():
            continue
        for p in sorted(trend_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(trend_dir.name)
    return paths, labels


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Bin predictions by confidence; ECE = weighted gap between mean conf and accuracy per bin."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        ece += (in_bin.sum() / len(confidences)) * abs(bin_conf - bin_acc)
    return float(ece)


def neighbor_purity_at_k(
    paths: list[Path],
    labels: list[str],
    embedder: FashionClipEmbedder,
    store: TrendQdrantStore,
    k: int,
) -> float:
    """For each ref, fraction of top-k neighbors (excluding self) sharing its label."""
    purities = []
    vectors = embedder.embed_images(paths)
    for path, label, vec in zip(paths, labels, vectors):
        # Default search is on image_vec (the image named vector)
        neighbors = store.search(vec, limit=k, exclude_filenames=[path.name])
        if not neighbors:
            continue
        same_class = sum(1 for n in neighbors if n.trend == label)
        purities.append(same_class / len(neighbors))
    return float(np.mean(purities)) if purities else 0.0


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (LOOCV)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_umap(vectors: np.ndarray, labels: list[str], out_path: Path) -> None:
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP plot")
        return
    n_neighbors = min(15, len(vectors) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for cls, color in zip(classes, colors):
        mask = np.array([lbl == cls for lbl in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=cls, s=80, alpha=0.8)
    ax.legend()
    ax.set_title("Reference corpus in fashion-CLIP space (UMAP)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(k: int, vote_method: str) -> None:
    paths, labels = discover_refs()
    classes = sorted(set(labels))
    logger.info(f"Loaded {len(paths)} reference images across {len(classes)} classes: {classes}")

    embedder = FashionClipEmbedder(model_name=settings.fashion_clip_model)
    store = TrendQdrantStore(
        path=PROJECT_ROOT / settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        embedding_dim=embedder.embedding_dim,
    )

    if store.count() == 0:
        raise RuntimeError(
            "Qdrant collection is empty. Run scripts/embed_reference_corpus.py first."
        )

    # --- LOOCV ---
    preds: list[str] = []
    confidences: list[float] = []
    for path, true_label in zip(paths, labels):
        prediction = predict_trend(
            image_path=path,
            embedder=embedder,
            store=store,
            k=k,
            vote_method=vote_method,
            search_mode="image",  # INTERIM: image-only until path-B (caption_vec) ships
            exclude_filenames=[path.name],
        )
        preds.append(prediction.trend_pred)
        confidences.append(prediction.confidence)

    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    conf_arr = np.array(confidences)
    correct_arr = (preds_arr == labels_arr).astype(int)

    accuracy = float(correct_arr.mean())
    macro_f1 = float(f1_score(labels_arr, preds_arr, labels=classes, average="macro"))
    cm = confusion_matrix(labels_arr, preds_arr, labels=classes)
    report = classification_report(
        labels_arr, preds_arr, labels=classes, output_dict=True, zero_division=0
    )
    ece = expected_calibration_error(conf_arr, correct_arr)
    purity = neighbor_purity_at_k(paths, labels, embedder, store, k=k)

    # --- Embedding-space sanity ---
    all_vectors = embedder.embed_images(paths)
    label_to_int = {c: i for i, c in enumerate(classes)}
    silhouette = float(
        silhouette_score(all_vectors, [label_to_int[lbl] for lbl in labels], metric="cosine")
    )

    logger.info(f"Accuracy:       {accuracy:.3f}")
    logger.info(f"Macro F1:       {macro_f1:.3f}")
    logger.info(f"ECE:            {ece:.3f}")
    logger.info(f"Purity@{k}:       {purity:.3f}")
    logger.info(f"Silhouette:     {silhouette:.3f}")
    logger.info(f"Confusion (rows=true, cols=pred): \n{cm}")

    with track_experiment(
        experiment_name="trend-classifier-eval",
        run_name=f"loocv-k{k}-{vote_method}",
        params={
            "embedding_model": settings.fashion_clip_model,
            "k": k,
            "distance": "cosine",
            "vote_method": vote_method,
            "n_refs": len(paths),
        },
        tags={"protocol": "leave-one-out", "classes": ",".join(classes)},
    ) as mlflow_tracker:
        mlflow_tracker.log_metric("accuracy", accuracy)
        mlflow_tracker.log_metric("macro_f1", macro_f1)
        mlflow_tracker.log_metric("ece", ece)
        mlflow_tracker.log_metric(f"neighbor_purity_at_{k}", purity)
        mlflow_tracker.log_metric("silhouette_cosine", silhouette)
        for cls in classes:
            mlflow_tracker.log_metric(f"precision_{cls}", report[cls]["precision"])
            mlflow_tracker.log_metric(f"recall_{cls}", report[cls]["recall"])
            mlflow_tracker.log_metric(f"f1_{cls}", report[cls]["f1-score"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cm_path = tmp / "confusion_matrix.png"
            umap_path = tmp / "umap.png"
            plot_confusion_matrix(cm, classes, cm_path)
            plot_umap(all_vectors, labels, umap_path)
            mlflow_tracker.log_artifact(str(cm_path))
            if umap_path.exists():
                mlflow_tracker.log_artifact(str(umap_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k neighbors for voting")
    parser.add_argument(
        "--vote",
        choices=["majority", "shepard"],
        default="shepard",
        help="Voting method: count-based majority or similarity-weighted (Shepard)",
    )
    args = parser.parse_args()
    main(k=args.k, vote_method=args.vote)
