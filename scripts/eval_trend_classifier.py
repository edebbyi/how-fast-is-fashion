"""Leave-one-out evaluation of the fashion-CLIP + Qdrant trend classifier.

For each labeled reference image: query Qdrant excluding itself, get top-k
neighbors, vote, compare to ground truth. Logs to MLflow:
  - confusion matrix PNG
  - per-class precision / recall / F1
  - macro-F1, accuracy
  - Expected Calibration Error (ECE)
  - silhouette score on the raw embeddings
  - neighbor purity @ k
  - 2D UMAP scatter of the embeddings
  - bootstrap 95% CIs on accuracy, macro-F1, ECE (when --bootstrap is set)

Search modes (one MLflow run per call):
  --mode image     pure visual k-NN (image_vec only)
  --mode text      pure text-to-text k-NN (caption_vec only); requires
                   data/02_reference_corpus/attributes.jsonl
  --mode hybrid    fuse both with alpha (default 0.6 weight on image)

Embedding model:
  --embedding-model HF model id (default: Settings.fashion_clip_model).
                    Use openai/clip-vit-base-patch32 for vanilla-CLIP baseline.
  --collection      Qdrant collection name (default: Settings.qdrant_collection).
                    Pair with a custom --embedding-model so different models
                    get their own collections.

Usage:
    .venv/bin/python scripts/eval_trend_classifier.py --mode image --k 5
    .venv/bin/python scripts/eval_trend_classifier.py --mode hybrid --k 5 --alpha 0.95 --bootstrap
    .venv/bin/python scripts/eval_trend_classifier.py --mode hybrid --alpha 0.95 \\
        --embedding-model openai/clip-vit-base-patch32 \\
        --collection trends_ref_v2_vanilla --bootstrap
"""

from __future__ import annotations

import argparse
import json
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
from fashion_forensics.normalization import attributes_to_text
from fashion_forensics.retrieval.classifier import predict_trend
from fashion_forensics.retrieval.embedder import FashionClipEmbedder
from fashion_forensics.retrieval.qdrant_store import TrendQdrantStore
from fashion_forensics.tracking import track_experiment

LABELED_ROOT = PROJECT_ROOT / "data" / "02_reference_corpus" / "labeled"
ATTRIBUTES_PATH = PROJECT_ROOT / "data" / "02_reference_corpus" / "attributes.jsonl"
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


def load_attributes_by_filename() -> dict[str, dict]:
    if not ATTRIBUTES_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in ATTRIBUTES_PATH.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["filename"]] = rec
    return out


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


def bootstrap_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    classes: list[str],
    n_iterations: int = 10000,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap-resample (with replacement) to compute 95% CIs on key metrics.

    Each iteration samples len(preds) indices with replacement, then recomputes
    accuracy, macro-F1, and ECE on the resampled outcomes. CIs are the 2.5 and
    97.5 percentiles across the iterations.

    Returns:
        {metric: {mean, lo, hi, ci_width}} for accuracy, macro_f1, ece.
    """
    rng = np.random.default_rng(random_seed)
    n = len(preds)
    accs, f1s, eces = [], [], []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        p, lbl, c = preds[idx], labels[idx], confidences[idx]
        correct_resample = (p == lbl).astype(int)
        accs.append(correct_resample.mean())
        f1s.append(f1_score(lbl, p, labels=classes, average="macro", zero_division=0))
        eces.append(expected_calibration_error(c, correct_resample))

    def summarize(values: list[float]) -> dict[str, float]:
        a = np.asarray(values)
        lo = float(np.percentile(a, 2.5))
        hi = float(np.percentile(a, 97.5))
        return {"mean": float(a.mean()), "lo": lo, "hi": hi, "ci_width": hi - lo}

    return {
        "accuracy": summarize(accs),
        "macro_f1": summarize(f1s),
        "ece": summarize(eces),
    }


def neighbor_purity_at_k(
    paths: list[Path],
    labels: list[str],
    embedder: FashionClipEmbedder,
    store: TrendQdrantStore,
    k: int,
) -> float:
    """For each ref, fraction of top-k neighbors (excluding self) sharing its label.

    Always uses image_vec — purity is a measure of the image-embedding space's
    local structure, not of the chosen retrieval mode.
    """
    purities = []
    vectors = embedder.embed_images(paths)
    for path, label, vec in zip(paths, labels, vectors):
        neighbors = store.search(vec, limit=k, exclude_filenames=[path.name])
        if not neighbors:
            continue
        same_class = sum(1 for n in neighbors if n.trend == label)
        purities.append(same_class / len(neighbors))
    return float(np.mean(purities)) if purities else 0.0


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
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


def main(
    k: int,
    vote_method: str,
    mode: str,
    alpha: float,
    embedding_model: str,
    collection: str,
    bootstrap: bool,
    bootstrap_iterations: int,
) -> None:
    paths, labels = discover_refs()
    classes = sorted(set(labels))
    per_class_counts = {c: labels.count(c) for c in classes}
    logger.info(f"Loaded {len(paths)} reference images: {per_class_counts}")
    logger.info(
        f"Mode: {mode} | k={k} | vote={vote_method} | alpha={alpha} | "
        f"model={embedding_model} | collection={collection}"
    )

    embedder = FashionClipEmbedder(model_name=embedding_model)
    store = TrendQdrantStore(
        path=PROJECT_ROOT / settings.qdrant_path,
        collection_name=collection,
        embedding_dim=embedder.embedding_dim,
    )

    if store.count() == 0:
        raise RuntimeError(
            "Qdrant collection is empty. Run scripts/embed_reference_corpus.py first."
        )

    # Load attributes for text/hybrid modes; build query_text per ref
    attrs_by_filename = load_attributes_by_filename()
    if mode in {"text", "hybrid"}:
        if not attrs_by_filename:
            raise RuntimeError(
                f"mode={mode!r} requires {ATTRIBUTES_PATH.name}. "
                "Run scripts/normalize_reference_corpus.py first."
            )
        missing = [p.name for p in paths if p.name not in attrs_by_filename]
        if missing:
            raise RuntimeError(
                f"mode={mode!r}: {len(missing)} images lack attributes records "
                f"(e.g. {missing[:3]}). Run scripts/normalize_reference_corpus.py."
            )

    # --- LOOCV ---
    preds: list[str] = []
    confidences: list[float] = []
    max_sims: list[float] = []
    for path, _ in zip(paths, labels):
        query_text = None
        if mode in {"text", "hybrid"}:
            query_text = attributes_to_text(attrs_by_filename[path.name])
        prediction = predict_trend(
            image_path=path,
            embedder=embedder,
            store=store,
            query_text=query_text,
            k=k,
            vote_method=vote_method,
            search_mode=mode,
            alpha=alpha,
            exclude_filenames=[path.name],
        )
        preds.append(prediction.trend_pred)
        confidences.append(prediction.confidence)
        # Track best-neighbor similarity for the open-set threshold suggestion
        max_sim = max((n["score"] for n in prediction.matched_refs), default=0.0)
        max_sims.append(max_sim)

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

    # --- Embedding-space sanity (image side) ---
    all_vectors = embedder.embed_images(paths)
    label_to_int = {c: i for i, c in enumerate(classes)}
    silhouette = float(
        silhouette_score(all_vectors, [label_to_int[lbl] for lbl in labels], metric="cosine")
    )

    # Suggest an open-set threshold from the distribution of correct-prediction
    # max-neighbor similarities. The 5th percentile keeps 95% of true-positive
    # neighborhoods above the threshold; products below are flagged as
    # `open_set_unknown`. Re-run after each taxonomy expansion since the
    # similarity distribution shifts as more reference classes are added.
    max_sims_arr = np.array(max_sims)
    correct_max_sims = max_sims_arr[correct_arr == 1]
    suggested_threshold = (
        float(np.percentile(correct_max_sims, 5)) if len(correct_max_sims) else 0.0
    )

    logger.info(f"Accuracy:       {accuracy:.3f}")
    logger.info(f"Macro F1:       {macro_f1:.3f}")
    logger.info(f"ECE:            {ece:.3f}")
    logger.info(f"Purity@{k}:       {purity:.3f}")
    logger.info(f"Silhouette:     {silhouette:.3f}")
    logger.info(f"Confusion (rows=true, cols=pred): \n{cm}")
    logger.info(
        f"Suggested open-set threshold (5th percentile of correct-prediction "
        f"max-similarity): {suggested_threshold:.3f}"
    )

    # Bootstrap 95% CIs (optional — adds a few seconds for n=74, 10k iterations)
    bootstrap_results = None
    if bootstrap:
        logger.info(f"Bootstrap: {bootstrap_iterations} iterations...")
        bootstrap_results = bootstrap_metrics(
            preds_arr, labels_arr, conf_arr, classes, n_iterations=bootstrap_iterations
        )
        for metric, stats in bootstrap_results.items():
            logger.info(
                f"  {metric:10s}  point={stats['mean']:.3f}  "
                f"95% CI [{stats['lo']:.3f}, {stats['hi']:.3f}]  width={stats['ci_width']:.3f}"
            )

    run_name_parts = [f"loocv-k{k}", mode, vote_method]
    if mode == "hybrid":
        run_name_parts.append(f"a{alpha:.2f}")
    if embedding_model != settings.fashion_clip_model:
        # Tag the run with a short model identifier so vanilla-CLIP runs are
        # distinguishable from fashion-CLIP runs in the MLflow UI.
        run_name_parts.append(embedding_model.split("/")[-1])
    run_name = "-".join(run_name_parts)

    params = {
        "embedding_model": embedding_model,
        "collection": collection,
        "k": k,
        "distance": "cosine",
        "vote_method": vote_method,
        "search_mode": mode,
        "alpha": alpha if mode == "hybrid" else None,
        "n_refs": len(paths),
        "bootstrap_iterations": bootstrap_iterations if bootstrap else 0,
        **{f"n_{cls}": cnt for cls, cnt in per_class_counts.items()},
    }
    tags = {
        "protocol": "leave-one-out",
        "classes": ",".join(classes),
        "ref_corpus_tag": "2026-04-30",
    }

    with track_experiment(
        experiment_name="trend-classifier-eval",
        run_name=run_name,
        params=params,
        tags=tags,
    ) as mlflow_tracker:
        mlflow_tracker.log_metric("accuracy", accuracy)
        mlflow_tracker.log_metric("macro_f1", macro_f1)
        mlflow_tracker.log_metric("ece", ece)
        mlflow_tracker.log_metric(f"neighbor_purity_at_{k}", purity)
        mlflow_tracker.log_metric("silhouette_cosine", silhouette)
        mlflow_tracker.log_metric("suggested_open_set_threshold", suggested_threshold)
        for cls in classes:
            mlflow_tracker.log_metric(f"precision_{cls}", report[cls]["precision"])
            mlflow_tracker.log_metric(f"recall_{cls}", report[cls]["recall"])
            mlflow_tracker.log_metric(f"f1_{cls}", report[cls]["f1-score"])
        if bootstrap_results is not None:
            for metric, stats in bootstrap_results.items():
                for stat_name, value in stats.items():
                    mlflow_tracker.log_metric(f"{metric}_bootstrap_{stat_name}", value)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cm_path = tmp / "confusion_matrix.png"
            umap_path = tmp / "umap.png"
            plot_confusion_matrix(cm, classes, cm_path, title=f"Confusion ({mode}, LOOCV)")
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
    parser.add_argument(
        "--mode",
        choices=["image", "text", "hybrid"],
        default="image",
        help="Search mode: image-only, text-only, or hybrid (image+text)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight on image score in hybrid mode; text gets (1-alpha). Ignored for non-hybrid.",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.fashion_clip_model,
        help="HF model id (default: fashion-CLIP). Use openai/clip-vit-base-patch32 for vanilla.",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection name. Pair with --embedding-model so models get separate stores.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap 95% CIs on accuracy, macro-F1, and ECE",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Bootstrap resample count (default 10000)",
    )
    args = parser.parse_args()
    main(
        k=args.k,
        vote_method=args.vote,
        mode=args.mode,
        alpha=args.alpha,
        embedding_model=args.embedding_model,
        collection=args.collection,
        bootstrap=args.bootstrap,
        bootstrap_iterations=args.bootstrap_iterations,
    )
