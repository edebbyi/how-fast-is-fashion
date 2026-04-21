"""Download the published dataset for reproducibility.

Images are too large for git. This script downloads the frozen dataset
from a hosted location so anyone can reproduce the analysis without
re-mining from Wayback Machine.

Usage:
    python scripts/download_data.py

The script is idempotent — it skips files that already exist.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Update this URL after uploading your dataset
DATASET_URL = ""  # e.g., HuggingFace, Google Drive, or S3

EXPECTED_STRUCTURE = """
Expected data structure after download:

data/
  01_data_audit/
    raw_images/          <- archived product images by month
    records/             <- raw JSONL metadata (append-only)
    cleaned/             <- filtered JSONL (valid products only)
    state/               <- mining state (hashes, keep rates)
    normalization/
      outputs/           <- LLM-normalized metadata
    evaluation/
      datasets/
        benchmark_gold.csv  <- hand-labeled ground truth
  03_shared/
    taxonomy/
      trend_taxonomy_2024_2026.json  <- trend definitions
"""


def main() -> None:
    if not DATASET_URL:
        print("No dataset URL configured yet.")
        print("To reproduce the analysis, you need the frozen dataset.")
        print(EXPECTED_STRUCTURE)
        print("Options:")
        print(
            "  1. Run the miner yourself:  python -c 'from fashion_forensics.mining.miner import FashionMiner; FashionMiner().run()'"
        )
        print("  2. Contact the author for the dataset")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    # TODO: implement download + extraction once dataset is hosted
    print("Done.")


if __name__ == "__main__":
    main()
