"""Real catalog precision from human ground-truth judgments (Discover's
👍/👎), not from reference-corpus LOOCV.

Implements RETROSPECTIVE.md section 9, item 2. classify_catalog.py's
open-set threshold (0.62) has only ever been checked against the
reference corpus, which notebook 03 section 14.5 already showed doesn't
transfer cleanly to real catalog images (a domain gap, re-confirmed in
section 3 of that same investigation). This script answers the question
that check couldn't: of catalog items a human actually judged, how many
were the prediction genuinely right?

Output: data/03_shared/catalog_distribution/catalog_ground_truth_precision.md
  overall precision, per-trend precision, and a threshold-sensitivity
  table over the judged sample - all explicitly labeled with the sample
  size behind them, since a small sample here is a real limitation, not
  a detail to bury.

Usage:
    .venv/bin/python scripts/catalog_ground_truth_precision.py
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from fashion_forensics.catalog_ground_truth import load_ground_truth
from fashion_forensics.config import DATA_DIR
from fashion_forensics.tracking import track_experiment

OUTPUT_PATH = DATA_DIR / "03_shared" / "catalog_distribution" / "catalog_ground_truth_precision.md"

# Below this many judgments, treat any precision number as suggestive,
# not conclusive - matches the caution notebook 03 already applies to
# small-n results elsewhere (e.g. section 14.3's n=14 caveat).
MIN_SAMPLE_FOR_CONFIDENCE = 30

# Candidate thresholds to check sensitivity across, alongside the shipped
# 0.62. Only makes sense to check values >= 0.62 - the judged sample only
# contains items that already cleared the current threshold, so there's no
# way to know precision below it from this data alone.
THRESHOLD_CANDIDATES = [0.62, 0.65, 0.68, 0.70]


def main() -> None:
    records = list(load_ground_truth().values())
    n = len(records)
    logger.info(f"Loaded {n} judged catalog items")

    if n == 0:
        logger.warning("No judgments yet - nothing to compute. Judge some items in Discover first.")
        return

    overall_correct = sum(1 for r in records if r["judged_correct"])
    overall_precision = overall_correct / n
    logger.info(
        f"Overall precision at threshold 0.62: {overall_precision:.3f} ({overall_correct}/{n})"
    )
    if n < MIN_SAMPLE_FOR_CONFIDENCE:
        logger.warning(
            f"n={n} is below {MIN_SAMPLE_FOR_CONFIDENCE} - treat this as a directional "
            "signal, not a number to act on yet."
        )

    # Per-trend breakdown - some trends may be far more reliable than others,
    # the same way reference-corpus LOOCV already showed per-trend precision
    # varies a lot (notebook 03 section 14.6).
    by_trend: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_trend[r["trend_pred"] or "unknown"].append(r)
    per_trend_precision = {
        trend: sum(1 for r in rows if r["judged_correct"]) / len(rows)
        for trend, rows in by_trend.items()
    }

    # Threshold sensitivity: among only the items that would still pass a
    # higher cutoff, is precision actually better? Real catalog evidence for
    # (or against) raising 0.62, not the reference-corpus-only calibration
    # notebook 03 section 14.5 already found doesn't transfer.
    threshold_rows = []
    for t in THRESHOLD_CANDIDATES:
        subset = [r for r in records if (r["max_sim"] or 0.0) >= t]
        precision = sum(1 for r in subset if r["judged_correct"]) / len(subset) if subset else None
        threshold_rows.append((t, len(subset), precision))

    sample_note = ""
    if n < MIN_SAMPLE_FOR_CONFIDENCE:
        sample_note = (
            f" **Below the {MIN_SAMPLE_FOR_CONFIDENCE}-item confidence threshold - "
            "directional only, not a number to act on yet.**"
        )

    lines = [
        "# Catalog ground-truth precision",
        "",
        f"From {n} human judgments collected in Discover (👍/👎).{sample_note}",
        "",
        f"**Overall precision at 0.62: {overall_precision:.3f}** ({overall_correct}/{n} correct)",
        "",
        "## Per-trend precision",
        "",
        "| Trend | Precision | n |",
        "|---|---|---|",
    ]
    for trend, precision in sorted(per_trend_precision.items()):
        lines.append(f"| {trend} | {precision:.3f} | {len(by_trend[trend])} |")

    lines += [
        "",
        "## Threshold sensitivity (within the judged sample)",
        "",
        "Only checks thresholds >= 0.62 - the judged sample only contains items that "
        "already cleared the current threshold, so precision below it isn't knowable "
        "from this data.",
        "",
        "| Threshold | Precision | n |",
        "|---|---|---|",
    ]
    for t, count, precision in threshold_rows:
        precision_str = f"{precision:.3f}" if precision is not None else "-"
        lines.append(f"| {t:.2f} | {precision_str} | {count} |")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {OUTPUT_PATH}")

    with track_experiment(
        experiment_name="catalog-ground-truth-precision",
        run_name="catalog-ground-truth-precision",
        params={"n_judgments": n, "min_sample_for_confidence": MIN_SAMPLE_FOR_CONFIDENCE},
        tags={"protocol": "catalog-ground-truth-precision"},
    ) as mlflow_tracker:
        mlflow_tracker.log_metric("overall_precision", overall_precision)
        mlflow_tracker.log_metric("n_judgments", n)
        for trend, precision in per_trend_precision.items():
            mlflow_tracker.log_metric(f"precision_{trend}", precision)
        for t, count, precision in threshold_rows:
            if precision is not None:
                mlflow_tracker.log_metric(f"precision_at_{t:.2f}", precision)
                mlflow_tracker.log_metric(f"n_at_{t:.2f}", count)
        mlflow_tracker.log_artifact(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
