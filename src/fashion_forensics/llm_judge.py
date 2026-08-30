"""LLM-assisted first-pass judging for Discover's ground-truth review.

Not a replacement for human review - reviewing 1000+ catalog items by hand
is genuinely slow (RETROSPECTIVE.md section 9). This gives a suggested
verdict, one-sentence reasoning, and a corrected trend (if wrong) as an
editable starting point, grounded in the actual taxonomy read live from
configs/trend_rules.yaml - not a second opinion pulled from nowhere. The
person reviewing still confirms or overrides it before submitting; nothing
here writes to catalog_ground_truth.jsonl directly.

Reuses fashion_forensics.prompt_lab.run_prompt() for the actual LLM call
(image loading, OpenRouter client, Langfuse tracing, JSON parsing) rather
than duplicating that plumbing - see that module for how it works. Each
distinct trend_pred gets its own compiled prompt (the taxonomy context
that matters - the PREDICTED trend's own rule - differs per trend), reused
across every item currently predicted that trend.

Each prompt also includes a few real, past human-corrected judgments for
that same trend_pred (text-only, no images - see MAX_FEW_SHOT_EXAMPLES) -
showing the model real examples of how a person reasons about this
taxonomy, rather than relying only on prose instructions that need
patching after every new kind of miss.
"""

from __future__ import annotations

from collections import defaultdict

from fashion_forensics.catalog_ground_truth import load_ground_truth
from fashion_forensics.config import load_yaml_config
from fashion_forensics.prompt_lab import run_prompt

VALID_TRENDS = ["basics", "mob_wife", "office_siren", "quiet_luxury", "unknown"]

# Text-only, not images - a few real examples of how a human reasons about
# this exact taxonomy transfer better than yet another prose instruction
# added after each new miss, and cost nothing extra in image tokens. Capped
# small on purpose: this is meant to nudge reasoning style, not turn into a
# retrieval system - see RETROSPECTIVE.md section 9 for why per-item
# similarity retrieval was deliberately deferred until there's a much
# bigger ground-truth pool than the ~135 judgments this shipped with.
MAX_FEW_SHOT_EXAMPLES = 3


def _select_few_shot_examples(
    trend_pred: str, max_examples: int = MAX_FEW_SHOT_EXAMPLES
) -> list[dict]:
    """Past judgments where a human corrected this same trend_pred - the
    most instructive kind ("here's a mistake and why"), not just any past
    judgment. Graceful with a small or empty pool: returns however many
    qualify, most recent first, down to zero."""
    ground_truth = load_ground_truth()
    candidates = [
        r
        for r in ground_truth.values()
        if r.get("trend_pred") == trend_pred
        and r.get("judged_correct") is False
        and (r.get("reason") or "").strip()
    ]
    candidates.sort(key=lambda r: r.get("judged_at") or "", reverse=True)
    return candidates[:max_examples]


def _format_few_shot_block(trend_pred: str) -> str:
    examples = _select_few_shot_examples(trend_pred)
    if not examples:
        return ""
    lines = []
    for e in examples:
        corrected = e.get("corrected_trend")
        verdict = f"actually {corrected}" if corrected else "marked incorrect"
        lines.append(f"- Predicted {trend_pred}, {verdict} - {e['reason']}")
    return (
        f"\nExamples of real corrections a human reviewer made for {trend_pred} "
        f"predictions:\n" + "\n".join(lines) + "\n"
    )


def _trend_signal_text(trend: dict) -> str:
    discriminating = trend.get("discriminating_details") or []
    if discriminating:
        return f"Look for: {', '.join(d.replace('_', ' ') for d in discriminating)}."
    # basics has no rule on purpose - defined by absence, per its own
    # config comment. Same framing already used in the Curate UI
    # (review_queue.py's _discriminating_detail_guidance).
    return (
        "Defined by absence, not presence - reject anything with a distinctive "
        "feature that belongs to another trend instead."
    )


def _trend_summaries() -> dict[str, str]:
    """Name + description + discriminating signal per active trend, read
    live from configs/trend_rules.yaml so this can't drift from the real
    taxonomy. Includes the signal, not just the one-line description - a
    first version of this prompt gave only descriptions for the "other
    trends" reference list, and a real test call missed a wool/cashmere
    sweater predicted basics precisely because nothing told the LLM that
    quiet_luxury's discriminating signal includes fine material alone,
    only its vague high-level description ("understated... minimalism")."""
    config = load_yaml_config("trend_rules")
    return {
        key: f"{t['name']}: {t['description'].strip()} {_trend_signal_text(t)}"
        for key, t in config["trends"].items()
    }


def _build_judge_prompt(trend_pred: str) -> str:
    config = load_yaml_config("trend_rules")
    trend = config["trends"][trend_pred]
    signal_text = _trend_signal_text(trend)

    summaries = _trend_summaries()
    other_trends = "\n".join(f"- {v}" for k, v in summaries.items() if k != trend_pred)
    few_shot_block = _format_few_shot_block(trend_pred)

    return f"""You are checking a fashion trend classifier's prediction against a real product.

The classifier predicted: {trend_pred}
{trend["name"]}: {trend["description"].strip()}
{signal_text}

For reference, the other active trends are:
{other_trends}
- unknown: doesn't clearly belong to any of the above.
{few_shot_block}
PRODUCT NAME: {{{{product_name}}}}
DESCRIPTION: {{{{description}}}}
COMPOSITION: {{{{composition}}}}

Read the product name/description/composition above carefully before deciding - a plain-\
looking item is NOT automatically "basics" if the text states a fine material (wool, \
cashmere, tweed, or silk) anywhere; that specific text signal alone is enough to make it \
quiet_luxury instead, even without a distinctive silhouette or hardware detail.

Look at the image. Does this item genuinely match "{trend_pred}" as defined above? Judge the \
photo as it actually looks - not what the garment might look like styled differently.

Respond as raw JSON only, no markdown:
{{"correct": true or false, "reasoning": "one short sentence", \
"corrected_trend": one of {VALID_TRENDS} if incorrect, else null}}"""


def judge_items(record_ids_by_trend: dict[str, list[str]]) -> dict[str, dict]:
    """record_ids_by_trend: {trend_pred: [record_id, ...]} - grouped so each
    distinct trend_pred only needs one compiled prompt, reused across all
    its items via run_prompt()'s own per-record substitution.

    Returns {record_id: {"correct": bool, "reasoning": str,
    "corrected_trend": str | None}}. Items where the LLM call failed or the
    response didn't parse into the expected shape are omitted entirely,
    not silently defaulted to some guessed value."""
    out: dict[str, dict] = {}
    for trend_pred, record_ids in record_ids_by_trend.items():
        if not record_ids:
            continue
        prompt = _build_judge_prompt(trend_pred)
        results = run_prompt(prompt, record_ids, label="ground_truth_judge")
        for r in results:
            if "error" in r:
                continue
            parsed = r.get("parsed", {})
            if "correct" not in parsed:
                continue
            corrected = parsed.get("corrected_trend")
            if corrected not in VALID_TRENDS:
                corrected = None
            out[r["record_id"]] = {
                "correct": bool(parsed["correct"]),
                "reasoning": str(parsed.get("reasoning") or ""),
                "corrected_trend": corrected,
            }
    return out


def group_by_trend(record_ids: list[str], trend_pred_by_id: dict[str, str]) -> dict[str, list[str]]:
    """Small helper so callers (Discover) don't need to hand-roll the
    groupby every time - record_ids missing from trend_pred_by_id are
    dropped rather than crashing on a KeyError."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        trend_pred = trend_pred_by_id.get(record_id)
        if trend_pred is not None:
            grouped[trend_pred].append(record_id)
    return dict(grouped)
