"""A cheap, zero-LLM-cost text check for one specific known failure mode:
image-only retrieval predicting `basics` for an item that's actually
`quiet_luxury`, when the retailer's own product name/description already
says the fabric plainly (e.g. "WOOL AND CASHMERE BLEND KNIT SWEATER").

Why this exists instead of hybrid mode: notebook 03 section 14.1 already
tested hybrid mode (image + LLM caption) on this exact confusion and found
it only marginally helps (+2.2pp accuracy) - the overlap is structural,
both trends' *captions* converge on similarly vague descriptions too, not
just their pixels. But a literal material keyword in the retailer's own
product name is a much stronger, more explicit signal than an LLM
caption guessing fabric from a photo - and unlike hybrid mode, it needs no
LLM normalization (which the catalog mostly doesn't have - only 59/2029
items are normalized), just the product_name/description every catalog
item already has for free. See RETROSPECTIVE.md section 9.

Deliberately narrow: only overrides a `basics` prediction, never touches
mob_wife/office_siren/quiet_luxury/unknown - a lone material keyword
shouldn't reroute an item the classifier is already confident about
something else (a wool-lined mob_wife jacket shouldn't get pulled toward
quiet_luxury just because "wool" appears in its name).
"""

from __future__ import annotations

from fashion_forensics.config import load_yaml_config


def load_fine_material_keywords() -> list[str]:
    """quiet_luxury's fine-material-alone rule block (configs/trend_rules.yaml
    - the any_of block with only a `material` condition, no subcategory or
    color requirement) - reused here as a text-keyword list so it can't
    drift from the taxonomy's own definition of what counts as a
    discriminating fine material."""
    config = load_yaml_config("trend_rules")
    rule = config["trends"]["quiet_luxury"]["rule"]
    for block in rule["any_of"]:
        if set(block.keys()) == {"material"}:
            return list(block["material"])
    raise ValueError("quiet_luxury rule's fine-material-alone block (material-only) not found")


def match_fine_material_keyword(record: dict, keywords: list[str]) -> str | None:
    """Case-insensitive substring search of a catalog record's product_name
    and description for a fine-material keyword. Returns the matched
    keyword (lowercased), or None.

    Plain substring match, not a word-boundary regex - deliberately
    simple. False-positive risk for these specific words (wool, cashmere,
    tweed, silk) appearing inside an unrelated word is negligible in real
    retail copy, and a strict word-boundary match would miss legitimate
    inflections like "woolen" or "silky"."""
    text = " ".join(filter(None, [record.get("product_name"), record.get("description")])).lower()
    for keyword in keywords:
        if keyword.lower() in text:
            return keyword.lower()
    return None
