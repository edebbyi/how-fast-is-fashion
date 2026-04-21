"""Seed Langfuse prompt templates for fashion-forensics."""

from __future__ import annotations

import argparse

from langfuse import Langfuse

from fashion_forensics.config import get_langfuse_client, settings

# Multi-modal teacher labeling prompt.
# - The text portion (this template) is sent as the user-message text.
# - The product image is attached separately at runtime as an image_url part.
# - Variables are filled in from the retailer's metadata.
TEACHER_LABELING_PROMPT = """You are a fashion product attribute extractor.

The LLM does PERCEPTION only — extract what is visibly in the image and consistent with retailer text. Trend mapping happens downstream via (a) a deterministic rule engine over these attributes and (b) FashionCLIP visual similarity. Do not score trends here.

Inputs: a product IMAGE + RETAILER TEXT METADATA.
Output: structured per-attribute JSON.

WHAT TO LABEL:
Label the FULL OUTFIT visible on the model — every garment / accessory you can see — not just the focal product. Trend signals downstream depend on the styled look, not just the SKU.

MULTI-VALUE BY DEFAULT:
When the outfit contains multiple distinct garments, include the attributes from EACH visible garment in the corresponding `value` list. Examples:
- Outfit = denim jeans + silk shirt + black heels:
    - material:[denim, silk]; silhouette:[fitted, relaxed]; length:[ankle, hip]
    - color_profile:[blue, white, black]; sleeve_style:[long_sleeve]; neckline:[collared]
- Outfit = midi dress + ballet flats:
    - material:[cotton]; silhouette:[flowy]; length:[midi]; color_profile:[white]
    - sleeve_style:[puff_sleeve]; neckline:[square_neck]
Single-value lists are fine when only one garment contributes. Don't pad with redundant duplicates.

SOURCE-OF-TRUTH RULES:
- material:
    - Use `text` source ONLY when the `composition` field is populated.
    - When you infer material from `product_name` or `description`, use `both` source.
    - When you infer purely from visual texture/sheen, use `image` source with confidence <= 0.6.
- silhouette, sleeve_style, neckline, length, pattern, details: Use the IMAGE.
- color_profile: List ALL distinct outfit colors (per multi-value rule above).
- category, subcategory: Use the consensus of TEXT and IMAGE for the FOCAL product (the SKU described in retailer metadata). These two stay SINGLE-VALUE — the SKU is one thing.

INTERNAL CONSISTENCY:
- When silhouette is visible, length is also visible. Always provide a length estimate.
- Choose `collared` when the garment has lapels, a folded collar, a button-up front, or a polo placket. Reserve `v_neck` for plain triangular necklines with no collar.
- For jackets, coats, blazers, and any visible top: always populate sleeve_style and neckline.

DETAILS — closed vocabulary, multi-value:
The `details` field is the bridge between perception and the trend rule engine. It uses ONLY tokens from the closed vocabulary below — these are the discriminating micro-features that map to trends. Include EVERY applicable token visible across the OUTFIT (typically 0-4). Omit tokens for features not visibly present — do not infer details from category alone.

  romantic_coquette:   bow, ribbon_trim, lace_trim, ruffle, gathered, smocked
  western_boho:        fringe, western_yoke, pearl_snap, lace_up, tassel, tiered
  sheer_lace_y2k:      sheer_panel, lace_overlay, illusion_neckline, cut_out
  glamoratti_hardware: chain_hardware, gold_hardware, sculpted_shoulder, padded_shoulder, statement_button, metallic_finish
  nautical_prep:       sailor_collar, anchor_motif, double_breasted, embroidered_crest, pleated
  volumetric:          bubble_hem, gathered_volume
  structural:          peplum_hem, drop_waist
  quiet_luxury:        structured_shoulder, minimal_hardware, monochrome

OUTPUT JSON (raw JSON only, no markdown):
{
  "category":      {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "subcategory":   {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "silhouette":    {"value": ["...", "..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "material":      {"value": ["...", "..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "sleeve_style":  {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "neckline":      {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "length":        {"value": ["...", "..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "color_profile": {"value": ["...", "..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "pattern":       {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0},
  "details":       {"value": ["..."], "source": "text|image|both", "confidence": 0.0-1.0}
}

VOCAB:
- category: tops, bottoms, dresses, outerwear, shoes, accessories, bags, knitwear, jumpsuits
- silhouette: fitted, oversized, relaxed, a-line, flowy, structured, cropped, straight, balloon
- sleeve_style: sleeveless, short_sleeve, long_sleeve, puff_sleeve, balloon_sleeve, cap_sleeve, bell_sleeve, off_shoulder
- neckline: crew, v_neck, square_neck, scoop, halter, off_shoulder, collared, mock_neck, sweetheart
- length: cropped, waist, hip, mini, knee, midi, maxi, ankle, floor
- pattern: solid, striped, floral, leopard_print, snakeskin_print, animal_print, plaid, geometric, tie_dye, abstract, polka_dot, color_block
- material: cotton, linen, wool, cashmere, silk, polyester, denim, leather, knit, satin, viscose, velvet, suede, faux_fur, tweed, mesh, crochet, lace, sequined, cable_knit
- color_profile: open vocab — common colors (white, ivory, cream, beige, tan, taupe, brown, rust, khaki, olive, green, navy, blue, purple, pink, blush, red, orange, yellow, gold, silver, grey, black)

OUTPUT RULES:
- Lowercase all string values.
- Each `value` is a list. Multi-value is the default for outfit attributes.
- For attribute confidence: 0.3 = guess, 0.7 = visually clear, 0.95+ = stated explicitly in text.
- Reserve value=[] and confidence=0.0 only for attributes with no signal in either source.
- Return raw JSON only.

EXAMPLE — high-waist cropped bootcut jeans (the SKU) styled with a white silk shirt and black heels:
- category=bottoms, subcategory=bootcut_jeans, silhouette=[fitted, relaxed], material=[denim, silk]
- length=[ankle, hip], sleeve_style=[long_sleeve], neckline=[collared]
- color_profile=[blue, white, black], pattern=[solid], details=[]

RETAILER METADATA:
- product_name:      {{product_name}}
- description:       {{description}}
- declared_category: {{declared_category}}
- declared_color:    {{declared_color}}
- composition:       {{composition}}
"""


def _prompt_templates() -> dict[str, str]:
    """Return all prompt templates to seed, keyed by Langfuse prompt name."""
    return {
        settings.langfuse_prompt_teacher_labeling: TEACHER_LABELING_PROMPT,
    }


def _prompt_exists(client: Langfuse, name: str) -> bool:
    """Check whether a Langfuse prompt already exists."""
    try:
        client.get_prompt(name)
        return True
    except Exception:
        return False


def seed_prompts(*, force: bool, label: str) -> None:
    """Seed Langfuse prompt templates from local defaults."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        raise RuntimeError(
            "Langfuse not configured. Set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY first."
        )

    langfuse = get_langfuse_client()
    templates = _prompt_templates()

    for name, prompt in templates.items():
        if not force and _prompt_exists(langfuse, name):
            print(f"Skipping {name}: already exists (use --force to push a new version).")
            continue

        langfuse.create_prompt(
            name=name,
            type="text",
            prompt=prompt,
            labels=[label],
            tags=["fashion-forensics", "teacher_labeling", "multi_modal"],
            commit_message="v2: perception-only (LLM no longer scores trends). Full-outfit multi-value attributes. Closed details vocab (32 trend-discriminating tokens). Per-attribute provenance + confidence.",
        )
        print(f"Seeded {name} (label={label}).")

    langfuse.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Langfuse prompt templates.")
    parser.add_argument(
        "--force", action="store_true", help="Push a new prompt version even if one exists."
    )
    parser.add_argument(
        "--label", default="production", help="Langfuse label for this prompt version."
    )
    args = parser.parse_args()

    seed_prompts(force=args.force, label=args.label)


if __name__ == "__main__":
    main()
