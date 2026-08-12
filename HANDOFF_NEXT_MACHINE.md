# Handoff — pick up here on the new machine

Written 2026-08-11 because the source machine's disk started failing mid-session
(repeated file corruption + I/O stalls, including inside `.git` itself — git
commands stopped completing at all). Rather than keep fighting it, the working
tree is being moved to a new machine via AirDrop. This file is the "what's
where and what's next" map for picking the work back up cold.

## What this is

`how-fast-is-fashion` / `fashion-forensics` — trend-intelligence pipeline for
fast-fashion retail. See `README.md` and `ARCHITECTURE.md` for the project
itself. `STATUS_NOTES_2026-07-29.md` is a running session log with much more
detail than this file — read that first for full context on everything done
across the last few sessions.

## Branch / git state — IMPORTANT, needs manual attention

Working branch: `feat/catalog-image-classification`, off `main`.

**3 commits made, NOT pushed to origin** (git broke before push could
complete):
1. `3170d4a` — Add `scripts/classify_catalog.py`
2. `0a22c56` — Add basics reference images (43) + expand quiet_luxury (25→43)
3. `c60b511` — Document basics reference expansion findings in notebook 03

**Also NOT yet committed at all**: `data/02_reference_corpus/attributes.jsonl`
now has all 135 reference images captioned (was 74 when last committed) — the
missing 61 were labeled with `openai/gpt-4o-mini` in this session, matching
the model used for the original 74. This is real, paid-for LLM work — don't
lose it. Commit it before doing anything else.

First things to do on the new machine, once git is confirmed healthy there:
```bash
git add data/02_reference_corpus/attributes.jsonl
git commit -m "Add captions for all 135 reference images (gpt-4o-mini)"
git push -u origin feat/catalog-image-classification
```
Then probably open a PR.

If `.git` in the transferred copy has the same corruption the source machine
had (files that show a correct size in `ls -la` but read back empty — check
`.git/HEAD`, `.git/config`, `.git/packed-refs` first), the fix that worked
repeatedly today: rewrite the file with its known-correct plain-text content
(`HEAD` = `ref: refs/heads/<branch>\n`; `config` = standard clone config with
the `origin` remote). If that doesn't resolve it, the safer move is a fresh
`git clone` of `origin` on the new machine and manually re-applying the 3
local commits' diffs plus the `attributes.jsonl` update from this working
tree, rather than continuing to patch a possibly-still-damaged `.git`.

## What's set up and working (as of transfer)

- Reference corpus: 135 labeled images (`basics` 43, `mob_wife` 25,
  `office_siren` 24, `quiet_luxury` 43), all with LLM-generated attribute
  captions in `attributes.jsonl`.
- Qdrant vector store: gitignored, not transferred — rebuild with
  `.venv/bin/python scripts/embed_reference_corpus.py` (takes ~1 min, no
  API cost, just needs the fashion-CLIP model which downloads on first run).
- `.venv`: NOT transferred (disposable). Recreate:
  ```bash
  python3.11 -m venv .venv
  .venv/bin/python -m pip install -e ".[training,dev]"
  ```
- `.env`: NOT transferred (secrets, never went through git). Needs
  `GEMINI_API_KEY`, `OPEN_ROUTER_API_KEY`, `OPEN_ROUTER_MODEL=openai/gpt-4o-mini`,
  `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`. Bring
  these over separately (password manager, or manually re-copy from the old
  machine's `.env` if it's still reachable).
- Local MLflow: not transferred (`mlartifacts/`, `mlflow.db` — regenerable,
  low value, just run history). Start a server before running evals:
  ```bash
  .venv/bin/mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db
  ```
  Use port 5001, not 5000 — macOS's AirPlay Receiver squats on 5000 by
  default and produces a confusing 403 instead of connection-refused.

## What's next — the actual open experiment

This was queued to run right when the machine started failing, never
completed:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 .venv/bin/python scripts/eval_trend_classifier.py \
  --mode hybrid --alpha 0.95 --k 5
```

Context: `basics` was added this session (previously had zero reference
images despite being defined as an active trend). It turned out to badly
confuse with `quiet_luxury` — image-only LOOCV accuracy went 0.865 (74 refs,
no basics) → 0.763 (114 refs) → 0.726 (135 refs, after *also* expanding
`quiet_luxury` to try to fix it — that made the confusion worse, not better).
Full writeup: `notebooks/03_trend_classification.ipynb` §14.

The open question this hybrid-mode run answers: does adding the text
captions (now that all 135 images have them) help separate `basics` from
`quiet_luxury`, where pixels alone couldn't? Compare against the 0.726
image-only baseline above. If it doesn't help, the taxonomy note in §14
(curate for `structured_shoulder`/`minimal_hardware`/`monochrome` rather than
raw volume, or just document the overlap as a known limitation) is the
fallback plan.

## Known gotchas from this session, worth knowing about

- The source machine had real, recurring disk problems today: files (in
  `.git`, in `.venv`, and in the image corpus itself) that read back as
  completely empty despite `ls -la` showing a correct size. A disk
  repair (Disk Utility → First Aid) fixed it once, but it partially came
  back later. If the new machine shows anything like this, don't assume
  it's a one-off — verify with a byte-level read (`ls -la` size vs. actual
  `open(path,'rb').read()` length), not just `ls`.
- Also saw `OSError: [Errno 89] Operation canceled` during batch image
  reads (`FashionClipEmbedder.embed_images`) — happened twice, both times
  during `embed_images()` looping over all 135 files. A plain retry
  succeeded both times. If it recurs, check for stuck background git/IDE
  processes first (Cursor's git integration left 3 stuck `git status`
  processes running for hours on the source machine, which is what
  actually broke `git status`/`git push` — not disk corruption, that time).
