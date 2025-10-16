# Copilot instructions for this repo

Purpose: this workspace scaffolds a CodeT5+ fine-tuning project for “class name prediction” from a given class/program snippet. It is intentionally minimal; follow the conventions below to keep the project coherent as you add code.

## Project layout and data flow

- `resources/`: reference papers and docs (e.g., `2305.07922v2.pdf`). No code here.
- `data/`: raw inputs scraped or downloaded from external sources (unprocessed). Keep PII-free; prefer JSONL or gz.
- `datasets/`: processed, model-ready splits (e.g., `train.jsonl`, `valid.jsonl`, `test.jsonl`). Fields described below.
- `model/`: training code, configs, and checkpoints (e.g., `config.yaml`, `checkpoints/`, `tokenizers/`).
- `scripts/`: runnable CLIs for dataset building, training, evaluation, and utilities.

Typical flow:
1) Download/collect repos → `data/raw/...`
2) Build labeled dataset → `datasets/{train,valid,test}.jsonl`
3) Fine-tune CodeT5+ → outputs in `model/checkpoints/`
4) Evaluate and export metrics → `model/metrics/`

# Copilot instructions for this repo

Purpose: Fine-tune CodeT5+ to predict class names from code. This repo already includes dataset building, training, and evaluation CLIs. Keep edits aligned with existing schema, prompts, and paths.

## Layout and flow
- `resources/` papers (e.g., `2305.07922v2.pdf`).
- `data/` raw and clones → clones at `data/repos/<language>/<owner__repo>` (via GitPython).
- `datasets/<language>/` splits: `train.jsonl`, `valid.jsonl`, `test.jsonl`.
- `model/checkpoints/<run>/` training outputs; `model/metrics/<run>/metrics.json` from eval.
- `scripts/`: `build_dataset.py`, `train.py`, `eval.py`.

Flow:
1) Provide repo list `data/repos.txt` → run builder (supports `--languages python,java`, `--mask`).
2) Train CodeT5+ on a language split directory (e.g., `datasets/python`).
3) Evaluate to get EM/top-k/Levenshtein metrics.

## Dataset schema and parsing
JSONL fields: `language`, `repo`, `path`, `class_span` {start,end}, `source`, `target`.
- Python/Java classes are found via regex heuristics; spans cover declaration → next declaration/EOF.
- Masking (`--mask`) replaces only the declared identifier with `____` in the header.
- Dedup by normalized `source`; splits via `--train/--valid/--test` with `--seed`.

## Training and evaluation specifics
- Prompt template:
  Predict class name:\n{source}\nName:
- Tokenizer uses SentencePiece with `use_fast=False`. Model: CodeT5+ (e.g., `Salesforce/codet5p-220m`).
- Trainer evaluates/saves every 1000 steps by default; optional W&B via `--wandb`.
- Eval uses deterministic beam search to produce top-k; EM considers first token of the generated string.

## Commands (copy/paste examples)
- Build (Python, masked):
  python scripts/build_dataset.py --repos-file data/repos.txt --in data --out datasets --languages python --mask --min-lines 3
- Train:
  python scripts/train.py --model Salesforce/codet5p-220m --data datasets/python --output model/checkpoints/run1-python --batch-size 8 --grad-accum 2 --epochs 3 --fp16
- Evaluate top-5:
  python scripts/eval.py --ckpt model/checkpoints/run1-python --data datasets/python --k 5

## Conventions and notes
- Supported languages: python, java. Prefer Python first for stability.
- Keep artifacts >100MB out of git. Install via `requirements.txt`. Set `GITHUB_TOKEN` if you hit clone limits.
- If adding languages/parsers, gate behind `--languages` and keep the same schema/masking rules.

Reference: `scripts/build_dataset.py`, `scripts/train.py`, `scripts/eval.py`, `README.md`.
- Repro: set a `--seed` and pass it through all steps; document it in `model/metrics/seed.txt`.
