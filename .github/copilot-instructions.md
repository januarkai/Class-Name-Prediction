
# Copilot Instructions: CodeT5+ Class Name Prediction

This repo is an end-to-end pipeline for fine-tuning CodeT5+ to predict class names from code snippets. It is intentionally minimal and highly script-driven. Follow these conventions for maximum productivity:

## Project Structure & Data Flow
- `resources/`: Reference papers and docs (no code).
- `data/`: Raw repo clones (`data/repos/<language>/<owner__repo>`) and `repos.txt` (list of URLs).
- `datasets/<language>/`: Model-ready splits (`train.jsonl`, `valid.jsonl`, `test.jsonl`).
- `model/checkpoints/<run>/`: Training outputs; `model/metrics/<run>/metrics.json` for evaluation.
- `scripts/`: All core CLIs (`build_dataset.py`, `train.py`, `eval.py`, `predict.py`).

**Data flow:**
1. List repos in `data/repos.txt` (one per line).
2. Run `scripts/build_dataset.py` to extract and mask class snippets into JSONL splits.
3. Train with `scripts/train.py` (see GPU notes below).
4. Evaluate with `scripts/eval.py` (metrics: EM, top-k, Levenshtein).

## Dataset Schema & Masking
- Each JSONL: `language`, `repo`, `path`, `class_span` {start,end}, `source`, `target`.
- Masking (`--mask`): Only the declared class identifier is replaced with `____` in the header.
- Deduplication: By normalized `source`.
- Splits: Deterministic via `--seed`.

## Training & Evaluation
- Prompt template: `Predict class name:\n{source}\nName:`
- Tokenizer: SentencePiece (`use_fast=False`). Model: CodeT5+ (e.g., `Salesforce/codet5p-220m`).
- Trainer: Evaluates/saves every 1000 steps. W&B logging via `--wandb`.
- Evaluation: Deterministic beam search, EM on first token.

## GPU Selection & CUDA Devices
- For multi-GPU systems, always set `CUDA_VISIBLE_DEVICES` before running `train.py`:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/train.py --model Salesforce/codet5p-220m --data datasets/python --output model/checkpoints/run1-python --batch-size 8 --grad-accum 2 --epochs 3 --fp16 --cuda-device 0
  ```
  The script's `--cuda-device` should be 0 when using `CUDA_VISIBLE_DEVICES` (the visible device is mapped to index 0).

## Example Commands
- Build dataset:
  ```bash
  python scripts/build_dataset.py --repos-file data/repos.txt --in data --out datasets --languages python --mask --min-lines 3
  ```
- Train:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model Salesforce/codet5p-220m --data datasets/python --output model/checkpoints/run1-python --batch-size 8 --grad-accum 2 --epochs 3 --fp16
  ```
- Evaluate:
  ```bash
  python scripts/eval.py --ckpt model/checkpoints/run1-python --data datasets/python --k 5
  ```

## Conventions & Notes
- Supported: Python, Java. Prefer Python for stability.
- Artifacts >100MB are gitignored. Install via `requirements.txt`.
- Use `GITHUB_TOKEN` for higher clone limits.
- If adding new languages/parsers, gate behind `--languages` and match schema/masking rules.

## Key Files
- `scripts/build_dataset.py`: Extracts/masks classes, builds splits.
- `scripts/train.py`: Fine-tunes CodeT5+ (see CUDA/GPU logic at top).
- `scripts/eval.py`: Evaluation and metrics export.
- `README.md`: End-to-end workflow, troubleshooting, and GPU notes.

Reference: `scripts/build_dataset.py`, `scripts/train.py`, `scripts/eval.py`, `README.md`.
- Repro: set a `--seed` and pass it through all steps; document it in `model/metrics/seed.txt`.
