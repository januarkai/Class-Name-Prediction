#!/usr/bin/env python
"""
Predict class name(s) from a code snippet using a fine-tuned CodeT5+ checkpoint.

Usage examples:

  # From a file
  python scripts/predict.py \
    --ckpt model/checkpoints/run1-python \
    --language python \
    --file path/to/MyWrongClass.py \
    --k 5

  # From stdin
  cat path/to/Foo.java | python scripts/predict.py --ckpt model/checkpoints/run1-java --language java

Notes:
  - By default, the script masks the declared class identifier in the header with '____' to match training.
    Disable with --no-mask if your model was trained without masking.
  - The prompt template matches training/eval:
      Predict class name:\n{source}\nName:
"""
import argparse
import sys
import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


PROMPT = "Predict class name:\n{source}\nName:"


# Regexes mirror scripts/build_dataset.py
PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(?", re.MULTILINE)
JAVA_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def mask_header(src: str, language: str) -> str:
    """Replace only the declared identifier in the class header with '____'.
    If no class header is found, return the original source.
    """
    if language == 'python':
        m = PY_CLASS_RE.search(src)
        if not m:
            return src
        name = m.group(1)
        return re.sub(rf"(class\s+){re.escape(name)}(\s*\(?)", r"\1____\2", src, count=1)
    elif language == 'java':
        m = JAVA_CLASS_RE.search(src)
        if not m:
            return src
        name = m.group(2)
        return re.sub(rf"(\b(class|interface|enum)\s+){re.escape(name)}\b", r"\1____", src, count=1)
    else:
        return src


def first_token(s: str) -> str:
    s = s.strip()
    return s.split()[0] if s else ''


def predict(ckpt: str, source: str, language: str, k: int = 1, max_new_tokens: int = 16, do_mask: bool = True) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    masked = mask_header(source, language) if do_mask else source
    prompt = PROMPT.format(source=masked)

    inputs = tokenizer([prompt], return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(k, 1),
            num_return_sequences=max(k, 1),
            do_sample=False,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds = [first_token(t) for t in texts]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in preds:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Path to fine-tuned checkpoint dir (e.g., model/checkpoints/run1-python)')
    ap.add_argument('--language', default='python', choices=['python', 'java'], help='Source language for masking rules')
    ap.add_argument('--file', help='Path to input code file. If omitted, read from stdin.')
    ap.add_argument('--k', type=int, default=1, help='Top-k predictions to return')
    ap.add_argument('--max-new-tokens', type=int, default=16, help='Max new tokens to generate')
    ap.add_argument('--no-mask', action='store_true', help='Disable header masking')
    args = ap.parse_args()

    if args.file:
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    preds = predict(
        ckpt=args.ckpt,
        source=source,
        language=args.language,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        do_mask=not args.no_mask,
    )

    if not preds:
        print('')
        return

    # Print a concise, human-friendly output
    print(preds[0])
    if len(preds) > 1:
        # Also show alternatives on a single line (optional)
        alts = ', '.join(preds[1:])
        print(f"alts: {alts}")


if __name__ == '__main__':
    main()
