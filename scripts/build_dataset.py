#!/usr/bin/env python
"""
Build a class-name prediction dataset from GitHub repositories.

Inputs:
    - --repos-file: Path to a text file with one Git repo URL per line.
    - --in: Root directory for raw data (will create data/repos/<lang> for clones).
    - --out: Output directory for datasets/ (writes datasets/<lang>/{train,valid,test}.jsonl).

Optional:
  - --languages: Comma-separated list (python,java). Default: python.
  - --mask: If set, replace class identifiers with '____' in source.
  - --min-lines: Minimum number of lines per class to include. Default: 3.
  - --seed: Random seed for split. Default: 42.
  - --train/--valid/--test: Split ratios. Default: 0.8/0.1/0.1.

Outputs:
    - datasets/<language>/{train,valid,test}.jsonl with fields: language, repo, path, class_span?, source, target

Notes:
  - Clones into data/repos/<owner>__<repo>
  - Basic parsing heuristics for Python and Java only.
"""
import argparse
import os
import re
import json
import random
from pathlib import Path
from typing import Iterator, List, Dict, Tuple

from git import Repo
from tqdm import tqdm


PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(?", re.MULTILINE)
JAVA_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def safe_repo_dir(url: str) -> str:
    # Convert https://github.com/user/repo(.git) to user__repo
    name = url.rstrip('/')
    if name.endswith('.git'):
        name = name[:-4]
    parts = name.split('/')
    return f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]


def clone_or_update(url: str, dst_root: Path) -> Path:
    dst = dst_root / safe_repo_dir(url)
    if dst.exists() and (dst / '.git').exists():
        # try to fetch latest
        try:
            repo = Repo(str(dst))
            repo.remote().fetch()
        except Exception:
            pass
        return dst
    dst_root.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, str(dst))
    return dst


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""


def iter_files(repo_dir: Path, exts: Tuple[str, ...]) -> Iterator[Path]:
    for root, _, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(exts):
                yield Path(root) / f


def extract_python_classes(src: str) -> List[Tuple[str, Tuple[int, int]]]:
    # Return list of (class_name, (start_idx, end_idx)) spans
    results = []
    for m in PY_CLASS_RE.finditer(src):
        name = m.group(1)
        # naive: capture from class line to next top-level class or EOF
        start = m.start()
        results.append((name, (start, -1)))
    # post-process to assign end positions
    spans = []
    for i, (name, (start, _)) in enumerate(results):
        end = results[i + 1][1][0] if i + 1 < len(results) else len(src)
        spans.append((name, (start, end)))
    return spans


def extract_java_classes(src: str) -> List[Tuple[str, Tuple[int, int]]]:
    # Very rough: find class/interface/enum declaration and capture until next declaration or EOF
    candidates = []
    for m in JAVA_CLASS_RE.finditer(src):
        cls_kw, name = m.group(1), m.group(2)
        start = m.start()
        candidates.append((name, (start, -1)))
    spans = []
    for i, (name, (start, _)) in enumerate(candidates):
        end = candidates[i + 1][1][0] if i + 1 < len(candidates) else len(src)
        spans.append((name, (start, end)))
    return spans


def normalize_source(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def build_examples(repo: str, path: Path, language: str, src: str, mask: bool, min_lines: int) -> List[Dict]:
    if language == 'python':
        spans = extract_python_classes(src)
    elif language == 'java':
        spans = extract_java_classes(src)
    else:
        return []
    examples = []
    for name, (start, end) in spans:
        snippet = src[start:end]
        if snippet.count('\n') + 1 < min_lines:
            continue
        masked = snippet
        if mask:
            # Replace class name at declaration only, keep rest intact
            if language == 'python':
                masked = re.sub(rf"(class\s+){re.escape(name)}(\s*\(?)", r"\1____\2", masked, count=1)
            elif language == 'java':
                masked = re.sub(rf"(\b(class|interface|enum)\s+){re.escape(name)}\b", r"\1____", masked, count=1)
        examples.append({
            'language': language,
            'repo': repo,
            'path': str(path),
            'class_span': {'start': start, 'end': end},
            'source': masked,
            'target': name,
        })
    return examples


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repos-file', type=str, required=True)
    ap.add_argument('--in', dest='in_dir', type=str, default='data')
    ap.add_argument('--out', dest='out_dir', type=str, default='datasets')
    ap.add_argument('--languages', type=str, default='python')
    ap.add_argument('--mask', action='store_true')
    ap.add_argument('--min-lines', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train', type=float, default=0.8)
    ap.add_argument('--valid', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.in_dir)
    repos_root = data_root / 'repos'
    out_root = Path(args.out_dir)

    languages = [s.strip().lower() for s in args.languages.split(',') if s.strip()]
    with open(args.repos_file, 'r', encoding='utf-8') as f:
        repo_urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    lang_to_rows: Dict[str, List[Dict]] = {lang: [] for lang in languages}

    for url in tqdm(repo_urls, desc='Cloning'):
        repo_name = safe_repo_dir(url)
        # Clone once per language into language-specific dir to keep caches separate
        for lang in languages:
            try:
                lang_repo_root = repos_root / lang
                lang_repo_root.mkdir(parents=True, exist_ok=True)
                repo_dir = clone_or_update(url, lang_repo_root)
            except Exception as e:
                print(f"Failed to clone {url} for {lang}: {e}")
                continue

            if lang == 'python':
                exts = ('.py',)
            elif lang == 'java':
                exts = ('.java',)
            else:
                continue
            for file_path in iter_files(repo_dir, exts):
                src = read_text(file_path)
                rows = build_examples(
                    repo=repo_name,
                    path=file_path.relative_to(repo_dir),
                    language=lang,
                    src=src,
                    mask=args.mask,
                    min_lines=args.min_lines,
                )
                if rows:
                    lang_to_rows[lang].extend(rows)

    # Write per-language splits
    for lang, examples in lang_to_rows.items():
        # Dedup within language
        uniq = {}
        for r in examples:
            key = normalize_source(r['source'])
            if key not in uniq:
                uniq[key] = r
        rows = list(uniq.values())
        random.shuffle(rows)

        n = len(rows)
        n_train = int(n * args.train)
        n_valid = int(n * args.valid)
        train_rows = rows[:n_train]
        valid_rows = rows[n_train:n_train+n_valid]
        test_rows = rows[n_train+n_valid:]

        out_dir = out_root / lang
        write_jsonl(out_dir / 'train.jsonl', train_rows)
        write_jsonl(out_dir / 'valid.jsonl', valid_rows)
        write_jsonl(out_dir / 'test.jsonl', test_rows)
        print(f"[{lang}] Wrote: {len(train_rows)} train, {len(valid_rows)} valid, {len(test_rows)} test → {out_dir}")


if __name__ == '__main__':
    main()
