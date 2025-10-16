# CodeT5+ Class Name Prediction

This repo scaffolds an end-to-end pipeline to fine-tune CodeT5+ to predict class names from code snippets. It includes dataset building from GitHub repos, training, and evaluation.

## Setup

Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build dataset from GitHub

1) Prepare a list of repos (one per line): `data/repos.txt`
```
https://github.com/pallets/flask
https://github.com/psf/requests
```

2) Run dataset builder (Python by default; or pass `--languages python,java`):
```bash
python scripts/build_dataset.py \
  --repos-file data/repos.txt \
  --in data \
  --out datasets \
  --mask \
  --min-lines 3
```

Outputs per language: `datasets/<language>/{train,valid,test}.jsonl` with fields `language, repo, path, class_span, source, target`.

## Train CodeT5+

Train on Python only (adjust path for Java):
```bash
python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16 
```

The prompt template is:
```
Predict class name:
{source}
Name:
```

## Evaluate

Evaluate the Python model (adjust for Java):
```bash
python scripts/eval.py \
  --ckpt model/checkpoints/run1-python \
  --data datasets/python \
  --k 5
```

Produces: `model/metrics/run1-python/metrics.json` with exact match, case-insensitive EM, top-k accuracy, and average Levenshtein distance.

Training artifacts (under your `--output`, e.g., `model/checkpoints/run1-python/`):
- `training_log.csv` — step-wise training/eval metrics.
- `training_curve.png` — training/eval loss plot.
- `checkpoints.txt` — discovered `checkpoint-*` directories and final model path.

## Notes
- Start with Python for best heuristic parsing; Java is supported with basic regex.
- Use `--mask` to replace the declared class identifier with `____` to avoid label leakage.
- Consider rate limits and licenses when mining GitHub; export `GITHUB_TOKEN` for higher clone limits.
- Large files and artifacts are ignored via `.gitignore`.

## Predict from a snippet

Use the trained checkpoint to predict a class name from a code snippet (file or stdin). By default, the header class name is masked to match training.

From a file (Python):
```bash
python scripts/predict.py \
  --ckpt model/checkpoints/run1-python \
  --language python \
  --file path/to/MyWrongClass.py \
  --k 5 \
  --mask-all \
  --out path/to/MyClass_fixed.py \
  --rename-all
```

From stdin (Java):
```bash
cat path/to/Foo.java | python scripts/predict.py --ckpt model/checkpoints/run1-java --language java
```

Examples included:
- `examples/wrong_name_lru_cache.py` — LRU cache with wrong class name.
- `examples/wrong_name_image.py` — image container with wrong class name.

## GPU Selection and CUDA Devices

If you have multiple GPUs and want to select which one to use for training, set the `CUDA_VISIBLE_DEVICES` environment variable before running the training script. For example, to use GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --model Salesforce/codet5p-220m \
  --data datasets/python \
  --output model/checkpoints/run1-python \
  --batch-size 8 --grad-accum 2 --epochs 3 --fp16 --cuda-device 0
```

- The script argument `--cuda-device` should be set to 0 when using `CUDA_VISIBLE_DEVICES`, as the visible device will be mapped to index 0.
- This ensures all CUDA allocations go to the correct GPU and avoids out-of-memory errors on the wrong device.

See [PyTorch CUDA documentation](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) for more details.