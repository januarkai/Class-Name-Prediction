#!/usr/bin/env python
"""
Fine-tune CodeT5+ for class name prediction.

Inputs:
  --model: Pretrained model name or path (e.g., Salesforce/codet5p-220m)
  --data: Path to datasets directory containing train.jsonl/valid.jsonl
  --output: Output directory for checkpoints (model/checkpoints/<run>)

Optional:
  --batch-size, --grad-accum, --lr, --epochs, --max-source-len, --max-target-len, --fp16
  --wandb: Enable Weights & Biases logging (WANDB_PROJECT env var recommended)
  --seed: Random seed

Mapping:
  Prompt template: "Predict class name:\n{source}\nName:"
  Label: target (string)
"""

import argparse
import os
import sys

def parse_cuda_device():
    # Parse only --cuda-device from sys.argv
    import argparse as _argparse
    ap = _argparse.ArgumentParser(add_help=False)
    ap.add_argument('--cuda-device', type=int, default=0)
    args, _ = ap.parse_known_args()
    return args.cuda_device

cuda_device = parse_cuda_device()
import torch
if torch.cuda.is_available():
    torch.cuda.set_device(cuda_device)

from dataclasses import dataclass
from typing import Dict, List
import datasets as hfds
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


PROMPT = "Predict class name:\n{source}\nName:"


def load_dataset(data_dir: str):
    ds = hfds.load_dataset('json', data_files={
        'train': os.path.join(data_dir, 'train.jsonl'),
        'validation': os.path.join(data_dir, 'valid.jsonl'),
    })
    return ds


@dataclass
class Preprocessor:
    tokenizer: AutoTokenizer
    max_source_len: int
    max_target_len: int

    def __call__(self, batch: Dict[str, List[str]]):
        sources = [PROMPT.format(source=s) for s in batch['source']]
        targets = batch['target']
        model_inputs = self.tokenizer(
            sources, max_length=self.max_source_len, truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, max_length=self.max_target_len, truncation=True
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs


def main():
    # Import CSVLoggerCallback
    from csv_logger import CSVLoggerCallback
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-source-len', type=int, default=1024)
    ap.add_argument('--max-target-len', type=int, default=32)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--cuda-device', type=int, default=0, help='CUDA device id (default: 0)')
    args = ap.parse_args()


    hfds.logging.set_verbosity_info()

    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    ds = load_dataset(args.data)
    proc = Preprocessor(tokenizer, args.max_source_len, args.max_target_len)
    cols = ds['train'].column_names
    ds = ds.map(proc, batched=True, remove_columns=cols)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        evaluation_strategy='steps',
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        logging_steps=50,
        report_to=['wandb'] if args.wandb else [],
        seed=args.seed,
    )

    csv_log_path = os.path.join(args.output, 'training_log.csv')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CSVLoggerCallback(csv_log_path)],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()
