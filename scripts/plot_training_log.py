import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, required=True, help='Path to training_log.csv')
parser.add_argument('--out', type=str, default=None, help='Output path for plot (PNG)')
args = parser.parse_args()

log_path = args.log
out_path = args.out or os.path.join(os.path.dirname(log_path), 'training_curve.png')

df = pd.read_csv(log_path)

plt.figure(figsize=(10, 6))
if 'step' in df.columns:
    x = df['step']
else:
    x = range(len(df))
if 'train_loss' in df.columns:
    plt.plot(x, df['train_loss'], label='Train Loss')
if 'eval_loss' in df.columns:
    plt.plot(x, df['eval_loss'], label='Eval Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training/Eval Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path)
print(f"Saved plot to {out_path}")
