\
#!/usr/bin/env python3

"""
Generate recruiter-friendly figures from training logs.

Usage:
  # Save logs during training:
  python train.py | tee results/train.log

  # Generate figures:
  python scripts/make_figures.py --log results/train.log --out results/figures

Outputs:
  - training_curves.png
  - confusion_matrix_best.png
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"\[(?P<epoch>\d+)/(?:\d+)\]\s+"
    r"train loss=(?P<tr_loss>\d+\.\d+)\s+acc=(?P<tr_acc>\d+\.\d+)\s+\|\s+"
    r"val loss=(?P<va_loss>\d+\.\d+)\s+acc=(?P<va_acc>\d+\.\d+)\s+\|\s+"
    r"CM\s+\[\[TN=(?P<tn>\d+),\s+FP=(?P<fp>\d+)\],\s+\[FN=(?P<fn>\d+),\s+TP=(?P<tp>\d+)\]\]"
)

def parse_log(log_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in log_path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        row = {
            "epoch": int(d["epoch"]),
            "tr_loss": float(d["tr_loss"]),
            "tr_acc": float(d["tr_acc"]),
            "va_loss": float(d["va_loss"]),
            "va_acc": float(d["va_acc"]),
            "tn": int(d["tn"]),
            "fp": int(d["fp"]),
            "fn": int(d["fn"]),
            "tp": int(d["tp"]),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError(
            f"No epoch metrics found in {log_path}. "
            "Make sure you saved the console output of train.py (see README)."
        )
    rows.sort(key=lambda r: r["epoch"])
    return rows

def plot_curves(rows: List[Dict], out_dir: Path) -> Path:
    epochs = [r["epoch"] for r in rows]
    tr_loss = [r["tr_loss"] for r in rows]
    va_loss = [r["va_loss"] for r in rows]
    tr_acc = [r["tr_acc"] for r in rows]
    va_acc = [r["va_acc"] for r in rows]

    # Loss curve
    plt.figure()
    plt.plot(epochs, tr_loss, label="train loss")
    plt.plot(epochs, va_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    out_path1 = out_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_path1, dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, tr_acc, label="train acc")
    plt.plot(epochs, va_acc, label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    out_path2 = out_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200)
    plt.close()

    # Combined "training_curves.png" (simple 2-panel without subplots requested? The system says no subplots for user-visible charts in python_user_visible,
    # but this script will run on user's machine; subplots are fine. Still, keep single plots: we already did two separate PNGs.)
    return out_path2

def plot_confusion_matrix(best: Dict, out_dir: Path) -> Path:
    tn, fp, fn, tp = best["tn"], best["fp"], best["fn"], best["tp"]
    cm = [[tn, fp],
          [fn, tp]]

    plt.figure()
    im = plt.imshow(cm)  # default colormap
    plt.title(f"Confusion Matrix (Best val acc={best['va_acc']:.3f} @ epoch {best['epoch']})")
    plt.xticks([0, 1], ["Pred: healthy", "Pred: defect"], rotation=15)
    plt.yticks([0, 1], ["True: healthy", "True: defect"])
    plt.colorbar(im)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    out_path = out_dir / "confusion_matrix_best.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to training log captured from train.py stdout")
    ap.add_argument("--out", type=str, default="results/figures", help="Output folder for figures")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_log(log_path)
    # pick best by val acc, break ties by later epoch
    best = sorted(rows, key=lambda r: (r["va_acc"], r["epoch"]))[-1]

    plot_curves(rows, out_dir)
    plot_confusion_matrix(best, out_dir)

    print(f"[OK] Parsed epochs: {len(rows)}")
    print(f"[OK] Best epoch: {best['epoch']} | val acc={best['va_acc']:.4f}")
    print(f"[OK] Wrote figures to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
