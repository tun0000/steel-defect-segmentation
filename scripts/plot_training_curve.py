#!/usr/bin/env python3
"""Plot training/validation curves from an Ultralytics results.csv.

Example:
    uv run python scripts/plot_training_curve.py \\
        --csv reports/training_results.csv --out reports/figures/training_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="path to Ultralytics results.csv")
    parser.add_argument("--out", type=Path, required=True, help="output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    # Ultralytics picks best.pt by fitness = box mAP50-95 + mask mAP50-95 (see
    # SegmentMetrics.fitness), not mask mAP50-95 alone -- match that here so
    # "best epoch" agrees with which checkpoint was actually saved as best.pt.
    fitness = df["metrics/mAP50-95(B)"] + df["metrics/mAP50-95(M)"]
    best_epoch = df.loc[fitness.idxmax(), "epoch"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(df["epoch"], df["train/box_loss"], label="box_loss")
    ax1.plot(df["epoch"], df["train/seg_loss"], label="seg_loss")
    ax1.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    ax1.set_ylabel("training loss")
    ax1.legend()
    ax1.set_title("Training losses")

    ax2.plot(df["epoch"], df["metrics/mAP50(M)"], label="mask mAP50")
    ax2.plot(df["epoch"], df["metrics/mAP50-95(M)"], label="mask mAP50-95")
    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best epoch ({int(best_epoch)})")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mask mAP (val)")
    ax2.legend()
    ax2.set_title("Validation mask mAP")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved chart to {args.out} (best epoch: {int(best_epoch)})")


if __name__ == "__main__":
    main()
