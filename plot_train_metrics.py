#!/usr/bin/env python3
"""Plot train and validation loss vs epoch from train_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = Path(
    "/pscratch/sd/d/divyam/hine_runs/paper_1gpu/csv/train_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG path (default: <run>/plots/loss_vs_epoch.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    epoch = data["epoch"]
    out = args.out
    if out is None:
        out = args.csv.parent.parent / "plots" / "loss_vs_epoch.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(epoch, data["loss"], label="train loss", lw=2)
    ax.plot(epoch, data["val_loss"], label="val loss", lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Train and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epoch, data["high_loss"], label="train high", lw=1.5)
    ax.plot(epoch, data["val_high_loss"], label="val high", lw=1.5, ls="--")
    ax.plot(epoch, data["mid_loss"], label="train mid", lw=1.5)
    ax.plot(epoch, data["val_mid_loss"], label="val mid", lw=1.5, ls="--")
    ax.plot(epoch, data["low_loss"], label="train low", lw=1.5)
    ax.plot(epoch, data["val_low_loss"], label="val low", lw=1.5, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("High / mid / low (solid=train, dashed=val)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"epochs {int(epoch[0])}..{int(epoch[-1])}  n={len(epoch)}")
    print(f"final train={data['loss'][-1]:.6g}  val={data['val_loss'][-1]:.6g}")


if __name__ == "__main__":
    main()
