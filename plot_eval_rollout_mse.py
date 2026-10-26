#!/usr/bin/env python3
"""Plot high-field MSE vs lead time from eval_hine_rollout_mse.py.

Autoregressive mean (solid) with 25-75 percentile band; teacher-forced mean dashed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path(
    "/pscratch/sd/d/divyam/hine_runs/paper_1gpu/eval_rollout_mse"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.out_dir / "rollout_mse.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    ar = np.asarray(data["ar_mse"], dtype=np.float64)
    tf = np.asarray(data["tf_mse"], dtype=np.float64)
    horizon = int(data["horizon"])
    stride = int(data["stride"])
    n_w = ar.shape[0]
    lead = np.arange(1, horizon + 1)

    ar_mean = ar.mean(axis=0)
    ar_p25 = np.percentile(ar, 25, axis=0)
    ar_p75 = np.percentile(ar, 75, axis=0)
    tf_mean = tf.mean(axis=0)

    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.fill_between(lead, ar_p25, ar_p75, color="C0", alpha=0.25, label="AR 25–75%")
    ax.plot(lead, ar_mean, color="C0", lw=2.2, label="AR mean")
    ax.plot(lead, tf_mean, color="C1", lw=2.0, ls="--", label="TF mean")
    ax.set_xlabel("lead time (steps)")
    ax.set_ylabel("high-field MSE")
    ax.set_title(
        f"Eval rollout error vs lead time  "
        f"(n={n_w} windows, stride={stride}, horizon={horizon})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    out = plot_dir / "mse_vs_lead_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    print(
        f"AR mean lead1={ar_mean[0]:.6g} lead{horizon}={ar_mean[-1]:.6g}  "
        f"TF mean lead1={tf_mean[0]:.6g} lead{horizon}={tf_mean[-1]:.6g}"
    )


if __name__ == "__main__":
    main()
