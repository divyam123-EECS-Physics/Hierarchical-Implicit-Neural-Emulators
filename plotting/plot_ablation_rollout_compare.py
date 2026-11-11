#!/usr/bin/env python3
"""Overlay fine-field autoregressive MSE for L=2 and the L=1 ablations."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l2-npz", required=True, help="Plus rollout_mse.npz (ar_mse_high)")
    p.add_argument("--l1-npz", required=True, help="ablation_mse.npz for plain L=1")
    p.add_argument("--aux-npz", required=True, help="ablation_mse.npz for L=1 aux")
    p.add_argument("--unroll-npz", required=True, help="ablation_mse.npz for P=5 unroll")
    p.add_argument(
        "--output",
        default="/pscratch/sd/d/divyam/hine_runs/ablation_compare/mse_vs_lead.png",
    )
    return p.parse_args()


def mean_curve(path: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    mse = np.asarray(data[key], dtype=np.float64)
    lead = np.asarray(data["lead"])
    if lead.shape[0] != mse.shape[1]:
        lead = np.arange(1, mse.shape[1] + 1)
    return lead, mse.mean(axis=0)


def main() -> None:
    args = parse_args()
    series = [
        ("L=2 HINE", *mean_curve(args.l2_npz, "ar_mse_high")),
        ("L=1", *mean_curve(args.l1_npz, "mse")),
        ("L=1 aux", *mean_curve(args.aux_npz, "mse")),
        ("L=1 unroll P=5 (partial checkpoint)", *mean_curve(args.unroll_npz, "mse")),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, lead, mean in series:
        ax.plot(lead, mean, lw=2, label=label)
    ax.set_xlabel("lead")
    ax.set_ylabel("fine-field MSE")
    ax.set_title("Autoregressive rollout error")
    ax.legend()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
