#!/usr/bin/env python3
"""Overlay the largest singular value of dz2/du, dz2/dz1, and dz2/dz2."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_frame_spectrum")
SERIES = (
    ("U-Z2", r"$\partial z_2/\partial u$"),
    ("Z1-Z2", r"$\partial z_2/\partial z_1$"),
    ("Z2-Z2", r"$\partial z_2/\partial z_2$"),
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    stems = None
    for key, label in SERIES:
        blob = np.load(BASE / f"n20_svmax_{key}" / "singular_values.npz")
        y = np.asarray(blob["singular_values"], dtype=np.float64)[:, 0]
        x = np.asarray(blob["high_stems"])
        if stems is None:
            stems = x
        elif not np.array_equal(stems, x):
            raise SystemExit(f"stem mismatch for {key}")
        ax.plot(np.arange(y.size), y, "o-", label=f"{label}  mean {np.mean(y):.3g}")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("largest singular value")
    ax.set_title(r"L=3 largest singular values of $\partial z_2$")
    ax.legend()
    fig.tight_layout()
    out = BASE / "n20_svmax_z2_partials" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "max_singular_values.png", dpi=140)
    print(f"wrote {out / 'max_singular_values.png'}", flush=True)


if __name__ == "__main__":
    main()
