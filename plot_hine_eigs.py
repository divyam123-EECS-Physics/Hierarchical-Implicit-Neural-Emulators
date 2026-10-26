#!/usr/bin/env python3
"""Argand-plane eigenvalue plots for eval_hine_eigs.py outputs.

Style matches the eigenanalysis paper (Im vs Re, dashed unit circle).
HINE is a direct-step map, so only one series is drawn.

Run on a CPU interactive node, not the login node / Cursor IDE.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path(
    "/pscratch/sd/d/divyam/hine_runs/jacobian_paper_1gpu/eigs_steps"
)
FNAME_RE = re.compile(r"eigs_start(\d+)_step(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eigs-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="PNG directory (default: <eigs-dir>/plots)",
    )
    return p.parse_args()


def load_spectra(eigs_dir: Path) -> list[dict]:
    specs = []
    for path in sorted(eigs_dir.glob("eigs_start*_step*.npz")):
        m = FNAME_RE.search(path.name)
        if m is None:
            continue
        data = np.load(path)
        vals = np.asarray(data["eigenvalues"])
        specs.append(
            {
                "path": path,
                "start": int(data["start"]),
                "file_index": int(data["file_index"]),
                "step": int(data["step"]),
                "eigenvalues": vals,
                "radius_arnoldi": float(data["radius_arnoldi"]),
                "radius_power": float(data["radius_power"]),
            }
        )
    specs.sort(key=lambda d: (d["start"], d["step"]))
    return specs


def draw_unit_circle(ax: plt.Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1.0, alpha=0.55, label="unit circle")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color="0.7", lw=0.6)


def scatter_eigs(ax: plt.Axes, vals: np.ndarray) -> None:
    re = vals.real
    im = vals.imag
    mag = np.abs(vals)
    outside = mag > 1.0 + 1e-8
    inside = ~outside
    if np.any(inside):
        ax.scatter(
            re[inside],
            im[inside],
            s=36,
            facecolors="C0",
            edgecolors="C0",
            zorder=3,
            label=r"$|\lambda|\leq 1$",
        )
    if np.any(outside):
        ax.scatter(
            re[outside],
            im[outside],
            s=48,
            marker="s",
            facecolors="none",
            edgecolors="C1",
            linewidths=1.4,
            zorder=4,
            label=r"$|\lambda|>1$",
        )


def plot_one(spec: dict, out_path: Path) -> None:
    vals = spec["eigenvalues"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

    ax = axes[0]
    draw_unit_circle(ax)
    scatter_eigs(ax, vals)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title("full scale")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    draw_unit_circle(ax)
    scatter_eigs(ax, vals)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title("unit-circle zoom")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"start={spec['start']} idx={spec['file_index']} step={spec['step']}  "
        f"|λ|_arnoldi={spec['radius_arnoldi']:.4g}  "
        f"|λ|_power={spec['radius_power']:.4g}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def plot_summary(specs: list[dict], out_path: Path) -> None:
    """One panel per start at the earliest available step (usually 0)."""
    by_start: dict[int, list[dict]] = {}
    for s in specs:
        by_start.setdefault(s["start"], []).append(s)

    starts = sorted(by_start)
    if not starts:
        return
    n = len(starts)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, start in zip(axes, starts):
        # Prefer step 0; else the first available step for that start.
        candidates = sorted(by_start[start], key=lambda d: d["step"])
        chosen = next((c for c in candidates if c["step"] == 0), candidates[0])
        draw_unit_circle(ax)
        scatter_eigs(ax, chosen["eigenvalues"])
        ax.set_title(
            f"start={start} idx={chosen['file_index']}\n"
            f"step={chosen['step']} |λ|={chosen['radius_arnoldi']:.4g}"
        )
        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Jacobian eigenvalues (direct-step map)", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    args = parse_args()
    eigs_dir = args.eigs_dir
    out_dir = args.out_dir or (eigs_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_spectra(eigs_dir)
    if not specs:
        print(f"No eigs_start*_step*.npz under {eigs_dir}")
        return
    print(f"found {len(specs)} spectra in {eigs_dir}")

    for spec in specs:
        name = (
            f"eigs_start{spec['start']}_idx{spec['file_index']}"
            f"_step{spec['step']}.png"
        )
        plot_one(spec, out_dir / name)

    plot_summary(specs, out_dir / "eigs_summary.png")


if __name__ == "__main__":
    main()
