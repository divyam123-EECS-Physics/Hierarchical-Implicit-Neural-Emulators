#!/usr/bin/env python3
"""CPU-side plots for the Jacobian smoke analysis outputs.

Same figures as plot_jacobian_smoke.ipynb, but as a script safe to run on a
Perlmutter CPU interactive node (not the login node / Cursor IDE).

Example (interactive CPU, shared):
  salloc -A m5326 -C cpu -q shared_interactive -t 00:30:00 -n 1 -c 8 --mem=32G
  module load conda && conda activate hine
  cd /global/homes/d/divyam/TACSLAB/Hierarchical-Implicit-Neural-Emulators
  python plot_jacobian_smoke.py

Or full exclusive interactive CPU node:
  salloc -A m5326 -C cpu -q interactive -t 00:30:00 -N 1
  ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUT = Path("/pscratch/sd/d/divyam/hine_runs/jacobian_smoke")
STEPS = [0, 5, 10, 19]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=STEPS,
        help="Horizon steps for field galleries",
    )
    return p.parse_args()


def field2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    return a[0] if a.ndim == 3 else a


def plot_scalars(traj: np.lib.npyio.NpzFile, plot_dir: Path) -> None:
    starts = traj["starts"]
    file_indices = traj["file_indices"]
    horizon = traj["ar_radius"].shape[1]
    t = np.arange(horizon)
    labels = [f"start={s} idx={int(fi)}" for s, fi in zip(starts, file_indices)]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(starts), endpoint=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        (r"$|\lambda|_{\max}$", "ar_radius", "tf_radius"),
        ("MSE", "ar_mse", "tf_residual_mse"),
        (r"$\|e\|$", "ar_pert_norm", "tf_pert_norm"),
    ]
    for ax, (title, ar_key, tf_key) in zip(axes, panels):
        for i, (c, lab) in enumerate(zip(cmap, labels)):
            ax.plot(t, traj[ar_key][i], color=c, lw=2, label=lab)
            ax.plot(t, traj[tf_key][i], color=c, lw=1.5, ls="--", alpha=0.85)
        ax.set_xlabel("horizon step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].text(
        0.02,
        0.02,
        "solid=AR, dashed=TF",
        transform=axes[0].transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.suptitle("Jacobian smoke: scalars vs horizon", fontsize=12)
    out = plot_dir / "scalars_vs_horizon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_eps_check(traj: np.lib.npyio.NpzFile, plot_dir: Path) -> None:
    eps = float(traj["eps"])
    eps_check = float(traj["eps_check"])
    r_eps = float(traj["radius_eps"])
    r_eps_check = float(traj["radius_eps_check"])
    print(f"radius(eps={eps:g}) = {r_eps:.6g}")
    print(f"radius(eps={eps_check:g}) = {r_eps_check:.6g}")
    print(f"ratio = {r_eps / max(r_eps_check, 1e-12):.4g}")

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.bar([f"eps={eps:g}", f"eps={eps_check:g}"], [r_eps, r_eps_check], color=["C0", "C1"])
    ax.set_ylabel(r"$|\lambda|_{\max}$")
    ax.set_title("FD eps check (first start, after warmup)")
    ax.grid(True, axis="y", alpha=0.3)
    out = plot_dir / "eps_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_start_gallery(
    h5: h5py.File,
    start_i: int,
    start: int,
    file_idx: int,
    steps: list[int],
    plot_dir: Path,
) -> None:
    truth = h5["truth_states"][start_i]
    ar = h5["autoreg_pred"][start_i]
    tf = h5["net_pred_truth"][start_i]
    res_ar = h5["residual_ar"][start_i]
    e_vec = h5["e_vec"][start_i]

    truth_stack = np.concatenate([field2d(truth[s + 1])[None] for s in steps], axis=0)
    vmax = max(float(np.percentile(np.abs(truth_stack), 99)), 1e-6)
    row_names = ["truth t+1", "AR pred", "AR residual", "TF pred", r"$|e_{AR}|$"]

    fig, axes = plt.subplots(
        len(row_names),
        len(steps),
        figsize=(3.0 * len(steps), 2.4 * len(row_names)),
        constrained_layout=True,
    )
    for j, s in enumerate(steps):
        panels = [
            (field2d(truth[s + 1]), "bwr", -vmax, vmax),
            (field2d(ar[s]), "bwr", -vmax, vmax),
            (field2d(res_ar[s]), "bwr", -vmax, vmax),
            (field2d(tf[s]), "bwr", -vmax, vmax),
            (np.abs(field2d(e_vec[s])), "magma", 0.0, None),
        ]
        for i, ((img, cmap, vmin, vmax_i), name) in enumerate(zip(panels, row_names)):
            ax = axes[i, j]
            if vmax_i is None:
                vmax_i = float(np.percentile(img, 99.5)) or 1e-6
                vmin = 0.0
            im = ax.imshow(
                img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i, aspect="equal"
            )
            if i == 0:
                ax.set_title(f"step {s}")
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == len(steps) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"start={int(start)}  file_index={int(file_idx)}", fontsize=12)
    out = plot_dir / f"fields_start{int(start)}_idx{int(file_idx)}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    del truth, ar, tf, res_ar, e_vec
    print("wrote", out)


def plot_latents_first_start(
    h5: h5py.File,
    start: int,
    file_idx: int,
    steps: list[int],
    plot_dir: Path,
) -> None:
    mid = h5["ar_mid"][0]
    low = h5["ar_low"][0]
    fig, axes = plt.subplots(
        2, len(steps), figsize=(3.0 * len(steps), 5.0), constrained_layout=True
    )
    for j, s in enumerate(steps):
        m = field2d(mid[s])
        l = field2d(low[s])
        vm = float(np.percentile(np.abs(m), 99)) or 1e-6
        vl = float(np.percentile(np.abs(l), 99)) or 1e-6
        im0 = axes[0, j].imshow(m, origin="lower", cmap="bwr", vmin=-vm, vmax=vm)
        im1 = axes[1, j].imshow(l, origin="lower", cmap="bwr", vmin=-vl, vmax=vl)
        axes[0, j].set_title(f"step {s}")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
        if j == 0:
            axes[0, j].set_ylabel("AR mid (32x32)")
            axes[1, j].set_ylabel("AR low (8x8)")
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)

    fig.suptitle(f"AR latents  start={int(start)}  file_index={int(file_idx)}")
    out = plot_dir / f"latents_start{int(start)}_idx{int(file_idx)}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    steps = list(args.steps)

    summary_path = out_dir / "summary.txt"
    npz_path = out_dir / "trajectories.npz"
    h5_path = out_dir / "jacobian_fields.h5"
    for path in (summary_path, npz_path, h5_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(summary_path.read_text())
    traj = np.load(npz_path)
    print("npz keys:", list(traj.keys()))
    starts = traj["starts"]
    file_indices = traj["file_indices"]

    plot_scalars(traj, plot_dir)
    plot_eps_check(traj, plot_dir)

    with h5py.File(h5_path, "r") as h5:
        print("h5 keys:", list(h5.keys()))
        for i, (s, fi) in enumerate(zip(starts, file_indices)):
            plot_start_gallery(h5, i, int(s), int(fi), steps, plot_dir)
        plot_latents_first_start(
            h5, int(starts[0]), int(file_indices[0]), steps, plot_dir
        )

    print("All plots under", plot_dir)
    print("files:", sorted(p.name for p in plot_dir.iterdir()))


if __name__ == "__main__":
    main()
