#!/usr/bin/env python3
"""Plots for the L=2 energy budget and the separate directional analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for entry in (_REPO, _EVAL):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from energy_metric import LOW_K_MAX, MID_K_MAX


def _series(ax, x, y, label: str, color: str | None = None) -> None:
    ax.plot(x, y, label=label, color=color, linewidth=1.4)


def _aggregate(ax, x, values: np.ndarray, label: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_valid = np.sum(np.isfinite(arr), axis=0)
    n_label = int(np.max(n_valid)) if n_valid.size else 0
    mean = np.full(arr.shape[1], np.nan, dtype=np.float64)
    low = np.full(arr.shape[1], np.nan, dtype=np.float64)
    high = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for index in range(arr.shape[1]):
        column = arr[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        mean[index] = float(np.mean(column))
        low[index] = float(np.percentile(column, 10))
        high[index] = float(np.percentile(column, 90))
    finite = np.isfinite(mean)
    if np.any(finite):
        ax.fill_between(x, np.where(finite, low, mean), np.where(finite, high, mean), alpha=0.25, linewidth=0)
    ax.plot(x, mean, linewidth=1.6, label=f"{label} (n={n_label}/{arr.shape[0]})")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _band_series(shells: np.ndarray, shell_k: np.ndarray) -> dict[str, np.ndarray]:
    """shells is (..., n_k). Returns low/middle/high sums on the leading axes."""
    masks = {
        "low": (shell_k > 0) & (shell_k <= LOW_K_MAX),
        "middle": (shell_k > LOW_K_MAX) & (shell_k <= MID_K_MAX),
        "high": shell_k > MID_K_MAX,
    }
    return {name: shells[..., mask].sum(axis=-1) for name, mask in masks.items()}


def plot_budget(npz_path: Path, plot_dir: Path) -> None:
    data = np.load(npz_path)
    lead = np.asarray(data["lead"])
    step = np.asarray(data["step"])
    shell_k = np.asarray(data["shell_k"])
    n_win = int(np.asarray(data["E_ref"]).shape[0])
    plot_dir.mkdir(parents=True, exist_ok=True)

    for index in range(n_win):
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex="col")
        _series(axes[0, 0], lead, data["E_ref"][index], "reference")
        _series(axes[0, 0], lead, data["E_pred"][index], "predicted")
        axes[0, 0].set_ylabel("kinetic energy")
        axes[0, 0].set_title(f"window {index} KE")
        axes[0, 0].legend(fontsize=8)
        _series(axes[0, 1], lead, data["signed_energy_error"][index], "E(uhat)-E(u)")
        _series(axes[0, 1], lead, data["L"][index], "L")
        _series(axes[0, 1], lead, data["Q"][index], "Q")
        axes[0, 1].set_title("exact energy split")
        axes[0, 1].legend(fontsize=8)
        _series(axes[1, 0], lead, data["d_E"][index], "d_E")
        axes[1, 0].axhline(float(data["d_E_threshold"]), color="0.3", linestyle="--", label="d_E=5")
        exit_at = int(data["first_exit"][index])
        axes[1, 0].set_title(f"d_E, first exit {exit_at}")
        axes[1, 0].legend(fontsize=8)
        _series(axes[1, 1], lead, data["vorticity_mse"][index], "vorticity MSE")
        axes[1, 1].set_title("vorticity MSE")
        _series(axes[2, 0], step, data["E_H"][index], "E_H")
        _series(axes[2, 0], step, data["E_Z"][index], "E_Z")
        _series(axes[2, 0], step, data["C_HZ"][index], "C_HZ")
        axes[2, 0].axhline(0.0, color="0.5", linewidth=0.8)
        axes[2, 0].set_title("quadratic pathway terms")
        axes[2, 0].set_xlabel("rollout step")
        axes[2, 0].legend(fontsize=8)
        _series(axes[2, 1], step, data["rho_HZ"][index], r"$\rho_{HZ}$")
        _series(axes[2, 1], step, data["eta_cancel"][index], r"$\eta$")
        _series(axes[2, 1], step, data["cos_euclid"][index], r"$\cos(a,b)$")
        n_rho = int(np.sum(data["valid_rho"][index]))
        axes[2, 1].set_ylim(-1.05, 1.05)
        axes[2, 1].set_title(rf"alignment, valid $\rho$ {n_rho}/{step.size}")
        axes[2, 1].set_xlabel("rollout step")
        axes[2, 1].legend(fontsize=8)
        _save(fig, plot_dir / f"window_{index:02d}_budget.png")

        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        field = np.asarray(data["shell_C_HZ"][index], dtype=np.float64)
        limit = float(np.nanmax(np.abs(field))) if field.size else 1.0
        limit = max(limit, 1e-30)
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(shell_k[0] - 0.5, shell_k[-1] + 0.5, step[0] - 0.5, step[-1] + 0.5),
        )
        ax.axvline(LOW_K_MAX, color="0.2", linestyle="--", linewidth=0.8)
        ax.axvline(MID_K_MAX, color="0.2", linestyle=":", linewidth=0.8)
        ax.set_xlabel("wavenumber bin")
        ax.set_ylabel("rollout step")
        ax.set_title(f"window {index} signed C_HZ by shell")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, plot_dir / f"window_{index:02d}_shells.png")

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0))
    _aggregate(axes[0, 0], lead, data["E_ref"], "reference")
    _aggregate(axes[0, 0], lead, data["E_pred"], "predicted")
    axes[0, 0].set_title("KE across windows")
    axes[0, 0].legend(fontsize=8)
    _aggregate(axes[0, 1], lead, data["signed_energy_error"], "signed energy error")
    _aggregate(axes[0, 1], lead, data["L"], "L")
    _aggregate(axes[0, 1], lead, data["Q"], "Q")
    axes[0, 1].set_title("energy split, 10–90% band")
    axes[0, 1].legend(fontsize=8)
    _aggregate(axes[1, 0], lead, data["d_E"], "d_E")
    axes[1, 0].axhline(float(data["d_E_threshold"]), color="0.3", linestyle="--", label="d_E=5")
    axes[1, 0].set_title("d_E")
    axes[1, 0].legend(fontsize=8)
    _aggregate(axes[1, 1], lead, data["vorticity_mse"], "vorticity MSE")
    axes[1, 1].set_title("vorticity MSE")
    _aggregate(axes[2, 0], step, data["E_H"], "E_H")
    _aggregate(axes[2, 0], step, data["E_Z"], "E_Z")
    _aggregate(axes[2, 0], step, data["C_HZ"], "C_HZ")
    axes[2, 0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[2, 0].set_xlabel("rollout step")
    axes[2, 0].set_title("pathway terms")
    axes[2, 0].legend(fontsize=8)
    _aggregate(axes[2, 1], step, data["rho_HZ"], r"$\rho_{HZ}$")
    _aggregate(axes[2, 1], step, data["eta_cancel"], r"$\eta$")
    n_rho = int(np.sum(data["valid_rho"]))
    axes[2, 1].set_ylim(-1.05, 1.05)
    axes[2, 1].set_xlabel("rollout step")
    axes[2, 1].set_title(rf"alignment, valid $\rho$ {n_rho}/{data['valid_rho'].size}")
    axes[2, 1].legend(fontsize=8)
    _save(fig, plot_dir / "aggregate_budget.png")

    bands = _band_series(np.asarray(data["shell_C_HZ"]), shell_k)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, series in bands.items():
        _aggregate(ax, step, series, name)
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("C_HZ")
    ax.set_title(f"C_HZ bands: low ≤{LOW_K_MAX}, middle ≤{MID_K_MAX}, high above")
    ax.legend(fontsize=8)
    _save(fig, plot_dir / "aggregate_bands.png")

    mean_shell = np.nanmean(np.asarray(data["shell_C_HZ"], dtype=np.float64), axis=0)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    limit = max(float(np.nanmax(np.abs(mean_shell))), 1e-30)
    image = ax.imshow(
        mean_shell,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=(shell_k[0] - 0.5, shell_k[-1] + 0.5, step[0] - 0.5, step[-1] + 0.5),
    )
    ax.axvline(LOW_K_MAX, color="0.2", linestyle="--", linewidth=0.8)
    ax.axvline(MID_K_MAX, color="0.2", linestyle=":", linewidth=0.8)
    ax.set_xlabel("wavenumber bin")
    ax.set_ylabel("rollout step")
    ax.set_title(f"mean signed C_HZ, {n_win} windows")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, plot_dir / "aggregate_shells.png")


def plot_svd(save_dir: Path, plot_dir: Path) -> None:
    paths = sorted(Path(save_dir).glob("state_w*_s*.npz"))
    if not paths:
        return
    rows = []
    for path in paths:
        data = np.load(path)
        rows.append(data)
    labels = [f"w{int(data['window'])}/s{int(data['step'])}" for data in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, key, title in (
        (axes[0, 0], "A_values", "weighted A singular values"),
        (axes[0, 1], "B_values", "weighted B singular values"),
        (axes[1, 0], "M_values", "M_HZ singular values"),
        (axes[1, 1], "Q_E_values", "Q_E eigenvalues"),
    ):
        for label, data in zip(labels, rows):
            values = np.asarray(data[key], dtype=np.float64)
            ax.plot(np.arange(1, values.size + 1), values, marker="o", label=label)
        ax.set_xlabel("index, largest first")
        ax.set_title(title)
        ax.legend(fontsize=7)
    _save(fig, plot_dir / "spectrum.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, data in zip(labels, rows):
        angles = np.degrees(np.asarray(data["principal_angles"], dtype=np.float64))
        ax.plot(np.arange(1, angles.size + 1), angles, marker="o", label=label)
    ax.set_xlabel("principal-angle index")
    ax.set_ylabel("degrees")
    ax.set_title("angles between leading left subspaces of A and B")
    ax.legend(fontsize=7)
    _save(fig, plot_dir / "principal_angles.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = ["fraction_e_H", "fraction_e_Z", "fraction_a", "fraction_b"]
    xpos = np.arange(len(rows))
    width = 0.18
    for offset, name in enumerate(names):
        vals = [float(np.asarray(data[name])) for data in rows]
        ax.bar(xpos + (offset - 1.5) * width, vals, width=width, label=name)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("captured fraction")
    ax.set_title("realized errors and responses in the leading subspaces")
    ax.legend(fontsize=8)
    _save(fig, plot_dir / "subspace_fractions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-npz", type=Path, default=None)
    parser.add_argument("--svd-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.budget_npz is not None:
        plot_budget(args.budget_npz, args.plot_dir)
    if args.svd_dir is not None:
        plot_svd(args.svd_dir, args.plot_dir)


if __name__ == "__main__":
    main()
