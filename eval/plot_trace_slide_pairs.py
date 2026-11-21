#!/usr/bin/env python3
"""Slide PNGs for the four trace comparisons.

Two series use the plasma and cool pairing from the eigenvalue strip
plots. Each dot is one saved snapshot. The black mark is the median and
the interquartile interval of that one trajectory.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/pscratch/sd/d/divyam/hine_runs/trace_compare/n20_loader")
OUT = Path("/global/homes/d/divyam/trace_compare_slides")
ZIP = Path("/global/homes/d/divyam/trace_compare_slides.zip")

LABELS = {
    "l1_k4": r"L=1, $k_f=4$",
    "l2_k4": r"L=2, $k_f=4$",
    "l1_k5": r"L=1, $k_f=5$",
    "l2_k5": r"L=2, $k_f=5$",
}
COLOR_A = plt.get_cmap("plasma_r")(0.32)
COLOR_B = plt.get_cmap("cool_r")(0.42)

SHARED = [
    ("01_direct_trace", "tau", "Normalized direct trace", True),
    ("02_energy", "energy", "Isolated energy change", True),
    ("03_error_direction_trace", "dir_tau", "Error-direction direct trace", True),
    ("04_error_direction_energy", "dir_energy", "Error-direction energy change", True),
    ("05_twostep_direct", "direct", "Two-step direct energy", False),
    ("06_twostep_total", "total", "Two-step total energy", False),
    ("07_scale_P", "scale_P", r"$\mathrm{Tr}(PTP)/\mathrm{rank}(P)$", False),
    ("08_scale_Q", "scale_Q", r"$\mathrm{Tr}(QTQ)/\mathrm{rank}(Q)$", False),
]
LATENT = [
    ("09_latent_trace", "tau_K", r"$\mathrm{Tr}(K-I)/M$", True),
    ("10_latent_energy", "energy_K", "Latent energy change", True),
    ("11_frozen_loop_trace", "tau_BC", r"Frozen $\mathrm{Tr}(B_n C_n)/N$", True),
    ("12_consecutive_loop_trace", "tau_loop_next", r"Consecutive $\mathrm{Tr}(B_{n+1}C_n)/N$", True),
    ("13_b_gain", "b_gain", r"$\|B\|_F^2/M$", False),
    ("14_interference", "interference", "Two-step interference", True),
    ("15_loop_energy", "loop", "Loop energy", False),
    ("16_loop_net", "loop_net", "Loop net effect", True),
    ("17_loop_along_error", "dir_bc", r"$e^{T}BCe/\|e\|^2$", True),
    ("18_latent_along_eta", "dir_k", "Latent trace along $\\eta$", True),
]
LINES = [
    ("19_rollout_error", "e_norm", r"Free-rollout $\|e_n\|$"),
    ("20_retention_gap", "R_value", r"$R_n=\|e_n\|-I_n$"),
    ("21_accumulated_residual", "I_norm", r"$I_n$"),
    ("22_rho", "rho_norm", r"$\|\rho_n\|$"),
    ("23_teacher_residual", "eps_u_norm", r"Teacher residual $\|\epsilon^u_n\|$"),
]


def _jitter(stem: int, scale: float = 0.12) -> float:
    value = (int(stem) * 1103515245 + 12345) & 0x7FFFFFFF
    unit = (value % 10000) / 10000.0 - 0.5
    return float(unit * scale)


def _load(name: str) -> dict:
    stored = np.load(ROOT / name / "metrics.npz")
    return {key: np.asarray(stored[key]) for key in stored.files}


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _strip(path: Path, runs: list[tuple[str, dict, object]], key: str, ylabel: str, signed: bool, title: str, force_zero: bool = False) -> None:
    usable = []
    for label, run, color in runs:
        if key not in run:
            continue
        values = _finite(run[key])
        if values.size:
            usable.append((label, run, color, values))
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    collected = []
    for position, (label, run, color, values) in enumerate(usable):
        stems = np.asarray(run["stem"])
        series = np.asarray(run[key], dtype=np.float64)
        for stem, value in zip(stems, series):
            if not np.isfinite(value):
                continue
            ax.scatter(
                position + _jitter(int(stem)),
                value,
                s=26,
                color=color,
                alpha=0.8,
                linewidths=0,
                zorder=3,
            )
        collected.append(values)
        low, med, high = np.percentile(values, [25, 50, 75])
        ax.plot([position, position], [low, high], color="black", lw=2.2, zorder=4)
        ax.plot([position - 0.16, position + 0.16], [med, med], color="black", lw=2.0, zorder=5)
        ax.scatter([], [], s=26, color=color, label=label)
    pads = np.concatenate(collected)
    low = float(pads.min())
    high = float(pads.max())
    if force_zero or (signed and low < 0.0 < high):
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    pad = 0.08 * span if span else 0.05 * max(abs(high), 1.0)
    ax.set_ylim(low - pad, high + pad)
    if signed and low - pad <= 0.0 <= high + pad:
        ax.axhline(0.0, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([item[0] for item in usable])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _lines(path: Path, runs: list[tuple[str, dict, object]], key: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    drew = False
    for label, run, color in runs:
        if key not in run:
            continue
        values = np.asarray(run[key], dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        ax.plot(np.arange(values.size), values, marker="o", ms=4.5, lw=1.4, color=color, label=label)
        drew = True
    if not drew:
        plt.close(fig)
        return
    if key == "R_value":
        ax.axhline(0.0, color="0.35", ls="--", lw=0.8)
    ax.set_xlabel("loader step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _memory(path: Path, runs: list[tuple[str, dict, object]], key: str, title: str) -> None:
    usable = []
    for label, run, color in runs:
        if key not in run:
            continue
        values = np.asarray(run[key], dtype=np.float64)
        if values.ndim == 2 and np.isfinite(values).any():
            usable.append((label, run, color, values))
    if not usable:
        return
    horizons = np.asarray(usable[0][1]["horizons"])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for series_index, (label, run, color, values) in enumerate(usable):
        shift = (series_index - (len(usable) - 1) / 2.0) * 0.16
        stems = np.asarray(run["stem"])
        medians = []
        for horizon_index, horizon in enumerate(horizons):
            column = values[:, horizon_index]
            finite = column[np.isfinite(column)]
            medians.append(float(np.median(finite)) if finite.size else np.nan)
            center = horizon_index + shift
            for stem, value in zip(stems, column):
                if np.isfinite(value):
                    ax.scatter(center + _jitter(int(stem), 0.05), value, s=22, color=color, alpha=0.75, linewidths=0, zorder=3)
        ax.plot(np.arange(len(horizons)) + shift, medians, color=color, lw=1.6, marker="o", ms=4.5, label=label, zorder=4)
    finite = np.concatenate([values[np.isfinite(values)] for _label, _run, _color, values in usable])
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    pad = 0.12 * span if span else 0.05
    ax.set_ylim(max(0.0, low - pad), high + pad)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([str(int(horizon)) for horizon in horizons])
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel(r"latent gain $\|K\cdots\|_F^2/M$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bundle(folder: Path, names: list[str], title: str, include_latent: bool) -> None:
    runs = []
    colors = (COLOR_A, COLOR_B)
    loaded = [_load(name) for name in names]
    for name, run, color in zip(names, loaded, colors):
        runs.append((LABELS[name], run, color))
    panels = list(SHARED)
    if include_latent:
        panels.extend(LATENT)
    for stem, key, ylabel, signed in panels:
        _strip(folder / f"{stem}.png", runs, key, ylabel, signed, title)
    for stem, key, ylabel in LINES:
        _lines(folder / f"{stem}.png", runs, key, ylabel, title)
    if include_latent:
        _memory(folder / "24_memory_varying.png", runs, "memory_varying", title + ", time-varying K")
        _memory(folder / "25_memory_frozen.png", runs, "memory_frozen", title + ", frozen K")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    _bundle(OUT / "01_k4_L1_vs_L2", ["l1_k4", "l2_k4"], r"Wavenumber 4: L=1 and L=2", True)
    _bundle(OUT / "02_k5_L1_vs_L2", ["l1_k5", "l2_k5"], r"Wavenumber 5: L=1 and L=2", True)
    _bundle(OUT / "03_L1_k4_vs_k5", ["l1_k4", "l1_k5"], r"L=1: $k_f=4$ and $k_f=5$", False)
    _bundle(OUT / "04_L2_k4_vs_k5", ["l2_k4", "l2_k5"], r"L=2: $k_f=4$ and $k_f=5$", True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(OUT.rglob("*.png")):
            handle.write(path, path.relative_to(OUT.parent))
    print(f"zip {ZIP} files {sum(1 for _ in OUT.rglob('*.png'))}")


if __name__ == "__main__":
    main()
