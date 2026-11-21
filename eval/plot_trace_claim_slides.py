#!/usr/bin/env python3
"""Slide PNGs for the three traces in HINE Trace Diagnostics.

Four comparisons, each in its own folder: L=1 against L=2 at wavenumber 4,
L=1 against L=2 at wavenumber 5, L=2 at wavenumber 4 against L=2 at
wavenumber 5, and L=1 at wavenumber 4 against L=1 at wavenumber 5. Latent
quantities are omitted when both sides are L=1.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_trace_slide_pairs import (
    COLOR_A,
    COLOR_B,
    LABELS,
    _finite,
    _jitter,
    _lines,
    _load,
    _strip,
)

OUT = Path("/global/homes/d/divyam/trace_compare_claim_slides")
ZIP = Path("/global/homes/d/divyam/trace_compare_claim_slides.zip")


N_FINE = 256 ** 2
RANK_P = 32 ** 2
RANK_Q = N_FINE - RANK_P


def _with_scales(name: str) -> dict:
    run = _load(name)
    scale_p = np.asarray(run["scale_P"], dtype=np.float64)
    scale_q = np.asarray(run["scale_Q"], dtype=np.float64)
    run["tau_P"] = RANK_P * (scale_p - 1.0) / N_FINE
    run["tau_Q"] = RANK_Q * (scale_q - 1.0) / N_FINE
    return run


def _orientation(path: Path, runs: list[tuple[str, dict, object]], iso_key: str, dir_key: str, ylabel: str, title: str) -> None:
    usable = []
    for label, run, color in runs:
        iso = _finite(run[iso_key]) if iso_key in run else np.array([])
        direc = _finite(run[dir_key]) if dir_key in run else np.array([])
        if iso.size or direc.size:
            usable.append((label, run, color))
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    collected = []
    for position, (label, run, color) in enumerate(usable):
        stems = np.asarray(run["stem"])
        for key, marker, shift, kind in (
            (iso_key, "o", -0.14, "isotropic"),
            (dir_key, "D", 0.14, "actual error"),
        ):
            if key not in run:
                continue
            series = np.asarray(run[key], dtype=np.float64)
            finite = series[np.isfinite(series)]
            if finite.size == 0:
                continue
            collected.append(finite)
            for stem, value in zip(stems, series):
                if np.isfinite(value):
                    ax.scatter(
                        position + shift + _jitter(int(stem), 0.06),
                        value,
                        s=26,
                        marker=marker,
                        color=color,
                        alpha=0.8,
                        linewidths=0,
                        zorder=3,
                    )
            low, med, high = np.percentile(finite, [25, 50, 75])
            center = position + shift
            ax.plot([center, center], [low, high], color="black", lw=1.6, zorder=4)
            ax.plot([center - 0.08, center + 0.08], [med, med], color="black", lw=1.8, zorder=5)
        ax.scatter([], [], s=26, color=color, label=label)
    ax.scatter([], [], s=26, marker="o", color="0.35", label="isotropic trace")
    ax.scatter([], [], s=26, marker="D", color="0.35", label="actual error")
    pads = np.concatenate(collected)
    low = float(pads.min())
    high = float(pads.max())
    span = high - low or max(abs(high), 1.0)
    near_zero = low < 0.0 < high or min(abs(low), abs(high)) < 0.3 * span
    if near_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
        span = high - low or 1.0
    pad = 0.08 * span
    ax.set_ylim(low - pad, high + pad)
    if low - pad <= 0.0 <= high + pad:
        ax.axhline(0.0, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([item[0] for item in usable])
    ax.set_ylabel(ylabel, linespacing=1.5)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _levels(names: list[str]) -> set[str]:
    return {name.split("_", 1)[0] for name in names}


def _direct_ylabel(names: list[str]) -> str:
    levels = _levels(names)
    if levels == {"l1"}:
        return r"$\mathrm{Tr}(J-I)/N$"
    if levels == {"l2"}:
        return r"$\mathrm{Tr}(A-I)/N$"
    return r"$\mathrm{Tr}(J-I)/N,\ \mathrm{Tr}(A-I)/N$"


def _scale_ylabel(names: list[str], projector: str) -> str:
    levels = _levels(names)
    if levels == {"l1"}:
        return rf"$\mathrm{{Tr}}({projector}(J-I){projector})/N$"
    if levels == {"l2"}:
        return rf"$\mathrm{{Tr}}({projector}(A-I){projector})/N$"
    return rf"$\mathrm{{Tr}}({projector}(J-I){projector})/N,\ \mathrm{{Tr}}({projector}(A-I){projector})/N$"


def _orientation_direct_ylabel(names: list[str]) -> str:
    levels = _levels(names)
    if levels == {"l1"}:
        return r"$e_n^{\top}(J-I)e_n/\|e_n\|^2$"
    if levels == {"l2"}:
        return r"$e_n^{\top}(A-I)e_n/\|e_n\|^2$"
    return r"$e_n^{\top}(J-I)e_n/\|e_n\|^2,\ e_n^{\top}(A-I)e_n/\|e_n\|^2$"


def _bundle(folder: Path, names: list[str], title: str) -> None:
    colors = (COLOR_A, COLOR_B)
    runs = [(LABELS[name], _with_scales(name), color) for name, color in zip(names, colors)]
    specs = [
        ("01_direct_trace", "tau", _direct_ylabel(names), False, False),
        ("02_latent_loop_TrBC", "tau_BC", r"$\mathrm{Tr}(BC)/N$", True, True),
        ("03_latent_memory_TrK", "tau_K", r"$\mathrm{Tr}(K-I)/M$", True, True),
        ("04_scale_coarse", "tau_P", _scale_ylabel(names, "P"), True, False),
        ("05_scale_fine", "tau_Q", _scale_ylabel(names, "Q"), True, False),
        ("06_twostep_cross_term", "interference", r"$2\mathrm{Tr}((A_{n+1}A_n)^{\top}B_{n+1}C_n)/N$", True, True),
    ]
    for stem, key, ylabel, signed, force_zero in specs:
        _strip(folder / f"{stem}.png", runs, key, ylabel, signed, title, force_zero=force_zero)
    _orientation(
        folder / "07_orientation_direct.png",
        runs,
        "tau",
        "dir_tau",
        _orientation_direct_ylabel(names),
        title,
    )
    _orientation(
        folder / "08_orientation_loop.png",
        runs,
        "tau_BC",
        "dir_bc",
        r"$e_n^{\top}BC\,e_n/\|e_n\|^2$",
        title,
    )
    _lines(folder / "09_retention_R.png", runs, "R_value", r"$R(t)=\|e\|-I(t)$", title)


def main() -> None:
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    _bundle(OUT / "01_k4_L1_vs_L2", ["l1_k4", "l2_k4"], r"Wavenumber 4: L=1 and L=2")
    _bundle(OUT / "02_k5_L1_vs_L2", ["l1_k5", "l2_k5"], r"Wavenumber 5: L=1 and L=2")
    _bundle(OUT / "03_L2_k4_vs_k5", ["l2_k4", "l2_k5"], r"L=2: wavenumber 4 and wavenumber 5")
    _bundle(OUT / "04_L1_k4_vs_k5", ["l1_k4", "l1_k5"], r"L=1: wavenumber 4 and wavenumber 5")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(OUT.rglob("*.png")):
            handle.write(path, path.relative_to(OUT.parent))
    print(f"zip {ZIP} files {sum(1 for _ in OUT.rglob('*.png'))}")


if __name__ == "__main__":
    main()
