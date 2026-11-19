"""Figures for the L=2 coarse-lift diagnostics.

T and U are lift-and-response probes. A marker inside the unit circle is not
a one-step contraction of the autonomous HINE map, and a set of markers from
separate states is not the spectrum of their product.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAVEAT = (
    "T and U are lift-and-response diagnostics, not the autonomous HINE rollout. "
    "Eigenvalues inside the unit circle do not establish one-step contraction. "
    "Spectra at separate states do not establish stability of a time-varying product."
)


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_argand(eigenvalues: np.ndarray, labels: list[str], converged: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.45", lw=1, label="unit circle")
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    vals = np.asarray(eigenvalues, dtype=np.complex128)
    flags = np.asarray(converged, dtype=bool)
    if flags.shape != vals.shape:
        flags = np.ones(vals.shape, dtype=bool)
    for i, lab in enumerate(labels):
        block = vals[i]
        ok = flags[i]
        ax.scatter(np.real(block[ok]), np.imag(block[ok]), s=28, label=lab)
        if np.any(~ok):
            ax.scatter(
                np.real(block[~ok]),
                np.imag(block[~ok]),
                s=36,
                facecolors="none",
                edgecolors="0.2",
                label=f"{lab} unconverged",
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re λ(T)")
    ax.set_ylabel("Im λ(T)")
    ax.set_title("Leading eigenvalues of T = C D†")
    ax.legend(fontsize=7, loc="best")
    fig.text(0.01, 0.01, CAVEAT, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)


def plot_eigen_summary(labels: list[str], magnitudes: np.ndarray, residuals: np.ndarray, converged: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    x = np.arange(len(labels))
    mags = np.asarray(magnitudes, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64)
    flags = np.asarray(converged, dtype=bool)
    for mode in range(mags.shape[1]):
        axes[0].plot(x, mags[:, mode], marker="o", label=f"mode {mode + 1}")
        axes[1].semilogy(x, np.maximum(res[:, mode], 1e-16), marker="o", label=f"mode {mode + 1}")
    for i, ok_row in enumerate(flags):
        if not np.all(ok_row):
            axes[0].axvline(i, color="0.85", lw=4, zorder=0)
    axes[0].axhline(1.0, color="0.4", ls="--", lw=1, label="magnitude 1")
    axes[0].set_ylabel("|λ(T)|")
    axes[0].set_title("Leading |λ(T)| across selected states")
    axes[0].legend(fontsize=7, ncol=3)
    axes[1].set_ylabel("eigenpair residual")
    axes[1].set_xlabel("state")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_title("Shaded states have at least one unconverged mode")
    fig.text(0.01, 0.01, CAVEAT, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, path)


def plot_singular_gains(labels, series: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """series[name] has euclidean, rms, directional_rms. Square operators keep equal Euclidean and RMS values."""
    names = list(series)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)
    x = np.arange(len(labels))
    for name in names:
        axes[0].plot(x, series[name]["euclidean"][:, 0], marker="o", label=f"{name} σ_max Euclidean")
        axes[1].plot(x, series[name]["rms"][:, 0], marker="o", label=f"{name} σ_max RMS")
        if "directional_rms" in series[name]:
            axes[1].plot(
                x,
                series[name]["directional_rms"],
                marker="x",
                ls="none",
                label=f"{name} sampled directional RMS",
            )
    axes[0].set_title("Euclidean singular gain")
    axes[1].set_title("Physical RMS singular gain")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylabel("gain")
        ax.legend(fontsize=7)
    fig.suptitle("Partial SVD gains. Sampled crosses are not σ_max. T and U RMS factors are 1; C and CQ use sqrt(N/M).")
    fig.text(0.01, 0.01, CAVEAT, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    _save(fig, path)


def plot_cp_cq(steps: np.ndarray, rms_cp: np.ndarray, rms_cq: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(steps, rms_cp.mean(axis=0), label="mean RMS of C P e_H")
    ax.fill_between(
        steps,
        np.nanpercentile(rms_cp, 10, axis=0),
        np.nanpercentile(rms_cp, 90, axis=0),
        alpha=0.2,
    )
    ax.plot(steps, rms_cq.mean(axis=0), label="mean RMS of C Q e_H")
    ax.fill_between(
        steps,
        np.nanpercentile(rms_cq, 10, axis=0),
        np.nanpercentile(rms_cq, 90, axis=0),
        alpha=0.2,
    )
    ax.set_xlabel("rollout step")
    ax.set_ylabel("physical vorticity RMS")
    ax.set_title("Resolved-coarse versus unresolved-fine response of C")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_budget(steps, rms_next, rms_c, rms_k, rms_eps, rms_remainder, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    series = {
        "RMS η_{n+1}": rms_next,
        "RMS C e_H": rms_c,
        "RMS K e_Z": rms_k,
        "RMS ε_Z": rms_eps,
        "RMS remainder": rms_remainder,
    }
    for label, block in series.items():
        ax.plot(steps, np.nanmean(block, axis=0), label=label)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("physical vorticity RMS")
    ax.set_title("Coarse-error budget at teacher-forced Jacobians")
    ax.legend(fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Remainder is η_{n+1} - (C e_H + K e_Z + ε_Z). It is the nonlinear discrepancy, not a second operator.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, path)


def plot_twostep(labels, norm_aa, norm_bc, cosine, rel, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, norm_aa, width, label="||A_{n+1} A_n e_H||")
    axes[0].bar(x + width / 2, norm_bc, width, label="||B_{n+1} C_n e_H||")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Euclidean norm, physical vorticity")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Two contributions, not the full response")
    axes[1].plot(x, cosine, marker="o", label="cosine(AA e, BC e)")
    axes[1].plot(x, rel, marker="s", label="||sum - composed fine|| / ||composed||")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Alignment and tangent identity")
    fig.text(
        0.01,
        0.01,
        "The fine component of the two-step augmented tangent with zero initial latent perturbation is the sum. B C alone is one path.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path)


def _panel(ax, field: np.ndarray, title: str) -> None:
    data = np.asarray(field, dtype=np.float64)
    limit = float(np.nanmax(np.abs(data))) if data.size else 1.0
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    image = ax.imshow(data, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.046)


def plot_direction_fields(panels: list[tuple[str, np.ndarray]], path: Path, suptitle: str) -> None:
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.8 * rows), squeeze=False)
    for i, (title, field) in enumerate(panels):
        _panel(axes[i // cols][i % cols], field, title)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(suptitle)
    fig.text(0.01, 0.01, CAVEAT, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, path)


def complex_panels(name: str, field: np.ndarray) -> list[tuple[str, np.ndarray]]:
    data = np.asarray(field)
    if np.iscomplexobj(data):
        imag_max = float(np.max(np.abs(np.imag(data)))) if data.size else 0.0
        real_max = float(np.max(np.abs(np.real(data)))) if data.size else 0.0
        tag = "numerically real" if imag_max <= 1e-8 * max(real_max, 1e-30) else "complex"
        return [
            (f"{name} real ({tag})", np.real(data)),
            (f"{name} imag ({tag})", np.imag(data)),
        ]
    return [(name, np.real(data))]
