"""Argand and modulus plots with rank shown by a plasma color scale.

Values are sorted by descending modulus before plotting. Rank 1 is the
largest and sits at one end of the colormap. The last rank sits at the other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def write_rank_plots(
    plot_dir: Path,
    values: np.ndarray,
    title: str,
    kind: str,
) -> np.ndarray:
    """Write argand.png and modulus.png. Returns the values in rank order."""
    if kind not in ("eigenvalue", "singular"):
        raise ValueError(f"kind must be eigenvalue or singular, got {kind}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(values)
    raw = raw[np.isfinite(raw)]
    if kind == "eigenvalue":
        ordered = np.asarray(raw, dtype=np.complex128)
        order = np.argsort(-np.abs(ordered))
        ordered = ordered[order]
        height = np.abs(ordered)
        ylabel = r"$|\lambda|$"
    else:
        ordered = np.asarray(raw, dtype=np.float64).real
        order = np.argsort(-np.abs(ordered))
        ordered = np.abs(ordered[order])
        height = ordered
        ylabel = "singular value"
    n = int(ordered.size)
    ranks = np.arange(1, n + 1)
    cmap = plt.get_cmap("plasma_r")
    norm = Normalize(vmin=1, vmax=max(n, 1))
    colors = cmap(norm(ranks if n else np.array([1.0])))

    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.45", lw=1.0, label=r"$|\lambda|=1$")
    ax.axhline(0.0, color="0.85", lw=0.6)
    ax.axvline(0.0, color="0.85", lw=0.6)
    if kind == "eigenvalue":
        xs = np.real(ordered)
        ys = np.imag(ordered)
    else:
        xs = ordered
        ys = np.zeros(n, dtype=np.float64)
    if n:
        ax.scatter(xs, ys, c=colors, s=28, zorder=3)
    ax.scatter([0.0], [0.0], marker="x", s=70, c="k", zorder=4, label="0")
    ax.scatter([1.0], [0.0], marker="+", s=120, c="C3", zorder=4, label="1")
    reach = float(np.max(np.hypot(xs, ys))) if n else 1.0
    radius = max(reach, 1.0) * 1.2
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    bar.set_label("rank, 1 = hottest")
    bar.ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plot_dir / "argand.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if n:
        ax.plot(ranks, height, color="0.75", lw=1.0, zorder=1)
        ax.scatter(ranks, height, c=colors, s=28, zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("rank by modulus")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    bar.set_label("rank, 1 = hottest")
    bar.ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plot_dir / "modulus.png", dpi=140)
    plt.close(fig)
    return ordered


def write_permutation_strip(
    path: Path,
    series: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    log_y: bool = False,
) -> None:
    """One x position per block. Fifty dots, plasma reversed so rank 1 is hottest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("plasma_r")
    norm = Normalize(vmin=1, vmax=50)
    rng = np.random.default_rng(0)
    fig_w = max(8.0, 1.35 * len(series) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.6))
    for index, (_label, values) in enumerate(series):
        raw = np.asarray(values)
        raw = raw[np.isfinite(raw)]
        height = np.abs(raw)
        order = np.argsort(-height)
        height = height[order]
        ranks = np.arange(1, height.size + 1)
        jitter = rng.uniform(-0.12, 0.12, size=height.size)
        ax.scatter(
            np.full(height.size, index, dtype=np.float64) + jitter,
            height,
            c=ranks,
            cmap=cmap,
            norm=norm,
            s=26,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels([label for label, _values in series], rotation=0, ha="center")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.axhline(1.0, color="0.35", ls="--", lw=0.8)
    ax.set_title(title)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    bar.set_label("rank, 1 = hottest")
    bar.ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_paired_permutation_strip(
    path: Path,
    series_a: list[tuple[str, np.ndarray]],
    series_b: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    log_y: bool = False,
) -> None:
    """One x position per block. Two rank color scales, offset so both sets are visible."""
    labels_a = [label for label, _values in series_a]
    labels_b = [label for label, _values in series_b]
    if labels_a != labels_b:
        raise ValueError("paired strips need the same blocks in the same order")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fig_w = max(9.2, 1.45 * len(series_a) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    groups = (
        (series_a, -0.18, "plasma_r", label_a),
        (series_b, 0.18, "cool_r", label_b),
    )
    mappables = []
    for series, shift, cmap_name, _name in groups:
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=1, vmax=50)
        for index, (_label, values) in enumerate(series):
            raw = np.asarray(values)
            raw = raw[np.isfinite(raw)]
            height = np.abs(raw)
            order = np.argsort(-height)
            height = height[order]
            ranks = np.arange(1, height.size + 1)
            jitter = rng.uniform(-0.06, 0.06, size=height.size)
            ax.scatter(
                np.full(height.size, index, dtype=np.float64) + shift + jitter,
                height,
                c=ranks,
                cmap=cmap,
                norm=norm,
                s=22,
                zorder=3,
            )
        mappables.append(ScalarMappable(norm=norm, cmap=cmap))
    ax.set_xticks(np.arange(len(series_a)))
    ax.set_xticklabels(labels_a, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.axhline(1.0, color="0.35", ls="--", lw=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    positions = ([0.81, 0.14, 0.025, 0.72], [0.90, 0.14, 0.025, 0.72])
    for mappable, cax_pos, (_series, _shift, _cmap, name) in zip(mappables, positions, groups):
        cax = fig.add_axes(cax_pos)
        bar = fig.colorbar(mappable, cax=cax)
        bar.set_label(f"{name}\nrank, 1 = hottest")
        bar.ax.invert_yaxis()
    fig.savefig(path, dpi=140)
    plt.close(fig)
