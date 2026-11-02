#!/usr/bin/env python3
"""Compare MSE, KE stability, and high-only Jacobian eigs across finished runs.

Writes PNGs under /pscratch/sd/d/divyam/hine_runs/plots/compare_all/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("/pscratch/sd/d/divyam/hine_runs")
OUT = RUNS / "plots" / "compare_all"
STEP_RE = re.compile(r"eigs_step(\d+)\.npz$")


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    family: str  # l3 | paper | plus | fno
    mid: int | None = None
    low: int | None = None


def build_registry() -> list[RunSpec]:
    l3 = [
        ("m128_l128", 128, 128),
        ("m128_l64", 128, 64),
        ("m128_l32", 128, 32),
        ("m128_l16", 128, 16),
        ("m128_l8", 128, 8),
        ("m64_l64", 64, 64),
        ("m64_l32", 64, 32),
        ("m64_l16", 64, 16),
        ("m64_l8", 64, 8),
        ("m32_l32", 32, 32),
        ("m32_l16", 32, 16),
        ("m32_l8", 32, 8),
        ("m16_l16", 16, 16),
        ("m16_l8", 16, 8),
        ("m8_l8", 8, 8),
    ]
    specs: list[RunSpec] = []
    for name, mid, low in l3:
        specs.append(
            RunSpec(
                label=name,
                path=RUNS / "l3_sweep" / name,
                family="l3",
                mid=mid,
                low=low,
            )
        )
    specs.append(
        RunSpec(
            label="plus_L4",
            path=RUNS / "plus_L4_m64_32_8",
            family="plus",
        )
    )
    specs.append(
        RunSpec(
            label="plus_L5",
            path=RUNS / "plus_L5_m128_64_32_8",
            family="plus",
        )
    )
    for name in ("fno_1to1", "fno_2to2", "fno_3to3"):
        specs.append(RunSpec(label=name, path=RUNS / name, family="fno"))
    return specs


def mse_path(run: RunSpec) -> Path:
    return run.path / "jacobian_manual_mse" / "rollout_mse.npz"


def ke_path(run: RunSpec) -> Path:
    return run.path / "jacobian_manual_mse" / "kinetic_energy_survival.npz"


def eig_root(run: RunSpec, kind: str = "high") -> Path:
    folder = "jacobian_high_eigs" if kind == "high" else "jacobian_full_eigs"
    return run.path / folder / "scipy_3_n4"


def hierarchy_specs() -> list[RunSpec]:
    """Paper L3 (256-32-8) plus L4 and L5. These three have full-state eigs."""
    return [
        RunSpec(label="L3", path=RUNS / "paper_1gpu", family="paper"),
        RunSpec(label="L4", path=RUNS / "plus_L4_m64_32_8", family="plus"),
        RunSpec(label="L5", path=RUNS / "plus_L5_m128_64_32_8", family="plus"),
    ]


def load_mean_mse(run: RunSpec) -> tuple[np.ndarray, np.ndarray] | None:
    path = mse_path(run)
    if not path.is_file():
        print(f"WARN skip MSE {run.label}: missing {path}")
        return None
    z = np.load(path)
    if "ar_mse_high" not in z.files:
        print(f"WARN skip MSE {run.label}: no ar_mse_high in {path}")
        return None
    mse = np.asarray(z["ar_mse_high"], dtype=np.float64)
    if "lead" in z.files:
        lead = np.asarray(z["lead"], dtype=np.float64).ravel()
    else:
        lead = np.arange(1, mse.shape[1] + 1, dtype=np.float64)
    if lead.size != mse.shape[1]:
        lead = np.arange(1, mse.shape[1] + 1, dtype=np.float64)
    return lead, mse.mean(axis=0)


def load_stability(run: RunSpec) -> tuple[np.ndarray, np.ndarray] | None:
    path = ke_path(run)
    if not path.is_file():
        print(f"WARN skip KE {run.label}: missing {path}")
        return None
    z = np.load(path)
    if "S_ar" not in z.files:
        print(f"WARN skip KE {run.label}: no S_ar in {path}")
        return None
    lead = np.asarray(z["lead"], dtype=np.float64).ravel()
    s_ar = np.asarray(z["S_ar"], dtype=np.float64).ravel()
    return lead, s_ar


def is_baseline(run: RunSpec) -> bool:
    """Paper L3 and the matching sweep cell are both 256-32-8."""
    if run.label == "L3":
        return True
    return run.family == "l3" and run.mid == 32 and run.low == 8


def load_eig_rows(run: RunSpec, kind: str = "high") -> list[dict]:
    root = eig_root(run, kind)
    if not root.is_dir():
        print(f"WARN skip eigs {run.label}: missing {root}")
        return []
    rows: list[dict] = []
    for section_dir in sorted(root.glob("section_*")):
        for path in sorted(section_dir.glob("eigs_step*.npz")):
            m = STEP_RE.search(path.name)
            if m is None:
                continue
            z = np.load(path)
            vals = np.asarray(z["eigenvalues"]).astype(np.complex128).ravel()
            if vals.size == 0:
                continue
            lam = vals[np.argmax(np.abs(vals))]
            radius = float(z["radius"]) if "radius" in z.files else float(np.abs(lam))
            step = int(z["step"]) if "step" in z.files else int(m.group(1))
            rows.append({"step": step, "lambda": complex(lam), "radius": radius})
    if not rows:
        print(f"WARN skip eigs {run.label}: no eigs_step*.npz under {root}")
    return rows


def family_style(family: str) -> dict:
    if family == "l3":
        return {"ls": "-", "alpha": 0.85}
    if family == "paper":
        return {"ls": "-", "lw": 2.4, "alpha": 1.0}
    if family == "plus":
        return {"ls": "--", "lw": 2.0, "alpha": 0.95}
    return {"ls": ":", "lw": 2.0, "alpha": 0.95}


def plot_mse(
    specs: list[RunSpec],
    out: Path,
    filename: str = "mse_vs_lead_all.png",
    title: str = "Rollout MSE vs lead (mean over trials)",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    n = 0
    for run in specs:
        loaded = load_mean_mse(run)
        if loaded is None:
            continue
        lead, mean_mse = loaded
        style = family_style(run.family)
        ax.plot(lead, mean_mse, label=run.label, **style)
        n += 1
    ax.set_xlabel("lead (AR step)")
    ax.set_ylabel(r"mean high-field MSE")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=3, loc="best")
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(f"wrote {out / filename} ({n} curves)")


def plot_ke(
    specs: list[RunSpec],
    out: Path,
    filename: str = "ke_stability_rate_all.png",
    title: str = r"KE stability rate ($|E-\mu_E|\leq 5\sigma_E$; once failed stays failed)",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    n = 0
    for run in specs:
        loaded = load_stability(run)
        if loaded is None:
            continue
        lead, s_ar = loaded
        style = family_style(run.family)
        ax.plot(lead, s_ar, label=run.label, marker="o", markersize=3, **style)
        n += 1
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("lead")
    ax.set_ylabel(r"stability rate $S(n)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=3, loc="best")
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(f"wrote {out / filename} ({n} curves)")


def load_all_eigs(run: RunSpec, kind: str = "high") -> np.ndarray:
    root = eig_root(run, kind)
    if not root.is_dir():
        print(f"WARN skip eigs {run.label}: missing {root}")
        return np.array([], dtype=np.complex128)
    chunks: list[np.ndarray] = []
    for section_dir in sorted(root.glob("section_*")):
        for path in sorted(section_dir.glob("eigs_step*.npz")):
            z = np.load(path)
            vals = np.asarray(z["eigenvalues"]).astype(np.complex128).ravel()
            if vals.size:
                chunks.append(vals)
    if not chunks:
        print(f"WARN skip eigs {run.label}: no eigs_step*.npz under {root}")
        return np.array([], dtype=np.complex128)
    return np.concatenate(chunks)


def plot_argand_all(
    specs: list[RunSpec],
    out: Path,
    kind: str,
    filename: str,
    title: str,
) -> None:
    """Every saved LM eigenvalue, so the spread across sections and steps is visible."""
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.45", ls="--", lw=1.0, label=r"$|\lambda|=1$")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)

    ordered = sorted(specs, key=lambda run: is_baseline(run))
    color_i = 0
    n = 0
    for run in ordered:
        vals = load_all_eigs(run, kind)
        if vals.size == 0:
            continue
        if is_baseline(run):
            color = "black"
            size = 28
            alpha = 0.9
            zorder = 5
        else:
            color = f"C{color_i % 10}"
            color_i += 1
            size = 16
            alpha = 0.55
            zorder = 3
        ax.scatter(
            vals.real,
            vals.imag,
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0,
            zorder=zorder,
            label=f"{run.label} (n={vals.size})",
        )
        n += 1

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc="best", ncols=2)
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(f"wrote {out / filename} ({n} models)")


def plot_argand_max(
    specs: list[RunSpec],
    out: Path,
    kind: str = "high",
    filename: str = "argand_max_eig_all.png",
    title: str = "High-only LM eigenvalue with largest |λ| (one point per model)",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1.0, alpha=0.6, label=r"$|\lambda|=1$")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)

    n = 0
    for i, run in enumerate(specs):
        rows = load_eig_rows(run, kind)
        if not rows:
            continue
        radii = np.array([r["radius"] for r in rows], dtype=float)
        imax = int(np.argmax(radii))
        lam = rows[imax]["lambda"]
        color = f"C{i % 10}"
        ax.scatter(
            [lam.real],
            [lam.imag],
            s=70,
            c=color,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            label=f"{run.label} (|λ|={radii[imax]:.3g})",
        )
        ax.annotate(
            run.label,
            (lam.real, lam.imag),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            alpha=0.9,
        )
        n += 1

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc="best", ncols=1)
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(f"wrote {out / filename} ({n} models)")


def mean_radius_by_step(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_step: dict[int, list[float]] = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(float(r["radius"]))
    steps = np.array(sorted(by_step), dtype=int)
    means = np.array([float(np.mean(by_step[s])) for s in steps], dtype=float)
    return steps, means


def plot_mean_radius(
    specs: list[RunSpec],
    out: Path,
    kind: str = "high",
    filename: str = "radius_mean_vs_step_all.png",
    title: str = "High-only LM radius vs step (mean across sections)",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    n = 0
    for run in specs:
        rows = load_eig_rows(run, kind)
        if not rows:
            continue
        steps, means = mean_radius_by_step(rows)
        style = family_style(run.family)
        ax.plot(steps, means, label=run.label, marker="o", markersize=3, **style)
        n += 1
    ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.75, label=r"$|\lambda|=1$")
    ax.set_xlabel("AR step")
    ax.set_ylabel(r"mean $|\lambda|_{\max}$ over sections")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=3, loc="best")
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)
    print(f"wrote {out / filename} ({n} curves)")


def l3_heatmap_value(run: RunSpec, mode: str) -> float | None:
    path = mse_path(run)
    if not path.is_file():
        return None
    z = np.load(path)
    if "ar_mse_high" not in z.files:
        return None
    mse = np.asarray(z["ar_mse_high"], dtype=np.float64)
    if mode == "mean":
        return float(mse.mean())
    if mode == "max":
        return float(mse.max())
    raise ValueError(mode)


def plot_l3_heatmaps(specs: list[RunSpec], out: Path) -> None:
    # Prefer sweep cells; paper_1gpu fills (32,8) only if that sweep cell is absent.
    cells: dict[tuple[int, int], tuple[str, float, float]] = {}
    for run in specs:
        if run.mid is None or run.low is None:
            continue
        if run.family not in ("l3", "paper"):
            continue
        mean_v = l3_heatmap_value(run, "mean")
        max_v = l3_heatmap_value(run, "max")
        if mean_v is None or max_v is None:
            print(f"WARN skip heatmap cell {run.label}")
            continue
        key = (run.mid, run.low)
        if key in cells and run.family == "paper":
            continue
        cells[key] = (run.label, mean_v, max_v)

    if not cells:
        print("WARN no L3 heatmap cells")
        return

    mids = sorted({m for m, _ in cells})
    lows = sorted({l for _, l in cells}, reverse=True)
    mean_mat = np.full((len(lows), len(mids)), np.nan)
    max_mat = np.full((len(lows), len(mids)), np.nan)
    for (mid, low), (_, mean_v, max_v) in cells.items():
        i = lows.index(low)
        j = mids.index(mid)
        mean_mat[i, j] = mean_v
        max_mat[i, j] = max_v

    def draw(mat: np.ndarray, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="equal", cmap="viridis", origin="upper")
        ax.set_xticks(range(len(mids)))
        ax.set_xticklabels([str(m) for m in mids])
        ax.set_yticks(range(len(lows)))
        ax.set_yticklabels([str(l) for l in lows])
        ax.set_xlabel("mid resolution")
        ax.set_ylabel("low resolution")
        ax.set_title(title)
        for i in range(len(lows)):
            for j in range(len(mids)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2g}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="w" if v > np.nanmedian(mat) else "k",
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("high-field MSE")
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
        print(f"wrote {out / filename}")

    draw(
        mean_mat,
        "L3 high-field MSE (mean over trials and leads)",
        "heatmap_l3_mse_mean.png",
    )
    draw(
        max_mat,
        "L3 high-field MSE (max over trials and leads)",
        "heatmap_l3_mse_max.png",
    )


def plot_hierarchy_high_heatmaps(specs: list[RunSpec], out: Path) -> None:
    """High-field MSE only, one cell per model, same statistic as the L3 grids."""
    labels = [run.label for run in specs]
    mean_row = np.full(len(specs), np.nan)
    max_row = np.full(len(specs), np.nan)
    for i, run in enumerate(specs):
        mean_v = l3_heatmap_value(run, "mean")
        max_v = l3_heatmap_value(run, "max")
        if mean_v is None or max_v is None:
            print(f"WARN skip high-field heatmap {run.label}")
            continue
        mean_row[i] = mean_v
        max_row[i] = max_v

    def draw(row: np.ndarray, title: str, filename: str) -> None:
        mat = row.reshape(1, -1)
        fig, ax = plt.subplots(figsize=(6.4, 2.4), constrained_layout=True)
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks([0])
        ax.set_yticklabels(["high field"])
        ax.set_xlabel("model")
        ax.set_title(title)
        finite = row[np.isfinite(row)]
        cutoff = float(np.median(finite)) if finite.size else 0.0
        for j, v in enumerate(row):
            if np.isfinite(v):
                ax.text(
                    j,
                    0,
                    f"{v:.3g}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="w" if v > cutoff else "k",
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("high-field MSE")
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
        print(f"wrote {out / filename}")

    draw(
        mean_row,
        "L3 / L4 / L5 high-field MSE (mean over trials and leads)",
        "heatmap_l3_l4_l5_mse_mean.png",
    )
    draw(
        max_row,
        "L3 / L4 / L5 high-field MSE (max over trials and leads)",
        "heatmap_l3_l4_l5_mse_max.png",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = build_registry()
    print(f"registry={len(specs)} out={OUT}")
    plot_mse(specs, OUT)
    plot_ke(specs, OUT)
    plot_argand_max(specs, OUT)
    plot_mean_radius(specs, OUT)
    plot_l3_heatmaps(specs, OUT)

    trio = hierarchy_specs()
    plot_mse(
        trio,
        OUT,
        filename="mse_vs_lead_l3_l4_l5.png",
        title="L3 / L4 / L5 rollout MSE vs lead (mean over trials)",
    )
    plot_ke(
        trio,
        OUT,
        filename="ke_stability_rate_l3_l4_l5.png",
        title="L3 / L4 / L5 KE stability rate",
    )
    plot_argand_all(
        trio,
        OUT,
        kind="full",
        filename="argand_full_l3_l4_l5.png",
        title="Full-state eigenvalues (L3 in black)",
    )
    sweep = [run for run in specs if run.family == "l3"]
    plot_argand_all(
        sweep,
        OUT,
        kind="high",
        filename="argand_all_eigs_l3_sweep.png",
        title="Resolution-ablation eigenvalues (256-32-8 in black)",
    )
    plot_mean_radius(
        trio,
        OUT,
        kind="full",
        filename="radius_full_mean_vs_step_l3_l4_l5.png",
        title="Full-state LM radius vs step (mean across sections)",
    )
    plot_hierarchy_high_heatmaps(trio, OUT)
    print("COMPARE_ALL_PLOTS_DONE")


if __name__ == "__main__":
    main()
