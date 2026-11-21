#!/usr/bin/env python3
"""Categorical snapshot strips for the four-setting trace comparison.

Each dot is one probe-averaged snapshot. Jitter has no numerical meaning.
Median and interquartile interval describe that one trajectory; they are
not confidence intervals. Monte Carlo standard errors stay in the tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ("l1_k4", "l2_k4", "l1_k5", "l2_k5")
COLORS = {
    "l1_k4": "#0072B2",
    "l2_k4": "#E69F00",
    "l1_k5": "#009E73",
    "l2_k5": "#D55E00",
}
SIGNED = {
    "tau",
    "energy",
    "tau_K",
    "energy_K",
    "tau_BC",
    "tau_loop_next",
    "interference",
    "loop_net",
    "scale_P",
    "scale_Q",
    "dir_tau",
    "dir_energy",
    "dir_bc",
    "dir_k",
    "dir_interference",
    "dir_net",
    "R_value",
}


def _jitter(stem: int, scale: float = 0.16) -> float:
    value = (int(stem) * 1103515245 + 12345) & 0x7FFFFFFF
    unit = (value % 10000) / 10000.0 - 0.5
    return float(unit * scale)


def load_runs(save_dir: Path) -> list[dict]:
    runs = []
    for name in ORDER:
        path = save_dir / name / "metrics.npz"
        meta_path = save_dir / name / "metadata.json"
        if not path.is_file():
            continue
        stored = np.load(path, allow_pickle=True)
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"label": name, "name": name}
        runs.append({"name": name, "label": meta.get("label", name), "meta": meta, "data": stored})
    if not runs:
        raise FileNotFoundError(f"no metrics under {save_dir}")
    return runs


def _column(run: dict, key: str) -> np.ndarray:
    if key not in run["data"]:
        return np.full(1, np.nan)
    return np.asarray(run["data"][key], dtype=np.float64)


def _usable(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).any())


def _limits(groups: list[np.ndarray], signed: bool) -> tuple[float, float]:
    finite = np.concatenate([values[np.isfinite(values)] for values in groups if np.isfinite(values).any()])
    if finite.size == 0:
        return -1.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    if signed:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if high == low:
        pad = 1.0 if high == 0.0 else 0.05 * abs(high)
    else:
        pad = 0.05 * (high - low)
    return low - pad, high + pad


def _draw_strip(ax, runs: list[dict], key: str, ylabel: str, signed: bool) -> int:
    groups = []
    labels = []
    drawn = 0
    for position, run in enumerate(runs):
        values = _column(run, key)
        if not _usable(values):
            continue
        stems = np.asarray(run["data"]["stem"])
        groups.append(values)
        labels.append((position, run["label"], run["name"]))
        for stem, value in zip(stems, values):
            if not np.isfinite(value):
                continue
            ax.scatter(
                position + _jitter(int(stem)),
                value,
                s=28,
                color=COLORS[run["name"]],
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
            drawn += 1
        finite = values[np.isfinite(values)]
        quartile_low, median, quartile_high = np.percentile(finite, [25, 50, 75])
        ax.plot([position, position], [quartile_low, quartile_high], color="black", lw=2.4, zorder=3)
        ax.plot([position - 0.12, position + 0.12], [median, median], color="black", lw=2.0, zorder=4)
    if not groups:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return 0
    positions = [item[0] for item in labels]
    ax.set_xticks(positions)
    ax.set_xticklabels([item[1] for item in labels], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    low, high = _limits(groups, signed)
    ax.set_ylim(low, high)
    if signed:
        ax.axhline(0.0, color="0.4", lw=0.8, zorder=1)
    return drawn


def _orientation(ax, runs: list[dict], isotropic: str, directional: str, ylabel: str) -> None:
    groups = []
    for position, run in enumerate(runs):
        left = _column(run, isotropic)
        right = _column(run, directional)
        if not (_usable(left) or _usable(right)):
            continue
        stems = np.asarray(run["data"]["stem"])
        groups.extend([left, right])
        for stem, iso, direc in zip(stems, left, right):
            shift = _jitter(int(stem), scale=0.08)
            if np.isfinite(iso):
                ax.scatter(
                    position - 0.12 + shift,
                    iso,
                    s=26,
                    marker="o",
                    color=COLORS[run["name"]],
                    alpha=0.55,
                    linewidths=0,
                    zorder=2,
                )
            if np.isfinite(direc):
                ax.scatter(
                    position + 0.12 + shift,
                    direc,
                    s=32,
                    marker="D",
                    color=COLORS[run["name"]],
                    alpha=0.75,
                    linewidths=0,
                    zorder=2,
                )
    ax.scatter([], [], marker="o", color="0.3", label="isotropic probe average")
    ax.scatter([], [], marker="D", color="0.3", label="actual error direction")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([run["label"] for run in runs], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if groups:
        low, high = _limits(groups, True)
        ax.set_ylim(low, high)
    ax.axhline(0.0, color="0.4", lw=0.8)


def _caption(runs: list[dict], text: str) -> str:
    counts = []
    for run in runs:
        stems = np.asarray(run["data"]["stem"])
        trajectories = len(set(np.asarray(run["data"]["trajectory_id"]).tolist()))
        counts.append(f"{run['label']}: {len(stems)} snapshots, {trajectories} trajectory")
    return text + " " + "; ".join(counts) + "."


def render(save_dir: Path, report_path: Path, smoke: bool = False) -> None:
    save_dir = Path(save_dir)
    runs = load_runs(save_dir)
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _one(
        plot_dir / "direct_trace.png",
        runs,
        "tau",
        "Normalized direct trace contribution",
        _caption(
            runs,
            "L=1 is Tr(J-I)/N. L=2 is Tr(A-I)/N with the latent input held fixed. "
            "A negative value is a negative isotropic linear energy contribution, not contraction. "
            "Dots are snapshots, not Hutchinson probes or eigenvalues. "
            "Black marks are the median and interquartile interval of this one trajectory.",
        ),
    )
    _one(
        plot_dir / "direct_energy.png",
        runs,
        "energy",
        r"$\mathrm{Tr}(T^{T}T)/d - 1$",
        _caption(
            runs,
            "T is J for L=1 and A for L=2. This is the expected relative squared-norm change "
            "for an isotropic perturbation of that isolated block.",
        ),
    )
    latent_runs = [run for run in runs if run["meta"].get("level", 2) == 2 and _usable(_column(run, "tau_K"))]
    _grid(
        plot_dir / "latent_diagnostics.png",
        latent_runs,
        [
            ("tau_K", r"$\mathrm{Tr}(K-I)/M$"),
            ("energy_K", r"$\mathrm{Tr}(K^{T}K)/M - 1$"),
            ("tau_BC", r"frozen $\mathrm{Tr}(B_n C_n)/N$"),
            ("tau_loop_next", r"consecutive $\mathrm{Tr}(B_{n+1}C_n)/N$"),
            ("b_gain", r"$\|B\|_F^2/M$"),
        ],
        _caption(
            latent_runs,
            "L=1 has no latent block, so it is omitted rather than drawn at zero. "
            "Frozen and consecutive products are different operators. "
            "The loop trace keeps denominator N.",
        ),
    )
    _grid(
        plot_dir / "two_step_energy.png",
        runs,
        [
            ("direct", r"direct $\|F\|_F^2/N$"),
            ("interference", r"interference $2\mathrm{Tr}(F^{T}L)/N$"),
            ("loop", r"loop $\|L\|_F^2/N$"),
            ("total", r"total $\|F+L\|_F^2/N$"),
            ("loop_net", "loop net effect"),
        ],
        _caption(
            runs,
            "F is A_{n+1}A_n, or J_{n+1}J_n for L=1. L is B_{n+1}C_n. "
            "L=1 has no latent loop, so interference, loop energy, and the net loop effect are N/A. "
            "Its direct and total energies are the fine two-step map.",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    pairs = [
        ("tau", "dir_tau", "Direct trace: isotropic vs error"),
        ("energy", "dir_energy", "Direct energy change"),
        ("tau_BC", "dir_bc", "Frozen loop along the fine error"),
        ("tau_K", "dir_k", "Latent trace along eta"),
    ]
    for ax, (iso, direc, title) in zip(axes.ravel(), pairs):
        _orientation(ax, runs, iso, direc, title)
    fig.suptitle("Circles are isotropic averages; diamonds are the realized error direction")
    fig.tight_layout()
    fig.savefig(plot_dir / "orientation.png", dpi=140)
    plt.close(fig)
    _memory(plot_dir / "latent_memory.png", latent_runs)
    _grid(
        plot_dir / "scale_traces.png",
        runs,
        [
            ("scale_P", r"$\mathrm{Tr}(PTP)/\mathrm{rank}(P)$"),
            ("scale_Q", r"$\mathrm{Tr}(QTQ)/\mathrm{rank}(Q)$"),
        ],
        _caption(
            runs,
            "P is the block-average projector at pool factor 8 and Q = I-P. "
            "The same projector is applied to L=1 even though D is not inside that forward map. "
            "These two pieces do not add to the full trace.",
        ),
    )
    _rollout(plot_dir / "rollout_secondary.png", runs)
    write_report(save_dir, report_path, runs, smoke)
    _write_table(save_dir, runs)


def _one(path: Path, runs: list[dict], key: str, ylabel: str, caption: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _draw_strip(ax, runs, key, ylabel, key in SIGNED)
    fig.text(0.01, 0.01, caption, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _grid(path: Path, runs: list[dict], panels: list[tuple[str, str]], caption: str) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.4, 2.5 * len(panels)), squeeze=False)
    for ax, (key, ylabel) in zip(axes.ravel(), panels):
        _draw_strip(ax, runs, key, ylabel, key in SIGNED)
    fig.text(0.01, 0.005, caption, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _memory(path: Path, runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, key, title in zip(
        axes,
        ("memory_frozen", "memory_varying"),
        (r"frozen $\|K_n^h\|_F^2/M$", r"time-varying $\|K_{n+h-1}\cdots K_n\|_F^2/M$"),
    ):
        groups = []
        for position, run in enumerate(runs):
            if key not in run["data"]:
                continue
            values = np.asarray(run["data"][key], dtype=np.float64)
            horizons = np.asarray(run["data"]["horizons"])
            stems = np.asarray(run["data"]["stem"])
            for horizon_index, horizon in enumerate(horizons):
                column = values[:, horizon_index]
                groups.append(column)
                center = position + (horizon_index - (len(horizons) - 1) / 2.0) * 0.12
                for stem, value in zip(stems, column):
                    if np.isfinite(value):
                        ax.scatter(
                            center + _jitter(int(stem), 0.04),
                            value,
                            s=18,
                            color=COLORS[run["name"]],
                            alpha=0.5,
                            linewidths=0,
                        )
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([run["label"] for run in runs], rotation=15, ha="right")
        ax.set_title(title)
        if groups:
            low, high = _limits(groups, False)
            ax.set_ylim(low, high)
    fig.suptitle("Latent gain above 1 amplifies an isotropic latent perturbation. tau_K < 0 does not forbid that.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _rollout(path: Path, runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for run in runs:
        lead = np.arange(_column(run, "e_norm").size)
        axes[0].plot(lead, _column(run, "e_norm"), color=COLORS[run["name"]], label=run["label"])
        axes[0].plot(lead, _column(run, "eps_u_norm"), color=COLORS[run["name"]], ls="--", alpha=0.7)
        axes[1].plot(lead, _column(run, "R_value"), color=COLORS[run["name"]], label=run["label"])
    axes[0].set_ylabel(r"physical $\|e_n\|$ and $\|\epsilon^u_n\|$")
    axes[0].set_title("Solid: free-rollout error. Dashed: teacher-forced residual.")
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set_ylabel(r"$R_n=\|e_n\|-I_n$")
    axes[1].set_xlabel("step along the one loader trajectory")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "fraction_negative": None, "fraction_positive": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "q25": float(np.percentile(finite, 25)),
        "q75": float(np.percentile(finite, 75)),
        "fraction_negative": float(np.mean(finite < 0.0)),
        "fraction_positive": float(np.mean(finite > 0.0)),
    }


def write_report(save_dir: Path, report_path: Path, runs: list[dict], smoke: bool) -> None:
    summary = {}
    lines = ["# HINE trace comparison", ""]
    if smoke:
        lines.append("This file is the smoke run. It is a pipeline check, not the scientific comparison.")
        lines.append("")
    lines.append(
        "Dots are probe-averaged Jacobian diagnostics at true inputs. "
        "They are not eigenvalue samples and not individual Hutchinson probes. "
        "Each category is one loader trajectory, so the median and interquartile interval are descriptive. "
        "Probe standard errors are in `metrics.npz` and are not the spread drawn on the plots."
    )
    lines.append("")
    lines.append("Negative trace values are negative isotropic linear energy contributions. They do not prove contraction or long-horizon stability.")
    lines.append("")
    lines.append("Wavenumber-4 L=1 versus L=2 is the controlled hierarchy comparison. Wavenumber 5 repeats that comparison on a different forcing. Cross-wavenumber differences are not attributed to hierarchy alone.")
    lines.append("")
    tau_medians = {}
    energy_medians = {}
    for run in runs:
        name = run["name"]
        tau = _stats(_column(run, "tau"))
        energy = _stats(_column(run, "energy"))
        tau_medians[name] = tau["median"]
        energy_medians[name] = energy["median"]
        summary[name] = {
            "label": run["label"],
            "tau": tau,
            "energy": energy,
            "tau_K": _stats(_column(run, "tau_K")),
            "tau_BC": _stats(_column(run, "tau_BC")),
            "tau_loop_next": _stats(_column(run, "tau_loop_next")),
            "loop_net": _stats(_column(run, "loop_net")),
            "interference": _stats(_column(run, "interference")),
            "b_gain": _stats(_column(run, "b_gain")),
            "dir_tau": _stats(_column(run, "dir_tau")),
            "R_value": _stats(_column(run, "R_value")),
            "final_error": _finite_last(_column(run, "e_norm")),
        }
        lines.append(f"## {run['label']}")
        lines.append("")
        lines.append(
            f"Direct trace median { _fmt(tau['median']) }, "
            f"interquartile {_fmt(tau['q25'])} to {_fmt(tau['q75'])}, "
            f"fraction below zero {_fmt(tau['fraction_negative'])}."
        )
        lines.append(
            f"Isolated energy change median {_fmt(energy['median'])}, "
            f"fraction below zero {_fmt(energy['fraction_negative'])}."
        )
        if summary[name]["tau_K"]["n"]:
            lines.append(
                f"Latent trace median {_fmt(summary[name]['tau_K']['median'])}. "
                f"Frozen loop median {_fmt(summary[name]['tau_BC']['median'])}. "
                f"Consecutive loop median {_fmt(summary[name]['tau_loop_next']['median'])}. "
                f"Loop net effect median {_fmt(summary[name]['loop_net']['median'])}, "
                f"fraction below zero {_fmt(summary[name]['loop_net']['fraction_negative'])}."
            )
            lines.append(
                "A negative net effect means the latent loop reduces average two-step energy relative to the direct path. "
                "Negative interference alone only says the paths oppose."
            )
        else:
            lines.append("Latent diagnostics are N/A for this fine-only model.")
        memory = _memory_stats(run)
        summary[name]["memory_frozen"] = memory["frozen"]
        summary[name]["memory_varying"] = memory["varying"]
        lines.append(
            f"Error-direction direct trace median {_fmt(summary[name]['dir_tau']['median'])} "
            f"(fraction below zero {_fmt(summary[name]['dir_tau']['fraction_negative'])}). "
            f"Final free-rollout error {_fmt(summary[name]['final_error'])}. "
            f"Retention R median {_fmt(summary[name]['R_value']['median'])}."
        )
        if memory["frozen"]:
            lines.append(
                "Frozen latent gain medians by horizon: "
                + ", ".join(f"h={item['horizon']} {_fmt(item['median'])}" for item in memory["frozen"])
                + ". Time-varying gain medians: "
                + ", ".join(f"h={item['horizon']} {_fmt(item['median'])}" for item in memory["varying"])
                + ". A gain above 1 amplifies an isotropic latent perturbation. A negative latent trace does not forbid that."
            )
        lines.append("")
    finite_tau = {name: value for name, value in tau_medians.items() if value is not None}
    if finite_tau:
        lowest = min(finite_tau, key=finite_tau.get)
        agrees = []
        for name, tau_value in finite_tau.items():
            energy_value = energy_medians[name]
            if energy_value is None:
                continue
            agrees.append(bool((tau_value < 0) == (energy_value < 0)))
        lines.append("## Answers from these states")
        lines.append("")
        lines.append(f"The most negative direct-trace median is {summary[lowest]['label']} at {_fmt(finite_tau[lowest])}.")
        lines.append(
            "Its isolated energy-change median is "
            f"{_fmt(energy_medians[lowest])}. "
            "Energy change equals twice the normalized trace plus the normalized squared shift, "
            "so a negative trace can still leave the isolated map energy-amplifying."
        )
        if agrees and all(agrees):
            lines.append("On these states, the sign of the direct trace and the sign of the isolated energy change agree for every setting.")
        else:
            lines.append("On these states, the sign of the direct trace and the sign of the isolated energy change do not agree for every setting.")
        loop_notes = []
        for name, item in summary.items():
            if not item["loop_net"]["n"]:
                continue
            reduces = item["loop_net"]["median"] is not None and item["loop_net"]["median"] < 0.0
            opposes = item["interference"]["median"] is not None and item["interference"]["median"] < 0.0
            if reduces:
                loop_notes.append(f"{item['label']} has a negative loop-net median, so the latent loop reduces two-step energy relative to the direct path")
            elif opposes:
                loop_notes.append(f"{item['label']} has negative interference but a non-negative loop-net median, so the paths oppose without the loop reducing two-step energy")
            else:
                loop_notes.append(f"{item['label']} does not show a negative loop-net median")
        if loop_notes:
            lines.append(" ".join(sentence + "." for sentence in loop_notes))
        lines.append(
            "Realized fine errors occupy a negative-energy direction where the error-direction trace or energy change is negative. "
            "That local direction is compared with the free-rollout error above; growth of the rollout error means the one-step linear contribution does not determine the trajectory."
        )
        lines.append("These local diagnostics do not establish long-horizon nonlinear stability.")
        lines.append("")
    command_path = save_dir / "command.json"
    if command_path.is_file():
        command = json.loads(command_path.read_text())
        lines.append(f"Git commit recorded with the run: {command.get('git_commit', 'unknown')}. Analysis code may be uncommitted.")
        lines.append("")
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    if not smoke:
        mirror = save_dir / "TRACE_COMPARE_REPORT.md"
        mirror.write_text(report_path.read_text())


def _memory_stats(run: dict) -> dict:
    result = {"frozen": [], "varying": []}
    if "horizons" not in run["data"]:
        return result
    horizons = np.asarray(run["data"]["horizons"])
    for key, bucket in (("memory_frozen", "frozen"), ("memory_varying", "varying")):
        if key not in run["data"]:
            continue
        values = np.asarray(run["data"][key], dtype=np.float64)
        if values.ndim != 2:
            continue
        for index, horizon in enumerate(horizons):
            stats = _stats(values[:, index])
            if stats["n"] == 0:
                continue
            result[bucket].append({"horizon": int(horizon), **stats})
    return result


def _finite_last(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.4g}"


def _write_table(save_dir: Path, runs: list[dict]) -> None:
    columns = [
        "setting",
        "stem",
        "trajectory_id",
        "tau",
        "tau_se",
        "energy",
        "energy_se",
        "tau_K",
        "tau_K_se",
        "energy_K",
        "tau_BC",
        "tau_BC_se",
        "tau_loop_next",
        "b_gain",
        "direct",
        "interference",
        "loop",
        "total",
        "loop_net",
        "dir_tau",
        "dir_energy",
        "dir_bc",
        "dir_k",
        "e_norm",
        "R_value",
        "rho_norm",
    ]
    lines = [",".join(columns)]
    for run in runs:
        count = len(np.asarray(run["data"]["stem"]))
        for index in range(count):
            row = []
            for column in columns:
                if column == "setting":
                    row.append(run["name"])
                elif column == "trajectory_id":
                    row.append(str(np.asarray(run["data"]["trajectory_id"])[index]))
                else:
                    value = np.asarray(run["data"][column], dtype=np.float64)[index] if column in run["data"] else np.nan
                    row.append("" if not np.isfinite(value) else f"{float(value):.8g}")
            lines.append(",".join(row))
    (save_dir / "snapshots.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = Path(args.report) if args.report else Path(args.save_dir) / "TRACE_COMPARE_REPORT.md"
    render(Path(args.save_dir), report, smoke=args.smoke)
