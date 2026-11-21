#!/usr/bin/env python3
"""Snapshot strips for the two-step feedback traces.

Model setup is on the x-axis and each saved snapshot is one dot. The three
route traces share a vertical scale. Probe standard errors stay in the npz;
the whisker is the interquartile range across snapshots.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/pscratch/sd/d/divyam/hine_runs/trace_feedback")
OUT = Path("/global/homes/d/divyam/trace_feedback_slides")
ZIP = Path("/global/homes/d/divyam/trace_feedback_slides.zip")
COLOR_K4 = plt.get_cmap("plasma_r")(0.32)
COLOR_K5 = plt.get_cmap("cool_r")(0.42)
COLOR_L1_K4 = plt.get_cmap("plasma_r")(0.72)
COLOR_L1_K5 = plt.get_cmap("cool_r")(0.75)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _jitter(stem: int, width: float = 0.12) -> float:
    return float(np.random.default_rng(int(stem) % 1_000_003).uniform(-width, width))


def _load(path: Path) -> dict:
    stored = np.load(path)
    return {key: np.asarray(stored[key]) for key in stored.files}


def _strip(ax, series: list[tuple[str, np.ndarray, np.ndarray, object]], ylabel: str, zero: bool) -> None:
    drawn = []
    for position, (label, values, stems, color) in enumerate(series):
        finite = []
        for stem, value in zip(np.asarray(stems).reshape(-1), np.asarray(values, dtype=np.float64).reshape(-1)):
            if not np.isfinite(value):
                continue
            finite.append(value)
            ax.scatter(position + _jitter(int(stem)), value, s=26, color=color, alpha=0.8, linewidths=0, zorder=3)
        if not finite:
            continue
        drawn.append(np.asarray(finite))
        low, med, high = np.percentile(finite, [25, 50, 75])
        ax.plot([position, position], [low, high], color="black", lw=2.2, zorder=4)
        ax.plot([position - 0.16, position + 0.16], [med, med], color="black", lw=2.0, zorder=5)
        ax.scatter([], [], s=26, color=color, label=label)
    if not drawn:
        ax.set_ylabel(ylabel)
        return
    pads = np.concatenate(drawn)
    low = float(pads.min())
    high = float(pads.max())
    if zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low or 1.0
    pad = 0.08 * span
    ax.set_ylim(low - pad, high + pad)
    if low - pad <= 0.0 <= high + pad:
        ax.axhline(0.0, color="0.35", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([item[0] for item in series], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)


def _paired(folder: Path, names: list[str], ylabel: str, zero: bool, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = []
    for run in runs:
        series.append((run["label"], run["data"][names[0]], run["data"]["stem"], run["color"]))
    _strip(ax, series, ylabel, zero)
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{names[0]}.png", dpi=140)
    plt.close(fig)


def _route_figure(folder: Path, runs: list[dict]) -> None:
    keys = (
        ("tau_a", r"$\mathrm{tr}(D^{+}C_n)/N$"),
        ("tau_b", r"$\mathrm{tr}((B_{n+1}-D^{+})C_n)/N$"),
        ("tau_loop", r"$\mathrm{tr}(B_{n+1}C_n)/N$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), sharey=True)
    collected = []
    prepared = []
    for key, ylabel in keys:
        series = []
        for run in runs:
            values = run["data"][key]
            series.append((run["label"], values, run["data"]["stem"], run["color"]))
            collected.append(_finite(values))
        prepared.append((series, ylabel))
    pads = np.concatenate(collected) if collected else np.asarray([0.0])
    low = min(float(pads.min()), 0.0)
    high = max(float(pads.max()), 0.0)
    span = high - low or 1.0
    pad = 0.08 * span
    for ax, (series, ylabel) in zip(axes, prepared):
        _strip(ax, series, ylabel, zero=False)
        ax.set_ylim(low - pad, high + pad)
        ax.axhline(0.0, color="0.35", ls="--", lw=0.8, zorder=1)
    fig.tight_layout()
    fig.savefig(folder / "route_traces.png", dpi=140)
    plt.close(fig)


def _direct_figure(folder: Path, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = []
    for run in runs:
        if "tau_J" in run["data"] and _finite(run["data"]["tau_J"]).size:
            series.append((run["l1_label"], run["data"]["tau_J"], run["data"]["stem"], run["l1_color"]))
        series.append((run["label"], run["data"]["tau_A"], run["data"]["stem"], run["color"]))
    _strip(ax, series, r"$\mathrm{tr}(J_n-I)/N,\ \mathrm{tr}(A_n-I)/N$", zero=False)
    fig.tight_layout()
    fig.savefig(folder / "direct_trace.png", dpi=140)
    plt.close(fig)


def _median(values) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _report(path: Path, runs: list[dict]) -> None:
    lines = [
        "# Two-step feedback traces",
        "",
        "Each number is the median across saved snapshots. Probe standard errors are in the npz files and are not confidence intervals on the median.",
        "",
    ]
    for run in runs:
        data = run["data"]
        lines.append(f"## {run['label']}")
        lines.append("")
        lines.append(
            f"tau_a {_median(data['tau_a']):.6g}, tau_b {_median(data['tau_b']):.6g}, "
            f"tau_loop {_median(data['tau_loop']):.6g}, delta_loop {_median(data['delta_loop']):.6g}."
        )
        lines.append(
            f"tr(A-I)/N {_median(data['tau_A']):.6g}, tr(K-I)/M {_median(data['tau_K']):.6g}, "
            f"tr(J-I)/N {_median(data['tau_J']):.6g}."
        )
        lines.append(
            f"Rollout RMS lift {_median(data['rms_lift']):.6g}, revise {_median(data['rms_revise']):.6g}, "
            f"loop {_median(data['rms_loop']):.6g}. "
            f"cosine(lift, revise) {_median(data['cos_lift_revise']):.6g}, "
            f"cosine(lift, loop) {_median(data['cos_lift_loop']):.6g}."
        )
        lines.append("")
    lines.extend(
        [
            "A negative tau_a or tau_b means that route opposes the starting fine probe on average. "
            "A negative delta_loop means the latent loop reduced the mean squared two-step fine response relative to the direct route, for these probes.",
            "",
            "RMS and the two cosines are the same operators applied to the free-rollout fine error at the teacher-forced Jacobian. "
            "A small revision next to a non-negligible loop, with cosine(lift, loop) near one, means the upsampled coarse prediction carries most of that feedback. "
            "Two large pieces with cosine(lift, revise) near minus one are a cancellation.",
            "",
            "These signs do not by themselves show that the coarse state drives the rollout. "
            "That comparison also needs the direct fine route and the realized state increments.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def plot_available(root: Path = ROOT, out: Path = OUT, zip_path: Path = ZIP) -> list[Path]:
    catalog = (
        ("l2_k4", r"L=2, $k_f=4$", r"L=1, $k_f=4$", COLOR_K4, COLOR_L1_K4),
        ("l2_k5", r"L=2, $k_f=5$", r"L=1, $k_f=5$", COLOR_K5, COLOR_L1_K5),
    )
    runs = []
    for name, label, l1_label, color, l1_color in catalog:
        path = root / name / "metrics.npz"
        if path.is_file():
            runs.append(
                {
                    "name": name,
                    "label": label,
                    "l1_label": l1_label,
                    "color": color,
                    "l1_color": l1_color,
                    "data": _load(path),
                }
            )
    if not runs:
        print(f"no metrics under {root}")
        return []
    if out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True)
    _route_figure(out, runs)
    _direct_figure(out, runs)
    _paired(out, ["delta_loop"], r"$\Delta_{\mathrm{loop}}$", True, runs)
    _paired(out, ["tau_K"], r"$\mathrm{tr}(K_n-I)/M$", True, runs)
    _paired(out, ["rms_lift"], "RMS(lift)", False, runs)
    _paired(out, ["rms_revise"], "RMS(revise)", False, runs)
    _paired(out, ["rms_loop"], "RMS(loop)", False, runs)
    _paired(out, ["cos_lift_revise"], r"$\cos(\mathrm{lift},\mathrm{revise})$", True, runs)
    _paired(out, ["cos_lift_loop"], r"$\cos(\mathrm{lift},\mathrm{loop})$", True, runs)
    _report(out / "TRACE_FEEDBACK_REPORT.md", runs)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(out.parent))
    written = sorted(out.glob("*.png"))
    print(f"zip {zip_path} files {len(written)}")
    return written


if __name__ == "__main__":
    plot_available()
