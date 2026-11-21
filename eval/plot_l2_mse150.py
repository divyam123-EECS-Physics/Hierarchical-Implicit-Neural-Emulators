#!/usr/bin/env python3
"""150-step autoregressive MSE for the two L=2 models.

Uses the same rollout as eval_jacobian_manual_mse_plus.py: one warmup step
from a zero coarse guess, then 150 steps at t=1. Windows are non-overlapping
blocks of the loader-kept eval frames (stem divisible by 5). The band is the
standard deviation across those windows, not a confidence interval on the mean.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_jacobian_manual_mse_plus import (
    choose_sections,
    load_kept_paths,
    load_model,
    load_trajectories,
    spatial_mse,
)

N_STEPS = 150
HORIZON = N_STEPS + 2
NORM = 10.0
SAVE = Path("/pscratch/sd/d/divyam/hine_runs/trace_compare/rollout_mse150")
SLIDE = Path("/global/homes/d/divyam/trace_compare_claim_slides/05_L2_rollout_mse")
ZIP = Path("/global/homes/d/divyam/trace_compare_claim_slides.zip")
RUNS = (
    (
        "l2_k4",
        r"L=2, $k_f=4$",
        "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
        "/pscratch/sd/d/divyam/training_data/paper_nx256_from10000/eval",
        plt.get_cmap("plasma_r")(0.32),
    ),
    (
        "l2_k5",
        r"L=2, $k_f=5$",
        "/pscratch/sd/d/divyam/hine_runs/k5/l2/checkpoint-last.pth",
        "/pscratch/sd/d/divyam/training_data/paper_nx256_fkx5/eval",
        plt.get_cmap("cool_r")(0.42),
    ),
)


@torch.no_grad()
def autoregressive_mse(ema, model, frames_np: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    k = int(model.num_coarse)
    if k != 1:
        raise RuntimeError(f"expected one coarse channel, found {k}")
    pieces = []
    for start in range(0, frames_np.shape[0], batch_size):
        batch = torch.from_numpy(frames_np[start : start + batch_size]).to(device).unsqueeze(2)
        count = batch.shape[0]
        high = batch[:, 0]
        coarse = torch.zeros(count, 1, model.latent_res[0], model.latent_res[0], device=device, dtype=high.dtype)
        t0 = torch.zeros(count, device=device, dtype=high.dtype)
        _, outs = ema(high, [coarse], t0)
        coarse = outs[0]
        t1 = torch.ones(count, device=device, dtype=high.dtype)
        errors = np.zeros((count, N_STEPS), dtype=np.float64)
        for step in range(N_STEPS):
            high, outs = ema(high, [coarse], t1)
            coarse = outs[0]
            errors[:, step] = spatial_mse(high, batch[:, step + 1])
        pieces.append(errors)
    return np.concatenate(pieces, axis=0)


def compute(device: torch.device) -> None:
    SAVE.mkdir(parents=True, exist_ok=True)
    for name, _label, checkpoint, eval_dir, _color in RUNS:
        path = SAVE / f"{name}.npz"
        if path.is_file():
            print(f"[skip] {path}", flush=True)
            continue
        kept = load_kept_paths(eval_dir, 5, "mod")
        chosen = choose_sections(len(kept) // HORIZON, None)
        print(f"[load] {name} windows={len(chosen)} frames={HORIZON}", flush=True)
        trajectories = load_trajectories(kept, chosen, HORIZON)
        frames = np.stack([np.stack(traj, axis=0) for traj in trajectories], axis=0).astype(np.float32) / np.float32(NORM)
        model, ema = load_model(checkpoint, device)
        mse = autoregressive_mse(ema, model, frames, 4, device)
        stems = np.asarray([kept[int(block) * HORIZON][0] for block in chosen], dtype=np.int64)
        np.savez_compressed(path, mse=mse, lead=np.arange(1, N_STEPS + 1), stem=stems, norm=np.float64(NORM))
        print(f"[wrote] {path} shape={mse.shape} final_mean={mse[:, -1].mean():.4g}", flush=True)
        del model, ema, frames
        if device.type == "cuda":
            torch.cuda.empty_cache()


def plot() -> None:
    SLIDE.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, label, _checkpoint, _eval_dir, color in RUNS:
        stored = np.load(SAVE / f"{name}.npz")
        mse = np.asarray(stored["mse"], dtype=np.float64)
        lead = np.asarray(stored["lead"])
        mean = mse.mean(axis=0)
        std = mse.std(axis=0, ddof=1)
        ax.plot(lead, mean, color=color, lw=2.0, label=f"{label} (n={mse.shape[0]})")
        ax.fill_between(lead, mean - std, mean + std, color=color, alpha=0.22, linewidth=0)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("spatial MSE")
    ax.set_title(r"L=2 autoregressive error, $k_f=4$ and $k_f=5$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = SLIDE / "mse_vs_step.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    if ZIP.exists():
        with zipfile.ZipFile(ZIP, "a", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.write(out, out.relative_to(ZIP.parent))
    print(f"wrote {out}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    compute(device)
    plot()


if __name__ == "__main__":
    main()
