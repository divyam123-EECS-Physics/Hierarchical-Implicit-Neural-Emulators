#!/usr/bin/env python3
"""Batched high + coarse rollout MSE for FutureINNPlus checkpoints.

Does not modify eval_jacobian_manual_mse.py. Horizon rule:
  n_steps = horizon - (K + 1),  horizon >= num_of_frame = K + 2.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch.nn import functional as F

from model_plus import FutureINNPlus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/checkpoint-last.pth",
    )
    p.add_argument("--n-initial-conditions", type=int, default=None)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/jacobian_manual_mse",
    )
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def load_kept_paths(eval_dir: str, every_nth_field: int) -> list[tuple[int, str]]:
    numbered: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(eval_dir, "*.mat")):
        try:
            numbered.append((int(Path(path).stem), path))
        except ValueError:
            continue
    numbered.sort(key=lambda item: item[0])
    return numbered[::every_nth_field]


def choose_sections(
    n_available: int, n_initial_conditions: int | None
) -> np.ndarray:
    if n_available < 1:
        raise ValueError("Need at least one full section")
    if n_initial_conditions is None:
        return np.arange(n_available)
    if n_initial_conditions > n_available:
        raise ValueError(
            f"Asked for {n_initial_conditions} sections, only {n_available} fit."
        )
    return np.arange(n_initial_conditions)


def load_trajectories(
    kept: list[tuple[int, str]], chosen: np.ndarray, horizon: int
) -> list[list[np.ndarray]]:
    trajectories = []
    for block in chosen:
        start = int(block) * horizon
        frames = [
            scipy.io.loadmat(kept[start + k][1])["Omega"] for k in range(horizon)
        ]
        trajectories.append(frames)
    return trajectories


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[FutureINNPlus, torch.nn.Module]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    model = FutureINNPlus(ckpt["args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model.ema_vit


def spatial_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (pred - target).pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()


@torch.no_grad()
def rollout(
    ema_vit: torch.nn.Module,
    model: FutureINNPlus,
    frames_np: np.ndarray,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    teacher_forced: bool,
) -> dict[str, np.ndarray]:
    k = model.num_coarse
    t_full = float(k)
    pred_high_all, gt_high_all, mse_high_all = [], [], []
    pred_c_all = [[] for _ in range(k)]
    gt_c_all = [[] for _ in range(k)]
    mse_c_all = [[] for _ in range(k)]
    ema_vit.eval()

    for start in range(0, frames_np.shape[0], batch_size):
        batch = torch.from_numpy(frames_np[start : start + batch_size]).to(device)
        batch = batch.unsqueeze(2)
        b = batch.shape[0]
        high = batch[:, 0]
        coarses = [
            torch.zeros(
                b, 1, model.latent_res[i], model.latent_res[i], device=device, dtype=high.dtype
            )
            for i in range(k)
        ]
        # Warmup indicators 0 .. K-1
        for c in range(k):
            t = float(c) * torch.ones(b, device=device, dtype=high.dtype)
            _, outs = ema_vit(high, coarses, t)
            coarses[c] = outs[c]

        tK = t_full * torch.ones(b, device=device, dtype=high.dtype)
        hs, gh = [], []
        cs = [[] for _ in range(k)]
        gcs = [[] for _ in range(k)]
        eh = np.zeros((b, n_steps), dtype=np.float64)
        ec = [np.zeros((b, n_steps), dtype=np.float64) for _ in range(k)]

        for step in range(n_steps):
            if teacher_forced:
                high = batch[:, step]
            high, coarses = ema_vit(high, coarses, tK)
            true_h = batch[:, step + 1]
            eh[:, step] = spatial_mse(high, true_h)
            hs.append(high.detach().float().cpu())
            gh.append(true_h.detach().float().cpu())
            for i in range(k):
                true_c = F.avg_pool2d(
                    batch[:, step + 2 + i],
                    kernel_size=model.ds_factors[i],
                    stride=model.ds_factors[i],
                )
                ec[i][:, step] = spatial_mse(coarses[i], true_c)
                cs[i].append(coarses[i].detach().float().cpu())
                gcs[i].append(true_c.detach().float().cpu())

        pred_high_all.append(torch.stack(hs, 1).numpy())
        gt_high_all.append(torch.stack(gh, 1).numpy())
        mse_high_all.append(eh)
        for i in range(k):
            pred_c_all[i].append(torch.stack(cs[i], 1).numpy())
            gt_c_all[i].append(torch.stack(gcs[i], 1).numpy())
            mse_c_all[i].append(ec[i])

    cat = lambda xs: np.concatenate(xs, 0)
    out = {
        "pred_high": cat(pred_high_all),
        "gt_high": cat(gt_high_all),
        "mse_high": cat(mse_high_all),
    }
    for i in range(k):
        out[f"pred_coarse_{i}"] = cat(pred_c_all[i])
        out[f"gt_coarse_{i}"] = cat(gt_c_all[i])
        out[f"mse_coarse_{i}"] = cat(mse_c_all[i])
    return out


def plot_mse_curves(save_dir: str, lead: np.ndarray, ar: dict, tf: dict, k: int) -> str:
    n_rows = 1 + k
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 2.8 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    series = [("high", "mse_high")] + [(f"coarse{i}", f"mse_coarse_{i}") for i in range(k)]
    for ax, (name, key) in zip(axes, series):
        ax.semilogy(lead, ar[key].mean(0), label=f"AR {name}")
        ax.semilogy(lead, tf[key].mean(0), label=f"TF {name}")
        ax.set_ylabel("MSE")
        ax.legend()
    axes[-1].set_xlabel("lead")
    out = os.path.join(save_dir, "mse_vs_lead.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    if args.every_nth_field < 1:
        raise ValueError(f"every-nth-field must be >= 1, got {args.every_nth_field}")
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {args.batch_size}")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU", file=sys.stderr)

    os.makedirs(args.save_dir, exist_ok=True)
    fields_dir = os.path.join(args.save_dir, "fields")
    os.makedirs(fields_dir, exist_ok=True)

    model, ema_vit = load_model(args.checkpoint, device)
    k = model.num_coarse
    min_horizon = k + 2
    if args.horizon < min_horizon:
        raise ValueError(
            f"horizon must be >= num_of_frame={min_horizon}, got {args.horizon}"
        )
    n_steps = args.horizon - (k + 1)
    print(
        f"checkpoint={args.checkpoint} L={k+1} K={k} "
        f"latent_res={model.latent_res} n_steps={n_steps} device={device}"
    )

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    if n_available < 1:
        raise ValueError(
            f"Need at least {args.horizon} frames after taking every "
            f"{args.every_nth_field}th, found {len(kept)} kept files."
        )
    chosen = choose_sections(n_available, args.n_initial_conditions)
    trajectories = load_trajectories(kept, chosen, args.horizon)
    print(
        f"sorted kept={len(kept)} every_{args.every_nth_field}th "
        f"full_sections={n_available} using={len(chosen)}"
    )

    n_frames = len(trajectories[0])
    frames_np = np.stack(
        [np.stack(traj, axis=0) for traj in trajectories], axis=0
    ).astype(np.float32) / np.float32(args.norm)

    file_stem = np.zeros((len(trajectories), n_frames), dtype=np.int64)
    kept_index = np.zeros_like(file_stem)
    for b, block in enumerate(chosen):
        start = int(block) * args.horizon
        for step in range(n_frames):
            kept_index[b, step] = start + step
            file_stem[b, step] = kept[start + step][0]
    lead = np.arange(1, n_steps + 1, dtype=np.int64)
    target_stem = file_stem[:, 1 : n_steps + 1]
    target_kept_index = kept_index[:, 1 : n_steps + 1]

    ar = rollout(
        ema_vit, model, frames_np, args.batch_size, n_steps, device, teacher_forced=False
    )
    tf = rollout(
        ema_vit, model, frames_np, args.batch_size, n_steps, device, teacher_forced=True
    )

    npz_path = os.path.join(args.save_dir, "rollout_mse.npz")
    payload = {
        "ar_pred_high": ar["pred_high"],
        "tf_pred_high": tf["pred_high"],
        "gt_high": ar["gt_high"],
        "ar_mse_high": ar["mse_high"],
        "tf_mse_high": tf["mse_high"],
        "lead": lead,
        "file_stem": file_stem,
        "target_stem": target_stem,
        "kept_index": kept_index,
        "target_kept_index": target_kept_index,
        "chosen": np.asarray(chosen),
        "checkpoint": np.asarray(args.checkpoint),
        "eval_dir": np.asarray(args.eval_dir),
        "horizon": np.asarray(args.horizon, dtype=np.int64),
        "every_nth_field": np.asarray(args.every_nth_field, dtype=np.int64),
        "norm": np.asarray(args.norm, dtype=np.float64),
        "latent_res": np.asarray(model.latent_res, dtype=np.int64),
        "num_coarse": np.asarray(k, dtype=np.int64),
    }
    for i in range(k):
        payload[f"ar_pred_coarse_{i}"] = ar[f"pred_coarse_{i}"]
        payload[f"tf_pred_coarse_{i}"] = tf[f"pred_coarse_{i}"]
        payload[f"gt_coarse_{i}"] = ar[f"gt_coarse_{i}"]
        payload[f"ar_mse_coarse_{i}"] = ar[f"mse_coarse_{i}"]
        payload[f"tf_mse_coarse_{i}"] = tf[f"mse_coarse_{i}"]
    # Paper-style aliases when K>=2
    if k >= 1:
        payload["ar_pred_mid"] = ar["pred_coarse_0"]
        payload["ar_mse_mid"] = ar["mse_coarse_0"]
        payload["gt_mid"] = ar["gt_coarse_0"]
    if k >= 2:
        payload["ar_pred_low"] = ar[f"pred_coarse_{k-1}"]
        payload["ar_mse_low"] = ar[f"mse_coarse_{k-1}"]
        payload["gt_low"] = ar[f"gt_coarse_{k-1}"]

    np.savez_compressed(npz_path, **payload)
    print(
        f"saved {npz_path}  B={ar['mse_high'].shape[0]} T={ar['mse_high'].shape[1]}  "
        f"lead1 high AR {ar['mse_high'][:, 0].mean():.4g}"
    )
    mse_plot = plot_mse_curves(args.save_dir, lead, ar, tf, k)
    print(f"wrote {mse_plot}")
    print("JACOBIAN_MANUAL_MSE_PLUS_DONE")


if __name__ == "__main__":
    main()
