#!/usr/bin/env python3
"""Autoregressive fine-field MSE vs lead for FineUNet ablations.

Reuses data loading and spatial_mse from eval_jacobian_manual_mse.py.
L=2/L=3 HINE evaluation stays on the existing Plus and L=3 scripts.
"""

from __future__ import annotations

import argparse
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
import torch

from eval.eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
    spatial_mse,
)
from model_ablation import AblationEmulator, FineUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--every-nth-field",
        type=int,
        default=5,
        help="After sorting by integer stem, keep every Nth file by position",
    )
    p.add_argument(
        "--eval-dir",
        default="/global/homes/d/divyam/TACSLAB/eval",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to ablation checkpoint-last.pth (or equivalent)",
    )
    p.add_argument(
        "--n-initial-conditions",
        type=int,
        default=None,
        help="How many sections to test. Omit to use all that fit --horizon.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Kept frames per non-overlapping section",
    )
    p.add_argument(
        "--norm",
        type=float,
        default=10.0,
        help="Divide Omega by this before the model and MSE",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    p.add_argument(
        "--save-dir",
        default="",
        help="Directory for npz and MSE curves (default: beside checkpoint)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    return p.parse_args()


def load_ablation_model(
    checkpoint_path: str, device: torch.device
) -> tuple[AblationEmulator, FineUNet]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    model = AblationEmulator(ckpt["args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model.ema_vit


@torch.no_grad()
def rollout_ar(
    ema_vit: FineUNet,
    frames_np: np.ndarray,
    batch_size: int,
    n_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll fine field autoregressively; return (pred [B,T,1,H,W], mse [B,T])."""
    pred_all = []
    mse_all = []
    ema_vit.eval()
    aux = getattr(ema_vit, "num_aux", 0) > 0
    for start in range(0, frames_np.shape[0], batch_size):
        batch = torch.from_numpy(frames_np[start : start + batch_size]).to(device)
        batch = batch.unsqueeze(2)  # B, T, 1, H, W
        high = batch[:, 0]
        preds = []
        mses = np.zeros((batch.shape[0], n_steps), dtype=np.float64)
        for k in range(n_steps):
            out = ema_vit(high)
            high = out[0] if aux else out
            true_h = batch[:, k + 1]
            mses[:, k] = spatial_mse(high, true_h)
            preds.append(high.detach().float().cpu())
        pred_all.append(torch.stack(preds, 1).numpy())
        mse_all.append(mses)
    return np.concatenate(pred_all, 0), np.concatenate(mse_all, 0)


def plot_mse_curve(save_dir: str, lead: np.ndarray, mse: np.ndarray) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    for row in range(mse.shape[0]):
        ax.plot(lead, mse[row], color="C0", alpha=0.25, lw=0.8)
    ax.plot(lead, mse.mean(axis=0), color="black", lw=2.5, label="AR mean")
    ax.set_xlabel("lead")
    ax.set_ylabel("MSE")
    ax.set_title("fine-field autoregressive MSE")
    ax.legend()
    out = os.path.join(save_dir, "mse_vs_lead.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    if args.every_nth_field < 1:
        raise ValueError(f"every-nth-field must be >= 1, got {args.every_nth_field}")
    if args.horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {args.horizon}")
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {args.batch_size}")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU", file=sys.stderr)

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "ablation_mse"
    )
    os.makedirs(save_dir, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    if n_available < 1:
        raise ValueError(
            f"Need at least {args.horizon} frames after every {args.every_nth_field}th, "
            f"found {len(kept)} kept files."
        )
    chosen = choose_sections(n_available, args.n_initial_conditions)
    trajectories = load_trajectories(kept, chosen, args.horizon)
    print(
        f"sorted kept={len(kept)} every_{args.every_nth_field}th "
        f"full_sections={n_available} using={len(chosen)}"
    )

    model, ema_vit = load_ablation_model(args.checkpoint, device)
    print(f"checkpoint={args.checkpoint} ablation={model.ablation} device={device}")

    frames = np.stack(
        [np.stack(traj, axis=0) for traj in trajectories], axis=0
    ).astype(np.float32)
    frames = frames / float(args.norm)
    n_steps = frames.shape[1] - 1
    lead = np.arange(1, n_steps + 1)

    pred, mse = rollout_ar(ema_vit, frames, args.batch_size, n_steps, device)
    np.savez_compressed(
        os.path.join(save_dir, "ablation_mse.npz"),
        lead=lead,
        mse=mse,
        pred=pred,
        chosen=chosen,
        ablation=np.array(model.ablation),
    )
    curve = plot_mse_curve(save_dir, lead, mse)
    print(f"saved {save_dir}/ablation_mse.npz and {curve}")
    print(
        "mse mean@leads:",
        {int(lead[i]): float(mse[:, i].mean()) for i in (0, min(4, n_steps - 1), n_steps - 1)},
    )


if __name__ == "__main__":
    main()
