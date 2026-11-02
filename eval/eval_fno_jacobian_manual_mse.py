#!/usr/bin/env python3
"""FNO high-field rollout MSE (separate from the HINE script).

Integer-sorts eval *.mat files, takes every Nth by position, tiles non-overlapping
horizon sections, and rolls FNO2d in autoregressive and teacher-forced modes.
Supports 1-to-1, 2-to-2, and 3-to-3 maps (channel counts read from the checkpoint).
Saves high-field predictions, ground truth, MSE arrays, field figures, and MSE curves.
Does not import or modify eval_jacobian_manual_mse.py beyond shared data helpers.
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

from eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
)
from fno_baseline import FNO2d


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
        default="/pscratch/sd/d/divyam/hine_runs/fno_baseline_debug/checkpoint-last.pth",
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
        help="Device to run the model on",
    )
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/fno_baseline_debug/jacobian_manual_mse",
        help="Directory for the npz, the field figures, and the MSE curves",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sections per forward chunk",
    )
    return p.parse_args()


def load_fno(checkpoint_path: str, device: torch.device) -> FNO2d:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    a = ckpt["args"]
    modes = int(getattr(a, "modes", 96))
    width = int(getattr(a, "width", 32))
    n_layers = int(getattr(a, "n_layers", 3))
    in_steps = int(getattr(a, "in_steps", 1))
    out_steps = int(getattr(a, "out_steps", 1))
    model = FNO2d(
        modes=modes,
        width=width,
        n_layers=n_layers,
        in_channels=in_steps,
        out_channels=out_steps,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def spatial_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (pred - target).pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()


@torch.no_grad()
def rollout_fno(
    net: FNO2d,
    frames_np: np.ndarray,
    batch_size: int,
    n_in: int,
    n_out: int,
    device: torch.device,
    teacher_forced: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block map with time as channels. frames_np: (B, T, H, W).

    Stores one lead per predicted frame as (B, leads, 1, H, W).
    """
    n_leads = ((frames_np.shape[1] - n_in) // n_out) * n_out
    if n_leads < 1:
        raise ValueError(
            f"Need horizon > n_in and (horizon - n_in) >= n_out; "
            f"got T={frames_np.shape[1]} n_in={n_in} n_out={n_out}"
        )
    pred_high: list[np.ndarray] = []
    gt_high: list[np.ndarray] = []
    mse_high: list[np.ndarray] = []
    net.eval()
    for start in range(0, frames_np.shape[0], batch_size):
        batch = torch.from_numpy(frames_np[start : start + batch_size]).to(device)
        b = batch.shape[0]
        ctx = batch[:, :n_in]
        hs: list[torch.Tensor] = []
        gh: list[torch.Tensor] = []
        eh = np.zeros((b, n_leads), dtype=np.float64)
        for t in range(0, n_leads, n_out):
            if teacher_forced:
                ctx = batch[:, t : t + n_in]
            pred = net(ctx)
            true = batch[:, t + n_in : t + n_in + n_out]
            for j in range(n_out):
                eh[:, t + j] = spatial_mse(pred[:, j : j + 1], true[:, j : j + 1])
                hs.append(pred[:, j : j + 1].detach().float().cpu())
                gh.append(true[:, j : j + 1].detach().float().cpu())
            ctx = pred
        pred_high.append(torch.stack(hs, 1).numpy())
        gt_high.append(torch.stack(gh, 1).numpy())
        mse_high.append(eh)
    cat = lambda xs: np.concatenate(xs, 0)
    return cat(pred_high), cat(gt_high), cat(mse_high)


def field_figure(
    b: int,
    k: int,
    chosen: np.ndarray,
    lead: np.ndarray,
    target_stem: np.ndarray,
    ar_pred_high: np.ndarray,
    tf_pred_high: np.ndarray,
    gt_high: np.ndarray,
):
    ar = ar_pred_high[b, k, 0]
    tf = tf_pred_high[b, k, 0]
    gt = gt_high[b, k, 0]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.2), constrained_layout=True)
    titles = ["autoregressive", "teacher forced", "ground truth", "|AR − GT|", "|TF − GT|"]
    field_lim = (
        min(ar.min(), tf.min(), gt.min()),
        max(ar.max(), tf.max(), gt.max()),
    )
    panels = [ar, tf, gt, np.abs(ar - gt), np.abs(tf - gt)]
    for c, panel in enumerate(panels):
        kw = {"vmin": field_lim[0], "vmax": field_lim[1]} if c < 3 else {}
        im = axs[c].imshow(panel, **kw)
        axs[c].set_xticks([])
        axs[c].set_yticks([])
        fig.colorbar(im, ax=axs[c], fraction=0.046)
        axs[c].set_title(titles[c])
    axs[0].set_ylabel("high")
    fig.suptitle(
        f"section {int(chosen[b])}  lead {int(lead[k])}  stem {int(target_stem[b, k])}"
    )
    return fig


def plot_mse_curves(
    save_dir: str,
    lead: np.ndarray,
    ar_mse_high: np.ndarray,
    tf_mse_high: np.ndarray,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ar_mean = ar_mse_high.mean(axis=0)
    tf_mean = tf_mse_high.mean(axis=0)
    for row in range(ar_mse_high.shape[0]):
        ax.plot(lead, ar_mse_high[row], color="C0", alpha=0.25, lw=0.8)
        ax.plot(lead, tf_mse_high[row], color="C1", alpha=0.25, lw=0.8)
    ax.plot(lead, ar_mean, color="black", lw=2.5, label="AR mean")
    ax.plot(lead, tf_mean, color="black", lw=2.5, ls="--", label="TF mean")
    ax.set_title("high")
    ax.set_xlabel("lead")
    ax.set_ylabel("MSE")
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

    os.makedirs(args.save_dir, exist_ok=True)
    fields_dir = os.path.join(args.save_dir, "fields")
    os.makedirs(fields_dir, exist_ok=True)

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
        f"full_sections={n_available} using={len(chosen)} "
        f"section_index={chosen.tolist()}"
    )

    net = load_fno(args.checkpoint, device)
    n_in = int(net.in_channels)
    n_out = int(net.out_channels)
    if n_in != n_out:
        raise ValueError(f"AR eval requires in_channels == out_channels, got {n_in}/{n_out}")
    if args.horizon <= n_in:
        raise ValueError(
            f"horizon must be > n_in ({n_in}), got {args.horizon}"
        )
    if (args.horizon - n_in) % n_out != 0:
        raise ValueError(
            f"(horizon - n_in) must be divisible by n_out; "
            f"got horizon={args.horizon} n_in={n_in} n_out={n_out}"
        )
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"checkpoint={args.checkpoint} modes={net.modes} width={net.width} "
        f"layers={net.n_layers} in={n_in} out={n_out} "
        f"params={n_params / 1e6:.2f}M device={device}"
    )

    n_frames = len(trajectories[0])
    n_leads = ((n_frames - n_in) // n_out) * n_out
    frames_np = np.stack(
        [np.stack(traj, axis=0) for traj in trajectories], axis=0
    ).astype(np.float32) / np.float32(args.norm)

    file_stem = np.zeros((len(trajectories), n_frames), dtype=np.int64)
    kept_index = np.zeros_like(file_stem)
    for b, block in enumerate(chosen):
        start = int(block) * args.horizon
        for k in range(n_frames):
            kept_index[b, k] = start + k
            file_stem[b, k] = kept[start + k][0]
    # Lead 1 = first predicted frame (index n_in in the section).
    lead = np.arange(1, n_leads + 1, dtype=np.int64)
    target_stem = file_stem[:, n_in : n_in + n_leads]
    target_kept_index = kept_index[:, n_in : n_in + n_leads]

    ar_pred_high, gt_high, ar_mse_high = rollout_fno(
        net, frames_np, args.batch_size, n_in, n_out, device, teacher_forced=False
    )
    tf_pred_high, _, tf_mse_high = rollout_fno(
        net, frames_np, args.batch_size, n_in, n_out, device, teacher_forced=True
    )

    npz_path = os.path.join(args.save_dir, "rollout_mse.npz")
    np.savez_compressed(
        npz_path,
        ar_pred_high=ar_pred_high,
        tf_pred_high=tf_pred_high,
        gt_high=gt_high,
        ar_mse_high=ar_mse_high,
        tf_mse_high=tf_mse_high,
        lead=lead,
        file_stem=file_stem,
        target_stem=target_stem,
        kept_index=kept_index,
        target_kept_index=target_kept_index,
        chosen=np.asarray(chosen),
        checkpoint=np.asarray(args.checkpoint),
        eval_dir=np.asarray(args.eval_dir),
        horizon=np.asarray(args.horizon, dtype=np.int64),
        every_nth_field=np.asarray(args.every_nth_field, dtype=np.int64),
        norm=np.asarray(args.norm, dtype=np.float64),
        in_steps=np.asarray(n_in, dtype=np.int64),
        out_steps=np.asarray(n_out, dtype=np.int64),
        model_type=np.asarray("fno"),
    )
    print(
        f"saved {npz_path}  B={ar_mse_high.shape[0]} T={ar_mse_high.shape[1]}  "
        f"lead1 high AR {ar_mse_high[:, 0].mean():.4g} TF {tf_mse_high[:, 0].mean():.4g}"
    )

    plot_leads = [0, n_leads // 2, n_leads - 1]
    for b in range(ar_mse_high.shape[0]):
        for k in plot_leads:
            fig = field_figure(
                b,
                k,
                chosen,
                lead,
                target_stem,
                ar_pred_high,
                tf_pred_high,
                gt_high,
            )
            fig.savefig(
                os.path.join(fields_dir, f"b{b:03d}_lead{int(lead[k]):03d}.png"),
                dpi=120,
            )
            plt.close(fig)

    mse_plot = plot_mse_curves(args.save_dir, lead, ar_mse_high, tf_mse_high)
    print(f"wrote fields under {fields_dir}")
    print(f"wrote {mse_plot}")
    print("FNO_JACOBIAN_MANUAL_MSE_DONE")


if __name__ == "__main__":
    main()
