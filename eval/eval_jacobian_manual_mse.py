#!/usr/bin/env python3
"""Batched high/mid/low rollout MSE from jacobian_manual.ipynb.

Integer-sorts eval *.mat files, takes every Nth by position, tiles non-overlapping
horizon sections, and rolls the t=2 ema_vit map in autoregressive and teacher-forced
modes. Saves predictions, ground truth, MSE arrays, field figures, and MSE curves.
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

from model import FutureINN


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
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/checkpoint-last.pth",
    )
    p.add_argument(
        "--model-type",
        choices=["hine"],
        default="hine",
        help="Checkpoint class. Only the hierarchical implicit neural emulator for now.",
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
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_manual_mse",
        help="Directory for the npz, the field figures, and the MSE curves",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sections per forward chunk",
    )
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
    return (np.arange(n_initial_conditions) * n_available) // n_initial_conditions


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


def _remap_low_res_out_keys(
    state_dict: dict, mid_res: int, low_res: int
) -> dict:
    """Map older resolution-keyed heads (e.g. low_res_out.32) to index keys (0/1)."""
    out: dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("vit.low_res_out.", "ema_vit.low_res_out."):
            if key.startswith(f"{prefix}{mid_res}."):
                new_key = f"{prefix}0." + key[len(f"{prefix}{mid_res}.") :]
                break
            if key.startswith(f"{prefix}{low_res}."):
                new_key = f"{prefix}1." + key[len(f"{prefix}{low_res}.") :]
                break
        out[new_key] = value
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[FutureINN, torch.nn.Module]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    model = FutureINN(ckpt["args"]).to(device)
    state = ckpt["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # paper_1gpu and other older L3 ckpts keyed heads by spatial size.
        state = _remap_low_res_out_keys(state, model.ds_res, model.low_res)
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model.ema_vit


def spatial_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (pred - target).pow(2).mean(dim=(1, 2, 3)).double().cpu().numpy()


@torch.no_grad()
def rollout(
    ema_vit: torch.nn.Module,
    model: FutureINN,
    frames_np: np.ndarray,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    teacher_forced: bool,
) -> tuple[np.ndarray, ...]:
    pred_high, pred_mid, pred_low = [], [], []
    gt_high, gt_mid, gt_low = [], [], []
    mse_high, mse_mid, mse_low = [], [], []
    ema_vit.eval()
    for start in range(0, frames_np.shape[0], batch_size):
        batch = torch.from_numpy(frames_np[start : start + batch_size]).to(device)
        batch = batch.unsqueeze(2)
        b = batch.shape[0]
        high = batch[:, 0]
        mid0 = torch.zeros(
            b, 1, model.ds_res, model.ds_res, device=device, dtype=high.dtype
        )
        low0 = torch.zeros(
            b, 1, model.low_res, model.low_res, device=device, dtype=high.dtype
        )
        _, mid, _ = ema_vit(
            high, mid0, low0, torch.zeros(b, device=device, dtype=high.dtype)
        )
        _, _, low = ema_vit(
            high, mid, low0, torch.ones(b, device=device, dtype=high.dtype)
        )
        t2 = 2 * torch.ones(b, device=device, dtype=high.dtype)
        hs, ms, ls, gh, gm, gl = [], [], [], [], [], []
        eh = np.zeros((b, n_steps), dtype=np.float64)
        em = np.zeros((b, n_steps), dtype=np.float64)
        el = np.zeros((b, n_steps), dtype=np.float64)
        for k in range(n_steps):
            if teacher_forced:
                high = batch[:, k]
            high, mid, low = ema_vit(high, mid, low, t2)
            true_h = batch[:, k + 1]
            true_m = F.avg_pool2d(
                batch[:, k + 2], kernel_size=model.ds_factor, stride=model.ds_factor
            )
            true_l = F.avg_pool2d(
                batch[:, k + 3],
                kernel_size=model.ds_factor_low,
                stride=model.ds_factor_low,
            )
            eh[:, k] = spatial_mse(high, true_h)
            em[:, k] = spatial_mse(mid, true_m)
            el[:, k] = spatial_mse(low, true_l)
            hs.append(high.detach().float().cpu())
            ms.append(mid.detach().float().cpu())
            ls.append(low.detach().float().cpu())
            gh.append(true_h.detach().float().cpu())
            gm.append(true_m.detach().float().cpu())
            gl.append(true_l.detach().float().cpu())
        pred_high.append(torch.stack(hs, 1).numpy())
        pred_mid.append(torch.stack(ms, 1).numpy())
        pred_low.append(torch.stack(ls, 1).numpy())
        gt_high.append(torch.stack(gh, 1).numpy())
        gt_mid.append(torch.stack(gm, 1).numpy())
        gt_low.append(torch.stack(gl, 1).numpy())
        mse_high.append(eh)
        mse_mid.append(em)
        mse_low.append(el)
    cat = lambda xs: np.concatenate(xs, 0)
    return tuple(
        cat(xs)
        for xs in (
            pred_high,
            pred_mid,
            pred_low,
            gt_high,
            gt_mid,
            gt_low,
            mse_high,
            mse_mid,
            mse_low,
        )
    )


def field_figure(
    b: int,
    k: int,
    chosen: np.ndarray,
    lead: np.ndarray,
    target_stem: np.ndarray,
    ar_pred_high: np.ndarray,
    ar_pred_mid: np.ndarray,
    ar_pred_low: np.ndarray,
    tf_pred_high: np.ndarray,
    tf_pred_mid: np.ndarray,
    tf_pred_low: np.ndarray,
    gt_high: np.ndarray,
    gt_mid: np.ndarray,
    gt_low: np.ndarray,
):
    rows = [
        ("high", ar_pred_high[b, k, 0], tf_pred_high[b, k, 0], gt_high[b, k, 0]),
        ("mid", ar_pred_mid[b, k, 0], tf_pred_mid[b, k, 0], gt_mid[b, k, 0]),
        ("low", ar_pred_low[b, k, 0], tf_pred_low[b, k, 0], gt_low[b, k, 0]),
    ]
    fig, axs = plt.subplots(3, 5, figsize=(16, 8), constrained_layout=True)
    titles = ["autoregressive", "teacher forced", "ground truth", "|AR − GT|", "|TF − GT|"]
    for r, (name, ar, tf, gt) in enumerate(rows):
        field_lim = (
            min(ar.min(), tf.min(), gt.min()),
            max(ar.max(), tf.max(), gt.max()),
        )
        panels = [ar, tf, gt, np.abs(ar - gt), np.abs(tf - gt)]
        for c, panel in enumerate(panels):
            kw = {"vmin": field_lim[0], "vmax": field_lim[1]} if c < 3 else {}
            im = axs[r, c].imshow(panel, **kw)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            fig.colorbar(im, ax=axs[r, c], fraction=0.046)
            if r == 0:
                axs[r, c].set_title(titles[c])
            if c == 0:
                axs[r, c].set_ylabel(name)
    fig.suptitle(
        f"section {int(chosen[b])}  lead {int(lead[k])}  stem {int(target_stem[b, k])}"
    )
    return fig


def plot_mse_curves(
    save_dir: str,
    lead: np.ndarray,
    ar_mse_high: np.ndarray,
    ar_mse_mid: np.ndarray,
    ar_mse_low: np.ndarray,
    tf_mse_high: np.ndarray,
    tf_mse_mid: np.ndarray,
    tf_mse_low: np.ndarray,
) -> str:
    series = [
        ("high", ar_mse_high, tf_mse_high),
        ("mid", ar_mse_mid, tf_mse_mid),
        ("low", ar_mse_low, tf_mse_low),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True, constrained_layout=True)
    for ax, (name, ar, tf) in zip(axs, series):
        ar_mean = ar.mean(axis=0)
        tf_mean = tf.mean(axis=0)
        for row in range(ar.shape[0]):
            ax.plot(lead, ar[row], color="C0", alpha=0.25, lw=0.8)
            ax.plot(lead, tf[row], color="C1", alpha=0.25, lw=0.8)
        ax.plot(lead, ar_mean, color="black", lw=2.5, label="AR mean")
        ax.plot(lead, tf_mean, color="black", lw=2.5, ls="--", label="TF mean")
        ax.set_title(name)
        ax.set_xlabel("lead")
        ax.set_ylabel("MSE")
        ax.legend()
    out = os.path.join(save_dir, "mse_vs_lead.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    if args.model_type != "hine":
        raise NotImplementedError(f"model type {args.model_type} is not wired up yet")
    if args.every_nth_field < 1:
        raise ValueError(f"every-nth-field must be >= 1, got {args.every_nth_field}")
    if args.horizon < 4:
        raise ValueError(f"horizon must be >= 4 (need n_steps = horizon - 3), got {args.horizon}")
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {args.batch_size}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU", file=sys.stderr)

    os.makedirs(args.save_dir, exist_ok=True)
    fields_dir = os.path.join(args.save_dir, "fields")
    os.makedirs(fields_dir, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    if n_available < 1:
        raise ValueError(
            f"Need at least {args.horizon} frames after taking every {args.every_nth_field}th, "
            f"found {len(kept)} kept files."
        )
    chosen = choose_sections(n_available, args.n_initial_conditions)
    trajectories = load_trajectories(kept, chosen, args.horizon)
    print(
        f"sorted kept={len(kept)} every_{args.every_nth_field}th "
        f"full_sections={n_available} using={len(chosen)} "
        f"section_index={chosen.tolist()}"
    )

    model, ema_vit = load_model(args.checkpoint, device)
    print(
        f"checkpoint={args.checkpoint} ds_res={model.ds_res} "
        f"low_res={model.low_res} device={device}"
    )

    n_frames = len(trajectories[0])
    n_steps = n_frames - 3
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
    lead = np.arange(1, n_steps + 1, dtype=np.int64)
    target_stem = file_stem[:, 1 : n_steps + 1]
    target_kept_index = kept_index[:, 1 : n_steps + 1]

    (
        ar_pred_high,
        ar_pred_mid,
        ar_pred_low,
        gt_high,
        gt_mid,
        gt_low,
        ar_mse_high,
        ar_mse_mid,
        ar_mse_low,
    ) = rollout(
        ema_vit, model, frames_np, args.batch_size, n_steps, device, teacher_forced=False
    )
    (
        tf_pred_high,
        tf_pred_mid,
        tf_pred_low,
        _,
        _,
        _,
        tf_mse_high,
        tf_mse_mid,
        tf_mse_low,
    ) = rollout(
        ema_vit, model, frames_np, args.batch_size, n_steps, device, teacher_forced=True
    )

    npz_path = os.path.join(args.save_dir, "rollout_mse.npz")
    np.savez_compressed(
        npz_path,
        ar_pred_high=ar_pred_high,
        ar_pred_mid=ar_pred_mid,
        ar_pred_low=ar_pred_low,
        tf_pred_high=tf_pred_high,
        tf_pred_mid=tf_pred_mid,
        tf_pred_low=tf_pred_low,
        gt_high=gt_high,
        gt_mid=gt_mid,
        gt_low=gt_low,
        ar_mse_high=ar_mse_high,
        ar_mse_mid=ar_mse_mid,
        ar_mse_low=ar_mse_low,
        tf_mse_high=tf_mse_high,
        tf_mse_mid=tf_mse_mid,
        tf_mse_low=tf_mse_low,
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
    )
    print(
        f"saved {npz_path}  B={ar_mse_high.shape[0]} T={ar_mse_high.shape[1]}  "
        f"lead1 high/mid/low AR "
        f"{ar_mse_high[:, 0].mean():.4g} / {ar_mse_mid[:, 0].mean():.4g} / {ar_mse_low[:, 0].mean():.4g}"
    )

    plot_leads = [0, n_steps // 2, n_steps - 1]
    for b in range(ar_mse_high.shape[0]):
        for k in plot_leads:
            fig = field_figure(
                b,
                k,
                chosen,
                lead,
                target_stem,
                ar_pred_high,
                ar_pred_mid,
                ar_pred_low,
                tf_pred_high,
                tf_pred_mid,
                tf_pred_low,
                gt_high,
                gt_mid,
                gt_low,
            )
            fig.savefig(
                os.path.join(fields_dir, f"b{b:03d}_lead{int(lead[k]):03d}.png"),
                dpi=120,
            )
            plt.close(fig)

    mse_plot = plot_mse_curves(
        args.save_dir,
        lead,
        ar_mse_high,
        ar_mse_mid,
        ar_mse_low,
        tf_mse_high,
        tf_mse_mid,
        tf_mse_low,
    )
    print(f"wrote fields under {fields_dir}")
    print(f"wrote {mse_plot}")
    print("JACOBIAN_MANUAL_MSE_DONE")


if __name__ == "__main__":
    main()
