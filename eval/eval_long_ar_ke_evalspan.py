#!/usr/bin/env python3
"""One long autoregressive fine-field rollout, then a KE-compatible rollout_mse.npz.

--span eval uses one eval directory (previous behavior).
--span full walks the continuous series train (stems <= 91015), validation
(<= 91620), and eval, keeping stem % every_nth == 0. Frames are read one at a
time so the full series is not resident on the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from eval.eval_ablation_mse import load_ablation_model
from eval.eval_jacobian_manual_mse import load_kept_paths, load_model as load_l3, spatial_mse
from fno_baseline import FNO2d
from eval.eval_jacobian_manual_mse_plus import load_model as load_l2
from eval.eval_kinetic_energy_survival import mat_path_for_stem

STEM_MIN = 10000
STEM_MAX = 105000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["l3", "l2", "l1", "fno"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--span",
        choices=["eval", "full"],
        default="eval",
        help="eval: one directory. full: train+validation+eval stems 10000..105000.",
    )
    p.add_argument("--save-dir", required=True)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--align", choices=["positional", "mod"], default="mod")
    p.add_argument("--start-index", type=int, default=0, help="Index into kept frames")
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p.parse_args()


def full_series_paths(every: int, align: str) -> list[tuple[int, str]]:
    if every < 1:
        raise ValueError(f"every-nth-field must be >= 1, got {every}")
    if align == "mod":
        start = STEM_MIN + (every - (STEM_MIN % every)) % every
        stems = range(start, STEM_MAX + 1, every)
    elif align == "positional":
        stems = range(STEM_MIN, STEM_MAX + 1, every)
    else:
        raise ValueError(f"align must be positional or mod, got {align}")
    kept: list[tuple[int, str]] = []
    for stem in stems:
        path = mat_path_for_stem(int(stem))
        if not path.is_file():
            raise FileNotFoundError(f"missing frame {path}")
        kept.append((int(stem), str(path)))
    return kept


def load_fno(checkpoint_path: str, device: torch.device) -> FNO2d:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    a = ckpt["args"]
    model = FNO2d(
        modes=int(getattr(a, "modes", 96)),
        width=int(getattr(a, "width", 32)),
        n_layers=int(getattr(a, "n_layers", 3)),
        in_channels=int(getattr(a, "in_steps", 1)),
        out_channels=int(getattr(a, "out_steps", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_high(path: str, norm: float, device: torch.device) -> torch.Tensor:
    omega = np.asarray(scipy.io.loadmat(path)["Omega"], dtype=np.float32)
    frame = torch.from_numpy(omega).div_(float(norm))
    return frame.to(device).view(1, 1, frame.shape[0], frame.shape[1])


def _empty_fields(n_steps: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (1, n_steps, 1, height, width)
    return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)


@torch.no_grad()
def roll_l3(ema_vit, model, kept, norm, device):
    n_steps = len(kept) - 1
    high = load_high(kept[0][1], norm, device)
    _, _, height, width = high.shape
    pred, gt = _empty_fields(n_steps, height, width)
    mid0 = torch.zeros(1, 1, model.ds_res, model.ds_res, device=device, dtype=high.dtype)
    low0 = torch.zeros(1, 1, model.low_res, model.low_res, device=device, dtype=high.dtype)
    _, mid, _ = ema_vit(high, mid0, low0, torch.zeros(1, device=device, dtype=high.dtype))
    _, _, low = ema_vit(high, mid, low0, torch.ones(1, device=device, dtype=high.dtype))
    t2 = 2 * torch.ones(1, device=device, dtype=high.dtype)
    mse = np.zeros((1, n_steps), dtype=np.float64)
    for k in tqdm(range(n_steps), desc="AR L=3"):
        high, mid, low = ema_vit(high, mid, low, t2)
        true_h = load_high(kept[k + 1][1], norm, device)
        mse[:, k] = spatial_mse(high, true_h)
        pred[0, k] = high.detach().float().cpu().numpy()
        gt[0, k] = true_h.detach().float().cpu().numpy()
    return pred, gt, mse


@torch.no_grad()
def roll_l2(ema_vit, model, kept, norm, device):
    n_steps = len(kept) - 1
    k = model.num_coarse
    high = load_high(kept[0][1], norm, device)
    _, _, height, width = high.shape
    pred, gt = _empty_fields(n_steps, height, width)
    coarses = [
        torch.zeros(1, 1, model.latent_res[i], model.latent_res[i], device=device, dtype=high.dtype)
        for i in range(k)
    ]
    for c in range(k):
        tt = float(c) * torch.ones(1, device=device, dtype=high.dtype)
        _, outs = ema_vit(high, coarses, tt)
        coarses[c] = outs[c]
    tK = float(k) * torch.ones(1, device=device, dtype=high.dtype)
    mse = np.zeros((1, n_steps), dtype=np.float64)
    for step in tqdm(range(n_steps), desc="AR L=2"):
        high, coarses = ema_vit(high, coarses, tK)
        true_h = load_high(kept[step + 1][1], norm, device)
        mse[:, step] = spatial_mse(high, true_h)
        pred[0, step] = high.detach().float().cpu().numpy()
        gt[0, step] = true_h.detach().float().cpu().numpy()
    return pred, gt, mse


@torch.no_grad()
def roll_l1(ema_vit, kept, norm, device):
    n_steps = len(kept) - 1
    high = load_high(kept[0][1], norm, device)
    _, _, height, width = high.shape
    pred, gt = _empty_fields(n_steps, height, width)
    aux = getattr(ema_vit, "num_aux", 0) > 0
    mse = np.zeros((1, n_steps), dtype=np.float64)
    for step in tqdm(range(n_steps), desc="AR L=1"):
        out = ema_vit(high)
        high = out[0] if aux else out
        true_h = load_high(kept[step + 1][1], norm, device)
        mse[:, step] = spatial_mse(high, true_h)
        pred[0, step] = high.detach().float().cpu().numpy()
        gt[0, step] = true_h.detach().float().cpu().numpy()
    return pred, gt, mse


@torch.no_grad()
def roll_fno(net, kept, norm, device):
    n_in = int(net.in_channels)
    n_out = int(net.out_channels)
    if n_in != n_out:
        raise ValueError(f"AR rollout needs in_channels == out_channels, got {n_in}/{n_out}")
    n_leads = ((len(kept) - n_in) // n_out) * n_out
    if n_leads < 1:
        raise ValueError(f"need more than {n_in} frames, got {len(kept)}")
    used = kept[: n_in + n_leads]
    ctx = torch.cat([load_high(used[i][1], norm, device) for i in range(n_in)], dim=1)
    _, _, height, width = ctx.shape
    pred, gt = _empty_fields(n_leads, height, width)
    mse = np.zeros((1, n_leads), dtype=np.float64)
    net.eval()
    for t in tqdm(range(0, n_leads, n_out), desc=f"AR FNO {n_in}to{n_out}"):
        out = net(ctx)
        for j in range(n_out):
            true_h = load_high(used[t + n_in + j][1], norm, device)
            piece = out[:, j : j + 1]
            mse[:, t + j] = spatial_mse(piece, true_h)
            pred[0, t + j] = piece.detach().float().cpu().numpy()
            gt[0, t + j] = true_h.detach().float().cpu().numpy()
        ctx = out
    return pred, gt, mse, used


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    os.makedirs(args.save_dir, exist_ok=True)

    if args.span == "full":
        kept_all = full_series_paths(args.every_nth_field, args.align)
    else:
        kept_all = load_kept_paths(args.eval_dir, args.every_nth_field, args.align)
    if args.start_index < 0 or args.start_index >= len(kept_all):
        raise ValueError(f"start-index {args.start_index} out of range for {len(kept_all)} frames")
    kept = kept_all[args.start_index :]

    print(
        f"kind={args.kind} span={args.span} align={args.align} start_index={args.start_index} "
        f"first_stem={kept[0][0]} last_stem={kept[-1][0]} n_kept={len(kept)} device={device}"
    )

    used = kept
    if args.kind == "l3":
        model, ema = load_l3(args.checkpoint, device)
        pred, gt, mse = roll_l3(ema, model, kept, args.norm, device)
    elif args.kind == "l2":
        model, ema = load_l2(args.checkpoint, device)
        pred, gt, mse = roll_l2(ema, model, kept, args.norm, device)
    elif args.kind == "l1":
        model, ema = load_ablation_model(args.checkpoint, device)
        pred, gt, mse = roll_l1(ema, kept, args.norm, device)
    else:
        net = load_fno(args.checkpoint, device)
        pred, gt, mse, used = roll_fno(net, kept, args.norm, device)

    n_steps = pred.shape[1]
    stems = np.asarray([stem for stem, _ in used], dtype=np.int64)
    n_ctx = stems.shape[0] - n_steps
    lead = np.arange(1, n_steps + 1, dtype=np.int64)
    npz_path = os.path.join(args.save_dir, "rollout_mse.npz")
    np.savez_compressed(
        npz_path,
        ar_pred_high=pred,
        gt_high=gt,
        ar_mse_high=mse,
        lead=lead,
        file_stem=stems[None, :],
        target_stem=stems[None, n_ctx:],
        chosen=np.asarray([args.start_index], dtype=np.int64),
        checkpoint=np.asarray(args.checkpoint),
        eval_dir=np.asarray(args.eval_dir if args.span == "eval" else "train+validation+eval"),
        horizon=np.asarray(stems.shape[0], dtype=np.int64),
        every_nth_field=np.asarray(args.every_nth_field, dtype=np.int64),
        norm=np.asarray(args.norm, dtype=np.float64),
        align=np.asarray(args.align),
        kind=np.asarray(args.kind),
        span=np.asarray(args.span),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    ax.plot(lead, mse[0], color="C0", lw=1.2)
    ax.set_xlabel("lead")
    ax.set_ylabel("MSE")
    ax.set_title(f"{args.kind} {args.span} AR MSE from stem {int(stems[0])}")
    ax.grid(True, alpha=0.3)
    curve = os.path.join(args.save_dir, "mse_vs_lead.png")
    fig.savefig(curve, dpi=150)
    plt.close(fig)
    print(
        f"saved {npz_path} shape={pred.shape} "
        f"first_target={int(stems[n_ctx])} last_stem={int(stems[-1])} "
        f"mse@1={mse[0, 0]:.6g} mse@end={mse[0, -1]:.6g} curve={curve}"
    )


if __name__ == "__main__":
    main()
