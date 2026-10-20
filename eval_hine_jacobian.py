#!/usr/bin/env python3
"""Online Jacobian / spectral-radius analysis for a FutureINN smoke checkpoint.

Differentiates the live t=2 ema_vit step with finite-difference products.
Does not call sample() (no_grad + autocast) and does not edit model/ufno.

Writes scalar summary/npz plus Spectral_Stability-aligned field dumps in HDF5.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

from model import FutureINN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_smoke_1gpu/checkpoint-last.pth",
        help="Path to checkpoint-last.pth from the paper smoke run",
    )
    p.add_argument(
        "--eval-dir",
        default="/global/homes/d/divyam/TACSLAB/eval",
        help="Directory of eval/*.mat vorticity frames",
    )
    p.add_argument(
        "--output-dir",
        default="/pscratch/sd/d/divyam/hine_runs/jacobian_smoke",
        help="Directory for summary.txt, trajectories.npz, and jacobian_fields.h5",
    )
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--power-iters", type=int, default=15)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--eps-check", type=float, default=1e-2)
    p.add_argument("--data-norm", type=float, default=10.0)
    p.add_argument("--time-step-keep", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def count_raw_mats(eval_dir: str) -> int:
    return len(glob.glob(os.path.join(eval_dir, "*.mat")))


def list_eval_paths(eval_dir: str, time_step_keep: int) -> list[str]:
    paths = glob.glob(os.path.join(eval_dir, "*.mat"))
    if not paths:
        raise FileNotFoundError(f"No .mat files under {eval_dir}")
    kept = []
    for path in paths:
        stem = Path(path).stem
        try:
            idx = int(stem)
        except ValueError:
            continue
        if idx % time_step_keep == 0:
            kept.append((idx, path))
    if not kept:
        raise FileNotFoundError(
            f"No .mat files with index % {time_step_keep} == 0 under {eval_dir}"
        )
    kept.sort(key=lambda t: t[0])
    return [p for _, p in kept]


def load_frame(path: str, img_size: int, data_norm: float, device: torch.device) -> torch.Tensor:
    omega = np.asarray(scipy.io.loadmat(path)["Omega"], dtype=np.float32)
    t = torch.from_numpy(omega)[None, None].to(device)
    if t.shape[-2] != img_size or t.shape[-1] != img_size:
        t = torch.nn.functional.interpolate(
            t, size=(img_size, img_size), mode="bilinear", align_corners=False, antialias=True
        )
    return t / data_norm


def load_model(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, object]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    args = ckpt["args"]
    model = FutureINN(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.ema_vit.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.ema_vit, args


def zeros_like_maps(high: torch.Tensor, ds_res: int, low_res: int) -> tuple[torch.Tensor, torch.Tensor]:
    b = high.shape[0]
    mid = torch.zeros(b, 1, ds_res, ds_res, device=high.device, dtype=high.dtype)
    low = torch.zeros(b, 1, low_res, low_res, device=high.device, dtype=high.dtype)
    return mid, low


@torch.no_grad()
def warmup_latents(
    ema_vit: torch.nn.Module, high: torch.Tensor, ds_res: int, low_res: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sample()'s first branch: t=0 then t=1, high unchanged."""
    b = high.shape[0]
    mid0, low0 = zeros_like_maps(high, ds_res, low_res)
    t0 = torch.zeros(b, device=high.device, dtype=high.dtype)
    t1 = torch.ones(b, device=high.device, dtype=high.dtype)
    _, mid, _ = ema_vit(high, mid0, low0, t0)
    _, _, low = ema_vit(high, mid, low0, t1)
    return mid, low


@torch.no_grad()
def phi(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive one-step map (t=2). Float32, no autocast."""
    b = high.shape[0]
    t = 2 * torch.ones(b, device=high.device, dtype=high.dtype)
    return ema_vit(high, mid, low, t)


def pack_state(high: torch.Tensor, mid: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    return torch.cat([high.reshape(-1), mid.reshape(-1), low.reshape(-1)])


def unpack_state(
    vec: torch.Tensor, shapes: tuple[torch.Size, torch.Size, torch.Size]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_h = int(np.prod(shapes[0]))
    n_m = int(np.prod(shapes[1]))
    high = vec[:n_h].reshape(shapes[0])
    mid = vec[n_h : n_h + n_m].reshape(shapes[1])
    low = vec[n_h + n_m :].reshape(shapes[2])
    return high, mid, low


def to_cpu_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def fd_jvp(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Finite-difference Jacobian-vector product of the packed t=2 map."""
    shapes = (high.shape, mid.shape, low.shape)
    x0 = pack_state(high, mid, low)
    y0 = pack_state(*phi(ema_vit, high, mid, low))
    high_p, mid_p, low_p = unpack_state(x0 + eps * v, shapes)
    y1 = pack_state(*phi(ema_vit, high_p, mid_p, low_p))
    return (y1 - y0) / eps


@torch.no_grad()
def spectral_radius(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    n_iters: int,
    eps: float,
    generator: torch.Generator,
) -> float:
    n = pack_state(high, mid, low).numel()
    v = torch.randn(n, device=high.device, dtype=high.dtype, generator=generator)
    v = v / (v.norm() + 1e-12)
    radius = 0.0
    for _ in range(n_iters):
        w = fd_jvp(ema_vit, high, mid, low, v, eps)
        radius = float(w.norm().item())
        v = w / (radius + 1e-12)
    return radius


@torch.no_grad()
def propagate_e_high(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    e_high: torch.Tensor,
    residual_high: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """One Spectral_Stability-style step: e <- J e + residual, high-supported e."""
    shapes = (high.shape, mid.shape, low.shape)
    n_h = int(np.prod(shapes[0]))
    n = pack_state(high, mid, low).numel()
    v = torch.zeros(n, device=high.device, dtype=high.dtype)
    v[:n_h] = e_high.reshape(-1)
    w = fd_jvp(ema_vit, high, mid, low, v, eps)
    w_high = w[:n_h].reshape(high.shape)
    return w_high + residual_high


def choose_starts(n_frames: int, horizon: int, num_starts: int) -> list[int]:
    max_start = n_frames - horizon - 1
    if max_start < 0:
        raise ValueError(
            f"Need at least horizon+1={horizon + 1} frames, have {n_frames}"
        )
    if num_starts == 1:
        return [0]
    return [int(round(i * max_start / (num_starts - 1))) for i in range(num_starts)]


@torch.no_grad()
def run_chain(
    ema_vit: torch.nn.Module,
    frames: list[torch.Tensor],
    start: int,
    horizon: int,
    ds_res: int,
    low_res: int,
    power_iters: int,
    eps: float,
    generator: torch.Generator,
    teacher_forced: bool,
) -> dict[str, np.ndarray]:
    high0 = frames[start]
    mid, low = warmup_latents(ema_vit, high0, ds_res, low_res)
    high = high0

    _, _, h, w = high0.shape
    radii = np.zeros(horizon, dtype=np.float64)
    residual_mse = np.zeros(horizon, dtype=np.float64)
    ar_mse = np.zeros(horizon, dtype=np.float64)
    pert_norms = np.zeros(horizon, dtype=np.float64)

    preds = np.zeros((horizon, 1, h, w), dtype=np.float32)
    mids = np.zeros((horizon, 1, ds_res, ds_res), dtype=np.float32)
    lows = np.zeros((horizon, 1, low_res, low_res), dtype=np.float32)
    residuals = np.zeros((horizon, 1, h, w), dtype=np.float32)
    eps_sums = np.zeros((horizon, 1, h, w), dtype=np.float32)
    e_vecs = np.zeros((horizon, 1, h, w), dtype=np.float32)
    truth_states = np.zeros((horizon + 1, 1, h, w), dtype=np.float32)

    for t_idx in range(horizon + 1):
        truth_states[t_idx] = to_cpu_f32(frames[start + t_idx])[0]

    e_high = torch.zeros_like(high0)
    eps_sum = torch.zeros_like(high0)

    for k in range(horizon):
        true_k = frames[start + k]
        true_next = frames[start + k + 1]
        if teacher_forced:
            high = true_k

        # Jacobian products use the pre-step state (high, mid, low).
        radii[k] = spectral_radius(
            ema_vit, high, mid, low, power_iters, eps, generator
        )

        pred_high, mid_out, low_out = phi(ema_vit, high, mid, low)
        residual = pred_high - true_next
        eps_sum = eps_sum + residual
        e_high = propagate_e_high(
            ema_vit, high, mid, low, e_high, residual, eps
        )

        step_mse = float((residual ** 2).mean().item())
        residual_mse[k] = step_mse
        ar_mse[k] = step_mse
        pert_norms[k] = float(e_high.norm().item())

        preds[k] = to_cpu_f32(pred_high)[0]
        mids[k] = to_cpu_f32(mid_out)[0]
        lows[k] = to_cpu_f32(low_out)[0]
        residuals[k] = to_cpu_f32(residual)[0]
        eps_sums[k] = to_cpu_f32(eps_sum)[0]
        e_vecs[k] = to_cpu_f32(e_high)[0]

        mid, low = mid_out, low_out
        if not teacher_forced:
            high = pred_high

    return {
        "radius": radii,
        "residual_mse": residual_mse,
        "ar_mse": ar_mse,
        "pert_norm": pert_norms,
        "pred_high": preds,
        "mid": mids,
        "low": lows,
        "residual": residuals,
        "eps_sum": eps_sums,
        "e_vec": e_vecs,
        "truth_states": truth_states,
    }


def write_hdf5(
    path: Path,
    starts: list[int],
    file_indices: np.ndarray,
    ar_list: list[dict[str, np.ndarray]],
    tf_list: list[dict[str, np.ndarray]],
) -> None:
    """Write Spectral_Stability-aligned field dumps plus HINE latents."""
    with h5py.File(path, "w") as hf:
        hf.create_dataset("starts", data=np.asarray(starts, dtype=np.int64))
        hf.create_dataset("file_indices", data=file_indices)

        truth = np.stack([ar["truth_states"] for ar in ar_list], axis=0)
        hf.create_dataset("truth_states", data=truth, compression="gzip")

        hf.create_dataset(
            "autoreg_pred",
            data=np.stack([ar["pred_high"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "net_pred_truth",
            data=np.stack([tf["pred_high"] for tf in tf_list], axis=0),
            compression="gzip",
        )
        # eps_sum / e_vec follow Spectral_Stability naming on the AR chain.
        hf.create_dataset(
            "eps_sum",
            data=np.stack([ar["eps_sum"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "e_vec",
            data=np.stack([ar["e_vec"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "eps_sum_tf",
            data=np.stack([tf["eps_sum"] for tf in tf_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "e_vec_tf",
            data=np.stack([tf["e_vec"] for tf in tf_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "residual_ar",
            data=np.stack([ar["residual"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "residual_tf",
            data=np.stack([tf["residual"] for tf in tf_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "ar_mid",
            data=np.stack([ar["mid"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "ar_low",
            data=np.stack([ar["low"] for ar in ar_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "tf_mid",
            data=np.stack([tf["mid"] for tf in tf_list], axis=0),
            compression="gzip",
        )
        hf.create_dataset(
            "tf_low",
            data=np.stack([tf["low"] for tf in tf_list], axis=0),
            compression="gzip",
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable; running on CPU", file=sys.stderr)

    raw_count = count_raw_mats(args.eval_dir)
    eval_paths = list_eval_paths(args.eval_dir, args.time_step_keep)
    kept_count = len(eval_paths)
    skipped_keep = raw_count - kept_count
    print(
        f"eval mats: raw={raw_count} kept={kept_count} "
        f"skipped_by_keep_rule(index%{args.time_step_keep}!=0)={skipped_keep} "
        f"dir={args.eval_dir}"
    )

    ema_vit, model_args = load_model(args.checkpoint, device)
    img_size = int(model_args.img_size)
    ds_res = int(model_args.img_size // model_args.ds_factor)
    low_res = int(model_args.img_size // model_args.ds_factor_low)
    print(
        f"checkpoint={args.checkpoint} img_size={img_size} "
        f"ds_res={ds_res} low_res={low_res} device={device}"
    )

    starts = choose_starts(len(eval_paths), args.horizon, args.num_starts)
    starts_not_used = kept_count - len(starts)
    needed = sorted({i for s in starts for i in range(s, s + args.horizon + 1)})
    print(
        f"starts={starts} (num_starts={len(starts)}; "
        f"kept frames not used as starts≈{starts_not_used}) "
        f"loading {len(needed)} frames"
    )

    frame_cache: dict[int, torch.Tensor] = {}
    for i in needed:
        frame_cache[i] = load_frame(eval_paths[i], img_size, args.data_norm, device)

    def relative_frames(start: int) -> list[torch.Tensor]:
        return [frame_cache[start + k] for k in range(args.horizon + 1)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # Two-eps check on the first start state after warmup.
    high0 = relative_frames(starts[0])[0]
    mid0, low0 = warmup_latents(ema_vit, high0, ds_res, low_res)
    r_eps = spectral_radius(
        ema_vit, high0, mid0, low0, args.power_iters, args.eps, gen
    )
    gen.manual_seed(args.seed)
    r_eps_check = spectral_radius(
        ema_vit, high0, mid0, low0, args.power_iters, args.eps_check, gen
    )
    print(
        f"eps check at start {starts[0]}: "
        f"radius(eps={args.eps})={r_eps:.6g} "
        f"radius(eps={args.eps_check})={r_eps_check:.6g}"
    )

    file_indices = np.asarray(
        [int(Path(eval_paths[s]).stem) for s in starts], dtype=np.int64
    )
    traj = {
        "starts": np.asarray(starts, dtype=np.int64),
        "file_indices": file_indices,
        "raw_mat_count": np.asarray(raw_count, dtype=np.int64),
        "kept_frame_count": np.asarray(kept_count, dtype=np.int64),
        "skipped_by_keep_rule": np.asarray(skipped_keep, dtype=np.int64),
        "eps": np.asarray(args.eps),
        "eps_check": np.asarray(args.eps_check),
        "radius_eps": np.asarray(r_eps),
        "radius_eps_check": np.asarray(r_eps_check),
        "tf_radius": [],
        "tf_residual_mse": [],
        "tf_pert_norm": [],
        "ar_radius": [],
        "ar_mse": [],
        "ar_pert_norm": [],
    }

    summary_lines = [
        "HINE Jacobian smoke analysis",
        f"checkpoint: {args.checkpoint}",
        f"eval_dir: {args.eval_dir}",
        f"raw_mat_count: {raw_count}",
        f"kept_frames (index%{args.time_step_keep}==0): {kept_count}",
        f"skipped_by_keep_rule: {skipped_keep}",
        f"num_starts: {len(starts)}",
        f"starts: {starts}",
        f"file_indices: {file_indices.tolist()}",
        f"horizon: {args.horizon}",
        f"power_iters: {args.power_iters}",
        f"eps: {args.eps}",
        f"eps_check: {args.eps_check}",
        f"eps_check_radius({args.eps})={r_eps:.6g}",
        f"eps_check_radius({args.eps_check})={r_eps_check:.6g}",
        "",
    ]

    ar_list: list[dict[str, np.ndarray]] = []
    tf_list: list[dict[str, np.ndarray]] = []

    for start in starts:
        file_idx = int(Path(eval_paths[start]).stem)
        print(f"=== start={start} file_index={file_idx} ===")
        frames = relative_frames(start)

        gen.manual_seed(args.seed + start)
        tf = run_chain(
            ema_vit,
            frames,
            0,
            args.horizon,
            ds_res,
            low_res,
            args.power_iters,
            args.eps,
            gen,
            teacher_forced=True,
        )
        gen.manual_seed(args.seed + start)
        ar = run_chain(
            ema_vit,
            frames,
            0,
            args.horizon,
            ds_res,
            low_res,
            args.power_iters,
            args.eps,
            gen,
            teacher_forced=False,
        )

        tf_list.append(tf)
        ar_list.append(ar)

        traj["tf_radius"].append(tf["radius"])
        traj["tf_residual_mse"].append(tf["residual_mse"])
        traj["tf_pert_norm"].append(tf["pert_norm"])
        traj["ar_radius"].append(ar["radius"])
        traj["ar_mse"].append(ar["ar_mse"])
        traj["ar_pert_norm"].append(ar["pert_norm"])

        summary_lines.extend(
            [
                f"start={start} file_index={file_idx}",
                f"  tf final residual MSE: {tf['residual_mse'][-1]:.6g}",
                f"  tf max |lambda|:       {tf['radius'].max():.6g}",
                f"  ar final MSE:          {ar['ar_mse'][-1]:.6g}",
                f"  ar max |lambda|:       {ar['radius'].max():.6g}",
                "",
            ]
        )
        print(summary_lines[-6])
        print(summary_lines[-5])
        print(summary_lines[-4])
        print(summary_lines[-3])

    for key in (
        "tf_radius",
        "tf_residual_mse",
        "tf_pert_norm",
        "ar_radius",
        "ar_mse",
        "ar_pert_norm",
    ):
        traj[key] = np.stack(traj[key], axis=0)

    npz_path = out_dir / "trajectories.npz"
    np.savez_compressed(npz_path, **traj)

    h5_path = out_dir / "jacobian_fields.h5"
    write_hdf5(h5_path, starts, file_indices, ar_list, tf_list)

    summary_path = out_dir / "summary.txt"
    summary_text = "\n".join(summary_lines) + "\n"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"wrote {summary_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {h5_path}")


if __name__ == "__main__":
    main()
