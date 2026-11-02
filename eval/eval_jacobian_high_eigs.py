#!/usr/bin/env python3
"""High-field Jacobian eigenvalue stress test along one AR trajectory.

Sources only eval_jacobian_manual_mse.py (data / load_model / latent warmup /
t=2 AR step) and jacobian_manual.ipynb (dense jacobian + matrix-free jvp/eigs).

At each of --horizon autoregressive steps:
  1) differentiate next high w.r.t. current high (mid/low fixed)
  2) estimate largest-magnitude eigenvalues
  3) save results
  4) take a no-grad t=2 step and repeat

--method torch:  torch.autograd.functional.jacobian + torch.linalg.eigvals
--method scipy:  jvp matvec + scipy.sparse.linalg.eigs (k=1, which=LM)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_model,
    load_trajectories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["torch", "scipy"],
        required=True,
        help="Dense torch eigvals, or matrix-free SciPy ARPACK",
    )
    p.add_argument(
        "--jacobian",
        choices=["high", "full"],
        default="high",
        help="high: next-high vs high (mid/low fixed). full: square map over (high,mid,low)",
    )
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
        "--horizon",
        type=int,
        default=20,
        help="Number of AR steps / kept frames in the section",
    )
    p.add_argument(
        "--norm",
        type=float,
        default=10.0,
        help="Divide Omega by this before the model",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_high_eigs",
        help="Directory for per-step npz and summary.txt",
    )
    p.add_argument(
        "--save-jacobian",
        action="store_true",
        help="Also write the dense Jacobian (torch method only; ~16 GiB/step)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose eigs_stepXX.npz already exists",
    )
    p.add_argument(
        "--ncv",
        type=int,
        default=4,
        help="Arnoldi subspace size for scipy eigs (ignored by torch)",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Max Arnoldi iterations for scipy eigs (ignored by torch)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Relative eigenvalue tolerance for scipy eigs (ignored by torch)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def step_path(save_dir: Path, step: int) -> Path:
    return save_dir / f"eigs_step{step:02d}.npz"


def peak_gpu_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))


def reset_peak_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def pack_l3_state(high: torch.Tensor, mid: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [high.reshape(-1), mid.reshape(-1), low.reshape(-1)],
        dim=0,
    )


def unpack_l3_state(
    flat: torch.Tensor,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nh = high.numel()
    nm = mid.numel()
    return (
        flat[:nh].reshape_as(high),
        flat[nh : nh + nm].reshape_as(mid),
        flat[nh + nm :].reshape_as(low),
    )


def time_one_jvp(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    jacobian: str = "high",
) -> float:
    """Wall time for one JVP of the selected eigen-operator."""
    dtype = high.dtype
    device = high.device
    mid_c = mid.detach().to(dtype)
    low_c = low.detach().to(dtype)
    h0 = high.detach().to(dtype)
    t = 2 * torch.ones(h0.shape[0], device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if jacobian == "high":
        v = torch.randn_like(h0)
        v = v / v.norm()
        _, jv = torch.autograd.functional.jvp(
            lambda h: ema_vit(h, mid_c, low_c, t)[0],
            (h0,),
            (v,),
        )
    else:
        v = torch.randn(pack_l3_state(h0, mid_c, low_c).numel(), device=device, dtype=dtype)
        v = v / v.norm()
        vh, vm, vl = unpack_l3_state(v, h0, mid_c, low_c)

        def f(h, m, l):
            oh, om, ol = ema_vit(h, m, l, t)
            return pack_l3_state(oh, om, ol)

        _, jv = torch.autograd.functional.jvp(f, (h0, mid_c, low_c), (vh, vm, vl))
    _ = jv.detach()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def compute_torch_eigs(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    save_jacobian: bool,
) -> dict[str, np.ndarray]:
    """Notebook dense path: jacobian then eigvals of high->high map."""
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    mid_c = mid.detach().to(dtype)
    low_c = low.detach().to(dtype)
    t = 2 * torch.ones(h0.shape[0], device=h0.device, dtype=dtype)

    J = torch.autograd.functional.jacobian(
        lambda h: ema_vit(h, mid_c, low_c, t)[0][0, 0, :, :],
        h0,
    )[:, :, 0, 0, :, :]
    # J is (H_out, W_out, H_in, W_in); flatten to (n, n) for eigvals
    n = h0[0, 0].numel()
    J_mat = J.reshape(n, n)
    vals = torch.linalg.eigvals(J_mat)
    order = torch.argsort(vals.abs(), descending=True)
    vals = vals[order]
    out: dict[str, np.ndarray] = {
        "eigenvalues": vals.detach().cpu().numpy().astype(np.complex128),
        "radius": np.asarray(float(vals.abs().max().item()), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "converged": np.asarray(True),
    }
    if save_jacobian:
        out["jacobian"] = J_mat.detach().float().cpu().numpy().astype(np.float32)
    return out


def compute_scipy_eigs(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    ncv: int,
    maxiter: int,
    tol: float,
    jacobian: str = "high",
) -> dict[str, np.ndarray]:
    """Notebook matrix-free path: jvp matvec + scipy.sparse.linalg.eigs."""
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    mid_c = mid.detach().to(dtype)
    low_c = low.detach().to(dtype)
    t = 2 * torch.ones(h0.shape[0], device=h0.device, dtype=dtype)
    if jacobian == "high":
        n = int(h0.numel())

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = (
                torch.from_numpy(np.ascontiguousarray(v_np))
                .to(device=h0.device, dtype=h0.dtype)
                .reshape_as(h0)
            )
            _, jv = torch.autograd.functional.jvp(
                lambda h: ema_vit(h, mid_c, low_c, t)[0],
                (h0,),
                (v,),
            )
            return jv.detach().reshape(-1).float().cpu().numpy()

    else:
        n = int(h0.numel() + mid_c.numel() + low_c.numel())

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(
                device=h0.device, dtype=h0.dtype
            )
            vh, vm, vl = unpack_l3_state(v, h0, mid_c, low_c)

            def f(h, m, l):
                oh, om, ol = ema_vit(h, m, l, t)
                return pack_l3_state(oh, om, ol)

            _, jv = torch.autograd.functional.jvp(
                f, (h0, mid_c, low_c), (vh, vm, vl)
            )
            return jv.detach().reshape(-1).float().cpu().numpy()

    ncv_eff = min(max(ncv, 2), n)
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float32)
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    converged = True
    try:
        vals = spla.eigs(
            op,
            k=1,
            which="LM",
            ncv=ncv_eff,
            maxiter=maxiter,
            tol=tol,
            v0=v0,
            return_eigenvectors=False,
        )
    except spla.ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues)
        vals = vals[np.isfinite(vals)]
        converged = False
        if vals.size == 0:
            raise
        print(
            f"    partial convergence: kept {vals.size}/1 eigenvalues",
            flush=True,
        )
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    return {
        "eigenvalues": vals.astype(np.complex128),
        "radius": np.asarray(float(np.max(np.abs(vals))), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "converged": np.asarray(converged),
        "k": np.asarray(1, dtype=np.int64),
        "ncv": np.asarray(ncv_eff, dtype=np.int64),
        "maxiter": np.asarray(maxiter, dtype=np.int64),
        "tol": np.asarray(tol, dtype=np.float64),
        "jacobian": np.asarray(jacobian),
    }


def main() -> None:
    args = parse_args()
    if args.method == "torch" and args.jacobian == "full":
        raise ValueError("--jacobian full is only supported with --method scipy")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    if n_available < 1:
        raise ValueError(
            f"Need at least {args.horizon} frames after every "
            f"{args.every_nth_field}th, found {len(kept)}"
        )
    chosen = choose_sections(n_available, 1)
    trajectories = load_trajectories(kept, chosen, args.horizon)
    frames = trajectories[0]
    print(
        f"kept={len(kept)} full_sections={n_available} "
        f"using_section={int(chosen[0])} horizon={args.horizon} "
        f"method={args.method} jacobian={args.jacobian} device={device} "
        f"ncv={args.ncv} maxiter={args.maxiter} tol={args.tol}",
        flush=True,
    )

    model, ema_vit = load_model(args.checkpoint, device)
    print(
        f"checkpoint={args.checkpoint} ds_res={model.ds_res} "
        f"low_res={model.low_res}",
        flush=True,
    )

    # One-section IC, same as notebook / rollout warmup
    init_np = np.asarray(frames[0], dtype=np.float32)
    high = torch.from_numpy(init_np).to(device).unsqueeze(0).unsqueeze(0) / args.norm
    dtype = next(ema_vit.parameters()).dtype
    high = high.to(dtype)
    b = high.shape[0]
    mid0 = torch.zeros(
        b, 1, model.ds_res, model.ds_res, device=device, dtype=high.dtype
    )
    low0 = torch.zeros(
        b, 1, model.low_res, model.low_res, device=device, dtype=high.dtype
    )
    with torch.no_grad():
        _, mid, _ = ema_vit(
            high, mid0, low0, torch.zeros(b, device=device, dtype=high.dtype)
        )
        _, _, low = ema_vit(
            high, mid, low0, torch.ones(b, device=device, dtype=high.dtype)
        )
    mid = mid.detach()
    low = low.detach()

    if args.jacobian == "high":
        n = int(high.numel())
    else:
        n = int(high.numel() + mid.numel() + low.numel())
    jvp_s = time_one_jvp(ema_vit, high, mid, low, jacobian=args.jacobian)
    print(
        f"one_jvp_s={jvp_s:.4f} jacobian={args.jacobian} "
        f"extrapolated_full_jacobian_s≈{jvp_s * n:.1f} "
        f"(n={n} reverse products) "
        f"extrapolated_scipy_eigs_s≈{jvp_s * args.maxiter:.1f} "
        f"(~{args.maxiter} Arnoldi iterations at one product each)",
        flush=True,
    )

    summary_lines = [
        f"method={args.method}",
        f"jacobian={args.jacobian}",
        f"checkpoint={args.checkpoint}",
        f"section={int(chosen[0])}",
        f"horizon={args.horizon}",
        f"state_dim={n}",
        f"one_jvp_s={jvp_s:.6f}",
        f"device={device}",
        f"ncv={args.ncv}",
        f"maxiter={args.maxiter}",
        f"tol={args.tol}",
    ]

    t2 = 2 * torch.ones(b, device=device, dtype=high.dtype)
    # Jacobian at the IC (step 0), then after each of the next (horizon-1) AR advances.
    n_jac_steps = args.horizon

    for k in range(n_jac_steps):
        out_path = step_path(save_dir, k)
        if args.skip_existing and out_path.is_file():
            print(f"step {k}: skip existing {out_path}", flush=True)
            continue

        print(f"step {k}: computing {args.method} eigs ...", flush=True)
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            if args.method == "torch":
                result = compute_torch_eigs(
                    ema_vit, high, mid, low, args.save_jacobian
                )
            else:
                result = compute_scipy_eigs(
                    ema_vit,
                    high,
                    mid,
                    low,
                    ncv=args.ncv,
                    maxiter=args.maxiter,
                    tol=args.tol,
                    jacobian=args.jacobian,
                )
        except torch.cuda.OutOfMemoryError as exc:
            fail_path = save_dir / f"step_{k}_failed.txt"
            fail_path.write_text(f"OutOfMemoryError at step {k}:\n{exc}\n")
            print(f"step {k}: OOM -> {fail_path}", flush=True)
            summary_lines.append(f"step_{k}=OOM")
            break
        except Exception as exc:
            fail_path = save_dir / f"step_{k}_failed.txt"
            fail_path.write_text(f"Error at step {k}:\n{type(exc).__name__}: {exc}\n")
            print(f"step {k}: failed -> {fail_path}", flush=True)
            summary_lines.append(f"step_{k}=FAILED:{type(exc).__name__}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mib = peak_gpu_mib()
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["peak_gpu_mib"] = np.asarray(peak_mib, dtype=np.float64)
        result["step"] = np.asarray(k, dtype=np.int64)
        result["method"] = np.asarray(args.method)
        result["jacobian"] = np.asarray(args.jacobian)

        np.savez_compressed(out_path, **result)
        radius = float(result["radius"])
        print(
            f"step {k}: radius={radius:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(
            f"step_{k}=ok radius={radius:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f}"
        )

        # Advance AR state for next jacobian evaluation (no grad)
        if k + 1 < n_jac_steps:
            with torch.no_grad():
                high, mid, low = ema_vit(high, mid, low, t2)
            high = high.detach()
            mid = mid.detach()
            low = low.detach()

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}", flush=True)
    print("JACOBIAN_HIGH_EIGS_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
