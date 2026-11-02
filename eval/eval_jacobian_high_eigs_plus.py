#!/usr/bin/env python3
"""High-field Jacobian eigenvalues for FutureINNPlus (coarses held fixed).

Does not modify eval_jacobian_high_eigs.py. Differentiates next high w.r.t.
current high with coarse latents frozen, then advances with the full AR step.
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

from eval_jacobian_manual_mse_plus import (
    choose_sections,
    load_kept_paths,
    load_model,
    load_trajectories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["torch", "scipy"], required=True)
    p.add_argument(
        "--jacobian",
        choices=["high", "full"],
        default="high",
        help="high: next-high vs high (coarses fixed). full: square map over (high,coarses)",
    )
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/checkpoint-last.pth",
    )
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/jacobian_high_eigs",
    )
    p.add_argument("--save-jacobian", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--ncv", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3)
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


def warmup_coarses(ema_vit, model, high: torch.Tensor):
    b = high.shape[0]
    device = high.device
    dtype = high.dtype
    k = model.num_coarse
    coarses = [
        torch.zeros(
            b, 1, model.latent_res[i], model.latent_res[i], device=device, dtype=dtype
        )
        for i in range(k)
    ]
    with torch.no_grad():
        for c in range(k):
            t = float(c) * torch.ones(b, device=device, dtype=dtype)
            _, outs = ema_vit(high, coarses, t)
            coarses[c] = outs[c].detach()
    return coarses


def _t_full(model, high: torch.Tensor) -> torch.Tensor:
    return float(model.num_coarse) * torch.ones(
        high.shape[0], device=high.device, dtype=high.dtype
    )


def pack_plus_state(high: torch.Tensor, coarses: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [high.reshape(-1)] + [c.reshape(-1) for c in coarses],
        dim=0,
    )


def unpack_plus_state(
    flat: torch.Tensor,
    high: torch.Tensor,
    coarses: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    nh = high.numel()
    h = flat[:nh].reshape_as(high)
    offset = nh
    out_c: list[torch.Tensor] = []
    for c in coarses:
        n = c.numel()
        out_c.append(flat[offset : offset + n].reshape_as(c))
        offset += n
    return h, out_c


def time_one_jvp(ema_vit, model, high, coarses, jacobian: str = "high") -> float:
    dtype = high.dtype
    h0 = high.detach().to(dtype)
    c_fixed = [c.detach().to(dtype) for c in coarses]
    t = _t_full(model, h0)
    if h0.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if jacobian == "high":
        v = torch.randn_like(h0)
        v = v / v.norm()
        _, jv = torch.autograd.functional.jvp(
            lambda h: ema_vit(h, c_fixed, t)[0],
            (h0,),
            (v,),
        )
    else:
        v = torch.randn(
            pack_plus_state(h0, c_fixed).numel(),
            device=h0.device,
            dtype=dtype,
        )
        v = v / v.norm()
        vh, vc = unpack_plus_state(v, h0, c_fixed)

        def f(h, *cs):
            oh, ocs = ema_vit(h, list(cs), t)
            return pack_plus_state(oh, ocs)

        _, jv = torch.autograd.functional.jvp(f, (h0, *c_fixed), (vh, *vc))
    _ = jv.detach()
    if h0.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def compute_torch_eigs(ema_vit, model, high, coarses, save_jacobian: bool):
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    c_fixed = [c.detach().to(dtype) for c in coarses]
    t = _t_full(model, h0)
    J = torch.autograd.functional.jacobian(
        lambda h: ema_vit(h, c_fixed, t)[0][0, 0, :, :],
        h0,
    )[:, :, 0, 0, :, :]
    n = h0[0, 0].numel()
    J_mat = J.reshape(n, n)
    vals = torch.linalg.eigvals(J_mat)
    order = torch.argsort(vals.abs(), descending=True)
    vals = vals[order]
    out = {
        "eigenvalues": vals.detach().cpu().numpy().astype(np.complex128),
        "radius": np.asarray(float(vals.abs().max().item()), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "converged": np.asarray(True),
    }
    if save_jacobian:
        out["jacobian"] = J_mat.detach().float().cpu().numpy().astype(np.float32)
    return out


def compute_scipy_eigs(
    ema_vit, model, high, coarses, ncv, maxiter, tol, jacobian: str = "high"
):
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    c_fixed = [c.detach().to(dtype) for c in coarses]
    t = _t_full(model, h0)
    if jacobian == "high":
        n = int(h0.numel())

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = (
                torch.from_numpy(np.ascontiguousarray(v_np))
                .to(device=h0.device, dtype=h0.dtype)
                .reshape_as(h0)
            )
            _, jv = torch.autograd.functional.jvp(
                lambda h: ema_vit(h, c_fixed, t)[0],
                (h0,),
                (v,),
            )
            return jv.detach().reshape(-1).float().cpu().numpy()

    else:
        n = int(h0.numel() + sum(c.numel() for c in c_fixed))

        def matvec(v_np: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(
                device=h0.device, dtype=h0.dtype
            )
            vh, vc = unpack_plus_state(v, h0, c_fixed)

            def f(h, *cs):
                oh, ocs = ema_vit(h, list(cs), t)
                return pack_plus_state(oh, ocs)

            _, jv = torch.autograd.functional.jvp(f, (h0, *c_fixed), (vh, *vc))
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
        f"kept={len(kept)} section={int(chosen[0])} horizon={args.horizon} "
        f"method={args.method} jacobian={args.jacobian} device={device}",
        flush=True,
    )

    model, ema_vit = load_model(args.checkpoint, device)
    print(
        f"checkpoint={args.checkpoint} latent_res={model.latent_res}",
        flush=True,
    )

    init_np = np.asarray(frames[0], dtype=np.float32)
    high = torch.from_numpy(init_np).to(device).unsqueeze(0).unsqueeze(0) / args.norm
    dtype = next(ema_vit.parameters()).dtype
    high = high.to(dtype)
    coarses = warmup_coarses(ema_vit, model, high)

    if args.jacobian == "high":
        n = int(high.numel())
    else:
        n = int(high.numel() + sum(c.numel() for c in coarses))
    jvp_s = time_one_jvp(ema_vit, model, high, coarses, jacobian=args.jacobian)
    print(
        f"one_jvp_s={jvp_s:.4f} jacobian={args.jacobian} state_dim={n}",
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
        f"latent_res={model.latent_res}",
    ]

    tK = _t_full(model, high)
    for step in range(args.horizon):
        out_path = step_path(save_dir, step)
        if args.skip_existing and out_path.is_file():
            print(f"step {step}: skip existing {out_path}", flush=True)
            continue
        print(f"step {step}: computing {args.method} eigs ...", flush=True)
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            if args.method == "torch":
                result = compute_torch_eigs(
                    ema_vit, model, high, coarses, args.save_jacobian
                )
            else:
                result = compute_scipy_eigs(
                    ema_vit,
                    model,
                    high,
                    coarses,
                    ncv=args.ncv,
                    maxiter=args.maxiter,
                    tol=args.tol,
                    jacobian=args.jacobian,
                )
        except Exception as exc:
            fail_path = save_dir / f"step_{step}_failed.txt"
            fail_path.write_text(f"Error at step {step}:\n{type(exc).__name__}: {exc}\n")
            print(f"step {step}: failed -> {fail_path}", flush=True)
            summary_lines.append(f"step_{step}=FAILED:{type(exc).__name__}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mib = peak_gpu_mib()
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["peak_gpu_mib"] = np.asarray(peak_mib, dtype=np.float64)
        result["step"] = np.asarray(step, dtype=np.int64)
        result["method"] = np.asarray(args.method)
        result["jacobian"] = np.asarray(args.jacobian)
        np.savez_compressed(out_path, **result)
        radius = float(result["radius"])
        print(
            f"step {step}: radius={radius:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(
            f"step_{step}=ok radius={radius:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f}"
        )

        if step + 1 < args.horizon:
            with torch.no_grad():
                high, coarses = ema_vit(high, coarses, tK)
            high = high.detach()
            coarses = [c.detach() for c in coarses]

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}", flush=True)
    print("JACOBIAN_HIGH_EIGS_PLUS_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
