#!/usr/bin/env python3
"""Hutchinson trace of the full ground-truth Jacobian (FutureINNPlus).

Linearizes the packed (high,coarses)->(next high,coarses) map at the training
stagger: high=frame k, coarse i=pool(k+1+i), t=num_coarse.
Estimates tr(J) ≈ mean(v · Jv) with Rademacher probes (±1), then reports
trace / state_dim. Does not normalize probes. Single-process only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from eval_jacobian_high_eigs_plus import (
    _t_full,
    gt_plus_state,
    pack_plus_state,
    peak_gpu_mib,
    reset_peak_gpu,
    unpack_plus_state,
)
from eval_jacobian_manual_mse_plus import (
    choose_sections,
    load_kept_paths,
    load_model,
    load_trajectories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/jacobian_trace_corrected_gt/n16",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=16,
        help="Number of Rademacher Hutchinson probes per step",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def step_path(save_dir: Path, step: int) -> Path:
    return save_dir / f"trace_step{step:02d}.npz"


def rademacher(n: int, device: torch.device, dtype: torch.dtype, rng: np.random.Generator) -> torch.Tensor:
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
    return torch.from_numpy(np.ascontiguousarray(signs)).to(device=device, dtype=dtype)


def compute_hutchinson_trace(
    ema_vit,
    model,
    high: torch.Tensor,
    coarses: list[torch.Tensor],
    n_probes: int,
    seed: int,
    step: int,
) -> dict[str, np.ndarray]:
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    c_fixed = [c.detach().to(dtype) for c in coarses]
    t = _t_full(model, h0)
    n = int(h0.numel() + sum(c.numel() for c in c_fixed))
    rng = np.random.default_rng(seed + step)

    def f(h, *cs):
        oh, ocs = ema_vit(h, list(cs), t)
        return pack_plus_state(oh, ocs)

    dots = np.zeros(n_probes, dtype=np.float64)
    for i in range(n_probes):
        v = rademacher(n, h0.device, h0.dtype, rng)
        vh, vc = unpack_plus_state(v, h0, c_fixed)
        _, jv = torch.autograd.functional.jvp(f, (h0, *c_fixed), (vh, *vc))
        dots[i] = float((v * jv.detach().reshape(-1)).sum().item())

    trace = float(dots.mean())
    trace_std = float(dots.std(ddof=1)) if n_probes > 1 else 0.0
    return {
        "probe_dots": dots,
        "trace": np.asarray(trace, dtype=np.float64),
        "trace_over_n": np.asarray(trace / n, dtype=np.float64),
        "trace_std": np.asarray(trace_std, dtype=np.float64),
        "trace_over_n_std": np.asarray(trace_std / n, dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "n_probes": np.asarray(n_probes, dtype=np.int64),
        "step": np.asarray(step, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    if args.n_probes < 1:
        raise ValueError(f"--n-probes must be >= 1, got {args.n_probes}")
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
        f"n_probes={args.n_probes} device={device}",
        flush=True,
    )

    model, ema_vit = load_model(args.checkpoint, device)
    print(
        f"checkpoint={args.checkpoint} latent_res={model.latent_res}",
        flush=True,
    )

    n_jac_steps = args.horizon - model.num_coarse
    if n_jac_steps < 1:
        raise ValueError(
            f"horizon must be > num_coarse={model.num_coarse}, got {args.horizon}"
        )
    dtype = next(ema_vit.parameters()).dtype
    high, coarses = gt_plus_state(frames, 0, model, device, dtype, args.norm)
    n = int(high.numel() + sum(c.numel() for c in coarses))

    summary_lines = [
        "method=hutchinson",
        "jacobian=full",
        f"checkpoint={args.checkpoint}",
        f"section={int(chosen[0])}",
        f"horizon={args.horizon}",
        f"n_jac_steps={n_jac_steps}",
        "frame_rule=high=k coarse_i=pool(k+1+i)",
        f"state_dim={n}",
        f"n_probes={args.n_probes}",
        f"device={device}",
        f"latent_res={model.latent_res}",
        f"num_coarse={model.num_coarse}",
        f"seed={args.seed}",
    ]

    for step in range(n_jac_steps):
        high, coarses = gt_plus_state(
            frames, step, model, device, dtype, args.norm
        )
        out_path = step_path(save_dir, step)
        print(
            f"step {step}: computing Hutchinson trace ({args.n_probes} probes) ...",
            flush=True,
        )
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            result = compute_hutchinson_trace(
                ema_vit,
                model,
                high,
                coarses,
                n_probes=args.n_probes,
                seed=args.seed,
                step=step,
            )
        except Exception as exc:
            fail_path = save_dir / f"step_{step}_failed.txt"
            fail_path.write_text(
                f"Error at step {step}:\n{type(exc).__name__}: {exc}\n"
            )
            print(f"step {step}: failed -> {fail_path}", flush=True)
            summary_lines.append(f"step_{step}=FAILED:{type(exc).__name__}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mib = peak_gpu_mib()
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["peak_gpu_mib"] = np.asarray(peak_mib, dtype=np.float64)
        result["method"] = np.asarray("hutchinson")
        result["jacobian"] = np.asarray("full")
        np.savez_compressed(out_path, **result)
        ton = float(result["trace_over_n"])
        print(
            f"step {step}: trace_over_n={ton:.6g} "
            f"trace={float(result['trace']):.6g} "
            f"wall_s={wall_s:.2f} peak_gpu_mib={peak_mib:.1f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(
            f"step_{step}=ok trace_over_n={ton:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f}"
        )

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {summary_path}", flush=True)
    print("JACOBIAN_TRACE_PLUS_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
