#!/usr/bin/env python3
"""Hutchinson trace of the fine-field Jacobian for Section 11 ablations.

The closed-loop step is fine input to fine output. Aux heads are dropped,
matching eval_ablation_eigs.fine_forward. Probes match eval_jacobian_trace.py:
16 unnormalized Rademacher vectors, tr(J) ≈ mean(v · Jv).

Also stores the Task 1 difference-quotient label
(tr(J) - N) / (N dt) with dt = 0.05.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from eval.eval_ablation_eigs import fine_forward, gt_fine, load_ablation
from eval.eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
)

RUNS = Path("/pscratch/sd/d/divyam/hine_runs")
CASES = [
    ("l1", RUNS / "ablation_l1/checkpoint-last.pth", RUNS / "ablation_l1/jacobian_trace"),
    (
        "aux1",
        RUNS / "ablation_l1_aux_1_b16/checkpoint-last.pth",
        RUNS / "ablation_l1_aux_1_b16/jacobian_trace",
    ),
    (
        "aux2",
        RUNS / "ablation_l1_aux_2/checkpoint-last.pth",
        RUNS / "ablation_l1_aux_2/jacobian_trace",
    ),
    (
        "p2",
        RUNS / "ablation_l1_unrolled_p2/checkpoint-last.pth",
        RUNS / "ablation_l1_unrolled_p2/jacobian_trace",
    ),
    (
        "p3_ep150",
        RUNS / "ablation_l1_unrolled_p3/checkpoint-150.pth",
        RUNS / "ablation_l1_unrolled_p3/jacobian_trace/checkpoint-150",
    ),
    (
        "p3_last",
        RUNS / "ablation_l1_unrolled_p3/checkpoint-last.pth",
        RUNS / "ablation_l1_unrolled_p3/jacobian_trace/checkpoint-last",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="", help="One of l1, aux1, aux2, p2, p3_ep150, p3_last. Empty runs all.")
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--n-sections", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def rademacher(n: int, device, dtype, rng: np.random.Generator) -> torch.Tensor:
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
    return torch.from_numpy(np.ascontiguousarray(signs)).to(device=device, dtype=dtype)


def compute_trace(ema_vit, high, n_probes: int, seed: int, dt: float) -> dict:
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    n = int(h0.numel())
    rng = np.random.default_rng(seed)

    def f(h):
        return fine_forward(ema_vit, h)

    dots = np.zeros(n_probes, dtype=np.float64)
    for i in range(n_probes):
        v = rademacher(n, h0.device, h0.dtype, rng).reshape_as(h0)
        _, jv = torch.autograd.functional.jvp(f, (h0,), (v,))
        dots[i] = float((v.reshape(-1) * jv.detach().reshape(-1)).sum().item())

    trace = float(dots.mean())
    trace_std = float(dots.std(ddof=1)) if n_probes > 1 else 0.0
    return {
        "probe_dots": dots,
        "trace": np.asarray(trace, dtype=np.float64),
        "trace_over_n": np.asarray(trace / n, dtype=np.float64),
        "trace_std": np.asarray(trace_std, dtype=np.float64),
        "trace_over_n_std": np.asarray(trace_std / n, dtype=np.float64),
        "fu_over_n": np.asarray((trace - n) / (n * dt), dtype=np.float64),
        "fu_over_n_std": np.asarray(trace_std / (n * dt), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "n_probes": np.asarray(n_probes, dtype=np.int64),
        "dt": np.asarray(dt, dtype=np.float64),
    }


def summarize_case(save_dir: Path) -> dict:
    files = sorted(save_dir.glob("section_*/trace_step*.npz"))
    if not files:
        raise FileNotFoundError(f"No trace files in {save_dir}")
    ton = np.array([float(np.load(p)["trace_over_n"]) for p in files])
    fu = np.array([float(np.load(p)["fu_over_n"]) for p in files])
    return {
        "n_points": int(ton.size),
        "trace_over_n_mean": float(ton.mean()),
        "trace_over_n_std": float(ton.std(ddof=1)) if ton.size > 1 else 0.0,
        "fu_over_n_mean": float(fu.mean()),
        "fu_over_n_std": float(fu.std(ddof=1)) if fu.size > 1 else 0.0,
    }


def run_case(name: str, checkpoint: Path, save_dir: Path, args, device, kept, chosen) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model, ema_vit = load_ablation(str(checkpoint), device)
    dtype = next(ema_vit.parameters()).dtype
    print(
        f"[{name}] checkpoint={checkpoint} ablation={model.ablation} "
        f"sections={chosen.tolist()} steps={args.n_steps}",
        flush=True,
    )
    trajectories = load_trajectories(kept, chosen, args.n_steps)
    for local_i, section_id in enumerate(chosen):
        frames = trajectories[local_i]
        section_dir = save_dir / f"section_{int(section_id):03d}"
        section_dir.mkdir(parents=True, exist_ok=True)
        for step in range(args.n_steps):
            out_path = section_dir / f"trace_step{step:02d}.npz"
            if args.skip_existing and out_path.is_file():
                print(f"[{name}] section {int(section_id)} step {step}: skip", flush=True)
                continue
            high = gt_fine(frames, step, device, dtype, args.norm)
            seed = args.seed + 10007 * int(section_id) + step
            t0 = time.perf_counter()
            result = compute_trace(ema_vit, high, args.n_probes, seed, args.dt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0
            result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
            result["section_id"] = np.asarray(int(section_id), dtype=np.int64)
            result["step"] = np.asarray(step, dtype=np.int64)
            result["case"] = np.asarray(name)
            result["checkpoint"] = np.asarray(str(checkpoint))
            result["jacobian"] = np.asarray("fine")
            result["method"] = np.asarray("hutchinson")
            np.savez_compressed(out_path, **result)
            print(
                f"[{name}] section {int(section_id)} step {step}: "
                f"trace_over_n={float(result['trace_over_n']):.6g} "
                f"fu_over_n={float(result['fu_over_n']):.6g} "
                f"wall_s={wall_s:.2f}",
                flush=True,
            )
    stats = summarize_case(save_dir)
    stats["case"] = name
    stats["checkpoint"] = str(checkpoint)
    (save_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(
        f"[{name}] SUMMARY n={stats['n_points']} "
        f"trace/N={stats['trace_over_n_mean']:.6g} ± {stats['trace_over_n_std']:.3g} "
        f"fu/N={stats['fu_over_n_mean']:.6g} ± {stats['fu_over_n_std']:.3g}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.n_probes < 2:
        raise SystemExit("--n-probes must be >= 2")
    if args.n_steps < 1 or args.n_sections < 1:
        raise SystemExit("n-steps and n-sections must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cases = [c for c in CASES if not args.case or c[0] == args.case]
    if not cases:
        known = ", ".join(c[0] for c in CASES)
        raise SystemExit(f"Unknown --case {args.case}. Known: {known}")

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.n_steps
    chosen = choose_sections(n_available, args.n_sections)
    print(
        f"kept={len(kept)} sections_available={n_available} chosen={chosen.tolist()} "
        f"device={device}",
        flush=True,
    )
    for name, checkpoint, save_dir in cases:
        run_case(name, checkpoint, save_dir, args, device, kept, chosen)
    print("ABLATION_TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
