#!/usr/bin/env python3
"""Hutchinson trace of the full ground-truth Jacobian (L=3).

Linearizes the packed (high,mid,low)->(next high,mid,low) map at the training
stagger: high=frame k, mid=pool(k+1), low=pool(k+2), t=2.
Estimates tr(J) ≈ mean(v · Jv) with Rademacher probes (±1), then reports
trace / state_dim. Does not normalize probes.

Orchestrator:
  python eval/eval_jacobian_trace.py --n-initial-conditions 4 --n-gpus 4

Worker:
  python eval/eval_jacobian_trace.py --worker --section-index 0 ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from eval_jacobian_high_eigs import gt_l3_state, pack_l3_state, unpack_l3_state
from eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_model,
    load_trajectories,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-initial-conditions", type=int, default=4)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--section-index", type=int, default=None)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/checkpoint-last.pth",
    )
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_trace_corrected_gt/n16",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=16,
        help="Number of Rademacher Hutchinson probes per step",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def step_path(save_dir: Path, step: int) -> Path:
    return save_dir / f"trace_step{step:02d}.npz"


def peak_gpu_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))


def reset_peak_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def rademacher(n: int, device: torch.device, dtype: torch.dtype, rng: np.random.Generator) -> torch.Tensor:
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
    return torch.from_numpy(np.ascontiguousarray(signs)).to(device=device, dtype=dtype)


def compute_hutchinson_trace(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    n_probes: int,
    seed: int,
    step: int,
) -> dict[str, np.ndarray]:
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    mid_c = mid.detach().to(dtype)
    low_c = low.detach().to(dtype)
    t = 2 * torch.ones(h0.shape[0], device=h0.device, dtype=dtype)
    n = int(h0.numel() + mid_c.numel() + low_c.numel())
    rng = np.random.default_rng(seed + step)

    def f(h, m, l):
        oh, om, ol = ema_vit(h, m, l, t)
        return pack_l3_state(oh, om, ol)

    dots = np.zeros(n_probes, dtype=np.float64)
    for i in range(n_probes):
        v = rademacher(n, h0.device, h0.dtype, rng)
        vh, vm, vl = unpack_l3_state(v, h0, mid_c, low_c)
        _, jv = torch.autograd.functional.jvp(f, (h0, mid_c, low_c), (vh, vm, vl))
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


def run_worker(args: argparse.Namespace) -> None:
    if args.section_index is None:
        raise ValueError("--worker requires --section-index")
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
    chosen = choose_sections(n_available, args.n_initial_conditions)
    if args.section_index < 0 or args.section_index >= len(chosen):
        raise ValueError(
            f"section-index {args.section_index} out of range for "
            f"{len(chosen)} chosen sections {chosen.tolist()}"
        )
    section_id = int(chosen[args.section_index])
    trajectories = load_trajectories(kept, np.asarray([section_id]), args.horizon)
    frames = trajectories[0]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[worker section_index={args.section_index} section_id={section_id}] "
        f"CUDA_VISIBLE_DEVICES={visible!r} device={device} "
        f"n_probes={args.n_probes} save_dir={save_dir}",
        flush=True,
    )

    model, ema_vit = load_model(args.checkpoint, device)
    print(
        f"[section {section_id}] checkpoint={args.checkpoint} "
        f"ds_res={model.ds_res} low_res={model.low_res}",
        flush=True,
    )

    n_jac_steps = args.horizon - 2
    if n_jac_steps < 1:
        raise ValueError(
            f"horizon must be >= 3 for ground-truth stagger, got {args.horizon}"
        )
    dtype = next(ema_vit.parameters()).dtype
    high, mid, low = gt_l3_state(frames, 0, model, device, dtype, args.norm)
    n = int(high.numel() + mid.numel() + low.numel())

    summary_lines = [
        "method=hutchinson",
        "jacobian=full",
        f"checkpoint={args.checkpoint}",
        f"section_index={args.section_index}",
        f"section_id={section_id}",
        f"horizon={args.horizon}",
        f"n_jac_steps={n_jac_steps}",
        "frame_rule=high=k mid=pool(k+1) low=pool(k+2)",
        f"state_dim={n}",
        f"n_probes={args.n_probes}",
        f"device={device}",
        f"CUDA_VISIBLE_DEVICES={visible}",
        f"seed={args.seed}",
    ]

    for k in range(n_jac_steps):
        high, mid, low = gt_l3_state(frames, k, model, device, dtype, args.norm)
        out_path = step_path(save_dir, k)
        print(
            f"[section {section_id}] step {k}: computing Hutchinson trace "
            f"({args.n_probes} probes) ...",
            flush=True,
        )
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            result = compute_hutchinson_trace(
                ema_vit,
                high,
                mid,
                low,
                n_probes=args.n_probes,
                seed=args.seed,
                step=k,
            )
        except torch.cuda.OutOfMemoryError as exc:
            fail_path = save_dir / f"step_{k}_failed.txt"
            fail_path.write_text(f"OutOfMemoryError at step {k}:\n{exc}\n")
            print(f"[section {section_id}] step {k}: OOM -> {fail_path}", flush=True)
            summary_lines.append(f"step_{k}=OOM")
            break
        except Exception as exc:
            fail_path = save_dir / f"step_{k}_failed.txt"
            fail_path.write_text(
                f"Error at step {k}:\n{type(exc).__name__}: {exc}\n"
            )
            print(
                f"[section {section_id}] step {k}: failed -> {fail_path}",
                flush=True,
            )
            summary_lines.append(f"step_{k}=FAILED:{type(exc).__name__}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mib = peak_gpu_mib()
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["peak_gpu_mib"] = np.asarray(peak_mib, dtype=np.float64)
        result["section_id"] = np.asarray(section_id, dtype=np.int64)
        result["section_index"] = np.asarray(args.section_index, dtype=np.int64)
        result["method"] = np.asarray("hutchinson")
        result["jacobian"] = np.asarray("full")

        np.savez_compressed(out_path, **result)
        ton = float(result["trace_over_n"])
        print(
            f"[section {section_id}] step {k}: trace_over_n={ton:.6g} "
            f"trace={float(result['trace']):.6g} "
            f"wall_s={wall_s:.2f} peak_gpu_mib={peak_mib:.1f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(
            f"step_{k}=ok trace_over_n={ton:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f}"
        )

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"[section {section_id}] wrote {summary_path}", flush=True)
    print(f"[section {section_id}] JACOBIAN_TRACE_WORKER_DONE", flush=True)


def _pump(stream, prefix: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    stream.close()


def run_orchestrator(args: argparse.Namespace) -> None:
    if args.n_initial_conditions != args.n_gpus:
        raise ValueError(
            f"--n-initial-conditions ({args.n_initial_conditions}) must equal "
            f"--n-gpus ({args.n_gpus})"
        )
    if args.n_initial_conditions < 1:
        raise ValueError("--n-initial-conditions must be >= 1")
    if args.n_probes < 1:
        raise ValueError(f"--n-probes must be >= 1, got {args.n_probes}")

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    chosen = choose_sections(n_available, args.n_initial_conditions)
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[orchestrator] n_available={n_available} chosen={chosen.tolist()} "
        f"n_gpus={args.n_gpus} n_probes={args.n_probes} save_root={save_root}",
        flush=True,
    )

    script = str(Path(__file__).resolve())
    procs: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []

    for i, section_id in enumerate(chosen):
        section_id = int(section_id)
        worker_dir = save_root / f"section_{section_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            script,
            "--worker",
            "--section-index",
            str(i),
            "--n-initial-conditions",
            str(args.n_initial_conditions),
            "--n-gpus",
            str(args.n_gpus),
            "--every-nth-field",
            str(args.every_nth_field),
            "--eval-dir",
            args.eval_dir,
            "--checkpoint",
            args.checkpoint,
            "--horizon",
            str(args.horizon),
            "--norm",
            str(args.norm),
            "--device",
            args.device,
            "--save-dir",
            str(worker_dir),
            "--n-probes",
            str(args.n_probes),
            "--seed",
            str(args.seed),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        print(
            f"[orchestrator] launching rank {i} section_id={section_id} "
            f"CUDA_VISIBLE_DEVICES={i} -> {worker_dir}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(
            target=_pump,
            args=(proc.stdout, f"[rank{i}/sec{section_id}] "),
            daemon=True,
        )
        t.start()
        pumps.append(t)

    codes = [proc.wait() for proc in procs]
    for t in pumps:
        t.join(timeout=5)
    if any(c != 0 for c in codes):
        raise RuntimeError(f"worker exit codes: {codes}")
    print("[orchestrator] all workers finished OK", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
