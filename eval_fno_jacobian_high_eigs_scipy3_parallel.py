#!/usr/bin/env python3
"""Multi-GPU scipy_3 Jacobian eigenvalue sweep for the L=1 FNO baseline.

Same orchestrator / ARPACK settings as eval_jacobian_high_eigs_scipy3_parallel.py,
but the map is FNO2d(high) with no mid/low warmup. Leaves the HINE script untouched.

Orchestrator:
  python eval_fno_jacobian_high_eigs_scipy3_parallel.py --n-initial-conditions 4 --n-gpus 4

Worker (set by the parent with CUDA_VISIBLE_DEVICES):
  python eval_fno_jacobian_high_eigs_scipy3_parallel.py --worker --section-index 0 ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Parent orchestrator must not bind GPUs; workers set CUDA_VISIBLE_DEVICES in env.
if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.sparse.linalg as spla
import torch

from eval_fno_jacobian_manual_mse import load_fno
from eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
)
from fno_baseline import FNO2d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-initial-conditions",
        type=int,
        default=4,
        help="Evenly spaced sections via choose_sections",
    )
    p.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help="Must equal --n-initial-conditions",
    )
    p.add_argument(
        "--worker",
        action="store_true",
        help="Run a single section (set CUDA_VISIBLE_DEVICES before launch)",
    )
    p.add_argument(
        "--section-index",
        type=int,
        default=None,
        help="Index into the choose_sections list (worker only)",
    )
    p.add_argument(
        "--every-nth-field",
        type=int,
        default=5,
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
        "--horizon",
        type=int,
        default=20,
    )
    p.add_argument(
        "--norm",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/fno_baseline_debug/jacobian_high_eigs/scipy_3_n4",
        help="Root for orchestrator; worker uses a section_* subdirectory",
    )
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


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


def time_one_jvp(net: FNO2d, high: torch.Tensor) -> float:
    h0 = high.detach().float()
    v = torch.randn_like(h0)
    v = v / v.norm()
    if h0.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, jv = torch.autograd.functional.jvp(
        lambda h: net(h),
        (h0,),
        (v,),
    )
    _ = jv.detach()
    if h0.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def compute_scipy_eigs(
    net: FNO2d,
    high: torch.Tensor,
    ncv: int,
    maxiter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    """scipy_3 path with eigenvectors: jvp of FNO2d + eigs(k=1, which=LM)."""
    h0 = high.detach().float()
    n = int(h0.numel())
    ncv_eff = min(max(ncv, 2), n)

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = (
            torch.from_numpy(np.ascontiguousarray(v_np))
            .to(device=h0.device, dtype=h0.dtype)
            .reshape_as(h0)
        )
        _, jv = torch.autograd.functional.jvp(
            lambda h: net(h),
            (h0,),
            (v,),
        )
        return jv.detach().reshape(-1).float().cpu().numpy()

    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float32)
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    converged = True
    vecs: np.ndarray | None = None
    try:
        vals, vecs = spla.eigs(
            op,
            k=1,
            which="LM",
            ncv=ncv_eff,
            maxiter=maxiter,
            tol=tol,
            v0=v0,
            return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues)
        vecs = np.asarray(exc.eigenvectors) if exc.eigenvectors is not None else None
        finite = np.isfinite(vals)
        vals = vals[finite]
        if vecs is not None and vecs.size and finite.any():
            vecs = np.asarray(vecs)[:, finite]
        converged = False
        if vals.size == 0:
            raise
        print(
            f"    partial convergence: kept {vals.size}/1 eigenvalues",
            flush=True,
        )
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    if vecs is None or vecs.size == 0:
        raise RuntimeError("ARPACK returned eigenvalues without eigenvectors")
    vecs = np.asarray(vecs)[:, order]
    evec = vecs[:, 0].astype(np.complex128)
    return {
        "eigenvalues": vals.astype(np.complex128),
        "eigenvectors": evec,
        "radius": np.asarray(float(np.max(np.abs(vals))), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "converged": np.asarray(converged),
        "k": np.asarray(1, dtype=np.int64),
        "ncv": np.asarray(ncv_eff, dtype=np.int64),
        "maxiter": np.asarray(maxiter, dtype=np.int64),
        "tol": np.asarray(tol, dtype=np.float64),
    }


def run_worker(args: argparse.Namespace) -> None:
    if args.section_index is None:
        raise ValueError("--worker requires --section-index")

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
        f"ncv={args.ncv} maxiter={args.maxiter} tol={args.tol} "
        f"save_dir={save_dir}",
        flush=True,
    )

    net = load_fno(args.checkpoint, device)
    print(
        f"[section {section_id}] checkpoint={args.checkpoint} "
        f"modes={net.modes} width={net.width} layers={net.n_layers}",
        flush=True,
    )

    init_np = np.asarray(frames[0], dtype=np.float32)
    high = torch.from_numpy(init_np).to(device).unsqueeze(0).unsqueeze(0) / args.norm
    high = high.float()

    n = int(high.numel())
    jvp_s = time_one_jvp(net, high)
    print(
        f"[section {section_id}] one_jvp_s={jvp_s:.4f} state_dim={n}",
        flush=True,
    )

    summary_lines = [
        "method=scipy",
        "model_type=fno",
        f"checkpoint={args.checkpoint}",
        f"section_index={args.section_index}",
        f"section_id={section_id}",
        f"horizon={args.horizon}",
        f"state_dim={n}",
        f"one_jvp_s={jvp_s:.6f}",
        f"device={device}",
        f"CUDA_VISIBLE_DEVICES={visible}",
        f"ncv={args.ncv}",
        f"maxiter={args.maxiter}",
        f"tol={args.tol}",
    ]

    n_jac_steps = args.horizon

    for k in range(n_jac_steps):
        out_path = step_path(save_dir, k)
        print(
            f"[section {section_id}] step {k}: computing scipy eigs ...",
            flush=True,
        )
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            result = compute_scipy_eigs(
                net,
                high,
                ncv=args.ncv,
                maxiter=args.maxiter,
                tol=args.tol,
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
        result["step"] = np.asarray(k, dtype=np.int64)
        result["section_id"] = np.asarray(section_id, dtype=np.int64)
        result["section_index"] = np.asarray(args.section_index, dtype=np.int64)
        result["method"] = np.asarray("scipy")
        result["model_type"] = np.asarray("fno")

        np.savez_compressed(out_path, **result)
        radius = float(result["radius"])
        print(
            f"[section {section_id}] step {k}: radius={radius:.6g} "
            f"wall_s={wall_s:.2f} peak_gpu_mib={peak_mib:.1f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(
            f"step_{k}=ok radius={radius:.6g} wall_s={wall_s:.2f} "
            f"peak_gpu_mib={peak_mib:.1f}"
        )

        if k + 1 < n_jac_steps:
            with torch.no_grad():
                high = net(high)
            high = high.detach()

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"[section {section_id}] wrote {summary_path}", flush=True)
    print(f"[section {section_id}] FNO_JACOBIAN_HIGH_EIGS_WORKER_DONE", flush=True)


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

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    chosen = choose_sections(n_available, args.n_initial_conditions)
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[orchestrator] n_available={n_available} chosen={chosen.tolist()} "
        f"n_gpus={args.n_gpus} save_root={save_root}",
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
            "--ncv",
            str(args.ncv),
            "--maxiter",
            str(args.maxiter),
            "--tol",
            str(args.tol),
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

    codes = []
    for proc in procs:
        codes.append(proc.wait())
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
