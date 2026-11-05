#!/usr/bin/env python3
"""Multi-GPU scipy Jacobian eigs for FutureINNPlus on ground-truth inputs.

Does not modify eval_jacobian_high_eigs_scipy3_parallel.py. Each worker
linearizes at staggered ground-truth states (n_jac_steps = horizon - num_coarse).
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
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_high_eigs_plus import (
    _t_full,
    gt_plus_state,
    pack_plus_state,
    peak_gpu_mib,
    reset_peak_gpu,
    step_path,
    time_one_jvp,
    unpack_plus_state,
)
from eval_jacobian_manual_mse_plus import (
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
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/checkpoint-last.pth",
    )
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu_plus/jacobian_high_eigs/scipy_3_n4",
    )
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument(
        "--jacobian",
        choices=["high", "full"],
        default="full",
        help="full (default): square map over (high,coarses). high: next-high vs high (coarses fixed)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def compute_scipy_eigs_with_vecs(
    ema_vit, model, high, coarses, ncv, maxiter, tol, jacobian: str = "high"
):
    """Same as compute_scipy_eigs but also returns the leading eigenvector."""
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
    vecs = None
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
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    if vecs is None or vecs.size == 0:
        raise RuntimeError("ARPACK returned eigenvalues without eigenvectors")
    vecs = np.asarray(vecs)[:, order]
    return {
        "eigenvalues": vals.astype(np.complex128),
        "eigenvectors": vecs[:, 0].astype(np.complex128),
        "radius": np.asarray(float(np.max(np.abs(vals))), dtype=np.float64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "converged": np.asarray(converged),
        "k": np.asarray(1, dtype=np.int64),
        "ncv": np.asarray(ncv_eff, dtype=np.int64),
        "maxiter": np.asarray(maxiter, dtype=np.int64),
        "tol": np.asarray(tol, dtype=np.float64),
        "jacobian": np.asarray(jacobian),
    }


def run_worker(args: argparse.Namespace) -> None:
    if args.section_index is None:
        raise ValueError("--worker requires --section-index")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    chosen = choose_sections(n_available, args.n_initial_conditions)
    section_id = int(chosen[args.section_index])
    trajectories = load_trajectories(kept, np.asarray([section_id]), args.horizon)
    frames = trajectories[0]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[worker section_index={args.section_index} section_id={section_id}] "
        f"CUDA_VISIBLE_DEVICES={visible!r} device={device}",
        flush=True,
    )

    model, ema_vit = load_model(args.checkpoint, device)
    n_jac_steps = args.horizon - model.num_coarse
    if n_jac_steps < 1:
        raise ValueError(
            f"horizon must be > num_coarse={model.num_coarse} for "
            f"ground-truth stagger, got {args.horizon}"
        )
    dtype = next(ema_vit.parameters()).dtype
    high, coarses = gt_plus_state(
        frames, 0, model, device, dtype, args.norm
    )

    if args.jacobian == "high":
        n = int(high.numel())
    else:
        n = int(high.numel() + sum(c.numel() for c in coarses))
    jvp_s = time_one_jvp(ema_vit, model, high, coarses, jacobian=args.jacobian)
    print(
        f"[section {section_id}] one_jvp_s={jvp_s:.4f} "
        f"jacobian={args.jacobian} state_dim={n} n_jac_steps={n_jac_steps}",
        flush=True,
    )

    summary_lines = [
        "method=scipy",
        f"jacobian={args.jacobian}",
        f"checkpoint={args.checkpoint}",
        f"section_index={args.section_index}",
        f"section_id={section_id}",
        f"horizon={args.horizon}",
        f"n_jac_steps={n_jac_steps}",
        "frame_rule=high=k coarse_i=pool(k+1+i)",
        f"state_dim={n}",
        f"one_jvp_s={jvp_s:.6f}",
        f"latent_res={model.latent_res}",
        f"num_coarse={model.num_coarse}",
    ]

    for step in range(n_jac_steps):
        high, coarses = gt_plus_state(
            frames, step, model, device, dtype, args.norm
        )
        out_path = step_path(save_dir, step)
        print(f"[section {section_id}] step {step}: computing scipy eigs ...", flush=True)
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            result = compute_scipy_eigs_with_vecs(
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
            print(f"[section {section_id}] step {step}: failed -> {fail_path}", flush=True)
            summary_lines.append(f"step_{step}=FAILED:{type(exc).__name__}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mib = peak_gpu_mib()
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["peak_gpu_mib"] = np.asarray(peak_mib, dtype=np.float64)
        result["step"] = np.asarray(step, dtype=np.int64)
        result["section_id"] = np.asarray(section_id, dtype=np.int64)
        result["section_index"] = np.asarray(args.section_index, dtype=np.int64)
        result["method"] = np.asarray("scipy")
        result["jacobian"] = np.asarray(args.jacobian)
        np.savez_compressed(out_path, **result)
        radius = float(result["radius"])
        print(
            f"[section {section_id}] step {step}: radius={radius:.6g} "
            f"wall_s={wall_s:.2f} -> {out_path}",
            flush=True,
        )
        summary_lines.append(f"step_{step}=ok radius={radius:.6g} wall_s={wall_s:.2f}")

    (save_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"[section {section_id}] JACOBIAN_HIGH_EIGS_PLUS_WORKER_DONE", flush=True)


def _pump(stream, prefix: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    stream.close()


def run_orchestrator(args: argparse.Namespace) -> None:
    if args.n_initial_conditions != args.n_gpus:
        raise ValueError("--n-initial-conditions must equal --n-gpus")
    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    n_available = len(kept) // args.horizon
    chosen = choose_sections(n_available, args.n_initial_conditions)
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[orchestrator] chosen={chosen.tolist()} n_gpus={args.n_gpus} save_root={save_root}",
        flush=True,
    )

    script = str(Path(__file__).resolve())
    procs, pumps = [], []
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
            "--jacobian",
            args.jacobian,
            "--seed",
            str(args.seed),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        procs.append(proc)
        t = threading.Thread(
            target=_pump, args=(proc.stdout, f"[rank{i}/sec{section_id}] "), daemon=True
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
