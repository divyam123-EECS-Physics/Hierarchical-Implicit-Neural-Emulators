#!/usr/bin/env python3
"""Leading Jacobian eigenvalue of FNO 1-to-1 at ground-truth frames.

Same protocol as eval_ablation_eigs.py: one eval section, state = frame k / norm,
matrix-free JVP, scipy.sparse.linalg.eigs(k=1, which="LM"). The state is never
replaced by a model prediction.

n_jac_steps = horizon - 1. --horizon 21 gives 20 steps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch

from eval.eval_fno_jacobian_manual_mse import load_fno
from eval.eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
)

DEFAULT_CHECKPOINT = "/pscratch/sd/d/divyam/hine_runs/fno_1to1/checkpoint-last.pth"
DEFAULT_SAVE = (
    "/pscratch/sd/d/divyam/hine_runs/fno_1to1/jacobian_high_eigs/gt_n20"
)
COPY_RADIUS = "/global/homes/d/divyam/jacobian_block_plots/fno_1to1_gt_radius.png"
COPY_ARGAND = "/global/homes/d/divyam/jacobian_block_plots/fno_1to1_gt_argand.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Frames in the section. n_jac_steps = horizon - 1.",
    )
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--save-dir", default=DEFAULT_SAVE)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def step_path(save_dir: Path, step: int) -> Path:
    return save_dir / f"eigs_step{step:02d}.npz"


def gt_fine(frames: list, step: int, device, dtype, norm: float) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(frames[step], dtype=np.float32))
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        / norm
    )


def compute_scipy_eigs(net, high, ncv, maxiter, tol, seed: int):
    dtype = next(net.parameters()).dtype
    h0 = high.detach().to(dtype)
    n = int(h0.numel())

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = (
            torch.from_numpy(np.ascontiguousarray(v_np))
            .to(device=h0.device, dtype=h0.dtype)
            .reshape_as(h0)
        )
        _, jv = torch.autograd.functional.jvp(net, (h0,), (v,))
        return jv.detach().reshape(-1).float().cpu().numpy()

    ncv_eff = min(max(ncv, 2), n)
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float32)
    rng = np.random.default_rng(seed)
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
        print(f"    partial convergence: kept {vals.size}/1 eigenvalues", flush=True)
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
        "jacobian": np.asarray("fine"),
        "linearization": np.asarray("ground_truth"),
    }


def load_rows(save_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(save_dir.glob("eigs_step*.npz")):
        data = np.load(path)
        step = int(data["step"]) if "step" in data.files else int(path.stem[-2:])
        lam = np.asarray(data["eigenvalues"]).astype(np.complex128).ravel()
        lam = lam[np.argmax(np.abs(lam))]
        rows.append({"step": step, "lambda": complex(lam), "radius": float(data["radius"])})
    rows.sort(key=lambda row: row["step"])
    return rows


def plot_results(save_dir: Path) -> tuple[Path, Path] | None:
    rows = load_rows(save_dir)
    if not rows:
        return None
    steps = np.array([row["step"] for row in rows])
    radii = np.array([row["radius"] for row in rows])
    lams = np.array([row["lambda"] for row in rows])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(steps, radii, marker="o", lw=1.5)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("largest |eigenvalue|")
    ax.set_title("FNO 1-to-1 at ground-truth inputs")
    radius_path = save_dir / "max_eig_vs_step.png"
    fig.savefig(radius_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="k", ls="--", lw=1.0, label="|λ|=1")
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.axvline(0.0, color="0.75", lw=0.6)
    ax.scatter(np.real(lams), np.imag(lams), s=28, zorder=3)
    r_max = float(max(np.abs(lams).max(), 1.0)) * 1.15
    ax.set_aspect("equal")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("FNO 1-to-1 leading eigenvalues")
    ax.legend(loc="best", fontsize=8)
    argand_path = save_dir / "argand.png"
    fig.savefig(argand_path, dpi=150)
    plt.close(fig)

    copy_dir = Path(COPY_RADIUS).parent
    copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(radius_path, COPY_RADIUS)
    shutil.copy2(argand_path, COPY_ARGAND)
    return radius_path, argand_path


def main() -> None:
    args = parse_args()
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
    frames = load_trajectories(kept, chosen, args.horizon)[0]
    n_jac_steps = args.horizon - 1
    if n_jac_steps < 1:
        raise ValueError(f"horizon must be >= 2, got {args.horizon}")

    net = load_fno(args.checkpoint, device)
    if int(net.in_channels) != 1 or int(net.out_channels) != 1:
        raise ValueError(
            f"This runner is for FNO 1-to-1, got in_channels={net.in_channels} "
            f"out_channels={net.out_channels}"
        )
    dtype = next(net.parameters()).dtype
    high0 = gt_fine(frames, 0, device, dtype, args.norm)
    n = int(high0.numel())
    print(
        f"section={int(chosen[0])} horizon={args.horizon} "
        f"n_jac_steps={n_jac_steps} state_dim={n} device={device} "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )

    summary = [
        "method=scipy",
        "jacobian=fine",
        "linearization=ground_truth",
        f"checkpoint={args.checkpoint}",
        f"section={int(chosen[0])}",
        f"horizon={args.horizon}",
        f"n_jac_steps={n_jac_steps}",
        "frame_rule=high=k",
        f"state_dim={n}",
        f"device={device}",
        f"ncv={args.ncv}",
        f"maxiter={args.maxiter}",
        f"tol={args.tol}",
    ]

    for step in range(n_jac_steps):
        out_path = step_path(save_dir, step)
        if args.skip_existing and out_path.is_file():
            print(f"step {step}: skip existing {out_path}", flush=True)
            continue
        high = gt_fine(frames, step, device, dtype, args.norm)
        print(f"step {step}: computing scipy eigs at ground truth ...", flush=True)
        t0 = time.perf_counter()
        try:
            result = compute_scipy_eigs(
                net, high, args.ncv, args.maxiter, args.tol, args.seed
            )
        except Exception as exc:
            fail_path = save_dir / f"step_{step}_failed.txt"
            fail_path.write_text(
                f"Error at step {step}:\n{type(exc).__name__}: {exc}\n"
            )
            print(f"step {step}: failed -> {fail_path}", flush=True)
            summary.append(f"step_{step}=FAILED:{type(exc).__name__}")
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        result["wall_s"] = np.asarray(wall_s, dtype=np.float64)
        result["step"] = np.asarray(step, dtype=np.int64)
        result["method"] = np.asarray("scipy")
        np.savez_compressed(out_path, **result)
        radius = float(result["radius"])
        print(
            f"step {step}: radius={radius:.6g} wall_s={wall_s:.2f} -> {out_path}",
            flush=True,
        )
        summary.append(f"step_{step}=ok radius={radius:.6g} wall_s={wall_s:.2f}")

    (save_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    plotted = plot_results(save_dir)
    if plotted:
        print(f"wrote {plotted[0]}", flush=True)
        print(f"wrote {plotted[1]}", flush=True)
        print(f"copied {COPY_RADIUS}", flush=True)
        print(f"copied {COPY_ARGAND}", flush=True)
    print("FNO_GT_EIGS_DONE", flush=True)


if __name__ == "__main__":
    main()
