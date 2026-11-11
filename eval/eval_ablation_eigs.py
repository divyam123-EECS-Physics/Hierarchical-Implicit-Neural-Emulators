#!/usr/bin/env python3
"""Largest-magnitude Jacobian eigenvalue of a FineUNet ablation.

Matches eval_jacobian_high_eigs.py: one ground-truth section, matrix-free
JVP, scipy.sparse.linalg.eigs(k=1, which="LM"). The state is the fine field
only. Aux heads are produced by the network but are not inputs and are not
part of the closed-loop state.

n_jac_steps = horizon - 1. --horizon 21 gives 20 steps.
"""

from __future__ import annotations

import argparse
import os
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

from eval.eval_jacobian_manual_mse import (
    choose_sections,
    load_kept_paths,
    load_trajectories,
)
from model_ablation import AblationEmulator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
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
    p.add_argument(
        "--save-dir",
        default="",
        help="Default: <checkpoint_dir>/jacobian_high_eigs",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def step_path(save_dir: Path, step: int) -> Path:
    return save_dir / f"eigs_step{step:02d}.npz"


def fine_forward(net: torch.nn.Module, high: torch.Tensor) -> torch.Tensor:
    out = net(high)
    if isinstance(out, tuple):
        return out[0]
    return out


def gt_fine(frames: list, step: int, device, dtype, norm: float) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(frames[step], dtype=np.float32))
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        / norm
    )


def compute_scipy_eigs(ema_vit, high, ncv, maxiter, tol, seed: int):
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    n = int(h0.numel())

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = (
            torch.from_numpy(np.ascontiguousarray(v_np))
            .to(device=h0.device, dtype=h0.dtype)
            .reshape_as(h0)
        )
        _, jv = torch.autograd.functional.jvp(
            lambda h: fine_forward(ema_vit, h),
            (h0,),
            (v,),
        )
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
    }


def plot_max_eig(save_dir: Path) -> str | None:
    steps, radii = [], []
    for path in sorted(save_dir.glob("eigs_step*.npz")):
        data = np.load(path)
        steps.append(int(data["step"]) if "step" in data.files else int(path.stem[-2:]))
        radii.append(float(data["radius"]))
    if not steps:
        return None
    order = np.argsort(steps)
    steps = np.asarray(steps)[order]
    radii = np.asarray(radii)[order]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(steps, radii, marker="o", lw=1.5)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("trajectory step")
    ax.set_ylabel("largest |eigenvalue|")
    ax.set_title(save_dir.parent.name)
    out = save_dir / "max_eig_vs_step.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def load_ablation(checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'args' and 'model'")
    model = AblationEmulator(ckpt["args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model.ema_vit


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "jacobian_high_eigs"
    ))
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

    model, ema_vit = load_ablation(args.checkpoint, device)
    dtype = next(ema_vit.parameters()).dtype
    high0 = gt_fine(frames, 0, device, dtype, args.norm)
    n = int(high0.numel())
    print(
        f"section={int(chosen[0])} horizon={args.horizon} "
        f"n_jac_steps={n_jac_steps} ablation={model.ablation} "
        f"state_dim={n} device={device}",
        flush=True,
    )

    summary = [
        "method=scipy",
        "jacobian=fine",
        f"checkpoint={args.checkpoint}",
        f"ablation={model.ablation}",
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
        print(f"step {step}: computing scipy eigs ...", flush=True)
        t0 = time.perf_counter()
        try:
            result = compute_scipy_eigs(
                ema_vit, high, args.ncv, args.maxiter, args.tol, args.seed
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
    curve = plot_max_eig(save_dir)
    if curve:
        print(f"wrote {curve}", flush=True)
    print("ABLATION_EIGS_DONE", flush=True)


if __name__ == "__main__":
    main()
