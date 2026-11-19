#!/usr/bin/env python3
"""Trace and partial spectrum of the full square step Jacobian on one eval frame.

L=2 FutureINNPlus: (high, low) -> (high, low) at t=num_coarse.
L=3 FutureINN: (high, mid, low) -> (high, mid, low) at t=2.
Coarse inputs are average-pools of the next kept eval frames, the same
training stagger as eval_jacobian_high_eigs.py and eval_jacobian_high_eigs_plus.py.

The trace is a Rademacher Hutchinson estimate, not an exact diagonal sum.
Eigenvalues are ARPACK largest-magnitude values. A small residual does not
prove that the returned set is the dominant one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# eval_jacobian_trace.py clears CUDA_VISIBLE_DEVICES unless --worker is present.
if "--worker" not in sys.argv:
    sys.argv.append("--worker")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.io
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_high_eigs import (
    gt_l3_state,
    load_kept_paths,
    pack_l3_state,
    unpack_l3_state,
)
from eval_jacobian_high_eigs_plus import (
    _t_full,
    gt_plus_state,
    pack_plus_state,
    unpack_plus_state,
)
from eval_jacobian_manual_mse import load_model as load_l3
from eval_jacobian_manual_mse_plus import load_model as load_plus
from eval_jacobian_trace import compute_hutchinson_trace as trace_l3
from eval_jacobian_trace_plus import compute_hutchinson_trace as trace_plus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stem", type=int, default=91836)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--n-probes", type=int, default=32)
    p.add_argument(
        "--ncv",
        type=int,
        default=0,
        help="Arnoldi subspace size. 0 means 2k+1.",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=0,
        help="Maximum Arnoldi iterations. 0 means 10*ncv.",
    )
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--save-dir", default="")
    p.add_argument(
        "--jvp-only",
        action="store_true",
        help="Time a few full-state JVPs and exit before the trace and ARPACK.",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def peek_family(checkpoint: str) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    if isinstance(args, dict):
        if args.get("num_coarse") is not None or args.get("latent_res") is not None:
            return "plus"
        return "l3"
    if getattr(args, "num_coarse", None) is not None or getattr(args, "latent_res", None) is not None:
        return "plus"
    return "l3"


def frames_at_stem(kept: list[tuple[int, str]], stem: int, n_future: int) -> tuple[list[np.ndarray], list[int]]:
    stems = [item[0] for item in kept]
    if stem not in stems:
        raise ValueError(f"stem {stem} is not in the kept eval list")
    pos = stems.index(stem)
    if pos + n_future >= len(kept):
        raise ValueError(f"stem {stem} needs {n_future} following kept frames")
    chosen = kept[pos : pos + n_future + 1]
    frames = [np.asarray(scipy.io.loadmat(path)["Omega"]) for _, path in chosen]
    return frames, [item[0] for item in chosen]


def _ci95(std: float, n_probes: int) -> float:
    if n_probes < 2:
        return float("nan")
    return float(1.96 * std / np.sqrt(n_probes))


def time_matvecs(matvec, n: int, repeats: int = 3) -> float:
    rng = np.random.default_rng(1)
    seconds = []
    for _ in range(repeats):
        v = rng.standard_normal(n).astype(np.float32)
        v /= np.linalg.norm(v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        matvec(v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    return float(np.mean(seconds))


def eigen_residuals(matvec, vals: np.ndarray, vecs: np.ndarray | None) -> np.ndarray:
    out = np.full(vals.shape[0], np.nan, dtype=np.float64)
    if vecs is None:
        return out
    for i, lam in enumerate(vals):
        v = np.asarray(vecs[:, i], dtype=np.complex128)
        norm = float(np.linalg.norm(v))
        if norm == 0.0 or not np.isfinite(lam):
            continue
        jv = matvec(np.real(v).astype(np.float32)) + 1j * matvec(np.imag(v).astype(np.float32))
        out[i] = float(np.linalg.norm(jv - lam * v) / norm)
    return out


def spectrum(matvec, n: int, k: int, ncv: int, maxiter: int, tol: float, seed: int) -> dict:
    if k < 1 or k >= n:
        raise ValueError(f"k must be in 1..{n - 1}, got {k}")
    ncv_eff = ncv if ncv > 0 else 2 * k + 1
    ncv_eff = min(max(ncv_eff, k + 2), n - 1)
    if ncv_eff <= k:
        raise ValueError(f"ncv={ncv_eff} is not larger than k={k}")
    maxiter_eff = maxiter if maxiter > 0 else 10 * ncv_eff
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float32)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    converged = True
    vecs = None
    try:
        vals, vecs = spla.eigs(
            op,
            k=k,
            which="LM",
            ncv=ncv_eff,
            maxiter=maxiter_eff,
            tol=tol,
            v0=v0,
            return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues)
        vecs = exc.eigenvectors
        finite = np.isfinite(vals)
        vals = vals[finite]
        if vecs is not None:
            vecs = np.asarray(vecs)[:, finite]
        converged = False
        if vals.size == 0:
            raise
        print(f"partial convergence: kept {vals.size}/{k} eigenvalues", flush=True)
    vals = np.asarray(vals, dtype=np.complex128)
    residuals = eigen_residuals(matvec, vals, None if vecs is None else np.asarray(vecs))
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    residuals = residuals[order]
    return {
        "eigenvalues": vals,
        "residual": residuals,
        "converged": np.asarray(converged),
        "k_requested": np.asarray(k, dtype=np.int64),
        "k_returned": np.asarray(vals.size, dtype=np.int64),
        "ncv": np.asarray(ncv_eff, dtype=np.int64),
        "maxiter": np.asarray(maxiter_eff, dtype=np.int64),
        "tol": np.asarray(tol, dtype=np.float64),
        "which": np.asarray("LM"),
    }


def plus_matvec(ema_vit, model, high, coarses):
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    c_fixed = [c.detach().to(dtype) for c in coarses]
    t = _t_full(model, h0)
    n = int(h0.numel() + sum(c.numel() for c in c_fixed))

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device=h0.device, dtype=h0.dtype)
        vh, vc = unpack_plus_state(v, h0, c_fixed)

        def f(h, *cs):
            oh, ocs = ema_vit(h, list(cs), t)
            return pack_plus_state(oh, ocs)

        _, jv = torch.autograd.functional.jvp(f, (h0, *c_fixed), (vh, *vc))
        return jv.detach().reshape(-1).float().cpu().numpy()

    return matvec, n, h0, c_fixed


def l3_matvec(ema_vit, high, mid, low):
    dtype = next(ema_vit.parameters()).dtype
    h0 = high.detach().to(dtype)
    mid_c = mid.detach().to(dtype)
    low_c = low.detach().to(dtype)
    t = 2 * torch.ones(h0.shape[0], device=h0.device, dtype=dtype)
    n = int(h0.numel() + mid_c.numel() + low_c.numel())

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device=h0.device, dtype=h0.dtype)
        vh, vm, vl = unpack_l3_state(v, h0, mid_c, low_c)

        def f(h, m, l):
            oh, om, ol = ema_vit(h, m, l, t)
            return pack_l3_state(oh, om, ol)

        _, jv = torch.autograd.functional.jvp(f, (h0, mid_c, low_c), (vh, vm, vl))
        return jv.detach().reshape(-1).float().cpu().numpy()

    return matvec, n, h0, mid_c, low_c


def write_plots(plot_dir: Path, vals: np.ndarray, title: str) -> None:
    from rank_spectrum_plots import write_rank_plots

    write_rank_plots(plot_dir, vals, title, kind="eigenvalue")


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
    if args.n_probes < 1:
        raise ValueError(f"--n-probes must be >= 1, got {args.n_probes}")
    family = peek_family(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    if family == "plus":
        model, ema_vit = load_plus(args.checkpoint, device)
        n_future = int(model.num_coarse)
        frames, stems = frames_at_stem(kept, args.stem, n_future)
        dtype = next(ema_vit.parameters()).dtype
        high, coarses = gt_plus_state(frames, 0, model, device, dtype, args.norm)
        matvec, n, h0, c_fixed = plus_matvec(ema_vit, model, high, coarses)
        level_shapes = {
            "high": list(h0.shape),
            "coarse": [list(c.shape) for c in c_fixed],
            "latent_res": [int(r) for r in model.latent_res],
        }
        map_name = "(high, low) -> (high, low)"
    else:
        model, ema_vit = load_l3(args.checkpoint, device)
        n_future = 2
        frames, stems = frames_at_stem(kept, args.stem, n_future)
        dtype = next(ema_vit.parameters()).dtype
        high, mid, low = gt_l3_state(frames, 0, model, device, dtype, args.norm)
        matvec, n, h0, mid_c, low_c = l3_matvec(ema_vit, high, mid, low)
        level_shapes = {
            "high": list(h0.shape),
            "mid": list(mid_c.shape),
            "low": list(low_c.shape),
        }
        map_name = "(high, mid, low) -> (high, mid, low)"

    jvp_s = time_matvecs(matvec, n)
    ncv_eff = args.ncv if args.ncv > 0 else 2 * args.k + 1
    ncv_eff = min(max(ncv_eff, args.k + 2), n - 1)
    maxiter_eff = args.maxiter if args.maxiter > 0 else 10 * ncv_eff
    print(
        f"family={family} map={map_name} stems={stems} state_dim={n} "
        f"jvp_s={jvp_s:.3f} k={args.k} ncv={ncv_eff} maxiter={maxiter_eff} "
        f"estimated_arpack_s={jvp_s * maxiter_eff:.0f}",
        flush=True,
    )
    if args.jvp_only:
        return

    save_dir = Path(args.save_dir) if args.save_dir else (
        Path(args.checkpoint).resolve().parent
        / "jacobian_frame_spectrum"
        / f"stem{args.stem}_k{args.k}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    if family == "plus":
        traced = trace_plus(ema_vit, model, h0, c_fixed, args.n_probes, args.seed, 0)
    else:
        traced = trace_l3(ema_vit, h0, mid_c, low_c, args.n_probes, args.seed, 0)
    half = _ci95(float(traced["trace_std"]), args.n_probes)
    trace = float(traced["trace"])
    print(
        f"trace={trace:.8g} std={float(traced['trace_std']):.8g} "
        f"ci95=[{trace - half:.8g}, {trace + half:.8g}] "
        f"trace_over_n={float(traced['trace_over_n']):.8g}",
        flush=True,
    )

    spec = spectrum(matvec, n, args.k, args.ncv, args.maxiter, args.tol, args.seed)
    vals = spec["eigenvalues"]
    print(
        f"returned={int(spec['k_returned'])}/{args.k} converged={bool(spec['converged'])} "
        f"leading_modulus={float(np.max(np.abs(vals))):.6g} "
        f"residual_max={float(np.nanmax(spec['residual'])):.3e}",
        flush=True,
    )
    payload = {
        "family": np.asarray(family),
        "map": np.asarray(map_name),
        "checkpoint": np.asarray(args.checkpoint),
        "stems": np.asarray(stems, dtype=np.int64),
        "state_dim": np.asarray(n, dtype=np.int64),
        "probe_dots": traced["probe_dots"],
        "trace": traced["trace"],
        "trace_std": traced["trace_std"],
        "trace_ci95_half": np.asarray(half, dtype=np.float64),
        "trace_over_n": traced["trace_over_n"],
        "trace_over_n_std": traced["trace_over_n_std"],
        "n_probes": traced["n_probes"],
        "eigenvalues": vals,
        "residual": spec["residual"],
        "converged": spec["converged"],
        "k_requested": spec["k_requested"],
        "k_returned": spec["k_returned"],
        "ncv": spec["ncv"],
        "maxiter": spec["maxiter"],
        "tol": spec["tol"],
        "which": spec["which"],
        "jvp_seconds": np.asarray(jvp_s, dtype=np.float64),
        "trace_method": np.asarray("hutchinson_rademacher"),
    }
    np.savez_compressed(save_dir / "spectrum.npz", **payload)
    meta = {
        "family": family,
        "map": map_name,
        "checkpoint": args.checkpoint,
        "stems": stems,
        "level_shapes": level_shapes,
        "state_dim": n,
        "trace": trace,
        "trace_std": float(traced["trace_std"]),
        "trace_ci95": [trace - half, trace + half],
        "trace_over_n": float(traced["trace_over_n"]),
        "n_probes": args.n_probes,
        "k_requested": args.k,
        "k_returned": int(spec["k_returned"]),
        "converged": bool(spec["converged"]),
        "residual_max": float(np.nanmax(spec["residual"])),
        "leading_modulus": float(np.max(np.abs(vals))),
        "which": "LM",
        "ncv": int(spec["ncv"]),
        "maxiter": int(spec["maxiter"]),
        "tol": args.tol,
        "jvp_seconds": jvp_s,
        "note": (
            "Trace is a Hutchinson estimate with Rademacher probes. "
            "Eigenvalues are ARPACK largest-magnitude values. "
            "A small residual does not prove they are the dominant set."
        ),
    }
    (save_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_plots(save_dir / "plots", vals, f"{family} full Jacobian, stem {args.stem}")
    print(f"wrote {save_dir}", flush=True)


if __name__ == "__main__":
    main()
