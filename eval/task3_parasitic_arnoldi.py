#!/usr/bin/env python3
"""Parasitic-block Arnoldi for the trained L=2 HINE, and Task 4's DB spectrum.

S_n v = K_n v - D Phi'_{n+1}(B_n v), with Phi' the unrestricted 512^2
solver tangent at the true field u_{n+1}. A partial Ritz spectrum is not
a certificate that every eigenvalue lies in the unit disk.

--db-only skips that solver and S. It estimates eigenvalues of
DB = D(dg/dz), the operator in Section 13.5.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_argv = sys.argv[1:]
if "--worker" not in _argv and "--n-gpus" in _argv:
    _n = int(_argv[_argv.index("--n-gpus") + 1])
    if _n > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_manual_mse_plus import load_model
from l2_model_jvp import ModelStepPhysical, assert_l2
from py2d_inmem import DNS_DATA, Py2DConfig, Py2DInMemory, pool_avg, rms
from task3_operators import hine_S_action
from task35_common import (
    base_metadata,
    load_omega,
    rel_rms,
    sample_windows,
    sha256_file,
    write_json,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task3_parasitic_debug_test/main",
    )
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--delta-t", type=float, default=0.05)
    p.add_argument("--k", type=int, default=8, help="Ritz pairs")
    p.add_argument("--ncv", type=int, default=24)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--fd-eps", type=float, default=1e-3)
    p.add_argument("--fd-ref-scale", type=float, default=1.0)
    p.add_argument("--solver-dt", type=float, default=1e-4)
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--model-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--db", action="store_true", help="Also estimate the spectrum of D B")
    p.add_argument(
        "--db-only",
        action="store_true",
        help=(
            "Task 4: eigenvalues of DB = D(dg/dz) only. "
            "Skips the py2d solver and the parasitic block S."
        ),
    )
    p.add_argument("--save-vectors", action="store_true")
    p.add_argument("--product-steps", type=int, default=0)
    p.add_argument("--burn-in", type=int, default=2)
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--product-start", type=int, default=0, help="Index into the kept list")
    p.add_argument(
        "--stability-subset",
        type=int,
        default=0,
        help="Repeat this many states over ncv, epsilon, and dt_dns",
    )
    args = p.parse_args(argv)
    if args.db_only and args.db:
        p.error("--db-only already computes DB; do not combine it with --db")
    if args.db_only and not (8 <= int(args.k) <= 16):
        p.error("--db-only asks for 8 to 16 eigenvalues per state (--k)")
    if args.db_only and (args.product_steps or args.stability_subset):
        p.error("--db-only does not run the parasitic product or the stability sweep")
    return args


def _solver(args, dt: float | None = None) -> Py2DInMemory:
    return Py2DInMemory(
        Py2DConfig(
            nx=args.nx,
            Re=args.re,
            fkx=args.fkx,
            fky=args.fky,
            alpha=args.alpha,
            beta=args.beta,
            dt_dns=args.solver_dt if dt is None else dt,
        )
    )


def _load_model(args):
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, ema = load_model(args.checkpoint, device)
    assert_l2(model)
    if args.model_dtype == "float64":
        model = model.double()
        ema = model.ema_vit
    before = sha256_file(args.checkpoint)
    return model, ModelStepPhysical(ema, args.norm), before


def _eigs(matvec, n: int, k: int, ncv: int, maxiter: int, tol: float, seed: int):
    k_eff = min(k, n - 2)
    ncv_eff = min(max(ncv, k_eff + 2), n - 1)
    counts = {"matvecs": 0}

    def counted(v):
        counts["matvecs"] += 1
        return np.asarray(matvec(v), dtype=np.float64)

    op = spla.LinearOperator((n, n), matvec=counted, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    v0 /= np.linalg.norm(v0)
    converged = True
    err = ""
    vecs = None
    try:
        vals, vecs = spla.eigs(
            op, k=k_eff, which="LM", ncv=ncv_eff, maxiter=maxiter, tol=tol,
            v0=v0, return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        converged = False
        err = f"{type(exc).__name__}: {exc}"
        vals = getattr(exc, "eigenvalues", None)
        vecs = getattr(exc, "eigenvectors", None)
    if vals is None or len(np.atleast_1d(vals)) == 0:
        return {
            "values": np.zeros(0, dtype=np.complex128),
            "vectors": None,
            "residuals": np.zeros(0),
            "converged": False,
            "error": err or "no Ritz values",
            "matvecs": counts["matvecs"],
            "k": k_eff,
            "ncv": ncv_eff,
        }
    vals = np.atleast_1d(np.asarray(vals, dtype=np.complex128))
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    if vecs is not None:
        vecs = np.asarray(vecs)[:, order]
    residuals = []
    for i, lam in enumerate(vals):
        if vecs is None or not np.isfinite(lam):
            residuals.append(np.nan)
            continue
        vec = vecs[:, i]
        image = counted(np.real(vec)) + 1j * counted(np.imag(vec))
        residuals.append(
            float(np.linalg.norm(image - lam * vec) / max(np.linalg.norm(vec), 1e-30))
        )
    return {
        "values": vals,
        "vectors": vecs,
        "residuals": np.asarray(residuals, dtype=np.float64),
        "converged": converged,
        "error": err,
        "matvecs": counts["matvecs"],
        "k": k_eff,
        "ncv": ncv_eff,
    }


def _reshape(v: np.ndarray, side: int) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(side, side)


def analyze_state(args, model, step, solver, sample, factor: int, tag: str = "S"):
    frames = sample["frames"]
    u_n = frames[0]["omega"]
    u_np1 = frames[1]["omega"]
    z = pool_avg(u_np1, factor)
    dns_path = DNS_DATA / f"{frames[1]['stem']}.mat"
    if not dns_path.is_file():
        raise FileNotFoundError(f"missing 512^2 frame {dns_path}")
    u_np1_512, _t = load_omega(dns_path)
    u_n_512, _tn = load_omega(DNS_DATA / f"{frames[0]['stem']}.mat")
    side = int(z.shape[0])
    n = side * side

    def matvec(v_flat: np.ndarray) -> np.ndarray:
        v = _reshape(v_flat, side)
        out = hine_S_action(
            step, solver, u_n, z, u_np1_512, v, args.delta_t,
            args.fd_eps, args.fd_ref_scale, factor, u_n.shape[0],
        )
        return out.reshape(-1)

    t0 = time.perf_counter()
    spec = _eigs(matvec, n, args.k, args.ncv, args.maxiter, args.tol, args.seed + sample["sample"])
    phi_pred = solver.advance_physical(u_n_512, args.delta_t)
    result = {
        "tag": tag,
        "sample": sample["sample"],
        "stems": [fr["stem"] for fr in frames],
        "times": [fr["time"] for fr in frames],
        "phi_consistency_rel": rel_rms(phi_pred, u_np1_512),
        "converged": spec["converged"],
        "error": spec["error"],
        "matvecs": spec["matvecs"],
        "wall_s": time.perf_counter() - t0,
        "values": spec["values"],
        "residuals": spec["residuals"],
        "largest_modulus": float(np.max(np.abs(spec["values"]))) if len(spec["values"]) else float("nan"),
        "largest_modulus_is_estimate": True,
        "partial_spectrum": True,
        "k": spec["k"],
        "ncv": spec["ncv"],
    }
    if args.save_vectors and spec["vectors"] is not None:
        result["vectors"] = spec["vectors"]
    if args.db:
        def db_vec(v_flat):
            bv, _kv = step.jvp_latent(u_n, z, _reshape(v_flat, side))
            return pool_avg(bv, factor).reshape(-1)

        def k_minus_db(v_flat):
            bv, kv = step.jvp_latent(u_n, z, _reshape(v_flat, side))
            return (kv - pool_avg(bv, factor)).reshape(-1)

        db = _eigs(db_vec, n, args.k, args.ncv, args.maxiter, args.tol, args.seed + 1000 + sample["sample"])
        approx = _eigs(k_minus_db, n, min(args.k, 4), args.ncv, args.maxiter, args.tol, args.seed + 2000)
        result["DB_values"] = db["values"]
        result["DB_converged"] = db["converged"]
        result["DB_largest_modulus"] = float(np.max(np.abs(db["values"]))) if len(db["values"]) else float("nan")
        result["S_approx_phi_identity_values"] = approx["values"]
        result["S_approx_note"] = "K - D B, which is S only if Phi' is the identity. Not the parasitic block."
    if len(spec["values"]) and spec["vectors"] is not None:
        lead = spec["vectors"][:, 0].reshape(side, side)
        result["leading_fourier_power"] = np.abs(np.fft.fft2(np.real(lead))) ** 2
    return result


def analyze_db_state(args, step, sample, factor: int) -> dict:
    """Spectrum of DB = D(dg/dz) at (u_n, D u_{n+1}). No solver and no S."""
    frames = sample["frames"]
    u_n = frames[0]["omega"]
    z = pool_avg(frames[1]["omega"], factor)
    side = int(z.shape[0])
    n = side * side

    def db_vec(v_flat: np.ndarray) -> np.ndarray:
        fine, _latent = step.jvp_latent(u_n, z, _reshape(v_flat, side))
        return pool_avg(fine, factor).reshape(-1)

    t0 = time.perf_counter()
    spec = _eigs(
        db_vec, n, args.k, args.ncv, args.maxiter, args.tol, args.seed + sample["sample"]
    )
    values = spec["values"]
    distances = np.abs(values - 1.0) if len(values) else np.zeros(0, dtype=np.float64)
    value_ok = np.isfinite(spec["residuals"]) & (spec["residuals"] <= float(args.tol))
    return {
        "tag": "DB",
        "operator": "DB",
        "sample": sample["sample"],
        "stems": [fr["stem"] for fr in frames],
        "times": [fr["time"] for fr in frames],
        "converged": spec["converged"],
        "value_converged": value_ok,
        "error": spec["error"],
        "matvecs": spec["matvecs"],
        "wall_s": time.perf_counter() - t0,
        "values": values,
        "residuals": spec["residuals"],
        "distance_from_1": distances,
        "largest_modulus": float(np.max(np.abs(values))) if len(values) else float("nan"),
        "partial_spectrum": True,
        "k": spec["k"],
        "ncv": spec["ncv"],
    }


def _save_db_state(save_dir: Path, result: dict) -> None:
    payload = {
        "operator": np.asarray("DB"),
        "sample": np.asarray(result["sample"]),
        "stems": np.asarray(result["stems"]),
        "times": np.asarray(result["times"], dtype=np.float64),
        "values_real": np.real(result["values"]),
        "values_imag": np.imag(result["values"]),
        "residuals": result["residuals"],
        "converged": np.asarray(result["converged"]),
        "value_converged": np.asarray(result["value_converged"], dtype=bool),
        "distance_from_1": np.asarray(result["distance_from_1"], dtype=np.float64),
        "largest_modulus": np.asarray(result["largest_modulus"]),
        "matvecs": np.asarray(result["matvecs"]),
        "wall_s": np.asarray(result["wall_s"]),
        "partial_spectrum": np.asarray(True),
        "k": np.asarray(result["k"]),
        "ncv": np.asarray(result["ncv"]),
    }
    path = save_dir / f"state_{int(result['sample']):02d}.npz"
    np.savez_compressed(path, **payload)
    nearest = (
        float(np.min(result["distance_from_1"])) if len(result["distance_from_1"]) else float("nan")
    )
    print(
        f"[state {result['sample']:02d}] DB |lambda|_max={result['largest_modulus']:.6g} "
        f"min|lambda-1|={nearest:.3e} converged={result['converged']} "
        f"wall_s={result['wall_s']:.1f} -> {path.name}",
        flush=True,
    )


def plot_db_argand(save_dir: Path) -> None:
    """One Argand diagram of every saved DB Ritz value, with 0 and 1 marked."""
    files = sorted(save_dir.glob("state_*.npz"))
    if not files:
        return
    pieces_re, pieces_im, pieces_dist = [], [], []
    pieces_id, pieces_res, pieces_ok = [], [], []
    for path in files:
        data = np.load(path)
        re = np.asarray(data["values_real"], dtype=np.float64)
        im = np.asarray(data["values_imag"], dtype=np.float64)
        pieces_re.append(re)
        pieces_im.append(im)
        pieces_dist.append(np.asarray(data["distance_from_1"], dtype=np.float64))
        pieces_id.append(np.full(re.shape, int(data["sample"]), dtype=np.int64))
        pieces_res.append(np.asarray(data["residuals"], dtype=np.float64))
        pieces_ok.append(np.asarray(data["value_converged"], dtype=bool))
    re = np.concatenate(pieces_re) if pieces_re else np.zeros(0)
    im = np.concatenate(pieces_im) if pieces_im else np.zeros(0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if len(re):
        ax.scatter(re, im, s=22, zorder=2, label="DB eigenvalues")
    ax.scatter([0.0], [0.0], marker="x", s=90, c="k", zorder=3, label="0")
    ax.scatter([1.0], [0.0], marker="+", s=140, c="C3", zorder=3, label="1")
    ax.axhline(0.0, color="0.85", lw=0.6)
    ax.axvline(0.0, color="0.85", lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(r"eigenvalue estimates of $DB = D\,\partial g/\partial z$")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "db_argand.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        save_dir / "db_eigenvalues.npz",
        values_real=re,
        values_imag=im,
        distance_from_1=np.concatenate(pieces_dist) if pieces_dist else np.zeros(0),
        residuals=np.concatenate(pieces_res) if pieces_res else np.zeros(0),
        value_converged=np.concatenate(pieces_ok) if pieces_ok else np.zeros(0, dtype=bool),
        sample=np.concatenate(pieces_id) if pieces_id else np.zeros(0, dtype=np.int64),
    )


def _save_state(save_dir: Path, result: dict) -> None:
    payload = {
        "sample": np.asarray(result["sample"]),
        "stems": np.asarray(result["stems"]),
        "times": np.asarray(result["times"], dtype=np.float64),
        "values_real": np.real(result["values"]),
        "values_imag": np.imag(result["values"]),
        "residuals": result["residuals"],
        "largest_modulus": np.asarray(result["largest_modulus"]),
        "converged": np.asarray(result["converged"]),
        "matvecs": np.asarray(result["matvecs"]),
        "wall_s": np.asarray(result["wall_s"]),
        "phi_consistency_rel": np.asarray(result["phi_consistency_rel"]),
        "partial_spectrum": np.asarray(True),
    }
    if "vectors" in result:
        payload["vectors"] = result["vectors"]
    if "DB_values" in result:
        payload["DB_values_real"] = np.real(result["DB_values"])
        payload["DB_values_imag"] = np.imag(result["DB_values"])
        payload["DB_largest_modulus"] = np.asarray(result["DB_largest_modulus"])
        payload["S_approx_phi_identity_real"] = np.real(result["S_approx_phi_identity_values"])
        payload["S_approx_phi_identity_imag"] = np.imag(result["S_approx_phi_identity_values"])
    if "leading_fourier_power" in result:
        payload["leading_fourier_power"] = result["leading_fourier_power"]
    path = save_dir / f"state_{int(result['sample']):02d}.npz"
    np.savez_compressed(path, **payload)
    print(
        f"[state {result['sample']:02d}] |lambda|_max={result['largest_modulus']:.6g} "
        f"converged={result['converged']} phi_rel={result['phi_consistency_rel']:.3e} "
        f"wall_s={result['wall_s']:.1f} -> {path.name}",
        flush=True,
    )


def product_growth(args, step, solver, factor, kept_samples_unused, eval_dir):
    """Renormalized product of S_n along consecutive kept frames."""
    from eval_jacobian_manual_mse import load_kept_paths

    kept = load_kept_paths(eval_dir, args.every_nth_field)
    n_steps = args.burn_in + args.product_steps
    if args.product_start + n_steps + 1 >= len(kept):
        raise ValueError("product segment does not fit in the kept list")
        rates = []
        for start_id in range(args.n_starts):
            rng = np.random.default_rng(args.seed + 17 + start_id)
            v = None
            gains = []
        underflow = False
        for j in range(n_steps):
            stem_n, path_n = kept[args.product_start + j]
            stem_1, path_1 = kept[args.product_start + j + 1]
            u_n, _t = load_omega(path_n)
            u_np1, _t1 = load_omega(path_1)
            z = pool_avg(u_np1, factor)
            if v is None:
                v = rng.standard_normal(z.shape)
                v /= np.linalg.norm(v)
            u512, _ = load_omega(DNS_DATA / f"{int(stem_1)}.mat")
            w = hine_S_action(
                step, solver, u_n, z, u512, v, args.delta_t,
                args.fd_eps, args.fd_ref_scale, factor, u_n.shape[0],
            )
            gain = float(np.linalg.norm(w))
            if gain < 1e-30 or not np.isfinite(gain):
                underflow = True
                break
            gains.append(np.log(gain))
            v = w / gain
        used = gains[args.burn_in :]
        duration = len(used) * args.delta_t
        rate = float(np.sum(used) / duration) if duration > 0 and not underflow else float("nan")
        rates.append({"start": start_id, "rate": rate, "underflow": underflow, "log_gains": gains})
    return rates


def plot_spectra(save_dir: Path) -> None:
    files = sorted(save_dir.glob("state_*.npz"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.5", lw=1, label="unit circle")
    moduli = []
    labels = []
    for path in files:
        data = np.load(path)
        re = data["values_real"]
        im = data["values_imag"]
        ax.scatter(re, im, s=18)
        moduli.append(float(data["largest_modulus"]))
        labels.append(int(data["sample"]))
    ax.set_aspect("equal")
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title("parasitic-block eigenvalue estimates (S_n)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "ritz_argand.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(labels, moduli, marker="o")
    ax.set_xlabel("reference sample")
    ax.set_ylabel("estimated largest modulus")
    ax.set_title("leading |Ritz value| of S_n (partial spectrum)")
    fig.tight_layout()
    fig.savefig(save_dir / "largest_modulus.png", dpi=140)
    plt.close(fig)

    data0 = np.load(files[0])
    if "leading_fourier_power" in data0:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        power = np.fft.fftshift(data0["leading_fourier_power"])
        im = ax.imshow(np.log10(power + 1e-16), origin="lower")
        ax.set_title("log Fourier power of leading Ritz vector")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(save_dir / "leading_vector_fourier.png", dpi=140)
        plt.close(fig)


def run_rank(args) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_windows(
        args.eval_dir, args.every_nth_field, args.n_states, lookahead=2, seed=args.seed
    )
    for sample in samples:
        if abs(sample["dt_kept"] - args.delta_t) > 1e-6:
            raise RuntimeError(
                f"kept-frame spacing {sample['dt_kept']} != --delta-t {args.delta_t}. "
                "Refusing to treat the model interval as exact."
            )
    model, step, ckpt_hash = _load_model(args)
    factor = int(model.ds_factors[0])
    mine = [s for s in samples if int(s["sample"]) % args.n_gpus == args.gpu_rank]
    if args.db_only:
        print(
            f"[rank {args.gpu_rank}] DB-only states {[s['sample'] for s in mine]} "
            f"factor={factor} k={args.k}",
            flush=True,
        )
        for sample in mine:
            _save_db_state(save_dir, analyze_db_state(args, step, sample, factor))
        meta = base_metadata(
            args,
            {
                "task": "L2 Task 4 spectrum of DB",
                "operator": "DB = D dg/dz",
                "section": "13.5",
                "py2d_solver": "not used",
                "parasitic_block": "not computed",
                "checkpoint_sha256": ckpt_hash,
                "delta_t_model": args.delta_t,
                "pool_factor": factor,
                "norm": args.norm,
                "k": args.k,
                "ncv": args.ncv,
                "derivative": "torch.autograd.functional.jvp for dg/dz, then average-pool D",
                "weights_unchanged": sha256_file(args.checkpoint) == ckpt_hash,
                "states": [
                    {"sample": s["sample"], "stems": [fr["stem"] for fr in s["frames"]]}
                    for s in samples
                ],
            },
        )
        if args.n_gpus == 1:
            plot_db_argand(save_dir)
        if args.gpu_rank == 0:
            write_json(save_dir / "metadata.json", meta)
        return

    solver = _solver(args)
    print(
        f"[rank {args.gpu_rank}] states {[s['sample'] for s in mine]} "
        f"factor={factor} delta_t={args.delta_t} dt_dns={args.solver_dt}",
        flush=True,
    )
    for sample in mine:
        result = analyze_state(args, model, step, solver, sample, factor)
        _save_state(save_dir, result)
        if args.stability_subset and sample["sample"] < args.stability_subset:
            for ncv in (16, 24, 40):
                for eps in (1e-4, 1e-3, 1e-2):
                    for dt in (args.solver_dt, 5e-5):
                        alt = argparse.Namespace(**vars(args))
                        alt.ncv = ncv
                        alt.fd_eps = eps
                        alt.solver_dt = dt
                        alt.db = False
                        alt.save_vectors = False
                        alt_solver = _solver(alt, dt=dt)
                        alt_result = analyze_state(
                            alt, model, step, alt_solver, sample, factor, tag="stability"
                        )
                        name = (
                            f"stability_s{sample['sample']:02d}_ncv{ncv}"
                            f"_eps{eps:.0e}_dt{dt:.0e}.npz"
                        )
                        np.savez_compressed(
                            save_dir / name,
                            values_real=np.real(alt_result["values"]),
                            values_imag=np.imag(alt_result["values"]),
                            largest_modulus=np.asarray(alt_result["largest_modulus"]),
                            converged=np.asarray(alt_result["converged"]),
                            ncv=np.asarray(ncv),
                            eps=np.asarray(eps),
                            dt_dns=np.asarray(dt),
                        )
                        print(f"[stability] {name} |lambda|={alt_result['largest_modulus']:.4g}", flush=True)
    if args.product_steps and args.gpu_rank == 0:
        rates = product_growth(args, step, solver, factor, samples, args.eval_dir)
        write_json(save_dir / "product_growth.json", {"rates": rates, "delta_t": args.delta_t})
        print(f"[product] {rates}", flush=True)
    if args.gpu_rank == 0:
        plot_spectra(save_dir)
    meta = base_metadata(
        args,
        {
            "task": "L2 parasitic block S_n",
            "checkpoint_sha256": ckpt_hash,
            "delta_t_model": args.delta_t,
            "delta_t_source": "difference of eval .mat time stamps at --every-nth-field",
            "pool_factor": factor,
            "norm": args.norm,
            "fd_eps_rel": args.fd_eps,
            "fd_ref_scale": args.fd_ref_scale,
            "fd_formula": "rms(v) * (F(x+hq)-F(x-hq))/(2h), q=v/rms(v), h=eps*max(rms(x),ref_scale)",
            "solver": solver.cfg.as_dict(),
            "derivative": "torch.autograd.functional.jvp for B and K; central FD for Phi'",
            "phi_time_index": "u_{n+1}, unrestricted interval [t_{n+1}, t_{n+2}]",
            "weights_unchanged": sha256_file(args.checkpoint) == ckpt_hash,
            "states": [
                {"sample": s["sample"], "stems": [fr["stem"] for fr in s["frames"]]}
                for s in samples
            ],
        },
    )
    if args.gpu_rank == 0:
        write_json(save_dir / "metadata.json", meta)


def _orchestrate(args) -> None:
    script = str(Path(__file__).resolve())
    procs = []
    for rank in range(args.n_gpus):
        cmd = [sys.executable, "-u", script, "--worker", "--gpu-rank", str(rank)]
        for key, value in vars(args).items():
            if key in ("worker", "gpu_rank"):
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[orchestrator] rank {rank}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"worker exit codes {codes}")
    save_dir = Path(args.save_dir)
    if args.db_only:
        plot_db_argand(save_dir)
    else:
        plot_spectra(save_dir)


def main() -> None:
    args = parse_args()
    if args.n_gpus > 1 and not args.worker:
        _orchestrate(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
