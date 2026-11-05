#!/usr/bin/env python3
"""Hypothesis H, second experiment: A_QQ versus the conditional solver map Psi.

The projector is average pooling, the same restriction the L=2 checkpoint
uses. A sharp Fourier screen is a different experiment and is not used here.
The latent input of g is held fixed, so this is not the total derivative.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coarse_projectors import PoolingProjector, q_model
from eval_jacobian_manual_mse_plus import load_model
from l2_model_jvp import ModelStepPhysical, assert_l2
from py2d_inmem import (
    DNS_DATA,
    Py2DConfig,
    Py2DInMemory,
    embed_field,
    fd_tangent,
    pool_avg,
    rms,
    truncate_field,
)
from task35_common import base_metadata, load_omega, rel_rms, sample_windows, sha256_file, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task5_hine_vs_cond_debug_test",
    )
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--n-directions", type=int, default=2)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--delta-t", type=float, default=0.05)
    p.add_argument("--fd-eps", type=float, default=1e-3)
    p.add_argument("--fd-ref-scale", type=float, default=1.0)
    p.add_argument("--eps-sweep", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--stability-subset", type=int, default=0)
    p.add_argument("--floor", type=float, default=1e-8)
    p.add_argument("--solver-dt", type=float, default=1e-4)
    p.add_argument("--also-dt", type=float, default=0.0)
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--model-dtype", choices=["float32", "float64"], default="float32")
    return p.parse_args(argv)


def _direction(rng, field, factor, eps_unused):
    v = q_model(rng.standard_normal(field.shape), factor)
    scale = rms(v)
    if scale == 0.0:
        raise RuntimeError("fine direction vanished")
    return v / scale


def compare_one(args, step, solver, projector, sample, direction, eps, factor, floor):
    frames = sample["frames"]
    u_n = frames[0]["omega"]
    u_np1 = frames[1]["omega"]
    z = pool_avg(u_np1, factor)
    dns_n = DNS_DATA / f"{frames[0]['stem']}.mat"
    u512, _t = load_omega(dns_n)
    trunc_err = rel_rms(truncate_field(u512, u_n.shape[0]), u_n)
    path = solver.record_coarse_path(u512, projector, args.delta_t)
    rng = np.random.default_rng(args.seed + 50 * sample["sample"] + direction)
    v = _direction(rng, u_n, factor, eps)
    a = step.jvp_fine_fixed_latent(u_n, z, v)
    a_v = q_model(a, factor)
    direction_512 = embed_field(v, solver.cfg.nx)

    def flow(u512_in):
        return solver.advance_conditional(u512_in, path, projector, args.delta_t)

    phi = fd_tangent(flow, u512, direction_512, eps, args.fd_ref_scale)
    p_v = q_model(truncate_field(phi, u_n.shape[0]), factor)
    base_out = flow(u512)
    end_coarse = pool_avg(truncate_field(base_out, u_n.shape[0]), factor)
    path_end = np.asarray(path[-1])
    coarse_err = rel_rms(end_coarse, path_end)
    na = rms(a_v)
    np_ = rms(p_v)
    nv = rms(v)
    mismatch = rms(a_v - p_v) / (na + np_ + floor)
    cosine = float("nan")
    if na > floor and np_ > floor:
        cosine = float(np.sum(a_v * p_v) / (np.linalg.norm(a_v) * np.linalg.norm(p_v)))
    return {
        "sample": int(sample["sample"]),
        "stem": int(frames[0]["stem"]),
        "direction": int(direction),
        "eps": float(eps),
        "gain_A": na / nv,
        "gain_Psi": np_ / nv,
        "mismatch": float(mismatch),
        "cosine": cosine,
        "floor": floor,
        "truncate_vs_eval_rel": trunc_err,
        "conditional_coarse_rel": coarse_err,
        "norm_v": nv,
    }


def _plot(save_dir: Path, rows: list[dict]) -> None:
    primary = [r for r in rows if abs(r["eps"] - rows[0]["eps"]) < 1e-15 or True]
    # One marker per (sample, direction) at the run's main epsilon: the first eps.
    main_eps = rows[0]["eps"]
    use = [r for r in rows if abs(r["eps"] - main_eps) < 1e-15]
    if not use:
        use = rows
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([r["gain_A"] for r in use], [r["gain_Psi"] for r in use], s=22)
    vals = [r["gain_A"] for r in use] + [r["gain_Psi"] for r in use]
    lo, hi = min(vals), max(vals)
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1)
    ax.set_xlabel("||A_QQ v|| / ||v||")
    ax.set_ylabel("||Psi v|| / ||v||")
    ax.set_title("paired fine-scale gains")
    fig.tight_layout()
    fig.savefig(save_dir / "paired_gains.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["mismatch"] for r in use], marker="o", label="action mismatch")
    ax.plot([r["cosine"] for r in use], marker="s", label="cosine")
    ax.set_xlabel("case")
    ax.set_title("A_QQ versus Psi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "mismatch_cosine.png", dpi=140)
    plt.close(fig)


def execute(args) -> list[dict]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, ema = load_model(args.checkpoint, device)
    assert_l2(model)
    if args.model_dtype == "float64":
        model = model.double()
        ema = model.ema_vit
    step = ModelStepPhysical(ema, args.norm)
    factor = int(model.ds_factors[0])
    solver = Py2DInMemory(
        Py2DConfig(
            nx=args.nx, Re=args.re, fkx=args.fkx, fky=args.fky,
            alpha=args.alpha, beta=args.beta, dt_dns=args.solver_dt,
        )
    )
    projector = PoolingProjector(args.nx, int(model.img_size), factor)
    samples = sample_windows(args.eval_dir, args.every_nth_field, args.n_states, 1, args.seed)
    for sample in samples:
        if abs(sample["dt_kept"] - args.delta_t) > 1e-6:
            raise RuntimeError(
                f"kept-frame spacing {sample['dt_kept']} != --delta-t {args.delta_t}"
            )
    rows = []
    for sample in samples:
        for direction in range(args.n_directions):
            row = compare_one(
                args, step, solver, projector, sample, direction, args.fd_eps, factor, args.floor
            )
            rows.append(row)
            print(
                f"[H] sample={row['sample']} dir={direction} "
                f"A={row['gain_A']:.4g} Psi={row['gain_Psi']:.4g} "
                f"mismatch={row['mismatch']:.4g} cos={row['cosine']:.4g}",
                flush=True,
            )
            if sample["sample"] < args.stability_subset:
                for eps in args.eps_sweep:
                    if abs(eps - args.fd_eps) < 1e-15:
                        continue
                    extra = compare_one(
                        args, step, solver, projector, sample, direction, eps, factor, args.floor
                    )
                    rows.append(extra)
    if args.also_dt > 0 and rows:
        alt = argparse.Namespace(**vars(args))
        alt.solver_dt = args.also_dt
        alt.also_dt = 0.0
        alt.n_states = min(1, args.n_states)
        alt.n_directions = 1
        alt.stability_subset = 0
        alt.save_dir = str(save_dir / f"dt_{args.also_dt:.0e}")
        execute(alt)
    np.savez_compressed(
        save_dir / "comparisons.npz",
        **{key: np.asarray([r[key] for r in rows]) for key in rows[0]},
    )
    _plot(save_dir, [r for r in rows if abs(r["eps"] - args.fd_eps) < 1e-15])
    inv_err = projector.right_inverse_error()
    meta = base_metadata(
        args,
        {
            "task": "A_QQ versus conditional Psi",
            "pool_factor": factor,
            "projector": projector.describe(),
            "right_inverse_rms": inv_err,
            "floor": args.floor,
            "delta_t_model": args.delta_t,
            "solver": solver.cfg.as_dict(),
            "note": (
                "Psi integrates the true 512^2 field with its pooled coarse path "
                "clamped every microstep. A_QQ differentiates g at the truncated "
                "256^2 field with the latent fixed."
            ),
            "weights_sha256": sha256_file(args.checkpoint),
        },
    )
    write_json(save_dir / "metadata.json", meta)
    return rows


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
