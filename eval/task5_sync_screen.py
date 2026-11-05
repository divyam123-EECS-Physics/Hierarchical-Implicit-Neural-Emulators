#!/usr/bin/env python3
"""Hypothesis H, first experiment: fine-scale decay under coarse clamping.

No checkpoint. Reference and follower are 512^2 py2d runs. kc=0 is the
unclamped control. The reported lambda_c is a finite-time product of
block gains after burn-in, not a claim that every direction contracts.
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

from coarse_projectors import PoolingProjector, SharpProjector
from py2d_inmem import DNS_DATA, Py2DConfig, Py2DInMemory, rms, steps_for
from task35_common import base_metadata, load_omega, sample_windows, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--n-segments", type=int, default=1)
    p.add_argument("--n-directions", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kc", type=int, nargs="+", default=[0, 8, 16, 32])
    p.add_argument("--cutoff-shape", choices=["square", "radial"], default="square")
    p.add_argument("--T", type=float, default=0.5, help="Physical time of one segment")
    p.add_argument("--tau", type=float, default=0.05, help="Renormalization interval")
    p.add_argument("--burn-in-blocks", type=int, default=2)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--delta-t-model", type=float, default=0.05)
    p.add_argument("--solver-dt", type=float, default=1e-4)
    p.add_argument("--also-dt", type=float, default=0.0, help="Repeat kc subset at this dt")
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/py2d_sync/debug_test",
    )
    return p.parse_args(argv)


def _make_solver(args, dt: float) -> Py2DInMemory:
    return Py2DInMemory(
        Py2DConfig(
            nx=args.nx, Re=args.re, fkx=args.fkx, fky=args.fky,
            alpha=args.alpha, beta=args.beta, dt_dns=dt,
        )
    )


def _projector(solver, kc: int, shape: str):
    if kc == 0:
        return None
    return SharpProjector(solver, kc, shape)


def _initial_follower(solver, omega, eps, projector, rng):
    pert = rng.standard_normal(omega.shape).astype(np.float64)
    hat = solver.jnp.asarray(np.fft.rfft2(pert))
    if projector is None:
        field = pert
        coarse_rms = 0.0
    else:
        field = np.asarray(solver.to_hat_phys(projector.Q_hat(hat)), dtype=np.float64)
        coarse_rms = float(rms(np.asarray(projector.coarse_hat(solver.jnp.fft.rfft2(field)))))
    scale = rms(field)
    if scale == 0.0:
        raise RuntimeError("projected perturbation vanished")
    field *= eps / scale
    return omega + field, coarse_rms


def run_segment(solver, omega, eps, tau, total, burn_in, projector, rng) -> dict:
    n_block = steps_for(solver.cfg.dt_dns, tau)
    n_steps = steps_for(solver.cfg.dt_dns, total)
    if n_steps % n_block:
        raise ValueError("--T must be an integer number of --tau intervals")
    n_blocks = n_steps // n_block
    follower, coarse_rms = _initial_follower(solver, omega, eps, projector, rng)
    ref = solver.startup(omega)
    fol = solver.startup(follower)
    gains = []
    first_series = None
    underflow = False
    for block in range(n_blocks):
        ref, fol, series = solver.clamped_block(ref, fol, n_block, projector)
        if first_series is None:
            first_series = series
        end = float(series[-1])
        if not np.isfinite(end) or end < 1e-30:
            underflow = True
            gains.append(float("nan"))
            break
        gains.append(float(np.log(end / eps)))
        fol, amp = solver.renormalize_with(ref, fol, eps, projector)
        if fol is None:
            underflow = True
            break
    used = [g for g in gains[burn_in:] if np.isfinite(g)]
    duration = len(used) * tau
    rate = float(np.sum(used) / duration) if duration > 0 else float("nan")
    return {
        "lambda_c": rate,
        "log_gains": gains,
        "used_blocks": len(used),
        "underflow": underflow,
        "coarse_rms_of_perturbation": coarse_rms,
        "first_block_E": first_series,
        "factor": float(np.exp(rate * tau)) if np.isfinite(rate) else float("nan"),
    }


def _summarize(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["kc"]), []).append(row["lambda_c"])
    summary = []
    for kc in sorted(grouped):
        vals = np.asarray(grouped[kc], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean = float(np.mean(finite)) if len(finite) else float("nan")
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        reliable_negative = bool(len(finite) and mean < 0.0 and mean + std < 0.0)
        summary.append(
            {
                "kc": kc,
                "n": int(len(finite)),
                "lambda_mean": mean,
                "lambda_std": std,
                "reliable_negative": reliable_negative,
                "inconclusive": int(len(finite)) < int(len(vals)) or len(finite) == 0,
                "step_factor": float(np.exp(mean * rows[0]["delta_t_model"])) if np.isfinite(mean) else float("nan"),
            }
        )
    negative = [s for s in summary if s["kc"] != 0 and s["reliable_negative"]]
    bracket = min(negative, key=lambda s: s["kc"])["kc"] if negative else None
    return {"by_kc": summary, "smallest_reliable_negative_kc": bracket}


def _plot(save_dir: Path, rows: list[dict], summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        series = row.get("first_block_E")
        if series is None:
            continue
        series = np.asarray(series)
        if series[0] == 0:
            continue
        ax.semilogy(np.arange(len(series)) * row["dt_dns"], series / series[0], lw=1)
    ax.set_xlabel("time within first block")
    ax.set_ylabel("E(t) / E(0)")
    ax.set_title("fine-error amplitude, first renormalization block")
    fig.tight_layout()
    fig.savefig(save_dir / "E_over_E0_semilog.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys, yerr = [], [], []
    for item in summary["by_kc"]:
        xs.append(item["kc"])
        ys.append(item["lambda_mean"])
        yerr.append(item["lambda_std"])
    ax.errorbar(xs, ys, yerr=yerr, marker="o", ls="none")
    ax.axhline(0.0, color="0.4", lw=1)
    ax.set_xlabel("cutoff kc (0 = unclamped)")
    ax.set_ylabel("lambda_c")
    ax.set_title("conditional growth rate")
    fig.tight_layout()
    fig.savefig(save_dir / "lambda_c_vs_kc.png", dpi=140)
    plt.close(fig)


def execute(args, dt: float, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    solver = _make_solver(args, dt)
    windows = sample_windows(args.eval_dir, args.every_nth_field, args.n_segments, 0, args.seed)
    rows = []
    for sample in windows:
        stem = sample["frames"][0]["stem"]
        omega, time = load_omega(DNS_DATA / f"{stem}.mat")
        for kc in args.kc:
            projector = _projector(solver, kc, args.cutoff_shape)
            for direction in range(args.n_directions):
                rng = np.random.default_rng(args.seed + 1000 * sample["sample"] + 10 * kc + direction)
                out = run_segment(
                    solver, omega, args.eps, args.tau, args.T,
                    args.burn_in_blocks, projector, rng,
                )
                row = {
                    "kc": int(kc),
                    "segment": int(sample["sample"]),
                    "stem": int(stem),
                    "time": time,
                    "direction": direction,
                    "dt_dns": dt,
                    "delta_t_model": args.delta_t_model,
                    "lambda_c": out["lambda_c"],
                    "step_factor": float(np.exp(out["lambda_c"] * args.delta_t_model))
                    if np.isfinite(out["lambda_c"]) else float("nan"),
                    "log_gains": out["log_gains"],
                    "underflow": out["underflow"],
                    "coarse_rms_of_perturbation": out["coarse_rms_of_perturbation"],
                    "first_block_E": out["first_block_E"],
                }
                rows.append(row)
                print(
                    f"[sync] kc={kc} stem={stem} dir={direction} "
                    f"lambda={out['lambda_c']:.4g} coarse_pert={out['coarse_rms_of_perturbation']:.2e}",
                    flush=True,
                )
                np.savez_compressed(
                    save_dir / f"kc{kc}_seg{sample['sample']}_dir{direction}.npz",
                    kc=np.asarray(kc),
                    stem=np.asarray(stem),
                    lambda_c=np.asarray(out["lambda_c"]),
                    log_gains=np.asarray(out["log_gains"], dtype=np.float64),
                    first_block_E=np.asarray(out["first_block_E"]),
                    dt_dns=np.asarray(dt),
                    coarse_rms=np.asarray(out["coarse_rms_of_perturbation"]),
                )
    summary = _summarize(rows)
    write_json(
        save_dir / "summary.json",
        {
            "summary": summary,
            "cutoff_shape": args.cutoff_shape,
            "inequality": "kc=0 is unclamped; otherwise |k_i| < kc (strict) in integer cycles per domain",
            "renormalization": "both AB2 levels scaled so the current fine difference has RMS eps",
            "rows": [
                {k: v for k, v in row.items() if k != "first_block_E"}
                for row in rows
            ],
        },
    )
    _plot(save_dir, rows, summary)
    return summary


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    summary = execute(args, args.solver_dt, save_dir)
    meta = base_metadata(
        args,
        {
            "task": "Hypothesis H synchronization screen",
            "solver": Py2DConfig(
                nx=args.nx, Re=args.re, fkx=args.fkx, fky=args.fky,
                alpha=args.alpha, beta=args.beta, dt_dns=args.solver_dt,
            ).as_dict(),
            "T": args.T,
            "tau": args.tau,
            "eps": args.eps,
            "burn_in_blocks": args.burn_in_blocks,
            "cutoff_shape": args.cutoff_shape,
            "summary": summary,
        },
    )
    write_json(save_dir / "metadata.json", meta)
    if args.also_dt > 0:
        execute(args, args.also_dt, save_dir / f"dt_{args.also_dt:.0e}")
    print(json_summary(summary), flush=True)


def json_summary(summary: dict) -> str:
    import json
    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    main()
