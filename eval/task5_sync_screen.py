#!/usr/bin/env python3
"""Hypothesis H, first experiment: fine-scale decay under coarse clamping.

No checkpoint. Reference and follower are 512^2 py2d runs. kc=0 is the
unclamped control. The reported lambda_c is a finite-time product of
block gains after burn-in, not a claim that every direction contracts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_argv = sys.argv[1:]
if "--worker" not in _argv and "--n-gpus" in _argv:
    _n = int(_argv[_argv.index("--n-gpus") + 1])
    if _n > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coarse_projectors import PoolingProjector, SharpProjector, q_model
from py2d_inmem import DNS_DATA, Py2DConfig, Py2DInMemory, rms, steps_for, truncate_field
from task35_common import base_metadata, load_omega, namespace_to_argv, sample_windows, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--n-segments", type=int, default=1)
    p.add_argument("--n-directions", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kc", type=int, nargs="+", default=[0, 8, 16, 32])
    p.add_argument("--cutoff-shape", choices=["square", "radial", "pooling"], default="square")
    p.add_argument("--pool-factors", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--nx-model", type=int, default=256)
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
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--worker", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep an npz that is already in --save-dir",
    )
    return p.parse_args(argv)


def sync_task_keys(n_segments: int, kcs, n_directions: int) -> list[tuple[int, int, int]]:
    """(segment, kc, direction) in the order one process would run them."""
    keys = []
    for segment in range(n_segments):
        for kc in kcs:
            for direction in range(n_directions):
                keys.append((int(segment), int(kc), int(direction)))
    return keys


def segment_filename(level: int, segment: int, direction: int, kind: str = "kc") -> str:
    return f"{kind}{level}_seg{segment}_dir{direction}.npz"


def screen_levels(args) -> tuple[list[int], str]:
    """Levels being swept, and the filename prefix (kc or pool)."""
    if args.cutoff_shape == "pooling":
        if not args.pool_factors:
            raise ValueError("--pool-factors is required when --cutoff-shape pooling")
        return [int(factor) for factor in args.pool_factors], "pool"
    return [int(kc) for kc in args.kc], "kc"


def dt_dirname(dt: float) -> str:
    return f"dt_{dt:.0e}"


def _make_solver(args, dt: float) -> Py2DInMemory:
    return Py2DInMemory(
        Py2DConfig(
            nx=args.nx, Re=args.re, fkx=args.fkx, fky=args.fky,
            alpha=args.alpha, beta=args.beta, dt_dns=dt,
        )
    )


def _projector(solver, level: int, shape: str, nx_model: int):
    if shape == "pooling":
        return PoolingProjector(solver.cfg.nx, nx_model, int(level))
    if level == 0:
        return None
    return SharpProjector(solver, level, shape)


def _final_fine_256(solver, ref, fol, projector, factor: int, nx_model: int) -> np.ndarray:
    """Unit-RMS fine difference on the model grid at the end of a segment."""
    diff = fol[1] - ref[1]
    fine_512 = np.array(projector.Q_phys_hat(diff), dtype=np.float64)
    fine = q_model(truncate_field(fine_512, nx_model), factor)
    scale = rms(fine)
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros((nx_model, nx_model), dtype=np.float64)
    return fine / scale


def _initial_follower(solver, omega, eps, projector, rng):
    pert = rng.standard_normal(omega.shape).astype(np.float64)
    hat = solver.jnp.asarray(np.fft.rfft2(pert))
    if projector is None:
        field = pert
        coarse_rms = 0.0
    else:
        field = np.array(solver.to_hat_phys(projector.Q_hat(hat)), dtype=np.float64, copy=True)
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
        "ref": ref,
        "fol": fol,
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


def _plot(save_dir: Path, rows: list[dict], summary: dict, kind: str) -> None:
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
    if kind == "pool":
        ax.set_xlabel("pooling factor r (nominal kc = 128/r)")
        plot_name = "lambda_c_vs_factor.png"
    else:
        ax.set_xlabel("cutoff kc (0 = unclamped)")
        plot_name = "lambda_c_vs_kc.png"
    ax.set_ylabel("lambda_c")
    ax.set_title("conditional growth rate")
    fig.tight_layout()
    fig.savefig(save_dir / plot_name, dpi=140)
    plt.close(fig)


def _save_segment(path: Path, row: dict, out: dict, final_fine: np.ndarray | None) -> None:
    payload = dict(
        kc=np.asarray(row["kc"]),
        segment=np.asarray(row["segment"]),
        stem=np.asarray(row["stem"]),
        direction=np.asarray(row["direction"]),
        time=np.asarray(row["time"]),
        lambda_c=np.asarray(out["lambda_c"]),
        log_gains=np.asarray(out["log_gains"], dtype=np.float64),
        first_block_E=np.asarray(out["first_block_E"]),
        dt_dns=np.asarray(row["dt_dns"]),
        coarse_rms=np.asarray(out["coarse_rms_of_perturbation"]),
        underflow=np.asarray(out["underflow"]),
        delta_t_model=np.asarray(row["delta_t_model"]),
    )
    if final_fine is not None:
        payload["final_fine_256"] = np.asarray(final_fine, dtype=np.float64)
    np.savez_compressed(path, **payload)


def _row_from_npz(path: Path, segment: int, direction: int, time: float, delta_t_model: float) -> dict:
    data = np.load(path)
    lam = float(np.asarray(data["lambda_c"]).reshape(-1)[0])
    dt_dns = float(np.asarray(data["dt_dns"]).reshape(-1)[0])
    underflow = (
        bool(np.asarray(data["underflow"]).reshape(-1)[0])
        if "underflow" in data.files
        else not np.isfinite(lam)
    )
    return {
        "kc": int(np.asarray(data["kc"]).reshape(-1)[0]),
        "segment": segment,
        "stem": int(np.asarray(data["stem"]).reshape(-1)[0]),
        "time": time,
        "direction": direction,
        "dt_dns": dt_dns,
        "delta_t_model": delta_t_model,
        "lambda_c": lam,
        "step_factor": float(np.exp(lam * delta_t_model)) if np.isfinite(lam) else float("nan"),
        "log_gains": np.asarray(data["log_gains"], dtype=np.float64),
        "underflow": underflow,
        "coarse_rms_of_perturbation": float(np.asarray(data["coarse_rms"]).reshape(-1)[0]),
        "first_block_E": np.asarray(data["first_block_E"], dtype=np.float64)
        if "first_block_E" in data.files
        else None,
    }


def _load_rows(args, dt: float, save_dir: Path) -> list[dict]:
    windows = sample_windows(args.eval_dir, args.every_nth_field, args.n_segments, 0, args.seed)
    by_segment = {int(sample["sample"]): sample for sample in windows}
    rows = []
    missing = []
    levels, kind = screen_levels(args)
    for segment, level, direction in sync_task_keys(args.n_segments, levels, args.n_directions):
        path = save_dir / segment_filename(level, segment, direction, kind)
        if not path.is_file():
            missing.append(path.name)
            continue
        sample = by_segment[segment]
        stem = int(sample["frames"][0]["stem"])
        _omega, time = load_omega(DNS_DATA / f"{stem}.mat")
        rows.append(_row_from_npz(path, segment, direction, time, args.delta_t_model))
    if missing:
        raise RuntimeError(f"{len(missing)} sync segments missing under {save_dir}: {missing[:8]}")
    bad = [row["kc"] for row in rows if abs(row["dt_dns"] - dt) > 1e-15]
    if bad:
        raise RuntimeError(f"saved dt_dns does not match requested {dt} for kc {bad[:8]}")
    return rows


def _finalize(args, dt: float, save_dir: Path) -> dict:
    rows = _load_rows(args, dt, save_dir)
    summary = _summarize(rows)
    _levels, kind = screen_levels(args)
    write_json(
        save_dir / "summary.json",
        {
            "summary": summary,
            "cutoff_shape": args.cutoff_shape,
            "inequality": (
                "pooling shares avg_pool(truncate(solver->model), r); nominal kc=128/r is a label only"
                if kind == "pool"
                else "kc=0 is unclamped; otherwise |k_i| < kc (strict) in integer cycles per domain"
            ),
            "renormalization": "both AB2 levels scaled so the current fine difference has RMS eps",
            "n_gpus": args.n_gpus,
            "rows": [
                {k: v for k, v in row.items() if k != "first_block_E"}
                for row in rows
            ],
        },
    )
    _plot(save_dir, rows, summary, kind)
    _write_meta(args, save_dir, dt, summary)
    print(json_summary(summary), flush=True)
    return summary


def _write_meta(args, save_dir: Path, dt: float, summary: dict) -> None:
    write_json(
        save_dir / "metadata.json",
        base_metadata(
            args,
            {
                "task": "Hypothesis H synchronization screen",
                "solver": Py2DConfig(
                    nx=args.nx, Re=args.re, fkx=args.fkx, fky=args.fky,
                    alpha=args.alpha, beta=args.beta, dt_dns=dt,
                ).as_dict(),
                "T": args.T,
                "tau": args.tau,
                "eps": args.eps,
                "burn_in_blocks": args.burn_in_blocks,
                "cutoff_shape": args.cutoff_shape,
                "n_gpus": args.n_gpus,
                "split": "segment-kc-direction index modulo n_gpus",
                "summary": summary,
            },
        ),
    )


def execute(args, dt: float, save_dir: Path, *, summarize: bool) -> dict | None:
    save_dir.mkdir(parents=True, exist_ok=True)
    solver = _make_solver(args, dt)
    windows = sample_windows(args.eval_dir, args.every_nth_field, args.n_segments, 0, args.seed)
    by_segment = {int(sample["sample"]): sample for sample in windows}
    levels, kind = screen_levels(args)
    owned = [
        key
        for index, key in enumerate(sync_task_keys(args.n_segments, levels, args.n_directions))
        if args.n_gpus <= 1 or index % args.n_gpus == args.gpu_rank
    ]
    print(
        f"[sync] rank {args.gpu_rank}/{args.n_gpus} dt={dt} owns {len(owned)} segments",
        flush=True,
    )
    loaded = {}
    for segment, level, direction in owned:
        path = save_dir / segment_filename(level, segment, direction, kind)
        if args.skip_existing and path.is_file():
            print(f"[sync] skip {path.name}", flush=True)
            continue
        sample = by_segment[segment]
        stem = int(sample["frames"][0]["stem"])
        if stem not in loaded:
            loaded[stem] = load_omega(DNS_DATA / f"{stem}.mat")
        omega, time = loaded[stem]
        projector = _projector(solver, level, args.cutoff_shape, args.nx_model)
        rng = np.random.default_rng(args.seed + 1000 * segment + 10 * level + direction)
        out = run_segment(
            solver, omega, args.eps, args.tau, args.T,
            args.burn_in_blocks, projector, rng,
        )
        row = {
            "kc": int(level),
            "segment": segment,
            "stem": stem,
            "time": time,
            "direction": direction,
            "dt_dns": dt,
            "delta_t_model": args.delta_t_model,
        }
        final_fine = None
        if kind == "pool" and projector is not None:
            final_fine = _final_fine_256(
                solver, out["ref"], out["fol"], projector, int(level), args.nx_model
            )
        print(
            f"[sync] {kind}={level} stem={stem} dir={direction} "
            f"lambda={out['lambda_c']:.4g} coarse_pert={out['coarse_rms_of_perturbation']:.2e}",
            flush=True,
        )
        _save_segment(path, row, out, final_fine)
    if not summarize:
        return None
    return _finalize(args, dt, save_dir)


def _run_distributed(args, dt: float, save_dir: Path) -> dict:
    save_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    script = str(Path(__file__).resolve())
    for rank in range(args.n_gpus):
        child = argparse.Namespace(**vars(args))
        child.worker = True
        child.gpu_rank = rank
        child.solver_dt = dt
        child.also_dt = 0.0
        child.save_dir = str(save_dir)
        cmd = [sys.executable, "-u", script, *namespace_to_argv(child)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[sync] launch rank {rank} dt={dt}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"sync workers exited {codes}")
    return _finalize(args, dt, save_dir)


def main() -> None:
    args = parse_args()
    if args.n_gpus < 1:
        raise SystemExit("--n-gpus must be >= 1")
    save_dir = Path(args.save_dir)
    if args.n_gpus > 1 and not args.worker:
        _run_distributed(args, args.solver_dt, save_dir)
        if args.also_dt > 0:
            _run_distributed(args, args.also_dt, save_dir / dt_dirname(args.also_dt))
        return
    summary = execute(args, args.solver_dt, save_dir, summarize=args.n_gpus <= 1)
    if args.n_gpus <= 1 and args.also_dt > 0:
        execute(args, args.also_dt, save_dir / dt_dirname(args.also_dt), summarize=True)


def json_summary(summary: dict) -> str:
    import json
    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    main()
