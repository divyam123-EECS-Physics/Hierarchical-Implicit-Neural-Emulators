#!/usr/bin/env python3
"""Hypothesis H, second experiment: A_QQ versus the conditional solver map Psi.

The projector is average pooling, the same restriction the L=2 checkpoint
uses. A sharp Fourier screen is a different experiment and is not used here.
The latent input of g is held fixed, so this is not the total derivative.
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
    steps_for,
    truncate_field,
)
from task35_common import (
    base_metadata,
    load_omega,
    namespace_to_argv,
    rel_rms,
    sample_windows,
    sha256_file,
    write_json,
)


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
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--direction-source", choices=["random", "sync-final"], default="random")
    p.add_argument(
        "--sync-dir",
        default="/pscratch/sd/d/divyam/hine_runs/py2d_sync/pool_factors",
    )
    p.add_argument("--sync-factor", type=int, default=8)
    p.add_argument("--sync-T", type=float, default=2.0)
    p.add_argument("--dns-save-dt", type=float, default=0.01)
    return p.parse_args(argv)


def hine_case_keys(n_states, n_directions, fd_eps, stability_subset, eps_sweep) -> list[tuple[int, int, float]]:
    """(sample, direction, eps), primary epsilon first, then the subset sweep."""
    keys = []
    primary = float(fd_eps)
    for sample in range(n_states):
        for direction in range(n_directions):
            keys.append((sample, direction, primary))
            if sample < stability_subset:
                for eps in eps_sweep:
                    eps = float(eps)
                    if abs(eps - primary) >= 1e-15:
                        keys.append((sample, direction, eps))
    return keys


def case_filename(sample: int, direction: int, eps: float) -> str:
    return f"s{sample:02d}_d{direction}_eps{eps:.1e}.npz"


def slow_case_name(segment: int, direction: int, probe: str) -> str:
    return f"s{segment:02d}_d{direction}_{probe}.npz"


def slow_cases(n_segments: int, n_directions: int) -> list[tuple[int, int, str]]:
    keys = []
    for segment in range(n_segments):
        for direction in range(n_directions):
            keys.append((segment, direction, "sync-final"))
            keys.append((segment, direction, "random"))
    return keys


def _direction(rng, field, factor, eps_unused):
    v = q_model(rng.standard_normal(field.shape), factor)
    scale = rms(v)
    if scale == 0.0:
        raise RuntimeError("fine direction vanished")
    return v / scale


def compare_one(args, step, solver, projector, sample, direction, eps, factor, floor, v=None, probe="random"):
    frames = sample["frames"]
    u_n = frames[0]["omega"]
    u_np1 = frames[1]["omega"]
    z = pool_avg(u_np1, factor)
    dns_n = DNS_DATA / f"{frames[0]['stem']}.mat"
    u512, _t = load_omega(dns_n)
    trunc_err = rel_rms(truncate_field(u512, u_n.shape[0]), u_n)
    path = solver.record_coarse_path(u512, projector, args.delta_t)
    if v is None:
        rng = np.random.default_rng(args.seed + 50 * sample["sample"] + direction)
        v = _direction(rng, u_n, factor, eps)
    else:
        v = q_model(np.asarray(v, dtype=np.float64), factor)
        scale = rms(v)
        if scale == 0.0:
            raise RuntimeError("supplied fine direction vanished")
        v = v / scale
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
        "probe": probe,
    }


def _plot(save_dir: Path, rows: list[dict]) -> None:
    primary = [r for r in rows if abs(r["eps"] - rows[0]["eps"]) < 1e-15 or True]
    # One marker per (sample, direction) at the run's main epsilon: the first eps.
    main_eps = rows[0]["eps"]
    use = [r for r in rows if abs(r["eps"] - main_eps) < 1e-15]
    if not use:
        use = rows
    fig, ax = plt.subplots(figsize=(5, 5))
    probes = sorted({str(r.get("probe", "random")) for r in use})
    if probes == ["random"]:
        ax.scatter([r["gain_A"] for r in use], [r["gain_Psi"] for r in use], s=22)
    else:
        for probe in probes:
            subset = [r for r in use if str(r.get("probe", "random")) == probe]
            ax.scatter(
                [r["gain_A"] for r in subset],
                [r["gain_Psi"] for r in subset],
                s=22,
                label=probe,
            )
        ax.legend()
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


def _save_case(path: Path, row: dict) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in row.items()})


def _expected_case_names(args) -> list[str]:
    if args.direction_source == "sync-final":
        return [slow_case_name(*key) for key in slow_cases(args.n_states, args.n_directions)]
    keys = hine_case_keys(
        args.n_states, args.n_directions, args.fd_eps, args.stability_subset, args.eps_sweep
    )
    return [case_filename(*key) for key in keys]


def _load_cases(args, save_dir: Path) -> list[dict]:
    names = _expected_case_names(args)
    rows = []
    missing = []
    for name in names:
        path = save_dir / name
        if not path.is_file():
            missing.append(path.name)
            continue
        data = np.load(path)
        row = {key: data[key] for key in data.files}
        parsed = {}
        for key, value in row.items():
            array = np.asarray(value)
            parsed[key] = array.reshape(-1)[0].item() if array.shape == () or array.size == 1 else array
        rows.append(parsed)
    if missing:
        raise RuntimeError(f"{len(missing)} HINE cases missing under {save_dir}: {missing[:8]}")
    return rows


def _finalize(args, save_dir: Path, note: dict) -> list[dict]:
    rows = _load_cases(args, save_dir)
    np.savez_compressed(
        save_dir / "comparisons.npz",
        **{key: np.asarray([r[key] for r in rows]) for key in rows[0]},
    )
    primary = [r for r in rows if abs(float(r["eps"]) - args.fd_eps) < 1e-15]
    _plot(save_dir, primary)
    meta = base_metadata(
        args,
        {
            "task": "A_QQ versus conditional Psi",
            "floor": args.floor,
            "delta_t_model": args.delta_t,
            "n_gpus": args.n_gpus,
            "split": "case index modulo n_gpus",
            "note": (
                "Psi integrates the true 512^2 field with its pooled coarse path "
                "clamped every microstep. A_QQ differentiates g at the truncated "
                "256^2 field with the latent fixed."
            ),
            "weights_sha256": sha256_file(args.checkpoint),
            **note,
        },
    )
    write_json(save_dir / "metadata.json", meta)
    return rows


def _prepare(args):
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
    note = {
        "pool_factor": factor,
        "projector": projector.describe(),
        "right_inverse_rms": projector.right_inverse_error(),
        "solver": solver.cfg.as_dict(),
    }
    return save_dir, step, solver, projector, factor, samples, note


def _end_state(stem: int, time: float, total: float, delta_t: float, dns_dt: float, nx_model: int):
    stem_end = int(stem) + steps_for(dns_dt, total)
    stem_next = stem_end + steps_for(dns_dt, delta_t)
    u512, t_end = load_omega(DNS_DATA / f"{stem_end}.mat")
    u_next, t_next = load_omega(DNS_DATA / f"{stem_next}.mat")
    if abs(t_end - (float(time) + total)) > 1e-6:
        raise RuntimeError(
            f"DNS stem {stem_end} is at t={t_end}, expected {float(time) + total}"
        )
    if abs((t_next - t_end) - delta_t) > 1e-6:
        raise RuntimeError(f"frame after the sync end is spaced {t_next - t_end}, not {delta_t}")
    return (
        truncate_field(u512, nx_model),
        truncate_field(u_next, nx_model),
        stem_end,
    )


def _execute_sync_final(args, *, summarize: bool):
    save_dir, step, solver, projector, factor, _samples, note = _prepare(args)
    if int(factor) != int(args.sync_factor):
        raise RuntimeError(
            f"checkpoint pool factor {factor} != --sync-factor {args.sync_factor}"
        )
    note["direction_source"] = "sync-final"
    note["sync_dir"] = args.sync_dir
    note["linearization"] = "true field at the end of the synchronization segment"
    keys = slow_cases(args.n_states, args.n_directions)
    owned = [
        key
        for index, key in enumerate(keys)
        if args.n_gpus <= 1 or index % args.n_gpus == args.gpu_rank
    ]
    print(f"[H] rank {args.gpu_rank}/{args.n_gpus} owns {len(owned)} slow-direction cases", flush=True)
    if args.gpu_rank == 0 or args.n_gpus <= 1:
        write_json(save_dir / "projector.json", note)
    nx_model = int(projector.nx_model)
    for segment, direction, probe in owned:
        path = save_dir / slow_case_name(segment, direction, probe)
        if args.skip_existing and path.is_file():
            print(f"[H] skip {path.name}", flush=True)
            continue
        sync_path = Path(args.sync_dir) / f"pool{factor}_seg{segment}_dir{direction}.npz"
        if not sync_path.is_file():
            raise RuntimeError(f"missing synchronization field {sync_path}")
        data = np.load(sync_path)
        if "final_fine_256" not in data.files:
            raise RuntimeError(f"{sync_path} has no final_fine_256")
        stem = int(np.asarray(data["stem"]).reshape(-1)[0])
        time = float(np.asarray(data["time"]).reshape(-1)[0])
        u_n, u_np1, stem_end = _end_state(
            stem, time, args.sync_T, args.delta_t, args.dns_save_dt, nx_model
        )
        sample = {
            "sample": segment,
            "frames": [
                {"omega": u_n, "stem": stem_end},
                {"omega": u_np1, "stem": stem_end},
            ],
        }
        if probe == "sync-final":
            direction_field = np.asarray(data["final_fine_256"], dtype=np.float64)
        else:
            rng = np.random.default_rng(args.seed + 10007 * segment + direction)
            direction_field = _direction(rng, u_n, factor, args.fd_eps)
        row = compare_one(
            args, step, solver, projector, sample, direction, args.fd_eps,
            factor, args.floor, v=direction_field, probe=probe,
        )
        print(
            f"[H] segment={segment} dir={direction} probe={probe} "
            f"A={row['gain_A']:.4g} Psi={row['gain_Psi']:.4g} "
            f"mismatch={row['mismatch']:.4g} cos={row['cosine']:.4g}",
            flush=True,
        )
        _save_case(path, row)
    if not summarize:
        return None
    return _finalize(args, save_dir, note)


def execute(args, *, summarize: bool) -> list[dict] | None:
    if args.direction_source == "sync-final":
        return _execute_sync_final(args, summarize=summarize)
    save_dir, step, solver, projector, factor, samples, note = _prepare(args)
    by_sample = {int(sample["sample"]): sample for sample in samples}
    keys = hine_case_keys(
        args.n_states, args.n_directions, args.fd_eps, args.stability_subset, args.eps_sweep
    )
    owned = [
        key
        for index, key in enumerate(keys)
        if args.n_gpus <= 1 or index % args.n_gpus == args.gpu_rank
    ]
    print(f"[H] rank {args.gpu_rank}/{args.n_gpus} owns {len(owned)} cases", flush=True)
    if args.gpu_rank == 0 or args.n_gpus <= 1:
        write_json(save_dir / "projector.json", note)
    for sample_index, direction, eps in owned:
        path = save_dir / case_filename(sample_index, direction, eps)
        if args.skip_existing and path.is_file():
            print(f"[H] skip {path.name}", flush=True)
            continue
        row = compare_one(
            args, step, solver, projector, by_sample[sample_index], direction, eps, factor, args.floor
        )
        print(
            f"[H] sample={row['sample']} dir={direction} eps={eps:.1e} "
            f"A={row['gain_A']:.4g} Psi={row['gain_Psi']:.4g} "
            f"mismatch={row['mismatch']:.4g} cos={row['cosine']:.4g}",
            flush=True,
        )
        _save_case(path, row)
    if not summarize:
        return None
    return _finalize(args, save_dir, note)


def _run_distributed(args) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    procs = []
    for rank in range(args.n_gpus):
        child = argparse.Namespace(**vars(args))
        child.worker = True
        child.gpu_rank = rank
        child.also_dt = 0.0
        cmd = [sys.executable, "-u", script, *namespace_to_argv(child)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[H] launch rank {rank}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"HINE workers exited {codes}")
    note_path = save_dir / "projector.json"
    note = {}
    if note_path.is_file():
        import json
        note = json.loads(note_path.read_text())
    _finalize(args, save_dir, note)


def _also(args) -> None:
    if args.also_dt <= 0 or args.worker:
        return
    alt = argparse.Namespace(**vars(args))
    alt.solver_dt = args.also_dt
    alt.also_dt = 0.0
    alt.n_states = min(1, args.n_states)
    alt.n_directions = 1
    alt.stability_subset = 0
    alt.save_dir = str(Path(args.save_dir) / f"dt_{args.also_dt:.0e}")
    if alt.n_gpus > 1:
        _run_distributed(alt)
    else:
        execute(alt, summarize=True)


def main() -> None:
    args = parse_args()
    if args.n_gpus < 1:
        raise SystemExit("--n-gpus must be >= 1")
    if args.n_gpus > 1 and not args.worker:
        _run_distributed(args)
        _also(args)
        return
    execute(args, summarize=args.n_gpus <= 1)
    if args.n_gpus <= 1:
        _also(args)


if __name__ == "__main__":
    main()
