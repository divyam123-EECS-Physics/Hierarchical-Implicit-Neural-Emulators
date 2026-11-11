#!/usr/bin/env python3
"""Scale-resolved trace of the fixed-latent fine Jacobian A = dg/du.

Probes are Rademacher fields masked to one radial wavenumber shell and
are not normalized. The mean of v·Av estimates tr(P_band A P_band).
Coarse shells are included: Section 14.5 asks whether damping is only
at fine scales. This is not Q A Q and not the packed-state Jacobian.
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

from coarse_projectors import apply_band, band_real_dof, radial_band_masks
from eval_jacobian_manual_mse_plus import load_model
from l2_model_jvp import ModelStepPhysical, assert_l2
from py2d_inmem import pool_avg
from task35_common import base_metadata, namespace_to_argv, sample_windows, sha256_file, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task5_scale_trace/n4",
    )
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--delta-t", type=float, default=0.05)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def band_tasks(n_states: int, n_bands: int) -> list[tuple[int, int]]:
    return [(state, band) for state in range(n_states) for band in range(n_bands)]


def _estimate(step, u, z, mask, n_probes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dots = np.empty(n_probes, dtype=np.float64)
    for probe in range(n_probes):
        signs = rng.integers(0, 2, size=u.shape) * 2 - 1
        direction = apply_band(signs.astype(np.float64), mask)
        image = step.jvp_fine_fixed_latent(u, z, direction)
        dots[probe] = float(np.sum(direction * image))
    return dots


def _run(args, *, summarize: bool) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, ema = load_model(args.checkpoint, device)
    assert_l2(model)
    step = ModelStepPhysical(ema, args.norm)
    factor = int(model.ds_factors[0])
    samples = sample_windows(args.eval_dir, args.every_nth_field, args.n_states, 1, args.seed)
    for sample in samples:
        if abs(sample["dt_kept"] - args.delta_t) > 1e-6:
            raise RuntimeError(
                f"kept-frame spacing {sample['dt_kept']} != --delta-t {args.delta_t}"
            )
    n = int(samples[0]["frames"][0]["omega"].shape[0])
    bands = radial_band_masks(n)
    tasks = band_tasks(args.n_states, len(bands))
    owned = [
        task
        for index, task in enumerate(tasks)
        if args.n_gpus <= 1 or index % args.n_gpus == args.gpu_rank
    ]
    print(f"[trace] rank {args.gpu_rank}/{args.n_gpus} owns {len(owned)} shells", flush=True)
    by_state = {int(sample["sample"]): sample for sample in samples}
    for state, band in owned:
        label, mask = bands[band]
        path = save_dir / f"state{state:02d}_band{band}.npz"
        if args.skip_existing and path.is_file():
            print(f"[trace] skip {path.name}", flush=True)
            continue
        frame = by_state[state]["frames"]
        u = frame[0]["omega"]
        z = pool_avg(frame[1]["omega"], factor)
        dots = _estimate(
            step, u, z, mask, args.n_probes, args.seed + 1000 * state + 17 * band
        )
        dof = band_real_dof(mask)
        mean = float(np.mean(dots))
        print(
            f"[trace] state={state} band={label} tr={mean:.6g} tr/n={mean / dof:.6g}",
            flush=True,
        )
        np.savez_compressed(
            path,
            state=np.asarray(state),
            band=np.asarray(band),
            label=np.asarray(label),
            stem=np.asarray(frame[0]["stem"]),
            dots=dots,
            trace=np.asarray(mean),
            real_dof=np.asarray(dof),
            trace_per_mode=np.asarray(mean / dof),
            n=np.asarray(n),
            pool_factor=np.asarray(factor),
        )
    if summarize:
        _finalize(args, save_dir, bands, factor, n)


def _finalize(args, save_dir: Path, bands, factor: int, n: int) -> None:
    rows = []
    for state, band in band_tasks(args.n_states, len(bands)):
        path = save_dir / f"state{state:02d}_band{band}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
        data = np.load(path)
        rows.append(
            {
                "state": int(data["state"]),
                "band": int(data["band"]),
                "label": str(data["label"]),
                "trace": float(data["trace"]),
                "real_dof": int(data["real_dof"]),
                "trace_per_mode": float(data["trace_per_mode"]),
            }
        )
    labels = [label for label, _mask in bands]
    means, stds, traces = [], [], []
    for band, label in enumerate(labels):
        vals = [row["trace_per_mode"] for row in rows if row["band"] == band]
        raw = [row["trace"] for row in rows if row["band"] == band]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        traces.append(float(np.mean(raw)))
    total = float(np.sum(traces))
    summary = {
        "labels": labels,
        "trace_per_mode_mean": means,
        "trace_per_mode_std_across_states": stds,
        "trace_mean": traces,
        "trace_over_n": total / float(n * n),
        "n": n * n,
        "pool_factor": factor,
        "probes": "Rademacher, Fourier-masked to the shell, not normalized",
        "operator": "dg/du with the latent held fixed",
    }
    write_json(save_dir / "summary.json", summary)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(np.arange(len(labels)), means, yerr=stds, marker="o", ls="none")
    ax.axhline(0.0, color="0.4", lw=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("tr(P A P) / dim(P)")
    ax.set_xlabel("radial wavenumber shell")
    ax.set_title(f"scale-resolved fine Jacobian  (sum/N = {total / (n * n):.4g})")
    fig.tight_layout()
    fig.savefig(save_dir / "trace_per_mode_vs_band.png", dpi=140)
    plt.close(fig)
    write_json(
        save_dir / "metadata.json",
        base_metadata(
            args,
            {
                "task": "scale-resolved trace of fixed-latent A",
                "summary": summary,
                "weights_sha256": sha256_file(args.checkpoint),
                "real_dof": [band_real_dof(mask) for _label, mask in bands],
            },
        ),
    )
    print(
        f"[trace] tr(A)/N = {total / (n * n):.6g} over {len(labels)} shells",
        flush=True,
    )


def _orchestrate(args) -> None:
    script = str(Path(__file__).resolve())
    procs = []
    for rank in range(args.n_gpus):
        child = argparse.Namespace(**vars(args))
        child.worker = True
        child.gpu_rank = rank
        cmd = [sys.executable, "-u", script, *namespace_to_argv(child)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[trace] launch rank {rank}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"trace workers exited {codes}")
    first = next(Path(args.save_dir).glob("state*_band*.npz"))
    data = np.load(first)
    n = int(data["n"])
    factor = int(data["pool_factor"])
    _finalize(args, Path(args.save_dir), radial_band_masks(n), factor, n)


def main() -> None:
    args = parse_args()
    if args.n_gpus > 1 and not args.worker:
        _orchestrate(args)
        return
    _run(args, summarize=args.n_gpus <= 1)


if __name__ == "__main__":
    main()
