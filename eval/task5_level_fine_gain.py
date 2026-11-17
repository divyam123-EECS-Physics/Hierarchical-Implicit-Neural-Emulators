#!/usr/bin/env python3
"""Fine-input gain for L=1 and L=3, on the same probes as L=2 A_QQ.

v is the pool-8 fine direction from the L=2 run. The gain is
||Q (fine response)|| / ||v|| in physical vorticity. Every latent input
is held fixed at the true pooled future field, so the derivative is only
with respect to the fine input. L=1 has no latent inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for _path in (_REPO, _EVAL):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coarse_projectors import q_model
from eval_jacobian_manual_mse import load_kept_paths
from eval_jacobian_manual_mse import load_model as load_l3
from eval_ablation_mse import load_ablation_model
from py2d_inmem import pool_avg, rms
from task35_common import base_metadata, load_omega, sample_windows, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--n-directions", type=int, default=4)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--pool-factor", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--l1-checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/ablation_l1/checkpoint-last.pth",
    )
    p.add_argument(
        "--l3-checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/checkpoint-last.pth",
    )
    p.add_argument(
        "--l2-comparisons",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task5_hine_vs_cond/n8/comparisons.npz",
    )
    p.add_argument(
        "--fno-gains",
        default="/pscratch/sd/d/divyam/hine_runs/fno_1to1/task5_fine_gain/n8/fine_gains.npz",
    )
    p.add_argument(
        "--compare-dir",
        default="/pscratch/sd/d/divyam/hine_runs/task5_fine_gain_compare",
    )
    return p.parse_args(argv)


def fine_direction(seed: int, sample: int, direction: int, shape, factor: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 50 * sample + direction)
    v = q_model(rng.standard_normal(shape), factor)
    scale = rms(v)
    if scale == 0.0:
        raise RuntimeError("fine direction vanished")
    return v / scale


def _image(field: np.ndarray, norm: float, device, dtype) -> torch.Tensor:
    arr = np.asarray(field, dtype=np.float64)
    return (
        torch.from_numpy(np.ascontiguousarray(arr))
        .to(device=device, dtype=dtype)
        .view(1, 1, arr.shape[0], arr.shape[1])
        / norm
    )


def _out(tangent: torch.Tensor, shape, norm: float) -> np.ndarray:
    return (tangent.detach().reshape(shape).float().cpu().numpy() * norm).astype(np.float64)


def jvp_l1(net, u_phys, v_phys, norm: float) -> np.ndarray:
    device = next(net.parameters()).device
    dtype = torch.float32

    def f(u_in):
        return net(u_in)

    _primal, tangent = torch.autograd.functional.jvp(
        f, (_image(u_phys, norm, device, dtype),), (_image(v_phys, norm, device, dtype),)
    )
    return _out(tangent, u_phys.shape, norm)


def jvp_l3(net, u_phys, mid_phys, low_phys, v_phys, norm: float) -> np.ndarray:
    """Fine head only. Both latent tensors are closed over, not JVP inputs."""
    device = next(net.parameters()).device
    dtype = torch.float32
    u = _image(u_phys, norm, device, dtype)
    v = _image(v_phys, norm, device, dtype)
    mid = _image(mid_phys, norm, device, dtype)
    low = _image(low_phys, norm, device, dtype)
    t = 2 * torch.ones(u.shape[0], device=device, dtype=dtype)

    def f(u_in):
        high, _mid_out, _low_out = net(u_in, mid, low, t)
        return high

    _primal, tangent = torch.autograd.functional.jvp(f, (u,), (v,))
    return _out(tangent, u_phys.shape, norm)


def gain_row(sample, direction, stem, response, v, factor, label) -> dict:
    fine = q_model(response, factor)
    return {
        "level": label,
        "sample": int(sample),
        "stem": int(stem),
        "direction": int(direction),
        "gain": rms(fine) / rms(v),
        "gain_full": rms(response) / rms(v),
        "coarse_response_rms": rms(response - fine),
    }


def run_l1(args, samples, device) -> list[dict]:
    _model, net = load_ablation_model(args.l1_checkpoint, device)
    if int(getattr(net, "num_aux", 0)) != 0:
        raise ValueError("L=1 checkpoint still has auxiliary latent heads")
    rows = []
    factor = args.pool_factor
    for sample in samples:
        u_n = sample["frames"][0]["omega"]
        for direction in range(args.n_directions):
            v = fine_direction(args.seed, sample["sample"], direction, u_n.shape, factor)
            response = jvp_l1(net, u_n, v, args.norm)
            row = gain_row(sample["sample"], direction, sample["frames"][0]["stem"], response, v, factor, "L1")
            rows.append(row)
            print(
                f"[L1] sample={row['sample']} dir={row['direction']} gain={row['gain']:.4f}",
                flush=True,
            )
    del net, _model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_l3(args, samples, kept, device) -> list[dict]:
    model, net = load_l3(args.l3_checkpoint, device)
    if int(model.ds_factor) != args.pool_factor:
        raise ValueError(
            f"L=3 finest pool factor is {model.ds_factor}, probes use {args.pool_factor}"
        )
    rows = []
    factor = args.pool_factor
    for sample in samples:
        start = int(sample["start_index"])
        if start + 2 >= len(kept):
            raise IndexError(f"sample {sample['sample']} has no u_n+2 kept frame")
        u_n = sample["frames"][0]["omega"]
        u_np1 = sample["frames"][1]["omega"]
        u_np2, time2 = load_omega(kept[start + 2][1])
        dt = sample["frames"][1]["time"] - sample["frames"][0]["time"]
        if abs((time2 - sample["frames"][1]["time"]) - dt) > 1e-6:
            raise RuntimeError(
                f"u_n+2 is not one model step later at sample {sample['sample']}"
            )
        mid = pool_avg(u_np1, int(model.ds_factor))
        low = pool_avg(u_np2, int(model.ds_factor_low))
        for direction in range(args.n_directions):
            v = fine_direction(args.seed, sample["sample"], direction, u_n.shape, factor)
            response = jvp_l3(net, u_n, mid, low, v, args.norm)
            row = gain_row(
                sample["sample"], direction, sample["frames"][0]["stem"], response, v, factor, "L3"
            )
            rows.append(row)
            print(
                f"[L3] sample={row['sample']} dir={row['direction']} gain={row['gain']:.4f}",
                flush=True,
            )
    return rows


def _save(save_dir: Path, rows: list[dict], args, note: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    gains = np.array([r["gain"] for r in rows], dtype=np.float64)
    np.savez(
        save_dir / "fine_gains.npz",
        sample=np.array([r["sample"] for r in rows]),
        stem=np.array([r["stem"] for r in rows]),
        direction=np.array([r["direction"] for r in rows]),
        gain=gains,
        gain_full=np.array([r["gain_full"] for r in rows]),
        coarse_response_rms=np.array([r["coarse_response_rms"] for r in rows]),
    )
    summary = {
        "n": int(gains.size),
        "gain_mean": float(gains.mean()),
        "gain_std": float(gains.std()),
        "gain_min": float(gains.min()),
        "gain_max": float(gains.max()),
        "pool_factor": int(args.pool_factor),
        "note": note,
    }
    write_json(save_dir / "summary.json", summary)
    write_json(save_dir / "metadata.json", base_metadata(args, {"task": note}))
    print(
        f"[done] {note} mean {summary['gain_mean']:.4f} "
        f"std {summary['gain_std']:.4f} "
        f"range {summary['gain_min']:.4f} {summary['gain_max']:.4f}",
        flush=True,
    )


def _paired(path: Path, key: str, eps: float | None = None) -> np.ndarray:
    data = np.load(path)
    if eps is None:
        order = np.lexsort((data["direction"], data["sample"]))
        return np.asarray(data[key], dtype=np.float64)[order]
    mask = np.abs(data["eps"] - eps) < 1e-15
    order = np.lexsort((data["direction"][mask], data["sample"][mask]))
    return np.asarray(data[key], dtype=np.float64)[mask][order]


def _plot_compare(path: Path, series: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = list(series)
    for i, name in enumerate(names):
        y = series[name]
        x = np.full(y.shape, i) + np.linspace(-0.12, 0.12, y.size)
        ax.scatter(x, y, s=18)
        ax.plot([i - 0.18, i + 0.18], [y.mean(), y.mean()], color="0.2", lw=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("||Q response|| / ||v||")
    ax.set_title("fine-input gain, same probes")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    samples = sample_windows(
        args.eval_dir, args.every_nth_field, args.n_states, lookahead=1, seed=args.seed
    )
    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    l1_rows = run_l1(args, samples, device)
    _save(
        Path("/pscratch/sd/d/divyam/hine_runs/ablation_l1/task5_fine_gain/n8"),
        l1_rows,
        args,
        "L=1 fine-input gain, no latent inputs",
    )
    l3_rows = run_l3(args, samples, kept, device)
    _save(
        Path("/pscratch/sd/d/divyam/hine_runs/paper_1gpu/task5_fine_gain/n8"),
        l3_rows,
        args,
        "L=3 fine-input gain, both latents fixed at the true pools",
    )
    l1 = np.array([r["gain"] for r in l1_rows])
    l3 = np.array([r["gain"] for r in l3_rows])
    l2 = _paired(Path(args.l2_comparisons), "gain_A", eps=1e-3)
    fno = _paired(Path(args.fno_gains), "gain_M")
    if not (l1.shape == l2.shape == l3.shape == fno.shape):
        raise RuntimeError(f"probe counts differ: L1 {l1.shape} L2 {l2.shape} L3 {l3.shape} FNO {fno.shape}")
    compare = Path(args.compare_dir)
    _plot_compare(
        compare / "fine_gain_models.png",
        {"L=1": l1, "L=2": l2, "L=3": l3, "FNO 1-to-1": fno},
    )
    write_json(
        compare / "summary.json",
        {
            "L1_mean": float(l1.mean()),
            "L2_mean": float(l2.mean()),
            "L3_mean": float(l3.mean()),
            "FNO_mean": float(fno.mean()),
            "n": int(l1.size),
            "probes": "same pool-8 fine directions as the L=2 A_QQ run",
        },
    )


if __name__ == "__main__":
    main()
