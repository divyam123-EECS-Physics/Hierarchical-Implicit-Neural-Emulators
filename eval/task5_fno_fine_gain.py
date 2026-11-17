#!/usr/bin/env python3
"""Fine-input gain of the one-to-one FNO, matched to the L=2 A_QQ probes.

v is the same pool-8 fine direction used for ||A_QQ v|| / ||v||. The FNO
has no latent. The reported gain is ||Q M'(u) v|| / ||v|| in physical
vorticity, with Q the same average-pool projector.
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
from eval_fno_jacobian_manual_mse import load_fno
from py2d_inmem import rms
from task35_common import base_metadata, sample_windows, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/fno_1to1/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/fno_1to1/task5_fine_gain/n8",
    )
    p.add_argument(
        "--hine-comparisons",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task5_hine_vs_cond/n8/comparisons.npz",
    )
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--n-directions", type=int, default=4)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--pool-factor", type=int, default=8)
    p.add_argument("--hine-eps", type=float, default=1e-3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p.parse_args(argv)


def fine_direction(seed: int, sample: int, direction: int, shape, factor: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 50 * sample + direction)
    v = q_model(rng.standard_normal(shape), factor)
    scale = rms(v)
    if scale == 0.0:
        raise RuntimeError("fine direction vanished")
    return v / scale


def fno_jvp(net: torch.nn.Module, u_phys: np.ndarray, v_phys: np.ndarray, norm: float) -> np.ndarray:
    """M'(u) v in physical vorticity. Parameters stay frozen."""
    device = next(net.parameters()).device
    dtype = torch.float32

    def image(field: np.ndarray) -> torch.Tensor:
        arr = np.asarray(field, dtype=np.float64)
        return (
            torch.from_numpy(np.ascontiguousarray(arr))
            .to(device=device, dtype=dtype)
            .view(1, 1, arr.shape[0], arr.shape[1])
            / norm
        )

    def f(u_in):
        return net(u_in)

    _primal, tangent = torch.autograd.functional.jvp(f, (image(u_phys),), (image(v_phys),))
    return (
        tangent.detach().reshape(u_phys.shape).float().cpu().numpy() * norm
    ).astype(np.float64)


def hine_gains(path: Path, eps: float) -> dict[tuple[int, int], float]:
    if not path.is_file():
        return {}
    data = np.load(path)
    out = {}
    for sample, direction, eps_i, gain in zip(
        data["sample"], data["direction"], data["eps"], data["gain_A"]
    ):
        if abs(float(eps_i) - eps) < 1e-15:
            out[(int(sample), int(direction))] = float(gain)
    return out


def _plot(save_dir: Path, rows: list[dict]) -> None:
    paired = [r for r in rows if np.isfinite(r["gain_A"])]
    fig, ax = plt.subplots(figsize=(5, 5))
    if paired:
        ax.scatter([r["gain_A"] for r in paired], [r["gain_M"] for r in paired], s=22)
        vals = [r["gain_A"] for r in paired] + [r["gain_M"] for r in paired]
        lo, hi = min(vals), max(vals)
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", lw=1)
        ax.set_xlabel("||A_QQ v|| / ||v||")
        ax.set_ylabel("||Q M'(u) v|| / ||v||")
    else:
        ax.hist([r["gain_M"] for r in rows], bins=12)
        ax.set_xlabel("||Q M'(u) v|| / ||v||")
    ax.set_title("fine-input gain, FNO 1-to-1")
    fig.tight_layout()
    fig.savefig(save_dir / "fine_gain.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    net = load_fno(args.checkpoint, device)
    if int(net.in_channels) != 1 or int(net.out_channels) != 1:
        raise ValueError(
            f"expected a 1-to-1 map, got in={net.in_channels} out={net.out_channels}"
        )
    samples = sample_windows(
        args.eval_dir, args.every_nth_field, args.n_states, lookahead=1, seed=args.seed
    )
    reference = hine_gains(Path(args.hine_comparisons), args.hine_eps)
    rows = []
    for sample in samples:
        u_n = sample["frames"][0]["omega"]
        for direction in range(args.n_directions):
            v = fine_direction(args.seed, sample["sample"], direction, u_n.shape, args.pool_factor)
            response = fno_jvp(net, u_n, v, args.norm)
            fine = q_model(response, args.pool_factor)
            gain = rms(fine) / rms(v)
            key = (int(sample["sample"]), int(direction))
            row = {
                "sample": key[0],
                "stem": int(sample["frames"][0]["stem"]),
                "direction": key[1],
                "gain_M": gain,
                "gain_full": rms(response) / rms(v),
                "gain_A": reference.get(key, float("nan")),
                "norm_v": rms(v),
                "coarse_response_rms": rms(response - fine),
            }
            rows.append(row)
            print(
                f"[fno] sample={key[0]} dir={key[1]} stem={row['stem']} "
                f"gain_M={gain:.4f} gain_A={row['gain_A']:.4f}",
                flush=True,
            )
    gains = np.array([r["gain_M"] for r in rows], dtype=np.float64)
    np.savez(
        save_dir / "fine_gains.npz",
        sample=np.array([r["sample"] for r in rows]),
        stem=np.array([r["stem"] for r in rows]),
        direction=np.array([r["direction"] for r in rows]),
        gain_M=gains,
        gain_full=np.array([r["gain_full"] for r in rows]),
        gain_A=np.array([r["gain_A"] for r in rows]),
        coarse_response_rms=np.array([r["coarse_response_rms"] for r in rows]),
    )
    summary = {
        "n": int(gains.size),
        "gain_M_mean": float(gains.mean()),
        "gain_M_std": float(gains.std()),
        "gain_M_min": float(gains.min()),
        "gain_M_max": float(gains.max()),
        "pool_factor": int(args.pool_factor),
        "operator": "Q M'(u) Q, physical vorticity, no latent",
    }
    paired = np.array([r["gain_A"] for r in rows], dtype=np.float64)
    if np.isfinite(paired).all():
        summary["gain_A_mean"] = float(paired.mean())
        summary["gain_A_on_same_probes_mean"] = float(paired.mean())
    write_json(save_dir / "summary.json", summary)
    _plot(save_dir, rows)
    meta = base_metadata(args, {"task": "FNO 1-to-1 fine-input gain", "pool_factor": args.pool_factor})
    write_json(save_dir / "metadata.json", meta)
    print(
        f"[fno] gain_M mean {summary['gain_M_mean']:.4f} "
        f"std {summary['gain_M_std']:.4f} "
        f"range {summary['gain_M_min']:.4f} {summary['gain_M_max']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
