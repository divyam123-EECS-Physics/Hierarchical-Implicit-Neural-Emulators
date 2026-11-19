#!/usr/bin/env python3
"""Fill latent cosines that the saved Task 1 results do not contain.

Forward rollout plus the latent Jacobian K. No Hutchinson traces and no DNS tangent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

_EVAL = Path(__file__).resolve().parent
_REPO = _EVAL.parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_jacobian_manual_mse_plus import load_model
from l2_fd_jvp import jvp_both_u, jvp_both_z, make_point, model_step
from l2_retention import _norm, latent_error_channels

KEEP = (
    "cos_eta_rho",
    "cos_rz_eta",
    "cos_rho_rz",
    "cos_rho_Iz",
    "channel_remainder_norm",
)


def _frames(entry: dict) -> list[np.ndarray]:
    frames = []
    for path in entry["paths"]:
        raw = scipy.io.loadmat(path)
        frames.append(np.asarray(raw["Omega"]))
    return frames


def _one_window(ema_vit, model, frames, device, dtype, norm: float, n_steps: int) -> dict:
    points = [
        make_point(frames[k : k + 3], model, device, dtype, norm) for k in range(n_steps)
    ]
    eps_z = []
    for pt in points:
        _g, h = model_step(ema_vit, pt.u, pt.z, pt.t)
        eps_z.append((h - pt.z_next).detach())
    high = points[0].u.clone()
    z = points[0].z.clone()
    e = [torch.zeros_like(high)]
    eta = [torch.zeros_like(z)]
    for k in range(n_steps):
        high, z = model_step(ema_vit, high, z, points[k].t)
        high = high.detach()
        z = z.detach()
        e.append(high - points[k].u_next)
        eta.append(z - points[k].z_next)
    e_phys = [t * norm for t in e]
    eta_phys = [t * norm for t in eta]
    ez_phys = [t * norm for t in eps_z]
    zeros = [torch.zeros_like(e_phys[0]) for _ in range(n_steps)]

    def apply_c(k: int, vec: torch.Tensor) -> torch.Tensor:
        if _norm(vec) == 0.0:
            return torch.zeros_like(points[k].z)
        _fine, latent = jvp_both_u(
            lambda u, zz, _t=points[k].t: model_step(ema_vit, u, zz, _t),
            points[k],
            vec / norm,
        )
        return latent * norm

    def apply_k(k: int, vec: torch.Tensor) -> torch.Tensor:
        if _norm(vec) == 0.0:
            return torch.zeros_like(points[k].z)
        _fine, latent = jvp_both_z(
            lambda u, zz, _t=points[k].t: model_step(ema_vit, u, zz, _t),
            points[k],
            vec / norm,
        )
        return latent * norm

    channels = latent_error_channels(
        e_phys, eta_phys, zeros, ez_phys, apply_c, apply_k, first_factor=False,
    )
    return {key: channels[key] for key in KEEP}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sample-index", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--n-gpus", type=int, required=True)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--rollout", type=int, default=40)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    payload = json.loads(Path(args.sample_index).read_text())
    samples = payload["samples"]
    mine = [s for s in samples if int(s["sample"]) % args.n_gpus == args.rank]
    model, ema = load_model(args.checkpoint, device)
    ema.double().eval()
    rows = {key: [] for key in KEEP}
    indices = []
    for sample in mine:
        t0 = time.time()
        frames = _frames(sample)
        if len(frames) < args.rollout + 2:
            raise RuntimeError(f"window {sample['sample']} has {len(frames)} frames")
        out = _one_window(ema, model, frames, device, torch.float64, args.norm, args.rollout)
        for key in KEEP:
            rows[key].append(out[key])
        indices.append(int(sample["sample"]))
        finite = out["cos_eta_rho"][np.isfinite(out["cos_eta_rho"])]
        rz = out["cos_rz_eta"][np.isfinite(out["cos_rz_eta"])]
        print(
            f"rank {args.rank} window {sample['sample']} "
            f"cos_eta_rho_mean={float(np.mean(finite)):.4f} "
            f"cos_eta_rz_mean={float(np.mean(rz)):.4f} "
            f"sec={time.time() - t0:.1f}",
            flush=True,
        )
    dest = Path(args.save_dir) / f"cosine_fill_rank{args.rank}.npz"
    np.savez_compressed(
        dest,
        indices=np.asarray(indices, dtype=np.int64),
        **{key: np.stack(rows[key]) for key in KEEP},
    )
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
