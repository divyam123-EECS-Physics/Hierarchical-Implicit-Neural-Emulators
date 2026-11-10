#!/usr/bin/env python3
"""One-window float64 finite-difference check. Proxy, no spectral radius.

Reports whether eps=0.01 agrees with eps/3 and eps/10 at the unchanged
tolerance. If it does not, compares 0.03, 0.003, and 0.001 on the same
states and directions, separately for the fine output g and the latent
output h. Does not launch the 20-window study.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_jacobian_manual_mse import load_kept_paths
from eval_jacobian_manual_mse_plus import load_model
from l2_fd_jvp import apply_D, fd_pair, make_point, model_step

CKPT = "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth"
EVAL_DIR = "/global/homes/d/divyam/TACSLAB/eval"
OUT = Path("/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_fd_precision/precision.json")
TOL = 1e-2
SEED = 0
NORM = 10.0
ROLLOUT = 40
PRIMARY = 0.01
COMPARE = (0.03, 0.003, 0.001)


def choose_sample_starts(n_kept: int, lookahead: int, n_samples: int, seed: int) -> np.ndarray:
    """Same draw as eval_jacobian_blocks.choose_sample_starts.

    Inlined so importing that module cannot hide the GPU.
    """
    max_start = n_kept - 1 - lookahead
    if max_start < 0:
        raise ValueError(f"Need at least {lookahead + 1} kept frames, have {n_kept}")
    rng = np.random.default_rng(seed)
    starts = rng.choice(max_start + 1, size=n_samples, replace=False)
    starts.sort()
    return starts.astype(np.int64)


def _enorm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().reshape(-1)))


def _audit(module: torch.nn.Module) -> dict:
    bad_params = [
        (name, str(tensor.dtype))
        for name, tensor in module.named_parameters()
        if tensor.dtype != torch.float64
    ]
    bad_buffers = [
        (name, str(tensor.dtype))
        for name, tensor in module.named_buffers()
        if tensor.is_floating_point() and tensor.dtype != torch.float64
    ]
    seen: list[dict] = []

    def hook(name):
        def _record(_mod, _inputs, output):
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for tensor in tensors:
                if (
                    torch.is_tensor(tensor)
                    and tensor.dtype != torch.float64
                    and len(seen) < 40
                ):
                    seen.append(
                        {
                            "module": name,
                            "dtype": str(tensor.dtype),
                            "shape": list(tensor.shape),
                            "autocast": bool(torch.is_autocast_enabled()),
                        }
                    )
        return _record

    handles = [
        submodule.register_forward_hook(hook(name))
        for name, submodule in module.named_modules()
    ]
    return {
        "bad_params": bad_params[:40],
        "n_bad_params": len(bad_params),
        "bad_buffers": bad_buffers[:40],
        "n_bad_buffers": len(bad_buffers),
        "bad_outputs": seen,
        "handles": handles,
    }


def _pair_row(name, block, left, right, eps, other, info, repeat_scale) -> dict:
    diff = _enorm(left - right)
    norm_left = _enorm(left)
    norm_right = _enorm(right)
    denom = max(norm_right, 1e-12)
    rel = diff / denom
    return {
        "name": name,
        "block": block,
        "eps": eps,
        "against": other,
        "rel": rel,
        "abs": diff,
        "norm": norm_left,
        "norm_against": norm_right,
        "ok_rel": bool(rel <= TOL),
        "h": float(info["h"]),
        "rel_u": float(info["rel_u"]),
        "rel_z": float(info["rel_z"]),
        "roundoff_scale": repeat_scale / max(float(info["h"]), 1e-30),
    }


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError(
            "probe_fd_precision requires one GPU; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    device = torch.device("cuda")
    kept = load_kept_paths(EVAL_DIR, 5)
    starts = choose_sample_starts(len(kept), ROLLOUT + 1, 1, SEED)
    start = int(starts[0])
    need = {0, 1, 2, ROLLOUT - 1, ROLLOUT, ROLLOUT + 1}
    frames = {}
    stems = {}
    for offset in sorted(need):
        stem, path = kept[start + offset]
        frames[offset] = np.asarray(scipy.io.loadmat(path)["Omega"])
        stems[offset] = int(stem)

    model, ema = load_model(CKPT, device)
    ema.double()
    ema.eval()
    audit = _audit(ema)

    def step(u, z, t):
        if torch.is_autocast_enabled():
            raise RuntimeError("autocast is enabled inside the finite-difference step")
        g, h = model_step(ema, u, z, t)
        if g.dtype != torch.float64 or h.dtype != torch.float64:
            raise RuntimeError(f"step returned {(g.dtype, h.dtype)}")
        return g, h

    points = []
    for name, offset in (("step0", 0), (f"step{ROLLOUT - 1}", ROLLOUT - 1)):
        point = make_point(
            [frames[offset], frames[offset + 1], frames[offset + 2]],
            model,
            device,
            torch.float64,
            NORM,
        )
        point.stems = (stems[offset], stems[offset + 1], stems[offset + 2])
        points.append((name, point))

    # One forward so the hooks see the real network, including attention.
    _g0, _h0 = step(points[0][1].u, points[0][1].z, points[0][1].t)
    for handle in audit.pop("handles"):
        handle.remove()
    g1, h1 = step(points[0][1].u, points[0][1].z, points[0][1].t)
    repeat = {
        "g": _enorm(_g0 - g1),
        "h": _enorm(_h0 - h1),
        "g_norm": _enorm(_g0),
        "h_norm": _enorm(_h0),
        "machine_eps": float(torch.finfo(torch.float64).eps),
    }

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    directions = []
    for name, point in points:
        v = torch.randn(point.u.shape, device=device, dtype=point.u.dtype, generator=gen)
        w = torch.randn(point.z.shape, device=device, dtype=point.z.dtype, generator=gen)
        v_graph = torch.randn(point.u.shape, device=device, dtype=point.u.dtype, generator=gen)
        directions.append((f"{name}/fine", point, v, torch.zeros_like(point.z)))
        directions.append((f"{name}/latent", point, torch.zeros_like(point.u), w))
        directions.append((f"{name}/graph", point, v_graph, apply_D(v_graph, int(model.ds_factors[0]))))

    epsilons = {
        "primary": PRIMARY,
        "primary/3": PRIMARY / 3.0,
        "primary/10": PRIMARY / 10.0,
        "e0.03": 0.03,
        "e0.003": 0.003,
        "e0.001": 0.001,
    }
    stored = {}
    for label, point, du, dz in directions:
        for tag, eps in epsilons.items():
            dg, dh, info = fd_pair(
                lambda u, z, _t=point.t: step(u, z, _t),
                point.u,
                point.z,
                du,
                dz,
                eps,
            )
            stored[(label, tag)] = (dg, dh, info, eps)

    primary_rows = []
    for label, _point, _du, _dz in directions:
        dg, dh, info, eps = stored[(label, "primary")]
        for tag in ("primary/3", "primary/10"):
            dg2, dh2, _, other = stored[(label, tag)]
            for block, left, right in (("g", dg, dg2), ("h", dh, dh2)):
                scale = repeat["g"] if block == "g" else repeat["h"]
                primary_rows.append(
                    _pair_row(label, block, left, right, eps, other, info, scale)
                )

    compare_tags = ("e0.03", "e0.003", "e0.001")
    compare_rows = []
    for label, _point, _du, _dz in directions:
        for i, tag_a in enumerate(compare_tags):
            for tag_b in compare_tags[i + 1 :]:
                ga, ha, info, eps_a = stored[(label, tag_a)]
                gb, hb, _, eps_b = stored[(label, tag_b)]
                for block, left, right in (("g", ga, gb), ("h", ha, hb)):
                    scale = repeat["g"] if block == "g" else repeat["h"]
                    compare_rows.append(
                        _pair_row(label, block, left, right, eps_a, eps_b, info, scale)
                    )

    def _summary(rows, block):
        chosen = [row for row in rows if row["block"] == block]
        return {
            "n": len(chosen),
            "n_failed_rel": sum(1 for row in chosen if not row["ok_rel"]),
            "max_rel": max((row["rel"] for row in chosen), default=None),
            "max_abs": max((row["abs"] for row in chosen), default=None),
            "min_norm": min((row["norm"] for row in chosen), default=None),
        }

    primary_ok = bool(primary_rows) and all(row["ok_rel"] for row in primary_rows)
    report = {
        "device": str(device),
        "model_dtype": "float64",
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "physical_jvp": "proxy",
        "spectral_radius": False,
        "fd_tol": TOL,
        "start_index": start,
        "stems": {str(k): v for k, v in stems.items()},
        "input_dtype": str(frames[0].dtype),
        "point_dtype": str(points[0][1].u.dtype),
        "audit": audit,
        "repeat": repeat,
        "primary_eps": PRIMARY,
        "primary_ok": primary_ok,
        "primary_g": _summary(primary_rows, "g"),
        "primary_h": _summary(primary_rows, "h"),
        "primary": primary_rows,
        "compare_eps": list(COMPARE),
        "compare_g": _summary(compare_rows, "g"),
        "compare_h": _summary(compare_rows, "h"),
        "compare": compare_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"primary_ok={primary_ok} "
        f"g_fail={report['primary_g']['n_failed_rel']}/{report['primary_g']['n']} "
        f"h_fail={report['primary_h']['n_failed_rel']}/{report['primary_h']['n']} "
        f"bad_outputs={len(audit['bad_outputs'])} "
        f"bad_params={audit['n_bad_params']}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)
    if audit["bad_outputs"] or audit["n_bad_params"] or audit["n_bad_buffers"]:
        raise SystemExit(3)
    if not primary_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
