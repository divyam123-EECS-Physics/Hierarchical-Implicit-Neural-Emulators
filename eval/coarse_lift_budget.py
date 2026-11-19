"""Realized coarse responses along the matched L=2 rollout.

Jacobians stay at the teacher-forced joint state. The vectors they hit are
the free-rollout errors. T and U are not applied here; this file measures
C, CP, CQ, K, and the two-step fine feedback.
"""

from __future__ import annotations

import numpy as np
import torch

from coarse_lift_ops import (
    BUDGET_LENGTH,
    TWOSTEP_STEPS,
    ZERO_FLOOR,
    LiftedBlocks,
    cosine,
    directional_gains,
    euclidean_norm,
    field_rms,
    two_step_parts,
)
from l2_fd_jvp import L2Point, make_point, model_step


def step_fn_at(ema, point: L2Point):
    t = point.t

    def step(u, z, _t=t):
        return model_step(ema, u, z, _t)

    return step


def teacher_points(frames, model, device, dtype, norm: float, n_steps: int) -> list[L2Point]:
    if len(frames) < n_steps + 2:
        raise ValueError(f"need {n_steps + 2} frames, got {len(frames)}")
    points = []
    for k in range(n_steps):
        points.append(make_point(frames[k : k + 3], model, device, dtype, norm))
    return points


def _image(tensor: torch.Tensor) -> np.ndarray:
    return np.squeeze(tensor.detach().cpu().numpy()).astype(np.float64, copy=False)


def _stats_pair(output: torch.Tensor, source) -> dict[str, float]:
    gains = directional_gains(output, source)
    return {
        "norm": euclidean_norm(output),
        "rms": field_rms(output),
        "gain_euclidean": gains["euclidean"],
        "gain_rms": gains["rms"],
    }


def rollout_budget(ema, model, frames, device, norm: float, n_steps: int = BUDGET_LENGTH, twostep_steps=TWOSTEP_STEPS) -> dict:
    """Free rollout from (u_0, D u_1) and the coarse linear budget at each step.

    e_0 = η_0 = 0. ε_Z is the teacher-forced coarse-head residual.
    The comparison is η_{n+1} against C e_n + K η_n + ε_Z, with C and K
    evaluated at the teacher-forced inputs.
    """
    dtype = torch.float64
    factor = int(model.ds_factors[0])
    points = teacher_points(frames, model, device, dtype, norm, n_steps)
    ops = [LiftedBlocks(step_fn_at(ema, pt), pt, factor, norm) for pt in points]

    high = points[0].u.clone()
    latent = points[0].z.clone()
    e_fields = [np.zeros(points[0].u.shape[-2:], dtype=np.float64)]
    eta_fields = [np.zeros(points[0].z.shape[-2:], dtype=np.float64)]
    eps_fields = []

    with torch.no_grad():
        for k, pt in enumerate(points):
            g, h = model_step(ema, pt.u, pt.z, pt.t)
            eps_fields.append(_image((h - pt.z_next) * norm))
            high, latent = model_step(ema, high, latent, pt.t)
            high = high.detach()
            latent = latent.detach()
            e_fields.append(_image((high - pt.u_next) * norm))
            eta_fields.append(_image((latent - pt.z_next) * norm))

    rows = []
    c_images = []
    cp_images = []
    cq_images = []
    k_images = []
    remainders = []
    for n in range(n_steps):
        e_n = e_fields[n]
        eta_n = eta_fields[n]
        eps = eps_fields[n]
        op = ops[n]
        c_e = op.C(e_n)
        cp_e = op.C(op.P(e_n))
        cq_e = op.CQ(e_n)
        k_e = op.K(eta_n)
        c_np = _image(c_e)
        cp_np = _image(cp_e)
        cq_np = _image(cq_e)
        k_np = _image(k_e)
        linear = c_np + k_np + eps
        remainder = eta_fields[n + 1] - linear
        add = c_np - cp_np - cq_np
        add_rel = float(np.linalg.norm(add) / max(np.linalg.norm(c_np), ZERO_FLOOR))
        c_stats = _stats_pair(c_e, e_n)
        cp_stats = _stats_pair(cp_e, op.P(e_n))
        cq_stats = _stats_pair(cq_e, op.Q(e_n))
        k_stats = _stats_pair(k_e, eta_n)
        rows.append(
            {
                "step": n,
                "norm_e_H": euclidean_norm(e_n),
                "rms_e_H": field_rms(e_n),
                "norm_e_Z": euclidean_norm(eta_n),
                "rms_e_Z": field_rms(eta_n),
                "norm_e_Z_next": euclidean_norm(eta_fields[n + 1]),
                "rms_e_Z_next": field_rms(eta_fields[n + 1]),
                "norm_C_e": c_stats["norm"],
                "rms_C_e": c_stats["rms"],
                "gain_C_euclidean": c_stats["gain_euclidean"],
                "gain_C_rms": c_stats["gain_rms"],
                "norm_CP_e": cp_stats["norm"],
                "rms_CP_e": cp_stats["rms"],
                "gain_CP_euclidean": cp_stats["gain_euclidean"],
                "gain_CP_rms": cp_stats["gain_rms"],
                "norm_CQ_e": cq_stats["norm"],
                "rms_CQ_e": cq_stats["rms"],
                "gain_CQ_euclidean": cq_stats["gain_euclidean"],
                "gain_CQ_rms": cq_stats["gain_rms"],
                "norm_K_e": k_stats["norm"],
                "rms_K_e": k_stats["rms"],
                "gain_K_euclidean": k_stats["gain_euclidean"],
                "gain_K_rms": k_stats["gain_rms"],
                "norm_eps_Z": euclidean_norm(eps),
                "rms_eps_Z": field_rms(eps),
                "norm_linear": euclidean_norm(linear),
                "rms_linear": field_rms(linear),
                "norm_remainder": euclidean_norm(remainder),
                "rms_remainder": field_rms(remainder),
                "add_rel": add_rel,
                "cosine_C_K": cosine(c_np, k_np),
            }
        )
        c_images.append(c_np)
        cp_images.append(cp_np)
        cq_images.append(cq_np)
        k_images.append(k_np)
        remainders.append(remainder)

    twostep = []
    for n in twostep_steps:
        n = int(n)
        if n < 0 or n + 1 >= n_steps:
            twostep.append({"step": n, "ok": False, "error": "step pair is outside the rollout"})
            continue
        parts = two_step_parts(ops[n], ops[n + 1], e_fields[n])
        twostep.append(
            {
                "step": n,
                "ok": parts["sum_vs_composed_rel"] <= 1e-6,
                "sum_vs_composed_rel": parts["sum_vs_composed_rel"],
                "norm_AA": parts["norm_AA"],
                "norm_BC": parts["norm_BC"],
                "norm_sum": parts["norm_sum"],
                "rms_AA": parts["rms_AA"],
                "rms_BC": parts["rms_BC"],
                "cosine_AA_BC": parts["cosine_AA_BC"],
                "initial_fine_norm": euclidean_norm(e_fields[n]),
                "AA": _image(parts["A_A_e"]),
                "BC": _image(parts["B_C_e"]),
            }
        )

    def column(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    names = list(rows[0].keys()) if rows else []
    packed = {name: column(name) for name in names if name != "step"}
    packed["step"] = column("step").astype(np.int64) if rows else np.zeros(0, dtype=np.int64)
    packed["C_e"] = np.stack(c_images, axis=0) if c_images else np.zeros((0,))
    packed["CP_e"] = np.stack(cp_images, axis=0) if cp_images else np.zeros((0,))
    packed["CQ_e"] = np.stack(cq_images, axis=0) if cq_images else np.zeros((0,))
    packed["K_e"] = np.stack(k_images, axis=0) if k_images else np.zeros((0,))
    packed["eps_Z"] = np.stack(eps_fields, axis=0)
    packed["remainder"] = np.stack(remainders, axis=0) if remainders else np.zeros((0,))
    packed["e_Z"] = np.stack(eta_fields, axis=0)
    packed["norm_e_H_series"] = np.asarray([euclidean_norm(e) for e in e_fields], dtype=np.float64)
    packed["rms_e_H_series"] = np.asarray([field_rms(e) for e in e_fields], dtype=np.float64)
    packed["twostep"] = twostep
    packed["add_rel_max"] = float(np.nanmax(packed["add_rel"])) if packed["add_rel"].size else float("nan")
    return packed
