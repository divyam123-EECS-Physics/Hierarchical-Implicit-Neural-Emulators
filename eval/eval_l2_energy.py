#!/usr/bin/env python3
"""L=2 kinetic-energy budget and energy-weighted Jacobian directions.

Does not retrain, does not edit Task 1 helpers, and writes only under
--save-dir. Stages:

  smoke   one window, four steps, finite-difference and adjoint checks
  budget  exact energy bookkeeping and realized A/B interactions
  svd     leading weighted singular/eigen directions after a passing budget
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for entry in (_REPO, _EVAL, _REPO / "plotting"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from energy_l2_model import (  # noqa: E402
    WeightedBlocks,
    downsample,
    load_l2_checkpoint,
    lock_precision,
    prepare_image,
    relative_error,
)
from energy_metric import (  # noqa: E402
    D_E_THRESHOLD,
    LOW_K_MAX,
    MID_K_MAX,
    REF_STEM_MAX,
    REF_STEM_MIN,
    EnergyMetric,
    d_E,
    degeneracy_flags,
    first_exit,
    orthonormality_error,
    principal_angles,
    ratio_or_nan,
    sort_singular_triplets,
    subspace_fraction,
)

DEFAULT_CHECKPOINT = "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth"
DEFAULT_SAMPLE_INDEX = (
    "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/"
    "task1_l2_n20_float64_py2d_jvp_20261116_204958/sample_index.json"
)
DEFAULT_REF_CACHE = (
    "/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_manual_mse/reference_kinetic_energy.npz"
)
EXACT_ATOL = 1e-8
EXACT_RTOL = 1e-8
DENOM_FLOOR = 0.0
SOURCE_FILES = (
    "eval/energy_metric.py",
    "eval/energy_l2_model.py",
    "eval/eval_l2_energy.py",
    "eval/ENERGY_L2_README.md",
    "plotting/plot_l2_energy.py",
    "tests/test_l2_energy.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "budget", "svd"), required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-index", default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--ref-cache", type=Path, default=DEFAULT_REF_CACHE)
    parser.add_argument("--budget-npz", type=Path, default=None)
    parser.add_argument("--norm", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--windows", default="")
    parser.add_argument("--svd-windows", default="0,10,19")
    parser.add_argument("--svd-steps", default="0,19,39")
    parser.add_argument("--k-singular", type=int, default=5)
    parser.add_argument("--svd-tol", type=float, default=1e-4)
    parser.add_argument("--svd-maxiter", type=int, default=300)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _ints(text: str) -> list[int]:
    if not str(text).strip():
        return []
    return [int(part) for part in str(text).split(",") if part.strip()]


def _field(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().reshape(tensor.shape[-2], tensor.shape[-1]).cpu().numpy().astype(np.float64, copy=False)


def _euclid(field: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(field, dtype=np.float64).ravel()))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = _euclid(left)
    b = _euclid(right)
    if a == 0.0 or b == 0.0:
        return float("nan")
    return float(np.dot(left.ravel(), right.ravel()) / (a * b))


def closure_ok(residual: float, scale: float) -> bool:
    return abs(float(residual)) <= EXACT_ATOL + EXACT_RTOL * max(abs(float(scale)), 0.0)


def load_window(entry: dict, dt: float, img_size: int = 256) -> tuple[list[np.ndarray], list[float], list[int]]:
    frames: list[np.ndarray] = []
    times: list[float] = []
    stems: list[int] = []
    for stem, path in zip(entry["stems"], entry["paths"]):
        raw = scipy.io.loadmat(path)
        if "time" not in raw:
            raise RuntimeError(f"{path} has no time value; refusing to infer spacing from the filename")
        omega = np.asarray(raw["Omega"], dtype=np.float64)
        if omega.shape != (img_size, img_size):
            raise RuntimeError(f"{path} shape {omega.shape} != {(img_size, img_size)}")
        frames.append(omega)
        times.append(float(np.asarray(raw["time"]).reshape(-1)[0]))
        stems.append(int(stem))
    for left, right in zip(times, times[1:]):
        gap = float(right) - float(left)
        if abs(gap - float(dt)) > 1e-6:
            raise RuntimeError(f"frame spacing {gap} != dt {dt} at stem {stems[0]}")
    for left, right in zip(stems, stems[1:]):
        if int(right) - int(left) != 5:
            raise RuntimeError(f"stem stride {right - left} != 5")
    return frames, times, stems


def verify_reference_cache(path: Path, metric: EnergyMetric) -> dict:
    """Read mu_E and sigma_E. Do not recompute them and do not rewrite the cache."""
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = np.load(path)
    stems = np.asarray(cache["stems"], dtype=np.int64)
    energies = np.asarray(cache["energies"], dtype=np.float64)
    mu = float(cache["mu_E"])
    sigma = float(cache["sigma_E"])
    if int(stems[0]) != REF_STEM_MIN or int(stems[-1]) != REF_STEM_MAX or stems.size != 95001:
        raise RuntimeError(
            f"reference interval {int(stems[0])}..{int(stems[-1])} n={stems.size} "
            f"!= {REF_STEM_MIN}..{REF_STEM_MAX} n=95001"
        )
    if abs(mu - float(energies.mean())) > 1e-12 or abs(sigma - float(energies.std(ddof=0))) > 1e-12:
        raise RuntimeError("cached mu_E/sigma_E do not match the cached reference energies")
    from eval_kinetic_energy_survival import kinetic_energy_torch, mat_path_for_stem

    device = torch.device("cpu")
    kx = torch.from_numpy(np.ascontiguousarray(metric.Kx))
    ky = torch.from_numpy(np.ascontiguousarray(metric.Ky))
    inv = torch.from_numpy(np.ascontiguousarray(metric.invKsq))
    rng = np.random.default_rng(0)
    picks = [int(v) for v in rng.choice(stems.size, size=3, replace=False)]
    spot = []
    for index in picks:
        stem = int(stems[index])
        omega = np.asarray(scipy.io.loadmat(mat_path_for_stem(stem))["Omega"], dtype=np.float64)
        ours = metric.energy(omega)
        torch_e = float(
            kinetic_energy_torch(
                torch.from_numpy(omega).view(1, metric.ny, metric.nx), kx, ky, inv
            ).reshape(-1)[0]
        )
        cached = float(energies[index])
        if abs(ours - torch_e) > 1e-8 * max(abs(torch_e), 1.0):
            raise RuntimeError(f"energy metric disagrees with kinetic_energy_torch at stem {stem}")
        rel_cache = abs(ours - cached) / max(abs(cached), 1e-30)
        if rel_cache > 1e-3:
            raise RuntimeError(f"reference cache disagrees with float64 KE at stem {stem}: rel {rel_cache}")
        spot.append({"stem": stem, "cached": cached, "float64": ours, "rel_cache": rel_cache})
    return {
        "path": str(path),
        "mu_E": mu,
        "sigma_E": sigma,
        "stem_min": REF_STEM_MIN,
        "stem_max": REF_STEM_MAX,
        "n_frames": int(stems.size),
        "population": "ddof=0 over reference stems only",
        "cache_dtype_note": "cache was built in float32; float64 spot checks are recorded",
        "spot_checks": spot,
    }


def analyze_window(
    step,
    metric: EnergyMetric,
    frames: list[np.ndarray],
    n_steps: int,
    mu_E: float,
    sigma_E: float,
) -> dict:
    """Free rollout from (u_0, D u_1) and teacher-forced A, B actions."""
    if len(frames) < n_steps + 2:
        raise ValueError(f"need {n_steps + 2} frames, got {len(frames)}")
    device = step.device
    truth = [prepare_image(frame, device) for frame in frames]
    u_hat = truth[0].clone()
    z_hat = downsample(truth[1], step.ds)
    init_e_h = float((u_hat - truth[0]).detach().reshape(-1).norm())
    init_e_z = float((z_hat - downsample(truth[1], step.ds)).detach().reshape(-1).norm())
    if init_e_h != 0.0 or init_e_z != 0.0:
        raise RuntimeError(f"initialization is not (u_0, D u_1): e_H={init_e_h} e_Z={init_e_z}")

    n_lead = n_steps + 1
    predicted = [_field(u_hat)]
    rows: list[dict] = []
    shell_e_h = []
    shell_e_z = []
    shell_c = []
    for n in range(n_steps):
        u_n = truth[n]
        u_next = truth[n + 1]
        z_true = downsample(u_next, step.ds)
        e_h = u_hat - u_n
        e_z = z_hat - z_true
        pred_h, _pred_z = step.step(u_n, z_true)
        epsilon = pred_h - u_next
        a = step.jvp_A(u_n, z_true, e_h)
        b = step.jvp_B(u_n, z_true, e_z)
        ell = a + b + epsilon
        u_roll, z_roll = step.step(u_hat, z_hat)
        delta = (u_roll - u_next) - ell

        a_np, b_np = _field(a), _field(b)
        eps_np, ell_np, delta_np = _field(epsilon), _field(ell), _field(delta)
        u_np1 = frames[n + 1]
        e_h_np, e_z_np = _field(e_h), _field(e_z)
        e_h_next = _field(u_roll - u_next)

        e_h_energy = metric.inner_G(a_np, a_np)
        e_z_energy = metric.inner_G(b_np, b_np)
        c_hz = metric.inner_G(a_np, b_np)
        e_h_term = 0.5 * e_h_energy
        e_z_term = 0.5 * e_z_energy
        eps_energy = metric.inner_G(eps_np, eps_np)
        ell_energy = 0.5 * metric.inner_G(ell_np, ell_np)
        quad_rhs = (
            e_h_term
            + e_z_term
            + c_hz
            + 0.5 * eps_energy
            + metric.inner_G(a_np, eps_np)
            + metric.inner_G(b_np, eps_np)
        )
        base = u_np1 + ell_np
        moved = base + delta_np
        delta_left = metric.energy(moved) - metric.energy(base)
        delta_right = metric.inner_G(base, delta_np) + 0.5 * metric.inner_G(delta_np, delta_np)
        e_lin = metric.energy(base)
        e_actual = metric.energy(frames[n + 1] + e_h_next)
        rho_denom = float(np.sqrt(max(e_h_energy, 0.0) * max(e_z_energy, 0.0)))
        eta_denom = e_h_term + e_z_term
        shell_e_h.append(metric.quad_shells(a_np))
        shell_e_z.append(metric.quad_shells(b_np))
        shell_c.append(metric.cross_shells(a_np, b_np))
        rows.append(
            {
                "E_H": e_h_term,
                "E_Z": e_z_term,
                "C_HZ": c_hz,
                "rho_HZ": ratio_or_nan(c_hz, rho_denom, DENOM_FLOOR),
                "eta_cancel": ratio_or_nan(-c_hz, eta_denom, DENOM_FLOOR),
                "cos_euclid": _cosine(a_np, b_np),
                "denom_rho": rho_denom,
                "denom_eta": eta_denom,
                "valid_rho": bool(np.isfinite(rho_denom) and rho_denom > DENOM_FLOOR),
                "valid_eta": bool(np.isfinite(eta_denom) and eta_denom > DENOM_FLOOR),
                "cross_u_a": metric.inner_G(u_np1, a_np),
                "cross_u_b": metric.inner_G(u_np1, b_np),
                "cross_u_eps": metric.inner_G(u_np1, eps_np),
                "E_linearized": e_lin,
                "quad_closure": ell_energy - quad_rhs,
                "quad_scale": max(abs(ell_energy), abs(quad_rhs)),
                "delta_closure": delta_left - delta_right,
                "delta_scale": max(abs(delta_left), abs(delta_right)),
                "abs_euclid_delta": _euclid(delta_np),
                "rel_euclid_delta": _euclid(delta_np) / max(_euclid(e_h_next), 1e-30),
                "abs_energy_delta": float(np.sqrt(max(metric.inner_G(delta_np, delta_np), 0.0))),
                "rel_energy_delta": float(np.sqrt(max(metric.inner_G(delta_np, delta_np), 0.0)))
                / max(float(np.sqrt(max(metric.inner_G(e_h_next, e_h_next), 0.0))), 1e-30),
                "abs_E_gap": abs(e_actual - e_lin),
                "rel_E_gap": abs(e_actual - e_lin) / max(abs(e_actual), 1e-30),
                "e_H_norm": _euclid(e_h_np),
                "e_Z_norm": _euclid(e_z_np),
                "epsilon_H_norm": _euclid(eps_np),
                "shell_E_H_closure": float(shell_e_h[-1].sum() - e_h_term),
                "shell_E_Z_closure": float(shell_e_z[-1].sum() - e_z_term),
                "shell_C_closure": float(shell_c[-1].sum() - c_hz),
            }
        )
        u_hat = u_roll.detach()
        z_hat = z_roll.detach()
        predicted.append(_field(u_hat))
        print(
            f"  step {n:02d}  ||e_H||={rows[-1]['e_H_norm']:.4e}  "
            f"C_HZ={rows[-1]['C_HZ']:.4e}  ||delta||={rows[-1]['abs_euclid_delta']:.4e}",
            flush=True,
        )

    lead_rows = []
    shell_ref, shell_pred, shell_l, shell_q = [], [], [], []
    for m in range(n_lead):
        split = metric.energy_split(frames[m], predicted[m])
        value = d_E(split["E_pred"], mu_E, sigma_E)
        split["d_E"] = value
        split["inside_band"] = bool(value <= D_E_THRESHOLD)
        shell_ref.append(metric.quad_shells(frames[m]))
        shell_pred.append(metric.quad_shells(predicted[m]))
        err = predicted[m] - frames[m]
        shell_l.append(metric.cross_shells(frames[m], err))
        shell_q.append(metric.quad_shells(err))
        split["shell_E_ref_closure"] = float(shell_ref[-1].sum() - split["E_ref"])
        split["shell_E_pred_closure"] = float(shell_pred[-1].sum() - split["E_pred"])
        split["shell_L_closure"] = float(shell_l[-1].sum() - split["L"])
        split["shell_Q_closure"] = float(shell_q[-1].sum() - split["Q"])
        lead_rows.append(split)

    d_values = np.array([row["d_E"] for row in lead_rows], dtype=np.float64)
    out = {
        "lead": np.arange(n_lead, dtype=np.int64),
        "step": np.arange(n_steps, dtype=np.int64),
        "init_e_H_norm": init_e_h,
        "init_e_Z_norm": init_e_z,
        "first_exit": first_exit(d_values),
        "shell_E_ref": np.stack(shell_ref),
        "shell_E_pred": np.stack(shell_pred),
        "shell_L": np.stack(shell_l),
        "shell_Q": np.stack(shell_q),
        "shell_E_H": np.stack(shell_e_h),
        "shell_E_Z": np.stack(shell_e_z),
        "shell_C_HZ": np.stack(shell_c),
    }
    for key in lead_rows[0]:
        if key == "inside_band":
            out[key] = np.array([row[key] for row in lead_rows], dtype=bool)
        else:
            out[key] = np.array([row[key] for row in lead_rows], dtype=np.float64)
    for key in rows[0]:
        if key.startswith("valid_"):
            out[key] = np.array([row[key] for row in rows], dtype=bool)
        else:
            out[key] = np.array([row[key] for row in rows], dtype=np.float64)
    out["closures_ok"] = window_closures_ok(out)
    return out


def window_closures_ok(result: dict) -> bool:
    if result["init_e_H_norm"] != 0.0 or result["init_e_Z_norm"] != 0.0:
        return False
    checks = [
        (result["closure"], np.maximum(np.abs(result["signed_energy_error"]), 1.0)),
        (result["quad_closure"], np.maximum(result["quad_scale"], 1.0)),
        (result["delta_closure"], np.maximum(result["delta_scale"], 1.0)),
        (result["shell_E_ref_closure"], np.maximum(np.abs(result["E_ref"]), 1.0)),
        (result["shell_E_pred_closure"], np.maximum(np.abs(result["E_pred"]), 1.0)),
        (result["shell_L_closure"], np.maximum(np.abs(result["L"]), 1.0)),
        (result["shell_Q_closure"], np.maximum(np.abs(result["Q"]), 1.0)),
        (result["shell_E_H_closure"], np.maximum(np.abs(result["E_H"]), 1.0)),
        (result["shell_E_Z_closure"], np.maximum(np.abs(result["E_Z"]), 1.0)),
        (result["shell_C_closure"], np.maximum(np.abs(result["C_HZ"]), 1.0)),
    ]
    for residual, scale in checks:
        if np.any(~np.isfinite(residual)):
            return False
        if np.any(np.abs(residual) > EXACT_ATOL + EXACT_RTOL * scale):
            return False
    return True


def stack_windows(results: list[dict]) -> dict:
    stacked = {}
    for key, value in results[0].items():
        if key in ("lead", "step"):
            stacked[key] = value
        elif np.ndim(value) == 0:
            stacked[key] = np.array([res[key] for res in results])
        else:
            stacked[key] = np.stack([res[key] for res in results], axis=0)
    stacked["closures_ok"] = np.array([bool(res["closures_ok"]) for res in results])
    return stacked


def probe_derivatives(step, frames: list[np.ndarray], eps_rel: float, seed: int) -> dict:
    device = step.device
    u = prepare_image(frames[0], device)
    z = downsample(prepare_image(frames[1], device), step.ds)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    v = torch.randn(u.shape, dtype=torch.float64, generator=generator).to(device)
    w = torch.randn(z.shape, dtype=torch.float64, generator=generator).to(device)
    cot_h = torch.randn(u.shape, dtype=torch.float64, generator=generator).to(device)
    jvp_a = step.jvp_A(u, z, v)
    jvp_b = step.jvp_B(u, z, w)
    fd_a = step.finite_difference_A(u, z, v, eps_rel)
    fd_b = step.finite_difference_B(u, z, w, eps_rel)
    back_a = step.vjp_A(u, z, cot_h)
    back_b = step.vjp_B(u, z, cot_h)
    adj_a_left = float(torch.sum(jvp_a * cot_h))
    adj_a_right = float(torch.sum(v * back_a))
    adj_b_left = float(torch.sum(jvp_b * cot_h))
    adj_b_right = float(torch.sum(w * back_b))
    report = {
        "fd_rel_A": relative_error(fd_a, jvp_a),
        "fd_rel_B": relative_error(fd_b, jvp_b),
        "adjoint_rel_A": abs(adj_a_left - adj_a_right) / max(abs(adj_a_left), abs(adj_a_right), 1e-30),
        "adjoint_rel_B": abs(adj_b_left - adj_b_right) / max(abs(adj_b_left), abs(adj_b_right), 1e-30),
        "fd_eps_rel": float(eps_rel),
    }
    report["derivative_ok"] = bool(
        report["fd_rel_A"] < 1e-3
        and report["fd_rel_B"] < 1e-3
        and report["adjoint_rel_A"] < 1e-6
        and report["adjoint_rel_B"] < 1e-6
    )
    return report


def _solver_triplets(operator, k: int, tol: float, maxiter: int, v0: np.ndarray):
    from scipy.sparse.linalg import svds

    n_small = min(int(operator.shape[0]), int(operator.shape[1]))
    ncv = min(max(2 * k + 1, 20), n_small - 1)
    if ncv <= k:
        raise ValueError(f"ncv={ncv} is not larger than k={k}")
    left, values, vh = svds(
        operator,
        k=k,
        ncv=ncv,
        which="LM",
        tol=tol,
        maxiter=maxiter,
        v0=np.asarray(v0, dtype=np.float64),
        return_singular_vectors=True,
    )
    solver_ok = True
    left, values, right = sort_singular_triplets(np.asarray(left, dtype=np.float64), np.asarray(values), np.asarray(vh, dtype=np.float64).T)
    residuals = []
    residuals_adj = []
    for i in range(values.size):
        image = np.asarray(operator.matvec(right[:, i]), dtype=np.float64)
        residuals.append(float(np.linalg.norm(image - values[i] * left[:, i]) / max(values[i], 1e-30)))
        if operator.rmatvec is not None:
            back = np.asarray(operator.rmatvec(left[:, i]), dtype=np.float64)
            residuals_adj.append(float(np.linalg.norm(back - values[i] * right[:, i]) / max(values[i], 1e-30)))
    residual = np.array(residuals, dtype=np.float64)
    residual_adj = np.array(residuals_adj, dtype=np.float64)
    converged = bool(solver_ok and residual.size == k and np.all(residual < 1e-3))
    if residual_adj.size:
        converged = bool(converged and np.all(residual_adj < 1e-3))
    return {
        "left": left,
        "values": values,
        "right": right,
        "residual": residual,
        "residual_adj": residual_adj,
        "ortho_left": orthonormality_error(left),
        "ortho_right": orthonormality_error(right),
        "degenerate": degeneracy_flags(values),
        "solver_ok": solver_ok,
        "converged": converged,
    }


def _linear_operator(shape, matvec, rmatvec):
    from scipy.sparse.linalg import LinearOperator

    return LinearOperator(
        shape,
        matvec=lambda x: np.asarray(matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64),
        rmatvec=lambda x: np.asarray(rmatvec(np.asarray(x, dtype=np.float64)), dtype=np.float64),
        dtype=np.float64,
    )


def teacher_state_at(step, frames: list[np.ndarray], target: int):
    """Free-rollout errors at input time `target`, with the teacher-forced base state."""
    device = step.device
    truth = [prepare_image(frame, device) for frame in frames]
    u_hat = truth[0].clone()
    z_hat = downsample(truth[1], step.ds)
    for n in range(target):
        u_roll, z_roll = step.step(u_hat, z_hat)
        u_hat = u_roll.detach()
        z_hat = z_roll.detach()
    u_n = truth[target]
    z_true = downsample(truth[target + 1], step.ds)
    e_h = u_hat - u_n
    e_z = z_hat - z_true
    return u_n, z_true, e_h, e_z


def analyze_directions(step, metric: EnergyMetric, frames, target: int, k: int, tol: float, maxiter: int, seed: int) -> dict:
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    u_n, z_true, e_h, e_z = teacher_state_at(step, frames, target)
    a = step.jvp_A(u_n, z_true, e_h)
    b = step.jvp_B(u_n, z_true, e_z)
    blocks = WeightedBlocks(step, metric, u_n, z_true)
    rng = np.random.default_rng(seed)
    op_a = _linear_operator(
        (blocks.n_out, blocks.n_fine), blocks.apply_A, blocks.apply_A_adjoint
    )
    op_b = _linear_operator(
        (blocks.n_out, blocks.n_latent), blocks.apply_B, blocks.apply_B_adjoint
    )
    op_m = _linear_operator(
        (blocks.n_fine, blocks.n_latent), blocks.apply_M_HZ, blocks.apply_M_HZ_adjoint
    )
    n_prod = blocks.n_fine + blocks.n_latent
    op_q = _linear_operator((n_prod, n_prod), blocks.apply_Q_E, blocks.apply_Q_E)
    ncv = min(max(2 * k + 1, 20), n_prod - 1)
    spec_a = _solver_triplets(op_a, k, tol, maxiter, rng.standard_normal(blocks.n_fine))
    spec_b = _solver_triplets(op_b, k, tol, maxiter, rng.standard_normal(blocks.n_latent))
    spec_m = _solver_triplets(op_m, k, tol, maxiter, rng.standard_normal(blocks.n_latent))
    try:
        evals, evecs = eigsh(op_q, k=k, which="LA", tol=tol, maxiter=maxiter, ncv=ncv)
        q_solver = True
    except ArpackNoConvergence as exc:
        evals = np.asarray(exc.eigenvalues, dtype=np.float64)
        evecs = np.asarray(exc.eigenvectors, dtype=np.float64)
        q_solver = False
    order = np.argsort(-np.asarray(evals, dtype=np.float64))
    evals = np.asarray(evals, dtype=np.float64)[order]
    evecs = np.asarray(evecs, dtype=np.float64)[:, order]
    q_resid = []
    for i in range(evals.size):
        image = blocks.apply_Q_E(evecs[:, i])
        q_resid.append(float(np.linalg.norm(image - evals[i] * evecs[:, i]) / max(abs(evals[i]), 1e-30)))
    q_resid = np.array(q_resid, dtype=np.float64)
    for i in range(evecs.shape[1]):
        idx = int(np.argmax(np.abs(evecs[:, i])))
        if evecs[idx, i] < 0.0:
            evecs[:, i] *= -1.0
    e_h_np, e_z_np = _field(e_h), _field(e_z)
    a_np, b_np = _field(a), _field(b)
    e_h_coord = metric.rms_coord_from_physical(e_h_np)
    e_z_coord = metric.rms_coord_from_physical(e_z_np)
    angles = principal_angles(spec_a["left"], spec_b["left"])
    fine_scale = np.sqrt(float(blocks.n_fine))
    latent_scale = np.sqrt(float(blocks.n_latent))
    return {
        "A_values": spec_a["values"],
        "B_values": spec_b["values"],
        "M_values": spec_m["values"],
        "Q_E_values": evals,
        "A_residual": spec_a["residual"],
        "A_residual_adj": spec_a["residual_adj"],
        "B_residual": spec_b["residual"],
        "B_residual_adj": spec_b["residual_adj"],
        "M_residual": spec_m["residual"],
        "M_residual_adj": spec_m["residual_adj"],
        "Q_E_residual": q_resid,
        "A_ortho_left": spec_a["ortho_left"],
        "A_ortho_right": spec_a["ortho_right"],
        "B_ortho_left": spec_b["ortho_left"],
        "B_ortho_right": spec_b["ortho_right"],
        "M_ortho_left": spec_m["ortho_left"],
        "M_ortho_right": spec_m["ortho_right"],
        "Q_E_ortho": orthonormality_error(evecs),
        "A_degenerate": spec_a["degenerate"],
        "B_degenerate": spec_b["degenerate"],
        "M_degenerate": spec_m["degenerate"],
        "Q_E_degenerate": degeneracy_flags(evals),
        "A_converged": spec_a["converged"],
        "B_converged": spec_b["converged"],
        "M_converged": spec_m["converged"],
        "Q_E_converged": bool(q_solver and q_resid.size == k and np.all(q_resid < 1e-3)),
        "principal_angles": angles,
        "fraction_e_H": subspace_fraction(spec_a["right"], e_h_coord),
        "fraction_e_Z": subspace_fraction(spec_b["right"], e_z_coord),
        "fraction_a": subspace_fraction(spec_a["left"], metric.pack(a_np)),
        "fraction_b": subspace_fraction(spec_b["left"], metric.pack(b_np)),
        "A_right_physical": np.stack(
            [fine_scale * spec_a["right"][:, i].reshape(metric.ny, metric.nx) for i in range(k)]
        ),
        "B_right_physical": np.stack(
            [latent_scale * spec_b["right"][:, i].reshape(blocks.latent_shape) for i in range(k)]
        ),
        "A_left_velocity": np.stack(
            [np.stack(metric.velocities_from_coord(spec_a["left"][:, i]), axis=0) for i in range(k)]
        ),
        "B_left_velocity": np.stack(
            [np.stack(metric.velocities_from_coord(spec_b["left"][:, i]), axis=0) for i in range(k)]
        ),
        "M_left_physical": np.stack(
            [fine_scale * spec_m["left"][:, i].reshape(metric.ny, metric.nx) for i in range(k)]
        ),
        "M_right_physical": np.stack(
            [latent_scale * spec_m["right"][:, i].reshape(blocks.latent_shape) for i in range(k)]
        ),
        "Q_E_fine_physical": np.stack(
            [fine_scale * evecs[: blocks.n_fine, i].reshape(metric.ny, metric.nx) for i in range(k)]
        ),
        "Q_E_latent_physical": np.stack(
            [latent_scale * evecs[blocks.n_fine :, i].reshape(blocks.latent_shape) for i in range(k)]
        ),
        "C_HZ": metric.inner_G(a_np, b_np),
        "input_scaling": "v_phys = sqrt(n_block) v_coord; unit coord has physical RMS 1",
        "output_scaling": "y_coord = pack(y); ||y_coord||^2 = <y, G y>",
    }


def code_identity(repo: Path) -> dict:
    digest = hashlib.sha256()
    for rel in SOURCE_FILES:
        path = repo / rel
        digest.update(rel.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    except Exception as exc:
        head = f"unavailable: {exc}"
        status = ""
    return {"git_head": head, "git_status": status, "source_sha256": digest.hexdigest()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def write_budget(path: Path, stacked: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in stacked.items()}
    payload["shell_k"] = np.arange(payload["shell_E_ref"].shape[-1], dtype=np.int64)
    payload["low_k_max"] = np.asarray(LOW_K_MAX)
    payload["mid_k_max"] = np.asarray(MID_K_MAX)
    payload["d_E_threshold"] = np.asarray(D_E_THRESHOLD)
    payload["mu_E"] = np.asarray(meta["reference"]["mu_E"])
    payload["sigma_E"] = np.asarray(meta["reference"]["sigma_E"])
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    (path.parent / "metadata.json").write_text(json.dumps(_jsonable(meta), indent=2) + "\n")


def _save_window(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in result.items()}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_window_result(path: Path) -> dict:
    loaded = np.load(path)
    result = {key: loaded[key] for key in loaded.files}
    for key in ("closures_ok",):
        if key in result:
            result[key] = bool(np.asarray(result[key]).reshape(-1)[0])
    for key in ("first_exit", "sample_id", "window_index"):
        if key in result and np.ndim(result[key]) == 0:
            result[key] = int(np.asarray(result[key]))
    return result


def run_budget_stage(args, step, metric, samples, meta) -> bool:
    indexes = _ints(args.windows)
    if not indexes:
        indexes = list(range(len(samples)))
    results = []
    manifest = []
    for index in indexes:
        entry = samples[index]
        dest = args.save_dir / "windows" / f"window_{index:02d}.npz"
        print(f"[window {index}] sample={entry.get('sample')} start={entry.get('start_index')}", flush=True)
        frames, times, stems = load_window(entry, args.dt, metric.nx)
        if len(frames) < args.n_steps + 2:
            raise RuntimeError(f"window {index} has {len(frames)} frames, need {args.n_steps + 2}")
        if dest.is_file():
            print(f"[window {index}] resume {dest}", flush=True)
            result = _load_window_result(dest)
        else:
            result = analyze_window(
                step, metric, frames, args.n_steps, meta["reference"]["mu_E"], meta["reference"]["sigma_E"]
            )
            result["sample_id"] = int(entry.get("sample", index))
            result["window_index"] = int(index)
            _save_window(dest, result)
        results.append(result)
        manifest.append(
            {
                "window_index": int(index),
                "sample": int(entry.get("sample", index)),
                "start_index": int(entry.get("start_index", -1)),
                "stems": stems,
                "times": times,
                "closures_ok": bool(result["closures_ok"]),
                "first_exit": int(result["first_exit"]),
            }
        )
        print(f"[window {index}] closures_ok={result['closures_ok']} first_exit={result['first_exit']}", flush=True)
    stacked = stack_windows(results)
    meta["windows"] = manifest
    meta["closures_ok"] = bool(np.all(stacked["closures_ok"]))
    if (args.save_dir / "budget.npz").is_file():
        raise RuntimeError(f"refusing to overwrite {args.save_dir / 'budget.npz'}")
    write_budget(args.save_dir / "budget.npz", stacked, meta)
    shutil.copyfile(args.sample_index, args.save_dir / "sample_index.json")
    from plot_l2_energy import plot_budget

    plot_budget(args.save_dir / "budget.npz", args.save_dir / "plots")
    return bool(meta["closures_ok"])


def run_svd_stage(args, step, metric, samples, meta) -> bool:
    if args.budget_npz is None or not Path(args.budget_npz).is_file():
        raise RuntimeError("svd stage requires --budget-npz from a finished budget run")
    budget = np.load(args.budget_npz)
    if "closures_ok" not in budget or not bool(np.all(budget["closures_ok"])):
        raise RuntimeError(f"budget closures did not pass: {args.budget_npz}")
    windows = _ints(args.svd_windows)
    steps = _ints(args.svd_steps)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    summaries = []
    for window in windows:
        frames, times, stems = load_window(samples[window], args.dt, metric.nx)
        for target in steps:
            if target < 0 or target + 2 > len(frames):
                raise RuntimeError(f"step {target} is outside window {window}")
            dest = args.save_dir / f"state_w{window}_s{target}.npz"
            if dest.is_file():
                existing = np.load(dest)
                if bool(np.asarray(existing["converged"]).reshape(-1)[0]):
                    print(f"[svd] keep existing {dest.name}", flush=True)
                    summaries.append({"window": window, "step": target, "converged": True, "skipped": True})
                    continue
            print(f"[svd] window {window} step {target}", flush=True)
            result = analyze_directions(
                step,
                metric,
                frames,
                target,
                args.k_singular,
                args.svd_tol,
                args.svd_maxiter,
                seed=args.seed + 1000 * window + target,
            )
            converged = bool(
                result["A_converged"]
                and result["B_converged"]
                and result["M_converged"]
                and result["Q_E_converged"]
            )
            all_ok = all_ok and converged
            payload = {key: np.asarray(value) for key, value in result.items()}
            payload["window"] = np.asarray(window)
            payload["step"] = np.asarray(target)
            payload["stem"] = np.asarray(stems[target])
            payload["time"] = np.asarray(times[target])
            payload["converged"] = np.asarray(converged)
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, dest)
            summaries.append(
                {
                    "window": window,
                    "step": target,
                    "stem": stems[target],
                    "time": times[target],
                    "converged": converged,
                    "A_values": result["A_values"].tolist(),
                    "B_values": result["B_values"].tolist(),
                    "M_values": result["M_values"].tolist(),
                    "Q_E_values": result["Q_E_values"].tolist(),
                    "principal_angles": result["principal_angles"].tolist(),
                    "fraction_e_H": result["fraction_e_H"],
                    "fraction_e_Z": result["fraction_e_Z"],
                    "fraction_a": result["fraction_a"],
                    "fraction_b": result["fraction_b"],
                    "C_HZ": result["C_HZ"],
                }
            )
    meta["svd"] = summaries
    meta["closures_ok"] = bool(all_ok)
    (args.save_dir / "metadata.json").write_text(json.dumps(_jsonable(meta), indent=2) + "\n")
    from plot_l2_energy import plot_svd

    plot_svd(args.save_dir, args.save_dir / "plots")
    return bool(all_ok)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "smoke":
        args.n_steps = 4
        args.windows = "0"
    lock_precision()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    metric = EnergyMetric(256, 256)
    reference = verify_reference_cache(Path(args.ref_cache), metric)
    _model, step, checkpoint_info = load_l2_checkpoint(args.checkpoint, device, args.norm)
    sample_payload = json.loads(Path(args.sample_index).read_text())
    samples = sample_payload["samples"]
    if len(samples) != 20 and args.stage != "smoke":
        raise RuntimeError(f"expected 20 windows, found {len(samples)}")
    meta = {
        "stage": args.stage,
        "checkpoint": checkpoint_info,
        "reference": reference,
        "dt": args.dt,
        "n_steps": args.n_steps,
        "norm": args.norm,
        "staggered": "input (u_n, D u_{n+1}) -> output (u_{n+1}, z_{n+2}); init (u_0, D u_1)",
        "blocks": "A=dF_H/du, B=dF_H/dz, C=dF_Z/du, K=dF_Z/dz; only A and B are applied; D is average pooling",
        "mean_mode": "invKsq[0,0]=0; constant vorticity has zero kinetic energy",
        "nyquist": "invKsq keeps the Nyquist column; irfft drops its imaginary velocity, matching kinetic_energy_torch",
        "inner_product": "<f,G g> = mean(u_f u_g + v_f v_g) after that projection; E = 0.5 <f, G f>",
        "bands": {"low": f"0 < |k| <= {LOW_K_MAX}", "middle": f"{LOW_K_MAX} < |k| <= {MID_K_MAX}", "high": f"|k| > {MID_K_MAX}"},
        "input_scaling": "v_phys = sqrt(n_block) v_coord",
        "output_scaling": "pack stacks Parseval coordinates of the two velocity components; ||pack(y)||^2 = <y, G y>",
        "code": code_identity(_REPO),
        "sample_index": str(args.sample_index),
        "d_E_threshold": D_E_THRESHOLD,
        "negative_C_HZ": "quadratic cancellation only; not a causal claim about the d_E band",
    }
    ok = True
    if args.stage in ("smoke", "budget"):
        if args.stage == "smoke":
            frames, _times, _stems = load_window(samples[0], args.dt, metric.nx)
            probe = probe_derivatives(step, frames, args.fd_eps, args.seed)
            meta["derivative_probe"] = probe
            print(f"[probe] {probe}", flush=True)
            ok = ok and bool(probe["derivative_ok"])
        ok = run_budget_stage(args, step, metric, samples, meta) and ok
    else:
        ok = run_svd_stage(args, step, metric, samples, meta) and ok
    print(f"[done] stage={args.stage} ok={ok} save_dir={args.save_dir}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
