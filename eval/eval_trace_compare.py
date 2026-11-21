#!/usr/bin/env python3
"""Trace and two-step energy diagnostics for four HINE settings.

Jacobians are taken at true inputs. L=2 uses (u_n, D u_{n+1}) and t=1.
L=1 is the complete fine map. Network outputs replace the state; identity
is subtracted once in the trace diagnostics. Latent diagnostics are NaN
for L=1 and are not treated as zeros.

The evaluated dots are the first ``--n-frames`` loader-kept frames
(stems divisible by 5). Later frames are lookahead for two-step products
and finite-horizon latent products. They are not extra dots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for _path in (_REPO, _EVAL):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eval_ablation_eigs import load_ablation
from eval_jacobian_manual_mse import load_kept_paths
from eval_l2_stability_task1 import _check_spacing, _mat_frame
from l2_fd_jvp import apply_D, jvp_both_u, jvp_both_uz, jvp_both_z, lock_precision, make_point, model_step
from eval_jacobian_manual_mse_plus import load_model as load_plus
from trace_compare_ops import (
    NORM_FLOOR,
    directional_energy,
    directional_quadratic,
    directional_two_step,
    finite_difference_check,
    mean_over_dim,
    project_blocks,
    rademacher,
    square_probe_metrics,
    two_step_energy,
)

K4_EVAL = "/pscratch/sd/d/divyam/training_data/paper_nx256_from10000/eval"
K5_EVAL = "/pscratch/sd/d/divyam/training_data/paper_nx256_fkx5/eval"
SETTINGS = {
    "l1_k4": {
        "label": "L=1 · k_f=4",
        "level": 1,
        "fkx": 4,
        "checkpoint": "/pscratch/sd/d/divyam/hine_runs/ablation_l1/checkpoint-last.pth",
        "eval_dir": K4_EVAL,
    },
    "l2_k4": {
        "label": "L=2 · k_f=4",
        "level": 2,
        "fkx": 4,
        "checkpoint": "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
        "eval_dir": K4_EVAL,
    },
    "l1_k5": {
        "label": "L=1 · k_f=5",
        "level": 1,
        "fkx": 5,
        "checkpoint": "/pscratch/sd/d/divyam/hine_runs/k5/l1/checkpoint-last.pth",
        "eval_dir": K5_EVAL,
    },
    "l2_k5": {
        "label": "L=2 · k_f=5",
        "level": 2,
        "fkx": 5,
        "checkpoint": "/pscratch/sd/d/divyam/hine_runs/k5/l2/checkpoint-last.pth",
        "eval_dir": K5_EVAL,
    },
}
ORDER = ("l1_k4", "l2_k4", "l1_k5", "l2_k5")
SCALAR_KEYS = (
    "tau",
    "tau_se",
    "energy",
    "energy_se",
    "two_tau",
    "shift_over_d",
    "identity_residual",
    "tau_K",
    "tau_K_se",
    "energy_K",
    "energy_K_se",
    "tau_BC",
    "tau_BC_se",
    "tau_loop_next",
    "tau_loop_next_se",
    "b_gain",
    "b_gain_se",
    "direct",
    "direct_se",
    "loop",
    "loop_se",
    "interference",
    "interference_se",
    "total",
    "total_se",
    "loop_net",
    "twostep_residual",
    "scale_P",
    "scale_P_se",
    "scale_Q",
    "scale_Q_se",
    "dir_tau",
    "dir_energy",
    "dir_bc",
    "dir_loop_next",
    "dir_k",
    "dir_direct",
    "dir_loop",
    "dir_interference",
    "dir_total",
    "dir_net",
    "eps_u_norm",
    "eps_z_norm",
    "e_norm",
    "eta_norm",
    "remainder_e",
    "remainder_eta",
    "I_norm",
    "R_value",
    "rho_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="all", choices=("all",) + ORDER)
    parser.add_argument("--checkpoint", default="", help="Override the checkpoint for a single setting")
    parser.add_argument("--eval-dir", default="", help="Override the eval directory for a single setting")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-horizons", default="1,2,4,8")
    parser.add_argument("--model-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--save-dir", default="/pscratch/sd/d/divyam/hine_runs/trace_compare/n20_loader")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--norm", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--identity-tol", type=float, default=1e-5)
    parser.add_argument("--fd-tol", type=float, default=5e-2)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_frames = 2
        args.n_probes = 2
        args.fd_check = True
    if args.checkpoint and args.setting == "all":
        raise ValueError("--checkpoint requires a single --setting")
    if args.eval_dir and args.setting == "all":
        raise ValueError("--eval-dir requires a single --setting")
    if args.n_frames < 1 or args.n_probes < 1:
        raise ValueError("frame and probe counts must be positive")
    return args


def _horizons(text: str) -> list[int]:
    values = []
    for piece in text.split(","):
        if not piece.strip():
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"memory horizon must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one memory horizon is required")
    return sorted(set(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _lock() -> None:
    lock_precision()
    if torch.is_autocast_enabled():
        raise RuntimeError("trace comparison requires autocast to be off")


def _field(array: np.ndarray, device: torch.device, dtype: torch.dtype, norm: float) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        / float(norm)
    )


def _flat(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().reshape(-1).double().cpu().numpy()


def _like(flat: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(flat))
        .to(device=reference.device, dtype=reference.dtype)
        .reshape_as(reference)
    )


def _norm(array: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(array, dtype=np.float64).reshape(-1)))


def load_setting_model(spec: dict, device: torch.device, dtype: torch.dtype):
    """Load one checkpoint and reject a mismatched architecture."""
    path = spec["checkpoint"]
    if spec["level"] == 1:
        model, ema = load_ablation(path, device)
        ablation = str(getattr(model, "ablation", ""))
        aux = int(getattr(model, "aux_levels", -1))
        if ablation != "l1" or aux != 0 or int(getattr(ema, "num_aux", -1)) != 0:
            raise RuntimeError(
                f"{path} is not a plain L=1 ablation "
                f"(ablation={ablation}, aux_levels={aux}, num_aux={getattr(ema, 'num_aux', None)})"
            )
        if int(model.img_size) != 256:
            raise RuntimeError(f"{path} img_size {model.img_size} != 256")
    else:
        model, ema = load_plus(path, device)
        latent = [int(value) for value in model.latent_res]
        if int(model.num_coarse) != 1 or latent != [32] or int(model.img_size) != 256:
            raise RuntimeError(
                f"{path} is not L=2 with latent 32 "
                f"(num_coarse={getattr(model, 'num_coarse', None)}, latent_res={latent})"
            )
        if int(model.ds_factors[0]) != 8:
            raise RuntimeError(f"{path} ds_factors[0]={model.ds_factors[0]} != 8")
    if dtype == torch.float64:
        model.double()
    elif dtype != torch.float32:
        raise RuntimeError(f"unsupported dtype {dtype}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ema = model.ema_vit
    weight_dtype = next(ema.parameters()).dtype
    if weight_dtype != dtype:
        raise RuntimeError(f"EMA dtype {weight_dtype} != requested {dtype}")
    return model, ema


def load_frames(eval_dir: str, n_frames: int, extra: int, dt: float):
    kept = load_kept_paths(eval_dir, 5, align="mod")
    need = n_frames + extra
    if len(kept) < need:
        raise RuntimeError(f"{eval_dir} has {len(kept)} kept frames, need {need}")
    frames, stems, times = [], [], []
    for stem, path in kept[:need]:
        omega, stamp = _mat_frame(path)
        frames.append(np.asarray(omega))
        stems.append(int(stem))
        times.append(float(stamp))
    _check_spacing(times, dt, eval_dir)
    return frames, stems, times


class Operator:
    """JVP closures at one teacher-forced state. L=1 leaves latent maps empty."""

    def __init__(self, ema, level: int, point, device, dtype, norm: float, frame, side: int, factor: int):
        self.ema = ema
        self.level = level
        self.point = point
        self.device = device
        self.dtype = dtype
        self.norm = float(norm)
        self.frame = frame
        self.side = side
        self.factor = factor
        self.u = frame if level == 1 else point.u
        self.fine_dim = int(self.u.numel())

    def _step_l2(self, u, z):
        return model_step(self.ema, u, z, self.point.t)

    def fine_forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.level == 1:
            return self.ema(state)
        high, _coarse = self._step_l2(state, self.point.z)
        return high

    def jvp_fine(self, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.level == 1:
            _value, tangent = torch.autograd.functional.jvp(
                lambda state: self.ema(state),
                (self.u,),
                (direction,),
            )
            return tangent.detach(), None
        fine, latent = jvp_both_u(self._step_l2, self.point, direction)
        return fine.detach(), latent.detach()

    def jvp_latent(self, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.level != 2:
            raise RuntimeError("L=1 has no latent Jacobian")
        fine, latent = jvp_both_z(self._step_l2, self.point, direction)
        return fine.detach(), latent.detach()

    def jvp_mixed(self, fine_direction: torch.Tensor, latent_direction: torch.Tensor):
        fine, latent = jvp_both_uz(self._step_l2, self.point, fine_direction, latent_direction)
        return fine.detach(), latent.detach()

    def teacher_residual(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.level == 1:
            prediction = self.ema(self.u).detach()
            target = _field(self.frame_next, self.device, self.dtype, self.norm)
            return _flat(prediction - target), None
        fine, latent = model_step(self.ema, self.point.u, self.point.z, self.point.t)
        return _flat(fine.detach() - self.point.u_next), _flat(latent.detach() - self.point.z_next)

    def set_next_frame(self, frame_next: np.ndarray) -> None:
        self.frame_next = frame_next


def _build_operators(ema, model, level, frames, device, dtype, norm: float) -> list[Operator]:
    operators = []
    side = int(frames[0].shape[-1])
    factor = 8 if level == 1 else int(model.ds_factors[0])
    last = len(frames) - 2
    for index in range(last):
        if level == 1:
            operator = Operator(ema, 1, None, device, dtype, norm, _field(frames[index], device, dtype, norm), side, factor)
        else:
            point = make_point(frames[index : index + 3], model, device, dtype, norm)
            operator = Operator(ema, 2, point, device, dtype, norm, point.u, side, factor)
        operator.set_next_frame(frames[index + 1])
        operators.append(operator)
    return operators


def _apply_chain(operators: list[Operator], start: int, direction: torch.Tensor, steps: int, frozen: bool) -> torch.Tensor | None:
    current = direction
    for offset in range(steps):
        index = start if frozen else start + offset
        if index >= len(operators) or operators[index].level != 2:
            return None
        _fine, current = operators[index].jvp_latent(current)
        current = current.detach()
    return current


def _nan_latent() -> dict:
    return {
        "tau_K": float("nan"),
        "tau_K_se": float("nan"),
        "energy_K": float("nan"),
        "energy_K_se": float("nan"),
        "tau_BC": float("nan"),
        "tau_BC_se": float("nan"),
        "tau_loop_next": float("nan"),
        "tau_loop_next_se": float("nan"),
        "b_gain": float("nan"),
        "b_gain_se": float("nan"),
        "loop": float("nan"),
        "loop_se": float("nan"),
        "interference": float("nan"),
        "interference_se": float("nan"),
        "loop_net": float("nan"),
        "twostep_residual": float("nan"),
        "dir_bc": float("nan"),
        "dir_loop_next": float("nan"),
        "dir_k": float("nan"),
        "dir_loop": float("nan"),
        "dir_interference": float("nan"),
        "dir_net": float("nan"),
        "eps_z_norm": float("nan"),
        "eta_norm": float("nan"),
        "remainder_eta": float("nan"),
    }


def evaluate_index(
    operators: list[Operator],
    index: int,
    probes: int,
    seed: int,
    horizons: list[int],
    e_model: np.ndarray | None,
    eta_model: np.ndarray | None,
    e_next: np.ndarray | None,
    eta_next: np.ndarray | None,
    norm: float,
) -> dict:
    """Probe-averaged diagnostics at one true state. Individual probes are not returned."""
    operator = operators[index]
    fine_dim = operator.fine_dim
    stem = int(operator.point.stems[0]) if operator.level == 2 and getattr(operator.point, "stems", None) else 0
    rng = np.random.default_rng(int(seed) + int(index) + 10007)
    fine_probe = rademacher(fine_dim, probes, rng)
    dots, images = [], []
    scale_p, scale_q = [], []
    bc_samples, loop_samples = [], []
    direct_samples, loop_energy_samples, cross_samples = [], [], []
    rank_p = (operator.side // operator.factor) ** 2
    rank_q = operator.side ** 2 - rank_p
    next_operator = operators[index + 1] if index + 1 < len(operators) else None

    for row in fine_probe:
        direction = _like(row, operator.u)
        fine_image, latent_image = operator.jvp_fine(direction)
        flat_image = _flat(fine_image)
        dots.append(float(np.dot(row, flat_image)))
        images.append(float(np.dot(flat_image, flat_image)))
        projected, complement = project_blocks(row, operator.side, operator.factor)
        p_image, _p_latent = operator.jvp_fine(_like(projected, operator.u))
        q_image, _q_latent = operator.jvp_fine(_like(complement, operator.u))
        scale_p.append(float(np.dot(projected, _flat(p_image))))
        scale_q.append(float(np.dot(complement, _flat(q_image))))
        if operator.level == 2 and latent_image is not None:
            response, _unused = operator.jvp_latent(latent_image)
            bc_samples.append(float(np.dot(row, _flat(response))))
            if next_operator is not None:
                direct_image, _direct_latent = next_operator.jvp_fine(fine_image)
                loop_image, _loop_latent = next_operator.jvp_latent(latent_image)
                direct_flat = _flat(direct_image)
                loop_flat = _flat(loop_image)
                direct_samples.append(float(np.dot(direct_flat, direct_flat)))
                loop_energy_samples.append(float(np.dot(loop_flat, loop_flat)))
                cross_samples.append(float(np.dot(direct_flat, loop_flat)))
                loop_samples.append(float(np.dot(row, loop_flat)))
        elif next_operator is not None:
            direct_image, _none = next_operator.jvp_fine(fine_image)
            direct_flat = _flat(direct_image)
            direct_samples.append(float(np.dot(direct_flat, direct_flat)))

    square = square_probe_metrics(np.asarray(dots), np.asarray(images), fine_dim)
    scale_p_metrics = mean_over_dim(np.asarray(scale_p), rank_p)
    scale_q_metrics = mean_over_dim(np.asarray(scale_q), rank_q)
    out = {
        "tau": square["tau"],
        "tau_se": square["tau_se"],
        "energy": square["energy"],
        "energy_se": square["energy_se"],
        "two_tau": square["two_tau"],
        "shift_over_d": square["shift_over_d"],
        "identity_residual": square["identity_residual"],
        "scale_P": scale_p_metrics["mean"],
        "scale_P_se": scale_p_metrics["se"],
        "scale_Q": scale_q_metrics["mean"],
        "scale_Q_se": scale_q_metrics["se"],
        "stem_seed_offset": stem,
    }
    out.update(_nan_latent())
    if direct_samples:
        if operator.level == 1:
            direct_metrics = mean_over_dim(np.asarray(direct_samples), fine_dim)
            out["direct"] = direct_metrics["mean"]
            out["direct_se"] = direct_metrics["se"]
            out["total"] = direct_metrics["mean"]
            out["total_se"] = direct_metrics["se"]
        elif loop_energy_samples:
            split = two_step_energy(
                np.asarray(direct_samples),
                np.asarray(loop_energy_samples),
                np.asarray(cross_samples),
                fine_dim,
            )
            out.update(
                {
                    "direct": split["direct"],
                    "direct_se": split["direct_se"],
                    "loop": split["loop"],
                    "loop_se": split["loop_se"],
                    "interference": split["interference"],
                    "interference_se": split["interference_se"],
                    "total": split["total"],
                    "total_se": split["total_se"],
                    "loop_net": split["loop_net"],
                    "twostep_residual": split["identity_residual"],
                }
            )
            loop_trace = mean_over_dim(np.asarray(loop_samples), fine_dim)
            out["tau_loop_next"] = loop_trace["mean"]
            out["tau_loop_next_se"] = loop_trace["se"]
    if bc_samples:
        bc_trace = mean_over_dim(np.asarray(bc_samples), fine_dim)
        out["tau_BC"] = bc_trace["mean"]
        out["tau_BC_se"] = bc_trace["se"]
    if operator.level == 2:
        latent_dim = int(operator.point.z.numel())
        latent_rng = np.random.default_rng(int(seed) + int(index) + 20011)
        latent_probe = rademacher(latent_dim, probes, latent_rng)
        k_dots, k_images, b_images = [], [], []
        horizon_frozen = {horizon: [] for horizon in horizons}
        horizon_varying = {horizon: [] for horizon in horizons}
        h_max = max(horizons)
        for row in latent_probe:
            direction = _like(row, operator.point.z)
            fine_response, latent_response = operator.jvp_latent(direction)
            k_flat = _flat(latent_response)
            k_dots.append(float(np.dot(row, k_flat)))
            k_images.append(float(np.dot(k_flat, k_flat)))
            b_flat = _flat(fine_response)
            b_images.append(float(np.dot(b_flat, b_flat)))
            current = latent_response
            for step in range(2, h_max + 1):
                _fine, current = operator.jvp_latent(current)
                current = current.detach()
                if step in horizon_frozen:
                    flat_current = _flat(current)
                    horizon_frozen[step].append(float(np.dot(flat_current, flat_current)))
            if 1 in horizon_frozen:
                horizon_frozen[1].append(float(np.dot(k_flat, k_flat)))
            current = latent_response
            for step in range(2, h_max + 1):
                if index + step - 1 >= len(operators):
                    break
                _fine, current = operators[index + step - 1].jvp_latent(current)
                current = current.detach()
                if step in horizon_varying and index + step - 1 < len(operators):
                    flat_current = _flat(current)
                    horizon_varying[step].append(float(np.dot(flat_current, flat_current)))
            if 1 in horizon_varying:
                horizon_varying[1].append(float(np.dot(k_flat, k_flat)))
        k_metrics = square_probe_metrics(np.asarray(k_dots), np.asarray(k_images), latent_dim)
        b_metrics = mean_over_dim(np.asarray(b_images), latent_dim)
        out["tau_K"] = k_metrics["tau"]
        out["tau_K_se"] = k_metrics["tau_se"]
        out["energy_K"] = k_metrics["energy"]
        out["energy_K_se"] = k_metrics["energy_se"]
        out["b_gain"] = b_metrics["mean"]
        out["b_gain_se"] = b_metrics["se"]
        out["memory_frozen"] = np.asarray(
            [
                mean_over_dim(np.asarray(horizon_frozen[horizon]), latent_dim)["mean"]
                if len(horizon_frozen[horizon]) == probes
                else float("nan")
                for horizon in horizons
            ],
            dtype=np.float64,
        )
        out["memory_varying"] = np.asarray(
            [
                mean_over_dim(np.asarray(horizon_varying[horizon]), latent_dim)["mean"]
                if len(horizon_varying[horizon]) == probes
                else float("nan")
                for horizon in horizons
            ],
            dtype=np.float64,
        )
    else:
        out["memory_frozen"] = np.full(len(horizons), np.nan)
        out["memory_varying"] = np.full(len(horizons), np.nan)

    if e_model is not None:
        e_image, e_latent = operator.jvp_fine(_like(e_model, operator.u))
        out["dir_tau"] = directional_quadratic(_flat(e_image), e_model, subtract_identity=True)
        out["dir_energy"] = directional_energy(_flat(e_image), e_model)
        if operator.level == 2 and e_latent is not None:
            bc_image, _bc_latent = operator.jvp_latent(e_latent)
            out["dir_bc"] = directional_quadratic(_flat(bc_image), e_model, subtract_identity=False)
            if next_operator is not None:
                direct_image, _unused = next_operator.jvp_fine(e_image)
                loop_image, _unused = next_operator.jvp_latent(e_latent)
                split = directional_two_step(_flat(direct_image), _flat(loop_image), e_model)
                out["dir_direct"] = split["direct"]
                out["dir_loop"] = split["loop"]
                out["dir_interference"] = split["interference"]
                out["dir_total"] = split["total"]
                out["dir_net"] = split["loop_net"]
                out["dir_loop_next"] = directional_quadratic(_flat(loop_image), e_model, subtract_identity=False)
            if eta_model is not None:
                _b_eta, k_eta = operator.jvp_latent(_like(eta_model, operator.point.z))
                out["dir_k"] = directional_quadratic(_flat(k_eta), eta_model, subtract_identity=True)
                if e_next is not None and eta_next is not None:
                    mixed_fine, mixed_latent = operator.jvp_mixed(
                        _like(e_model, operator.u),
                        _like(eta_model, operator.point.z),
                    )
                    eps_u, eps_z = operator.teacher_residual()
                    predicted_e = _flat(mixed_fine) + eps_u
                    predicted_eta = _flat(mixed_latent) + eps_z
                    out["remainder_e"] = _norm(e_next - predicted_e) * norm
                    out["remainder_eta"] = _norm(eta_next - predicted_eta) * norm
        elif next_operator is not None:
            direct_image, _none = next_operator.jvp_fine(e_image)
            out["dir_direct"] = directional_energy(_flat(direct_image), e_model) + 1.0
            out["dir_total"] = out["dir_direct"]
            if e_next is not None:
                eps_u, _none = operator.teacher_residual()
                predicted_e = _flat(e_image) + eps_u
                out["remainder_e"] = _norm(e_next - predicted_e) * norm
    for key in SCALAR_KEYS:
        out.setdefault(key, float("nan"))
    return out


def rollout_errors(operators: list[Operator], n_frames: int, norm: float) -> dict[str, np.ndarray]:
    """Free rollout from the true first state, plus teacher-forced residuals.

    L=2 starts at (u_0, D u_1). Retention uses the existing fine definitions
    I_n = ||sum_{i<n} eps^u_i|| and R_n = ||e_n|| - I_n. rho_n is the vector
    difference, not R_n.
    """
    level = operators[0].level
    e_model = [np.zeros(operators[0].fine_dim, dtype=np.float64)]
    eta_model: list[np.ndarray | None]
    if level == 1:
        eta_model = [None] * n_frames
    else:
        eta_model = [np.zeros(int(operators[0].point.z.numel()), dtype=np.float64)]
    eps_u, eps_z = [], []
    high = operators[0].u.detach().clone()
    latent = None if level == 1 else operators[0].point.z.detach().clone()
    steps = n_frames - 1
    for index in range(steps):
        eps_fine, eps_latent = operators[index].teacher_residual()
        eps_u.append(eps_fine)
        eps_z.append(eps_latent)
        if level == 1:
            high = operators[index].ema(high).detach()
            target = _field(
                operators[index].frame_next,
                high.device,
                high.dtype,
                operators[index].norm,
            )
            e_model.append(_flat(high - target))
        else:
            high, latent = model_step(operators[index].ema, high, latent, operators[index].point.t)
            high = high.detach()
            latent = latent.detach()
            e_model.append(_flat(high - operators[index].point.u_next))
            eta_model.append(_flat(latent - operators[index].point.z_next))
    # Residual at the last evaluated state, not included in the retention sum.
    last_eps_u, last_eps_z = operators[n_frames - 1].teacher_residual()
    physical = float(norm)
    e_norm = np.asarray([_norm(item) * physical for item in e_model], dtype=np.float64)
    eta_norm = np.full(n_frames, np.nan)
    if level == 2:
        eta_norm = np.asarray([_norm(item) * physical for item in eta_model], dtype=np.float64)
    eps_u_norm = np.asarray([_norm(item) * physical for item in eps_u + [last_eps_u]], dtype=np.float64)
    eps_z_norm = np.full(n_frames, np.nan)
    if level == 2:
        eps_z_norm = np.asarray(
            [_norm(item) * physical for item in eps_z + [last_eps_z]],
            dtype=np.float64,
        )
    i_norm = np.zeros(n_frames)
    r_value = np.zeros(n_frames)
    rho_norm = np.zeros(n_frames)
    running = np.zeros_like(e_model[0])
    for index in range(n_frames):
        i_norm[index] = _norm(running) * physical
        r_value[index] = e_norm[index] - i_norm[index]
        rho_norm[index] = _norm(e_model[index] - running) * physical
        if index < len(eps_u):
            running = running + eps_u[index]
    return {
        "e_model": e_model,
        "eta_model": eta_model,
        "e_norm": e_norm,
        "eta_norm": eta_norm,
        "eps_u_norm": eps_u_norm,
        "eps_z_norm": eps_z_norm,
        "I_norm": i_norm,
        "R_value": r_value,
        "rho_norm": rho_norm,
    }


def _record_stems(operators, stems: list[int]) -> None:
    for index, operator in enumerate(operators):
        if operator.level == 2:
            operator.point.stems = (stems[index], stems[index + 1], stems[index + 2])


def fd_gate(operator: Operator, seed: int, tolerance: float) -> dict:
    rng = np.random.default_rng(seed + 17)
    direction_np = rng.normal(size=operator.fine_dim)
    direction_np /= max(np.linalg.norm(direction_np), NORM_FLOOR)
    direction = _like(direction_np, operator.u)
    image, _latent = operator.jvp_fine(direction)
    report = finite_difference_check(operator.fine_forward, operator.u, direction, image)
    report["ok"] = bool(report["best_relative"] <= tolerance)
    if not report["ok"]:
        raise RuntimeError(
            f"finite-difference JVP check failed: relative {report['best_relative']:.3e} > {tolerance}"
        )
    return report


def _frame_path(directory: Path, index: int, stem: int) -> Path:
    return directory / f"frame_{index:04d}_{stem}.npz"


def _save_frame(path: Path, row: dict, stem: int, index: int) -> None:
    payload = {key: np.asarray(row.get(key, np.nan), dtype=np.float64) for key in SCALAR_KEYS}
    payload["memory_frozen"] = np.asarray(row["memory_frozen"], dtype=np.float64)
    payload["memory_varying"] = np.asarray(row["memory_varying"], dtype=np.float64)
    payload["stem"] = np.asarray(stem, dtype=np.int64)
    payload["index"] = np.asarray(index, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_frame(path: Path) -> dict:
    stored = np.load(path)
    row = {
        key: float(np.asarray(stored[key]).reshape(-1)[0]) if key in stored else float("nan")
        for key in SCALAR_KEYS
    }
    row["memory_frozen"] = np.asarray(stored["memory_frozen"], dtype=np.float64)
    row["memory_varying"] = np.asarray(stored["memory_varying"], dtype=np.float64)
    return row


def _stack(rows: list[dict], stems: list[int], horizons: list[int]) -> dict:
    stacked = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in SCALAR_KEYS}
    stacked["stem"] = np.asarray(stems, dtype=np.int64)
    stacked["trajectory_id"] = np.asarray(["loader-0"] * len(rows))
    stacked["memory_frozen"] = np.stack([row["memory_frozen"] for row in rows])
    stacked["memory_varying"] = np.stack([row["memory_varying"] for row in rows])
    stacked["horizons"] = np.asarray(horizons, dtype=np.int64)
    return stacked


def run_setting(spec: dict, args: argparse.Namespace, device: torch.device, dtype: torch.dtype, horizons: list[int]) -> Path:
    directory = Path(args.save_dir) / spec["name"]
    directory.mkdir(parents=True, exist_ok=True)
    combined = directory / "metrics.npz"
    if args.resume and combined.is_file() and not args.smoke:
        print(f"[skip] {spec['name']} metrics already exist", flush=True)
        return combined
    print(f"[load] {spec['name']} {spec['checkpoint']}", flush=True)
    model, ema = load_setting_model(spec, device, dtype)
    extra = max(3, max(horizons) + 1)
    frames, stems, times = load_frames(spec["eval_dir"], args.n_frames, extra, args.dt)
    operators = _build_operators(ema, model, spec["level"], frames, device, dtype, args.norm)
    if len(operators) < args.n_frames + 1:
        raise RuntimeError(
            f"need {args.n_frames + 1} operator states for two-step products, built {len(operators)}"
        )
    _record_stems(operators, stems)
    fd_report = None
    if args.fd_check:
        fd_report = fd_gate(operators[0], args.seed, args.fd_tol)
        print(f"[fd] {spec['name']} relative={fd_report['best_relative']:.3e}", flush=True)
        (directory / "fd_check.json").write_text(json.dumps(fd_report, indent=2))
    errors = rollout_errors(operators, args.n_frames, args.norm)
    rows = []
    dot_stems = stems[: args.n_frames]
    for index, stem in enumerate(dot_stems):
        path = _frame_path(directory, index, stem)
        if args.resume and path.is_file():
            rows.append(_load_frame(path))
            print(f"[resume] {spec['name']} stem {stem}", flush=True)
            continue
        started = time.time()
        e_now = errors["e_model"][index]
        e_next = errors["e_model"][index + 1] if index + 1 < len(errors["e_model"]) else None
        eta_now = None if errors["eta_model"][index] is None else errors["eta_model"][index]
        eta_next = None
        if index + 1 < len(errors["eta_model"]) and errors["eta_model"][index + 1] is not None:
            eta_next = errors["eta_model"][index + 1]
        row = evaluate_index(
            operators,
            index,
            args.n_probes,
            args.seed + int(stem),
            horizons,
            e_now,
            eta_now,
            e_next,
            eta_next,
            args.norm,
        )
        for key in ("e_norm", "eta_norm", "eps_u_norm", "eps_z_norm", "I_norm", "R_value", "rho_norm"):
            row[key] = float(errors[key][index])
        if abs(row["identity_residual"]) > args.identity_tol:
            raise RuntimeError(
                f"{spec['name']} stem {stem} energy identity residual {row['identity_residual']}"
            )
        if np.isfinite(row["twostep_residual"]) and abs(row["twostep_residual"]) > args.identity_tol:
            raise RuntimeError(
                f"{spec['name']} stem {stem} two-step identity residual {row['twostep_residual']}"
            )
        _save_frame(path, row, stem, index)
        rows.append(row)
        print(
            f"[frame] {spec['name']} {index + 1}/{args.n_frames} stem={stem} "
            f"tau={row['tau']:.4e} energy={row['energy']:.4e} "
            f"seconds={time.time() - started:.1f}",
            flush=True,
        )
    payload = _stack(rows, dot_stems, horizons)
    payload["level"] = np.asarray(spec["level"], dtype=np.int64)
    payload["fkx"] = np.asarray(spec["fkx"], dtype=np.int64)
    payload["dt"] = np.asarray(args.dt)
    payload["norm"] = np.asarray(args.norm)
    payload["n_probes"] = np.asarray(args.n_probes, dtype=np.int64)
    payload["seed"] = np.asarray(args.seed, dtype=np.int64)
    np.savez_compressed(combined, **payload)
    meta = {
        "name": spec["name"],
        "label": spec["label"],
        "level": spec["level"],
        "fkx": spec["fkx"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": _sha256(Path(spec["checkpoint"])),
        "eval_dir": spec["eval_dir"],
        "stems": dot_stems,
        "times": times[: args.n_frames],
        "n_frames": args.n_frames,
        "n_probes": args.n_probes,
        "seed": args.seed,
        "horizons": horizons,
        "dtype": str(dtype).replace("torch.", ""),
        "dt": args.dt,
        "norm": args.norm,
        "fd_check": fd_report,
        "coordinate": (
            "Network inputs and outputs are vorticity / norm. "
            "Input and output share that scale, so the Jacobian matches the physical map. "
            "Error norms are multiplied back by norm. Trace ratios are dimensionless."
        ),
    }
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
    del model, ema, operators
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return combined


def selected_settings(args: argparse.Namespace) -> list[dict]:
    names = ORDER if args.setting == "all" else (args.setting,)
    chosen = []
    for name in names:
        spec = dict(SETTINGS[name])
        spec["name"] = name
        if args.checkpoint:
            spec["checkpoint"] = args.checkpoint
        if args.eval_dir:
            spec["eval_dir"] = args.eval_dir
        chosen.append(spec)
    return chosen


def main() -> None:
    args = parse_args()
    horizons = _horizons(args.memory_horizons)
    _lock()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was requested but is not available")
    device = torch.device(args.device)
    dtype = torch.float64 if args.model_dtype == "float64" else torch.float32
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "settings": [spec["name"] for spec in selected_settings(args)],
        "smoke": bool(args.smoke),
        "n_frames": args.n_frames,
        "n_probes": args.n_probes,
        "seed": args.seed,
        "horizons": horizons,
        "dtype": args.model_dtype,
        "dt": args.dt,
        "norm": args.norm,
        "note": (
            "Dots are probe-averaged snapshot estimates. "
            "The 20 frames are one loader trajectory, so their spread is not a confidence interval. "
            "L=1 latent entries are missing, not zero."
        ),
    }
    (save_dir / "command.json").write_text(json.dumps(command, indent=2))
    for spec in selected_settings(args):
        run_setting(spec, args, device, dtype, horizons)
    if not args.skip_plots:
        from plot_trace_compare import render

        report = _REPO / "eval" / "TRACE_COMPARE_REPORT.md"
        render(save_dir, report if not args.smoke else save_dir / "SMOKE_REPORT.md", smoke=bool(args.smoke))
    print(f"[done] {save_dir}", flush=True)


if __name__ == "__main__":
    main()
