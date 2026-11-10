"""Centered finite-difference Jacobian actions for direct-step L=2 HINE.

Does not use torch.autograd.functional.jvp or torch.func.jvp.
The map is ema_vit(u, [z], t=1) -> (g, h) in model coordinates (Omega / norm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_EPS_LIST = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)

# Floor is 1e-8 times the Euclidean norm of a field of ones of that shape.
FD_ABS_FLOOR = 1e-8
FD_OUTPUT_FLOOR = 1e-12
FD_SCALING = (
    "One step h is chosen so the larger block-relative perturbation equals "
    "eps_rel. rel_block = h * ||d(block)|| / max(||block||, floor), "
    "floor = 1e-8 * sqrt(numel). A zero direction returns zero derivatives."
)


@dataclass
class L2Point:
    """Teacher-forced linearization point at emulator step n.

    u = u_n / norm, z = D(u_{n+1}) / norm.
    u_next = u_{n+1} / norm, z_next = D(u_{n+2}) / norm.
    """

    u: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    u_next: torch.Tensor
    z_next: torch.Tensor
    stems: tuple[int, int, int]


def assert_l2(model) -> None:
    if int(getattr(model, "num_coarse", -1)) != 1:
        raise ValueError(
            f"Task 1 is L=2 only (num_coarse=1); got num_coarse="
            f"{getattr(model, 'num_coarse', None)}"
        )


def make_point(
    frames: list[np.ndarray],
    model,
    device: torch.device,
    dtype: torch.dtype,
    norm: float,
) -> L2Point:
    """frames is [u_n, u_{n+1}, u_{n+2}] as Omega arrays."""
    if len(frames) < 3:
        raise ValueError(f"L=2 point needs 3 frames, got {len(frames)}")
    ds = int(model.ds_factors[0])

    def _field(arr: np.ndarray) -> torch.Tensor:
        src = np.ascontiguousarray(np.asarray(arr))
        return (
            torch.from_numpy(src)
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            / norm
        )

    u = _field(frames[0])
    u_next = _field(frames[1])
    u_next2 = _field(frames[2])
    z = F.avg_pool2d(u_next, kernel_size=ds, stride=ds)
    z_next = F.avg_pool2d(u_next2, kernel_size=ds, stride=ds)
    t = torch.ones(u.shape[0], device=device, dtype=dtype)
    return L2Point(
        u=u.detach(),
        z=z.detach(),
        t=t,
        u_next=u_next.detach(),
        z_next=z_next.detach(),
        stems=(0, 0, 0),
    )


def apply_D(v: torch.Tensor, ds: int) -> torch.Tensor:
    """Average-pool restriction. Same operator as FutureINNPlus._pool_seq."""
    return F.avg_pool2d(v, kernel_size=ds, stride=ds)


@torch.no_grad()
def model_step(ema_vit: torch.nn.Module, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
    if torch.is_autocast_enabled():
        raise RuntimeError("Task 1 finite differences require autocast to be off")
    high, coarses = ema_vit(u, [z], t)
    return high, coarses[0]


def _enorm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().reshape(-1).double()))


def block_floor(x: torch.Tensor) -> float:
    """Absolute floor: 1e-8 times ||ones|| for this block."""
    return FD_ABS_FLOOR * float(x.numel()) ** 0.5


def block_step(u, z, du, dz, eps_rel: float) -> dict:
    """Scalar step whose larger block-relative size equals eps_rel.

    Returns h, rel_u, rel_z, and zero. A zero direction has h = 0 and does
    not divide. The derivative is still Df @ (du, dz), not a rescaled vector.
    """
    scale_u = max(_enorm(u), block_floor(u))
    scale_z = max(_enorm(z), block_floor(z))
    nu = _enorm(du)
    nz = _enorm(dz)
    info = {
        "scale_u": scale_u,
        "scale_z": scale_z,
        "floor_u": block_floor(u),
        "floor_z": block_floor(z),
        "norm_du": nu,
        "norm_dz": nz,
    }
    if nu == 0.0 and nz == 0.0:
        info.update(h=0.0, rel_u=0.0, rel_z=0.0, zero=True)
        return info
    rate = 0.0
    if nu > 0.0:
        rate = max(rate, nu / scale_u)
    if nz > 0.0:
        rate = max(rate, nz / scale_z)
    h = float(eps_rel) / rate
    info.update(
        h=h,
        rel_u=(h * nu / scale_u) if nu > 0.0 else 0.0,
        rel_z=(h * nz / scale_z) if nz > 0.0 else 0.0,
        zero=False,
    )
    return info


@torch.no_grad()
def fd_pair(step_fn, u, z, du, dz, eps_rel: float):
    """Centered FD of step_fn(u, z) -> (g, h) in direction (du, dz).

    Returns (dg, dh, info). dg, dh approximate Df @ (du, dz). info records
    the relative perturbation actually applied to each input block.
    """
    info = block_step(u, z, du, dz, eps_rel)
    if info["zero"]:
        return torch.zeros_like(u), torch.zeros_like(z), info
    h = info["h"]
    gp, hp = step_fn(u + h * du, z + h * dz)
    gm, hm = step_fn(u - h * du, z - h * dz)
    inv = 1.0 / (2.0 * h)
    return (gp - gm) * inv, (hp - hm) * inv, info


@torch.no_grad()
def apply_A(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float) -> torch.Tensor:
    dg, _, _ = fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)
    return dg


@torch.no_grad()
def apply_B(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float) -> torch.Tensor:
    dg, _, _ = fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)
    return dg


@torch.no_grad()
def apply_C(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float) -> torch.Tensor:
    _, dh, _ = fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)
    return dh


@torch.no_grad()
def apply_K(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float) -> torch.Tensor:
    _, dh, _ = fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)
    return dh


@torch.no_grad()
def fd_both_u(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float):
    """One centered pair: (A v, C v)."""
    dg, dh, _ = fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)
    return dg, dh


@torch.no_grad()
def fd_both_z(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float):
    """One centered pair: (B w, K w)."""
    dg, dh, _ = fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)
    return dg, dh


@torch.no_grad()
def fd_both_uz(step_fn, point: L2Point, v: torch.Tensor, w: torch.Tensor, eps_rel: float):
    """One centered pair in a mixed direction: ((A v + B w), (C v + K w))."""
    dg, dh, _ = fd_pair(step_fn, point.u, point.z, v, w, eps_rel)
    return dg, dh


def _generator(seed: int, like: torch.Tensor) -> torch.Generator:
    gen = torch.Generator(device=like.device)
    gen.manual_seed(int(seed))
    return gen


def _rand_like(like: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=gen)


def _rel_change(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(_enorm(b), FD_OUTPUT_FLOOR)
    return _enorm(a - b) / denom


@torch.no_grad()
def _check_one(step_fn, point, du, dz, eps: float, tol: float, name: str) -> list[dict]:
    dg, dh, info = fd_pair(step_fn, point.u, point.z, du, dz, eps)
    checks = []
    for factor, label in ((1.0 / 3.0, "eps/3"), (0.1, "eps/10")):
        dg2, dh2, _ = fd_pair(step_fn, point.u, point.z, du, dz, eps * factor)
        for block, left, right in (("g", dg, dg2), ("h", dh, dh2)):
            rel = _rel_change(left, right)
            checks.append(
                {
                    "name": name,
                    "block": block,
                    "against": label,
                    "rel": rel,
                    "ok": bool(rel <= tol),
                    "rel_u": float(info["rel_u"]),
                    "rel_z": float(info["rel_z"]),
                    "h": float(info["h"]),
                    "zero": bool(info["zero"]),
                }
            )
    return checks


@torch.no_grad()
def validate_fd_eps(
    step_fn,
    points: list[tuple[str, L2Point]],
    working_eps: float,
    tol: float,
    seed: int,
    ds: int,
    graph_w=None,
    extra_directions: list | None = None,
) -> dict:
    """Validate the selected epsilon against nearby smaller steps.

    Checks fine-only, latent-only, and graph directions, and the fine and
    latent outputs separately. Random directions come from `seed`.
    """
    if graph_w is None:
        graph_w = lambda _point, vec, _ds=ds: apply_D(vec, _ds)
    gen = None
    checks: list[dict] = []
    for name, point in points:
        if gen is None:
            gen = _generator(seed, point.u)
        v = _rand_like(point.u, gen)
        w = _rand_like(point.z, gen)
        v_graph = _rand_like(point.u, gen)
        w_graph = graph_w(point, v_graph)
        checks.extend(_check_one(step_fn, point, v, torch.zeros_like(point.z), working_eps, tol, f"{name}/fine"))
        checks.extend(_check_one(step_fn, point, torch.zeros_like(point.u), w, working_eps, tol, f"{name}/latent"))
        checks.extend(_check_one(step_fn, point, v_graph, w_graph, working_eps, tol, f"{name}/graph"))
    for item in extra_directions or []:
        checks.extend(
            _check_one(
                step_fn,
                item["point"],
                item["du"],
                item["dz"],
                working_eps,
                tol,
                item["name"],
            )
        )
    ok = bool(checks) and all(c["ok"] for c in checks)
    worst = max(checks, key=lambda c: c["rel"]) if checks else None
    return {
        "working_eps": float(working_eps),
        "tol": float(tol),
        "seed": int(seed),
        "ok": ok,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "worst": worst,
        "checks": checks,
        "scaling": FD_SCALING,
        "compared_with": ["eps/3", "eps/10"],
    }


@torch.no_grad()
def fd_additivity(step_fn, point: L2Point, eps_rel: float, seed: int) -> dict:
    """Relative residual of FD(v+w) - FD(v) - FD(w) on the nonlinear map."""
    gen = _generator(seed, point.u)
    v1 = _rand_like(point.u, gen)
    z1 = _rand_like(point.z, gen)
    v2 = _rand_like(point.u, gen)
    z2 = _rand_like(point.z, gen)
    ag, ah, _ = fd_pair(step_fn, point.u, point.z, v1, z1, eps_rel)
    bg, bh, _ = fd_pair(step_fn, point.u, point.z, v2, z2, eps_rel)
    sg, sh, _ = fd_pair(step_fn, point.u, point.z, v1 + v2, z1 + z2, eps_rel)
    num = _enorm(torch.cat([(sg - ag - bg).reshape(-1), (sh - ah - bh).reshape(-1)]))
    den = max(_enorm(torch.cat([sg.reshape(-1), sh.reshape(-1)])), FD_OUTPUT_FLOOR)
    return {"rel": num / den, "seed": int(seed), "eps_rel": float(eps_rel)}
