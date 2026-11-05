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
        return (
            torch.from_numpy(np.asarray(arr, dtype=np.float32))
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
    high, coarses = ema_vit(u, [z], t)
    return high, coarses[0]


def _scale(u: torch.Tensor, z: torch.Tensor, du: torch.Tensor, dz: torch.Tensor, eps_rel: float) -> float:
    state = torch.sqrt(u.reshape(-1).pow(2).sum() + z.reshape(-1).pow(2).sum())
    direc = torch.sqrt(du.reshape(-1).pow(2).sum() + dz.reshape(-1).pow(2).sum())
    if float(direc) == 0.0:
        raise ValueError("finite-difference direction is zero")
    return float(eps_rel) * float(state) / float(direc)


@torch.no_grad()
def fd_pair(step_fn, u, z, du, dz, eps_rel: float):
    """Centered FD of step_fn(u, z) -> (g, h) in direction (du, dz).

    delta = eps_rel * ||(u, z)|| / ||(du, dz)||.
    Returns (dg, dh), the directional derivatives.
    """
    delta = _scale(u, z, du, dz, eps_rel)
    gp, hp = step_fn(u + delta * du, z + delta * dz)
    gm, hm = step_fn(u - delta * du, z - delta * dz)
    inv = 1.0 / (2.0 * delta)
    return (gp - gm) * inv, (hp - hm) * inv


@torch.no_grad()
def apply_A(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float) -> torch.Tensor:
    dg, _ = fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)
    return dg


@torch.no_grad()
def apply_B(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float) -> torch.Tensor:
    dg, _ = fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)
    return dg


@torch.no_grad()
def apply_C(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float) -> torch.Tensor:
    _, dh = fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)
    return dh


@torch.no_grad()
def apply_K(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float) -> torch.Tensor:
    _, dh = fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)
    return dh


@torch.no_grad()
def fd_both_u(step_fn, point: L2Point, v: torch.Tensor, eps_rel: float):
    """One centered pair: (A v, C v)."""
    return fd_pair(step_fn, point.u, point.z, v, torch.zeros_like(point.z), eps_rel)


@torch.no_grad()
def fd_both_z(step_fn, point: L2Point, w: torch.Tensor, eps_rel: float):
    """One centered pair: (B w, K w)."""
    return fd_pair(step_fn, point.u, point.z, torch.zeros_like(point.u), w, eps_rel)


@torch.no_grad()
def fd_both_uz(step_fn, point: L2Point, v: torch.Tensor, w: torch.Tensor, eps_rel: float):
    """One centered pair in a mixed direction: ((A v + B w), (C v + K w))."""
    return fd_pair(step_fn, point.u, point.z, v, w, eps_rel)


@torch.no_grad()
def eps_sweep(step_fn, point: L2Point, eps_list, tol: float, generator: torch.Generator | None = None):
    """Pairwise relative change of a random (u, z) directional derivative.

    Fails when no two consecutive eps values agree within `tol`.
    """
    du = torch.randn(point.u.shape, device=point.u.device, dtype=point.u.dtype, generator=generator)
    dz = torch.randn(point.z.shape, device=point.z.device, dtype=point.z.dtype, generator=generator)
    cols = []
    for eps in eps_list:
        dg, dh = fd_pair(step_fn, point.u, point.z, du, dz, float(eps))
        cols.append(torch.cat([dg.reshape(-1), dh.reshape(-1)]))
    rels = []
    for a, b in zip(cols, cols[1:]):
        rels.append(float((a - b).norm() / b.norm().clamp(min=1e-12)))
    ok = any(r <= tol for r in rels) if rels else False
    return {
        "eps_list": [float(e) for e in eps_list],
        "consecutive_rel": rels,
        "ok": bool(ok),
        "tol": float(tol),
    }
