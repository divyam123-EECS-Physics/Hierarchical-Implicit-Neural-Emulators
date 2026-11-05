"""Injection and first-factor retention channels for direct-step L=2 HINE.

I_n, R_n, rho_n are exact bookkeeping on the rollout (Eq. 5.1, 5.2, 10.2).
r_H and r_z are first-factor approximations, not an exact split of rho.
Proxy mode uses xi = eps_z - D eps_u and J_phys = A + B D.
"""

from __future__ import annotations

import numpy as np
import torch

from l2_fd_jvp import L2Point, apply_D, fd_both_uz, fd_both_z, model_step
from l2_operators import _phi_image


def _norm(x: torch.Tensor) -> float:
    return float(x.reshape(-1).double().norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()) / (na * nb))


def running_channels(
    e: list[torch.Tensor],
    eps_u: list[torch.Tensor],
    xi: list[torch.Tensor],
    apply_J_minus_I,
    apply_B,
) -> dict[str, np.ndarray]:
    """First-factor sums. e[0] is the initial fine error (zero on a fresh start).

    e has length T+1. eps_u and xi have length T.
    apply_J_minus_I(k, vec) and apply_B(k, vec) act at step k.
    """
    t = len(eps_u)
    if len(e) != t + 1 or len(xi) != t:
        raise ValueError(f"length mismatch e={len(e)} eps={t} xi={len(xi)}")
    S = torch.zeros_like(e[0])
    Z = torch.zeros_like(xi[0])
    rH = torch.zeros_like(e[0])
    rZ = torch.zeros_like(e[0])
    I = np.zeros(t + 1)
    R = np.zeros(t + 1)
    rho_n = np.zeros(t + 1)
    rH_n = np.zeros(t + 1)
    rZ_n = np.zeros(t + 1)
    resid = np.zeros(t + 1)
    align = np.full(t + 1, np.nan)
    e_n = np.zeros(t + 1)
    for n in range(t + 1):
        rho = e[n] - S
        e_n[n] = _norm(e[n])
        I[n] = _norm(S)
        R[n] = e_n[n] - I[n]
        rho_n[n] = _norm(rho)
        rH_n[n] = _norm(rH)
        rZ_n[n] = _norm(rZ)
        resid[n] = _norm(rho - rH - rZ)
        align[n] = _cos(rH, rZ)
        if n == t:
            break
        rH = rH + apply_J_minus_I(n, S)
        if n >= 1:
            rZ = rZ + apply_B(n, Z)
        S = S + eps_u[n]
        Z = Z + xi[n]
    return {
        "I": I,
        "R": R,
        "e_norm": e_n,
        "rho_norm": rho_n,
        "rH_norm": rH_n,
        "rZ_norm": rZ_n,
        "recon_norm": resid,
        "cos_rH_rZ": align,
    }


@torch.no_grad()
def rollout_diagnostics(
    ema_vit,
    frames: list[np.ndarray],
    model,
    device,
    dtype,
    norm: float,
    eps_rel: float,
    tangent,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Free AR rollout plus teacher-forced injections on one window.

    frames has length >= n_steps + 2: indices 0 .. n_steps+1.
    Errors are converted to physical units (multiplied by norm) before norms.
    """
    if len(frames) < n_steps + 2:
        raise ValueError(
            f"need {n_steps + 2} frames for rollout {n_steps}, got {len(frames)}"
        )
    ds = int(model.ds_factors[0])
    points = []
    for k in range(n_steps):
        from l2_fd_jvp import make_point

        pt = make_point(frames[k : k + 3], model, device, dtype, norm)
        points.append(pt)

    # Teacher-forced injections, model coordinates.
    eps_u = []
    eps_z = []
    xi = []
    for pt in points:
        g, h = model_step(ema_vit, pt.u, pt.z, pt.t)
        eu = g - pt.u_next
        ez = h - pt.z_next
        if tangent.is_exact:
            img = _phi_image(tangent, pt.u_next, eu, norm)
            xi_k = ez - apply_D(img, ds)
        else:
            xi_k = ez - apply_D(eu, ds)
        eps_u.append(eu)
        eps_z.append(ez)
        xi.append(xi_k)

    # Free rollout from (u_0, D u_1). e_0 = 0.
    high = points[0].u.clone()
    z = points[0].z.clone()
    e = [torch.zeros_like(high)]
    for k in range(n_steps):
        high, z = model_step(ema_vit, high, z, points[k].t)
        e.append(high - points[k].u_next)

    def apply_J_minus_I(k: int, vec: torch.Tensor) -> torch.Tensor:
        if _norm(vec) == 0.0:
            return torch.zeros_like(vec)
        v_for_z = _phi_image(tangent, points[k].u, vec, norm)
        Jv, _ = fd_both_uz(
            lambda u, z: model_step(ema_vit, u, z, points[k].t),
            points[k],
            vec,
            apply_D(v_for_z, ds),
            eps_rel,
        )
        return Jv - vec

    def apply_B(k: int, w: torch.Tensor) -> torch.Tensor:
        if _norm(w) == 0.0:
            return torch.zeros_like(e[0])
        Bw, _Kw = fd_both_z(
            lambda u, z: model_step(ema_vit, u, z, points[k].t),
            points[k],
            w,
            eps_rel,
        )
        return Bw

    # Scale to physical units for the reported norms.
    e_phys = [t * norm for t in e]
    eu_phys = [t * norm for t in eps_u]
    xi_phys = [t * norm for t in xi]

    def J_phys(k, vec_phys):
        return apply_J_minus_I(k, vec_phys / norm) * norm

    def B_phys(k, w_phys):
        return apply_B(k, w_phys / norm) * norm

    out = running_channels(e_phys, eu_phys, xi_phys, J_phys, B_phys)
    out["eps_u_norm"] = np.asarray([_norm(t) for t in eu_phys])
    out["eps_z_norm"] = np.asarray([_norm(t * norm) for t in eps_z])
    out["approximation"] = np.asarray("exact" if tangent.is_exact else "proxy")
    out["first_factor"] = np.asarray(True)
    return out
