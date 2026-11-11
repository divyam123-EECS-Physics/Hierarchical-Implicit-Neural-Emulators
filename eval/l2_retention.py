"""Injection and first-factor retention channels for direct-step L=2 HINE.

I_n, R_n, rho_n are bookkeeping on the rollout.
r_H and r_z are a truncated one-factor approximation, not a split of rho.
A large reconstruction residual is not automatically an implementation bug.
Proxy mode uses xi = eps_z - D eps_u and J_phys = A + B D.
"""

from __future__ import annotations

import numpy as np
import torch

from l2_fd_jvp import FD_OUTPUT_FLOOR, L2Point, apply_D, fd_both_uz, fd_both_z, model_step
from l2_operators import _phi_image
from l2_physical_jvp import is_proxy


def _norm(x: torch.Tensor) -> float:
    return float(x.reshape(-1).double().norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()) / (na * nb))


def relative_residual(numer: np.ndarray, denom: np.ndarray, floor: float = FD_OUTPUT_FLOOR) -> np.ndarray:
    """Per-entry numer/denom. Entries with denom at or below the floor are masked."""
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    rel = np.full(numer.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(denom) & (denom > floor)
    rel[mask] = numer[mask] / np.maximum(denom[mask], floor)
    return rel


_FINE_COS = (
    "cos_rH_rZ",
    "cos_e_rho",
    "cos_e_rH",
    "cos_e_rZ",
    "cos_rho_rH",
    "cos_rho_rZ",
    "cos_e_inj",
    "cos_rho_inj",
    "cos_rH_inj",
    "cos_rZ_inj",
)
_LATENT_COS = (
    "cos_eta_inj",
    "cos_eta_eta_lin",
)


def running_channels(
    e: list[torch.Tensor],
    eps_u: list[torch.Tensor],
    xi: list[torch.Tensor],
    apply_J_minus_I,
    apply_B,
    e_lin: list[torch.Tensor] | None = None,
    eta: list[torch.Tensor] | None = None,
    eta_lin: list[torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """First-factor sums. e[0] is the initial fine error (zero on a fresh start).

    e has length T+1. eps_u and xi have length T.
    apply_J_minus_I(k, vec) and apply_B(k, vec) act at step k, on injections
    accumulated before step k. r_H and r_z are a truncated approximation.
    If e_lin is given, also records ||rho_lin - r_H - r_z||.
    Cosines use the same vectors as the norms. S is the accumulated fine
    injection sum_{i<n} eps_u. Z is the accumulated latent injection.
    A zero vector leaves that cosine undefined (NaN).
    """
    t = len(eps_u)
    if len(e) != t + 1 or len(xi) != t:
        raise ValueError(f"length mismatch e={len(e)} eps={t} xi={len(xi)}")
    if e_lin is not None and len(e_lin) != t + 1:
        raise ValueError(f"e_lin length {len(e_lin)} != {t + 1}")
    if eta is not None and len(eta) != t + 1:
        raise ValueError(f"eta length {len(eta)} != {t + 1}")
    if eta_lin is not None and len(eta_lin) != t + 1:
        raise ValueError(f"eta_lin length {len(eta_lin)} != {t + 1}")
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
    lin_resid = np.full(t + 1, np.nan)
    cos = {name: np.full(t + 1, np.nan) for name in (*_FINE_COS, *_LATENT_COS)}
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
        if e_lin is not None:
            lin_resid[n] = _norm(e_lin[n] - S - rH - rZ)
        cos["cos_rH_rZ"][n] = _cos(rH, rZ)
        cos["cos_e_rho"][n] = _cos(e[n], rho)
        cos["cos_e_rH"][n] = _cos(e[n], rH)
        cos["cos_e_rZ"][n] = _cos(e[n], rZ)
        cos["cos_rho_rH"][n] = _cos(rho, rH)
        cos["cos_rho_rZ"][n] = _cos(rho, rZ)
        cos["cos_e_inj"][n] = _cos(e[n], S)
        cos["cos_rho_inj"][n] = _cos(rho, S)
        cos["cos_rH_inj"][n] = _cos(rH, S)
        cos["cos_rZ_inj"][n] = _cos(rZ, S)
        if eta is not None:
            cos["cos_eta_inj"][n] = _cos(eta[n], Z)
            if eta_lin is not None:
                cos["cos_eta_eta_lin"][n] = _cos(eta[n], eta_lin[n])
        if n == t:
            break
        rH = rH + apply_J_minus_I(n, S)
        if n >= 1:
            rZ = rZ + apply_B(n, Z)
        S = S + eps_u[n]
        Z = Z + xi[n]
    out = {
        "I": I,
        "R": R,
        "e_norm": e_n,
        "rho_norm": rho_n,
        "rH_norm": rH_n,
        "rZ_norm": rZ_n,
        "recon_norm": resid,
        **cos,
    }
    if e_lin is not None:
        out["recon_lin_norm"] = lin_resid
    return out


def propagate_augmented(step_at, points: list[L2Point], eps_u, eps_z, eps_rel: float):
    """Full linearized fine and latent errors along teacher-forced points.

    step_at(n) is the map at teacher-forced step n.
    e_lin,0 = eta_lin,0 = 0. One mixed centered difference supplies both
    block rows. The initial zero direction adds only the injections.
    """
    if len(points) != len(eps_u) or len(eps_u) != len(eps_z):
        raise ValueError("points, eps_u, and eps_z must have the same length")
    e_lin = [torch.zeros_like(points[0].u)]
    eta_lin = [torch.zeros_like(points[0].z)]
    for n, pt in enumerate(points):
        if _norm(e_lin[n]) == 0.0 and _norm(eta_lin[n]) == 0.0:
            dg = torch.zeros_like(e_lin[n])
            dh = torch.zeros_like(eta_lin[n])
        else:
            dg, dh = fd_both_uz(step_at(n), pt, e_lin[n], eta_lin[n], eps_rel)
        e_lin.append(dg + eps_u[n])
        eta_lin.append(dh + eps_z[n])
    return e_lin, eta_lin


def validation_directions(points, eps_u, xi, e_lin, eta_lin, tangent, ds: int, norm: float) -> list:
    """Directions the retention and linearized updates actually apply.

    r_H uses (S_n, D Phi'_n S_n), not (S_n, 0). Checks the first nonzero
    step, the middle step, and the last step. Zero slots are skipped.
    """
    t = len(eps_u)
    S = torch.zeros_like(eps_u[0])
    Z = torch.zeros_like(xi[0])
    states = []
    for n in range(t):
        states.append((S.clone(), Z.clone()))
        S = S + eps_u[n]
        Z = Z + xi[n]
    chosen: list[int] = []
    first = next(
        (
            n
            for n, (s, _z) in enumerate(states)
            if _norm(s) > 0.0 or _norm(e_lin[n]) > 0.0 or _norm(eta_lin[n]) > 0.0
        ),
        None,
    )
    for n in (first, t // 2, t - 1):
        if n is None or n in chosen or not (0 <= int(n) < t):
            continue
        chosen.append(int(n))
    out = []
    for n in chosen:
        s_n, z_n = states[n]
        pt = points[n]
        if _norm(s_n) > 0.0:
            image = _phi_image(tangent, pt.u, s_n, norm)
            out.append(
                {
                    "name": f"rH_step{n}",
                    "point": pt,
                    "du": s_n,
                    "dz": apply_D(image, ds),
                }
            )
        if n >= 1 and _norm(z_n) > 0.0:
            out.append(
                {
                    "name": f"rZ_step{n}",
                    "point": pt,
                    "du": torch.zeros_like(pt.u),
                    "dz": z_n,
                }
            )
        if _norm(e_lin[n]) > 0.0 or _norm(eta_lin[n]) > 0.0:
            out.append(
                {
                    "name": f"lin_step{n}",
                    "point": pt,
                    "du": e_lin[n].detach().clone(),
                    "dz": eta_lin[n].detach().clone(),
                }
            )
    return out


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
    direction_sink: list | None = None,
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
        if is_proxy(tangent):
            xi_k = ez - apply_D(eu, ds)
        else:
            img = _phi_image(tangent, pt.u_next, eu, norm)
            xi_k = ez - apply_D(img, ds)
        eps_u.append(eu)
        eps_z.append(ez)
        xi.append(xi_k)

    # Free rollout from (u_0, D u_1). e_0 = eta_0 = 0.
    high = points[0].u.clone()
    z = points[0].z.clone()
    e = [torch.zeros_like(high)]
    eta = [torch.zeros_like(z)]
    for k in range(n_steps):
        high, z = model_step(ema_vit, high, z, points[k].t)
        e.append(high - points[k].u_next)
        eta.append(z - points[k].z_next)

    def step_at(k: int):
        return lambda u, z, _t=points[k].t: model_step(ema_vit, u, z, _t)

    e_lin, eta_lin = propagate_augmented(step_at, points, eps_u, eps_z, eps_rel)
    if direction_sink is not None:
        direction_sink.extend(
            validation_directions(points, eps_u, xi, e_lin, eta_lin, tangent, ds, norm)
        )

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

    # Scale to physical units for the reported norms and vector differences.
    e_phys = [t * norm for t in e]
    eta_phys = [t * norm for t in eta]
    e_lin_phys = [t * norm for t in e_lin]
    eta_lin_phys = [t * norm for t in eta_lin]
    eu_phys = [t * norm for t in eps_u]
    xi_phys = [t * norm for t in xi]

    def J_phys(k, vec_phys):
        return apply_J_minus_I(k, vec_phys / norm) * norm

    def B_phys(k, w_phys):
        return apply_B(k, w_phys / norm) * norm

    out = running_channels(
        e_phys, eu_phys, xi_phys, J_phys, B_phys,
        e_lin=e_lin_phys, eta=eta_phys, eta_lin=eta_lin_phys,
    )
    out["eps_u_norm"] = np.asarray([_norm(t) for t in eu_phys])
    out["eps_z_norm"] = np.asarray([_norm(t * norm) for t in eps_z])
    out["e_lin_norm"] = np.asarray([_norm(t) for t in e_lin_phys])
    out["eta_norm"] = np.asarray([_norm(t) for t in eta_phys])
    out["eta_lin_norm"] = np.asarray([_norm(t) for t in eta_lin_phys])
    out["e_lin_err"] = np.asarray([_norm(a - b) for a, b in zip(e_phys, e_lin_phys)])
    out["eta_lin_err"] = np.asarray([_norm(a - b) for a, b in zip(eta_phys, eta_lin_phys)])
    out["approximation"] = np.asarray("proxy" if is_proxy(tangent) else "py2d_fd")
    out["first_factor"] = np.asarray("truncated")
    return out
