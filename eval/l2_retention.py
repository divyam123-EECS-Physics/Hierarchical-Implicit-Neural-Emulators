"""Injection and first-factor retention channels for direct-step L=2 HINE.

I_n, R_n, rho_n are bookkeeping on the rollout.
r_H and r_z are a truncated one-factor approximation, not a split of rho.
A large reconstruction residual is not automatically an implementation bug.
Proxy mode uses xi = eps_z - D eps_u and J_phys = A + B D.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from l2_fd_jvp import (
    FD_OUTPUT_FLOOR,
    L2Point,
    apply_D,
    fd_both_uz,
    jvp_both_u,
    jvp_both_uz,
    jvp_both_z,
    model_step,
)
from l2_operators import _phi_image
from l2_physical_jvp import is_proxy


def _norm(x: torch.Tensor) -> float:
    return float(x.reshape(-1).double().norm())


def _rms(x: torch.Tensor) -> float:
    flat = x.detach().reshape(-1).double()
    if flat.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(flat * flat)))


def combined_rms(p: torch.Tensor, q: torch.Tensor) -> float:
    """Block-balanced size sqrt(RMS(p)^2 + RMS(q)^2)."""
    return math.sqrt(_rms(p) ** 2 + _rms(q) ** 2)


def apply_shared_rescale(p: torch.Tensor, q: torch.Tensor, cum_log: float, lo: float = 1e-6, hi: float = 1e6):
    """Divide p and q by one shared factor when their combined RMS leaves [lo, hi].

    The returned cumulative log is the log of the product of those factors.
    Physical tensors are stored * exp(cum_log). Neither block is scaled alone.
    """
    scale = combined_rms(p, q)
    if not math.isfinite(scale) or scale == 0.0 or lo <= scale <= hi:
        return p, q, float(cum_log), False
    return p / scale, q / scale, float(cum_log) + math.log(scale), True


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return float("nan")
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return float("nan")
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
    "cos_e_e_lin",
)


LEGACY_COSINE_NAMES = {
    "cos_e_inj": r"\cos(e_n, \sum_{j<n} \epsilon^{u}_j)",
    "cos_rho_inj": r"\cos(\rho_n, \sum_{j<n} \epsilon^{u}_j)",
    "cos_rH_inj": r"\cos(r_H, \sum_{j<n} \epsilon^{u}_j)",
    "cos_rZ_inj": r"\cos(r_z, \sum_{j<n} \epsilon^{u}_j)",
    "cos_eta_inj": r"\cos(\eta_n, \sum_{j<n} \xi_j)",
    "I": r"\|\sum_{j<n} \epsilon^{u}_j\|",
    "R": r"\|e_n\|-I(t_n)",
}


def account_injections(
    eta: list[torch.Tensor],
    eps_z: list[torch.Tensor],
    xi: list[torch.Tensor],
    nu: list[torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """Norms of vector sums and the two latent injection cosines.

    cos_eta_inj is cos(eta_n, sum_{j<n} xi_j). That legacy name is unchanged.
    cos_eta_eps_z is cos(eta_n, sum_{j<n} epsilon^z_j).
    cos_nu_xi is cos(nu_n, sum_{j<n} xi_j). A missing nu stays NaN.
    Sums are norms of sums, not sums of norms.
    """
    t = len(eps_z)
    if len(eta) != t + 1 or len(xi) != t:
        raise ValueError(f"length mismatch eta={len(eta)} eps_z={t} xi={len(xi)}")
    if nu is not None and len(nu) != t + 1:
        raise ValueError(f"nu length {len(nu)} != {t + 1}")
    sum_z = torch.zeros_like(eta[0])
    sum_xi = torch.zeros_like(eta[0])
    cos_eta_xi = np.full(t + 1, np.nan)
    cos_eta_ez = np.full(t + 1, np.nan)
    cos_nu = np.full(t + 1, np.nan)
    ez_sum = np.zeros(t + 1)
    xi_sum = np.zeros(t + 1)
    for n in range(t + 1):
        ez_sum[n] = _norm(sum_z)
        xi_sum[n] = _norm(sum_xi)
        cos_eta_xi[n] = _cos(eta[n], sum_xi)
        cos_eta_ez[n] = _cos(eta[n], sum_z)
        if nu is not None:
            cos_nu[n] = _cos(nu[n], sum_xi)
        if n == t:
            break
        sum_z = sum_z + eps_z[n]
        sum_xi = sum_xi + xi[n]
    nu_norm = (
        np.asarray([_norm(item) for item in nu], dtype=np.float64)
        if nu is not None
        else np.full(t + 1, np.nan)
    )
    return {
        "eps_z_sum_norm": ez_sum,
        "xi_sum_norm": xi_sum,
        "xi_norm": np.asarray([_norm(item) for item in xi], dtype=np.float64),
        "cos_eta_eps_z": cos_eta_ez,
        "cos_eta_inj": cos_eta_xi,
        "cos_nu_xi": cos_nu,
        "nu_norm": nu_norm,
    }


def dphi_latent(tangent, point, vec: torch.Tensor, norm: float, ds: int) -> torch.Tensor:
    """D Phi'(point) vec in the latent space. A zero direction returns zero.

    Proxy Phi' is the identity. A physical tangent must already carry dns.
    This does not replace a failed tangent with eta or with a finite difference.
    """
    if _norm(vec) == 0.0:
        return torch.zeros_like(point.z)
    image = _phi_image(
        tangent,
        point.u,
        vec,
        norm,
        dns=getattr(point, "dns", None),
        hats=getattr(point, "hats", None),
    )
    return apply_D(image, ds)


def nu_series(tangent, points: list, e: list[torch.Tensor], eta: list[torch.Tensor], norm: float, ds: int):
    """nu_n = eta_n - D Phi'_n e_n. Returns (nu, available).

    available is false when a physical tangent has no DNS field. Those steps
    are NaN. They are not filled with eta.
    """
    if len(points) != len(e) or len(e) != len(eta):
        raise ValueError("points, e, and eta must have the same length")
    nu = []
    available = True
    for point, err, latent in zip(points, e, eta):
        if not is_proxy(tangent) and getattr(point, "dns", None) is None:
            nu.append(torch.full_like(latent, float("nan")))
            available = False
            continue
        nu.append(latent - dphi_latent(tangent, point, err, norm, ds))
    return nu, available


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
    Cosines use the same vectors as the norms. S is sum_{j<n} epsilon^u_j.
    Z, stored as cos_eta_inj, is sum_{j<n} xi_j, not sum epsilon^z_j.
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
        if e_lin is not None:
            cos["cos_e_e_lin"][n] = _cos(e[n], e_lin[n])
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
    e_lin,0 = eta_lin,0 = 0. One mixed torch JVP supplies both block rows.
    The initial zero direction adds only the injections.
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
            dg, dh = jvp_both_uz(step_at(n), pt, e_lin[n], eta_lin[n], eps_rel)
        e_lin.append((dg + eps_u[n]).detach())
        eta_lin.append((dh + eps_z[n]).detach())
    return e_lin, eta_lin


def propagate_homogeneous(
    step_at,
    bases: list[L2Point],
    p0: torch.Tensor,
    q0: torch.Tensor,
    eps_rel: float,
    rescale: bool = True,
    lo: float = 1e-6,
    hi: float = 1e6,
):
    """Homogeneous fine/latent tangent. No prediction-error injection.

    p_next = A p + B q and q_next = C p + K q, from one mixed centered
    difference. Stored states times exp(log_scale) reconstruct the
    unscaled tensors when rescale is on. log_scale[0] is 0.
    """
    if len(bases) == 0:
        raise ValueError("homogeneous propagation needs at least one base point")
    p = p0.detach().clone()
    q = q0.detach().clone()
    ps = [p.detach().clone()]
    qs = [q.detach().clone()]
    log_scale = [0.0]
    cum = 0.0
    for n, pt in enumerate(bases):
        if _norm(p) == 0.0 and _norm(q) == 0.0:
            dg = torch.zeros_like(p)
            dh = torch.zeros_like(q)
        else:
            dg, dh = fd_both_uz(step_at(n), pt, p, q, eps_rel)
        p = dg.detach()
        q = dh.detach()
        if rescale:
            p, q, cum, _applied = apply_shared_rescale(p, q, cum, lo=lo, hi=hi)
        ps.append(p.detach().clone())
        qs.append(q.detach().clone())
        log_scale.append(float(cum))
    return ps, qs, log_scale


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
            image = _phi_image(
                tangent, pt.u, s_n, norm,
                dns=getattr(pt, "dns", None),
                hats=getattr(pt, "hats", None),
            )
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


def _nan_series(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


LATENT_CHANNEL_KEYS = (
    "p_norm",
    "q_norm",
    "b_norm",
    "remainder_norm",
    "p_rms",
    "q_rms",
    "b_rms",
    "remainder_rms",
    "cross_bp",
    "p_sq",
    "delta_fine",
    "I_z_norm",
    "r_u_to_z_norm",
    "r_z_to_z_norm",
    "rho_eta_norm",
    "eta_recon_norm",
    "r_u_to_z_first_norm",
    "r_z_to_z_first_norm",
    "cos_p_eta",
    "cos_p_eta_next",
    "cos_p_Keta",
    "cos_p_eps_z",
    "cos_p_b",
    "cos_ru_Iz",
    "cos_rz_Iz",
    "cos_ru_rz",
    "cos_ru_eta",
    "cos_ru_rho",
    "cos_eta_rho",
    "cos_rz_eta",
    "cos_rho_rz",
    "cos_rho_Iz",
    "channel_remainder_norm",
    "cos_Qe_eta",
    "cos_p_nu",
    "eta_sq",
    "Iz_sq",
    "eta_sq_gap",
    "cross_Iz_rho",
    "rho_eta_sq",
    "cross_Iz_ru",
    "cross_Iz_rz",
    "cross_Iz_rem",
    "budget_resid",
)

LATENT_LIN_KEYS = tuple(f"lin_{name}" for name in LATENT_CHANNEL_KEYS if name not in {
    "r_u_to_z_first_norm",
    "r_z_to_z_first_norm",
    "cos_Qe_eta",
    "cos_p_nu",
})


def latent_error_channels(
    e: list[torch.Tensor],
    eta: list[torch.Tensor],
    eps_u: list[torch.Tensor],
    eps_z: list[torch.Tensor],
    apply_C,
    apply_K,
    *,
    Qe: list[torch.Tensor] | None = None,
    nu: list[torch.Tensor] | None = None,
    first_factor: bool = True,
) -> dict[str, np.ndarray]:
    """Fine-error contribution to the latent error at teacher-forced Jacobians.

    p_n = C_n e_n, q_n = (K_n - I) eta_n, b_n = K_n eta_n + eps_z,n.
    remainder_n = eta_{n+1} - (p_n + b_n).
    Accumulated sums are strictly before n and start at zero.
    r_H^eta (r_u_to_z) is sum_{k<n} C_k e_k.
    r_z^eta (r_z_to_z) is sum_{k<n} (K_k - I) eta_k.
    rho^eta is eta_n - sum_{k<n} eps_z. It is not the off-graph sum of xi.
    On the linearized recursion with zero initial error,
    rho^eta = r_H^eta + r_z^eta. For a nonlinear rollout the difference
    rho^eta - r_H^eta - r_z^eta is the Taylor remainder, stored as
    channel_remainder_norm. eta_recon_norm also subtracts the accumulated
    one-step remainder and is the closure residual, not that Taylor remainder.
    First-factor sums apply C and (K-I) to accumulated injections only.
    They are a different construction and are not these telescoping channels.
    """
    steps = len(eps_z)
    if len(e) != steps + 1 or len(eta) != steps + 1 or len(eps_u) != steps:
        raise ValueError(
            f"length mismatch e={len(e)} eta={len(eta)} eps_u={len(eps_u)} eps_z={steps}"
        )
    length = steps + 1
    I_z = torch.zeros_like(eta[0])
    r_u = torch.zeros_like(eta[0])
    r_z = torch.zeros_like(eta[0])
    rem_sum = torch.zeros_like(eta[0])
    sum_u = torch.zeros_like(e[0])
    sum_z = torch.zeros_like(eta[0])
    r_u_first = torch.zeros_like(eta[0])
    r_z_first = torch.zeros_like(eta[0])
    out = {name: _nan_series(length) for name in LATENT_CHANNEL_KEYS}
    if not first_factor:
        out["r_u_to_z_first_norm"] = _nan_series(length)
        out["r_z_to_z_first_norm"] = _nan_series(length)

    def _act(op, vec, like):
        if not torch.isfinite(vec).all() or _norm(vec) == 0.0:
            return torch.zeros_like(like)
        return op(vec)

    for n in range(length):
        rho = eta[n] - I_z
        out["I_z_norm"][n] = _norm(I_z)
        out["r_u_to_z_norm"][n] = _norm(r_u)
        out["r_z_to_z_norm"][n] = _norm(r_z)
        out["rho_eta_norm"][n] = _norm(rho)
        out["eta_recon_norm"][n] = _norm(rho - r_u - r_z - rem_sum)
        if first_factor:
            out["r_u_to_z_first_norm"][n] = _norm(r_u_first)
            out["r_z_to_z_first_norm"][n] = _norm(r_z_first)
        out["cos_ru_Iz"][n] = _cos(r_u, I_z)
        out["cos_rz_Iz"][n] = _cos(r_z, I_z)
        out["cos_ru_rz"][n] = _cos(r_u, r_z)
        out["cos_ru_eta"][n] = _cos(r_u, eta[n])
        out["cos_ru_rho"][n] = _cos(r_u, rho)
        out["cos_eta_rho"][n] = _cos(eta[n], rho)
        out["cos_rz_eta"][n] = _cos(r_z, eta[n])
        out["cos_rho_rz"][n] = _cos(rho, r_z)
        out["cos_rho_Iz"][n] = _cos(rho, I_z)
        out["channel_remainder_norm"][n] = _norm(rho - r_u - r_z)
        if Qe is not None:
            out["cos_Qe_eta"][n] = _cos(Qe[n], eta[n])
        eta_sq = _dot(eta[n], eta[n])
        iz_sq = _dot(I_z, I_z)
        rho_sq = _dot(rho, rho)
        cross_rho = 2.0 * _dot(I_z, rho)
        out["eta_sq"][n] = eta_sq
        out["Iz_sq"][n] = iz_sq
        out["eta_sq_gap"][n] = eta_sq - iz_sq
        out["cross_Iz_rho"][n] = cross_rho
        out["rho_eta_sq"][n] = rho_sq
        out["cross_Iz_ru"][n] = 2.0 * _dot(I_z, r_u)
        out["cross_Iz_rz"][n] = 2.0 * _dot(I_z, r_z)
        out["cross_Iz_rem"][n] = 2.0 * _dot(I_z, rem_sum)
        out["budget_resid"][n] = (eta_sq - iz_sq) - (cross_rho + rho_sq)
        if n == steps:
            break
        p = _act(lambda vec, k=n: apply_C(k, vec), e[n], eta[0])
        k_eta = _act(lambda vec, k=n: apply_K(k, vec), eta[n], eta[0])
        q = k_eta - eta[n]
        b = k_eta + eps_z[n]
        remainder = eta[n + 1] - (p + b)
        out["p_norm"][n] = _norm(p)
        out["q_norm"][n] = _norm(q)
        out["b_norm"][n] = _norm(b)
        out["remainder_norm"][n] = _norm(remainder)
        out["p_rms"][n] = _rms(p)
        out["q_rms"][n] = _rms(q)
        out["b_rms"][n] = _rms(b)
        out["remainder_rms"][n] = _rms(remainder)
        cross = 2.0 * _dot(b, p)
        p_sq = _dot(p, p)
        out["cross_bp"][n] = cross
        out["p_sq"][n] = p_sq
        out["delta_fine"][n] = cross + p_sq
        out["cos_p_eta"][n] = _cos(p, eta[n])
        out["cos_p_eta_next"][n] = _cos(p, eta[n + 1])
        out["cos_p_Keta"][n] = _cos(p, k_eta)
        out["cos_p_eps_z"][n] = _cos(p, eps_z[n])
        out["cos_p_b"][n] = _cos(p, b)
        if nu is not None:
            out["cos_p_nu"][n] = _cos(p, nu[n])
        if first_factor:
            p_first = _act(lambda vec, k=n: apply_C(k, vec), sum_u, eta[0])
            k_sum = _act(lambda vec, k=n: apply_K(k, vec), sum_z, eta[0])
            r_u_first = r_u_first + p_first
            r_z_first = r_z_first + (k_sum - sum_z)
        r_u = r_u + p
        r_z = r_z + q
        rem_sum = rem_sum + remainder
        I_z = I_z + eps_z[n]
        sum_u = sum_u + eps_u[n]
        sum_z = sum_z + eps_z[n]
    return out


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
    stems: list[int] | None = None,
    retention: bool = True,
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
        if stems is not None and not is_proxy(tangent):
            from l2_physical_jvp import load_dns_frame

            now = load_dns_frame(stems[k])
            nxt = load_dns_frame(stems[k + 1])
            pt.dns = now["omega"]
            pt.hats = now["hats"]
            pt.dns_next = nxt["omega"]
            pt.hats_next = nxt["hats"]
        points.append(pt)

    # Teacher-forced injections, model coordinates.
    eps_u = []
    eps_z = []
    xi = []
    for pt in points:
        g, h = model_step(ema_vit, pt.u, pt.z, pt.t)
        eu = (g - pt.u_next).detach()
        ez = (h - pt.z_next).detach()
        if is_proxy(tangent):
            xi_k = ez - apply_D(eu, ds)
        else:
            img = _phi_image(
                tangent, pt.u_next, eu, norm,
                dns=getattr(pt, "dns_next", None),
                hats=getattr(pt, "hats_next", None),
            )
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
        high = high.detach()
        z = z.detach()
        e.append(high - points[k].u_next)
        eta.append(z - points[k].z_next)

    e_lin = eta_lin = None
    if retention:
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
        v_for_z = _phi_image(
            tangent, points[k].u, vec, norm,
            dns=getattr(points[k], "dns", None),
            hats=getattr(points[k], "hats", None),
        )
        Jv, _ = jvp_both_uz(
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
        Bw, _Kw = jvp_both_z(
            lambda u, z: model_step(ema_vit, u, z, points[k].t),
            points[k],
            w,
            eps_rel,
        )
        return Bw

    # Scale to physical units for the reported norms and vector differences.
    e_phys = [t * norm for t in e]
    eta_phys = [t * norm for t in eta]
    eu_phys = [t * norm for t in eps_u]
    xi_phys = [t * norm for t in xi]
    out: dict = {}
    if retention:
        e_lin_phys = [t * norm for t in e_lin]
        eta_lin_phys = [t * norm for t in eta_lin]

        def J_phys(k, vec_phys):
            return apply_J_minus_I(k, vec_phys / norm) * norm

        def B_phys(k, w_phys):
            return apply_B(k, w_phys / norm) * norm

        out = running_channels(
            e_phys, eu_phys, xi_phys, J_phys, B_phys,
            e_lin=e_lin_phys, eta=eta_phys, eta_lin=eta_lin_phys,
        )
        out["e_lin_norm"] = np.asarray([_norm(t) for t in e_lin_phys])
        out["eta_lin_norm"] = np.asarray([_norm(t) for t in eta_lin_phys])
        out["e_lin_err"] = np.asarray([_norm(a - b) for a, b in zip(e_phys, e_lin_phys)])
        out["eta_lin_err"] = np.asarray([_norm(a - b) for a, b in zip(eta_phys, eta_lin_phys)])
    # nu at the last index needs the DNS field at that stem. The teacher
    # points cover 0..n_steps-1; the final error sits on frames[n_steps].
    points_nu = list(points)
    k_last = n_steps
    tail = make_point(
        [frames[k_last], frames[k_last + 1], frames[k_last + 1]],
        model,
        device,
        dtype,
        norm,
    )
    if stems is not None and not is_proxy(tangent):
        from l2_physical_jvp import load_dns_frame

        tail_dns = load_dns_frame(stems[k_last])
        tail.dns = tail_dns["omega"]
        tail.hats = tail_dns["hats"]
    points_nu.append(tail)
    nu_model, nu_available = nu_series(tangent, points_nu, e, eta, norm, ds)
    nu_phys = [item * norm for item in nu_model]
    ez_phys = [item * norm for item in eps_z]

    def apply_C_phys(k: int, vec_phys: torch.Tensor) -> torch.Tensor:
        if _norm(vec_phys) == 0.0:
            return torch.zeros_like(points[k].z)
        _fine, latent = jvp_both_u(
            lambda u, z, _t=points[k].t: model_step(ema_vit, u, z, _t),
            points[k],
            vec_phys / norm,
            eps_rel,
        )
        return latent * norm

    def apply_K_phys(k: int, vec_phys: torch.Tensor) -> torch.Tensor:
        if _norm(vec_phys) == 0.0:
            return torch.zeros_like(points[k].z)
        _fine, latent = jvp_both_z(
            lambda u, z, _t=points[k].t: model_step(ema_vit, u, z, _t),
            points[k],
            vec_phys / norm,
            eps_rel,
        )
        return latent * norm

    Qe_phys = []
    for point, err in zip(points_nu, e):
        missing = (not is_proxy(tangent)) and getattr(point, "dns", None) is None
        if missing or not torch.isfinite(err).all():
            Qe_phys.append(torch.full_like(point.z, float("nan")))
        else:
            Qe_phys.append(
                dphi_latent(tangent, point, err, norm, ds) * norm
            )
    channels = latent_error_channels(
        e_phys, eta_phys, eu_phys, ez_phys, apply_C_phys, apply_K_phys,
        Qe=Qe_phys, nu=nu_phys, first_factor=True,
    )
    out.update(channels)
    if retention:
        lin_channels = latent_error_channels(
            e_lin_phys, eta_lin_phys, eu_phys, ez_phys, apply_C_phys, apply_K_phys,
            first_factor=False,
        )
        out.update({f"lin_{key}": value for key, value in lin_channels.items()})
    else:
        blank = _nan_series(n_steps + 1)
        out.update({key: blank.copy() for key in LATENT_LIN_KEYS})
    out["q_operator"] = np.asarray("D" if is_proxy(tangent) else "DPhi")
    out.update(account_injections(eta_phys, ez_phys, xi_phys, nu_phys))
    out["nu_available"] = np.asarray(bool(nu_available))
    out["eps_u_norm"] = np.asarray([_norm(t) for t in eu_phys])
    out["eps_z_norm"] = np.asarray([_norm(t * norm) for t in eps_z])
    out["e_norm"] = np.asarray([_norm(t) for t in e_phys])
    out["eta_norm"] = np.asarray([_norm(t) for t in eta_phys])
    if retention:
        out["e_lin_rel"] = relative_residual(out["e_lin_err"], out["e_norm"])
        out["eta_lin_rel"] = relative_residual(out["eta_lin_err"], out["eta_norm"])
    out["approximation"] = np.asarray("proxy" if is_proxy(tangent) else "py2d_jvp")
    out["first_factor"] = np.asarray("truncated")
    return out
