"""Matrix-free L=2 blocks, Hutchinson traces, amplification, rho(S).

Proxy compositions use D Phi' ≈ D and are named proxy.
py2d_fd compositions use a finite-difference tangent through the py2d step.
That finite difference is not an exact tangent.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
import torch

from l2_fd_jvp import (
    FD_OUTPUT_FLOOR,
    L2Point,
    apply_D,
    fd_additivity,
    fd_both_u,
    fd_both_uz,
    fd_both_z,
)
from l2_physical_jvp import is_proxy
def _stats(samples: list[float] | np.ndarray) -> dict:
    x = np.asarray(samples, dtype=np.float64)
    m = int(x.size)
    mean = float(x.mean()) if m else float("nan")
    std = float(x.std(ddof=1)) if m > 1 else 0.0
    half = float(1.96 * std / np.sqrt(m)) if m > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": half, "probes": x}


def _rademacher(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n).astype(np.float32)


def _as_image(flat: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(flat))
        .to(device=like.device, dtype=like.dtype)
        .reshape_as(like)
    )


def quotient_halfwidth(trace_ci95, dim, dt):
    """CI of (Tr - dim) / (dim * dt). Tr(F_u)/N uses tr_A_ci95 / (N * dt)."""
    return np.asarray(trace_ci95, dtype=np.float64) / (
        np.asarray(dim, dtype=np.float64) * float(dt)
    )


def _phi_image(tangent, u_model: torch.Tensor, v_model: torch.Tensor, norm: float) -> torch.Tensor:
    """Phi'(u) v in model coordinates. The proxy returns v."""
    if is_proxy(tangent):
        return v_model
    u_phys = (u_model[0, 0].detach().float().cpu().numpy()) * norm
    v_phys = (v_model[0, 0].detach().float().cpu().numpy()) * norm
    jv = tangent.apply(u_phys, v_phys)
    out = torch.from_numpy(np.asarray(jv, dtype=np.float32)) / norm
    return out.to(device=v_model.device, dtype=v_model.dtype).view_as(v_model)


def _rel_residual(residual: torch.Tensor, reference: torch.Tensor) -> float:
    denom = max(float(reference.reshape(-1).double().norm()), FD_OUTPUT_FLOOR)
    return float(residual.reshape(-1).double().norm()) / denom


def diagnose_point(
    step_fn,
    point: L2Point,
    ds: int,
    eps_rel: float,
    tangent,
    dt: float,
    norm: float,
    n_probes: int,
    seed: int,
    spectral_radius: bool,
    ncv: int,
    maxiter: int,
    tol: float,
) -> dict:
    """Hutchinson diagnostics at one teacher-forced point.

    Fine probes estimate Tr(A), Tr(B D Phi'_n), and Tr(J). Coarse probes
    estimate Tr(K), Tr(DB), and Tr(S). Tr(DB) always uses apply_D(Bw).
    S uses D Phi' at u_{n+1}. Those two are not the same operator.
    """
    proxy = is_proxy(tangent)
    tag = "proxy" if proxy else "py2d_fd"
    rng = np.random.default_rng(seed)
    n_fine = int(point.u.numel())
    n_coarse = int(point.z.numel())

    tr_A, tr_feedback, tr_J, amp, off, comp = [], [], [], [], [], []
    for _ in range(n_probes):
        v = _as_image(_rademacher(n_fine, rng), point.u)
        # w = D Phi'_n v. Proxy Phi' is the identity, so w = D v.
        w = apply_D(_phi_image(tangent, point.u, v, norm), ds)
        Av, _Cv = fd_both_u(step_fn, point, v, eps_rel)
        Bw, _Kw = fd_both_z(step_fn, point, w, eps_rel)
        Jv, mixed_h = fd_both_uz(step_fn, point, v, w, eps_rel)
        if proxy:
            lifted = Jv
        else:
            lifted = _phi_image(tangent, point.u_next, Jv, norm)
        # (C + K D Phi'_n) v - D Phi'_{n+1} J v
        L = mixed_h - apply_D(lifted, ds)
        vf = v.reshape(-1).double()
        tr_A.append(float(torch.dot(vf, Av.reshape(-1).double())))
        tr_feedback.append(float(torch.dot(vf, Bw.reshape(-1).double())))
        tr_J.append(float(torch.dot(vf, Jv.reshape(-1).double())))
        amp.append(float(Jv.reshape(-1).double().pow(2).sum() / n_fine))
        off.append(float(L.reshape(-1).double().pow(2).sum() / n_fine))
        comp.append(_rel_residual(Jv - Av - Bw, Jv))

    tr_K, tr_DB, tr_DPhi_next, tr_S = [], [], [], []
    for _ in range(n_probes):
        w = _as_image(_rademacher(n_coarse, rng), point.z)
        Bw, Kw = fd_both_z(step_fn, point, w, eps_rel)
        DBw = apply_D(Bw, ds)
        if proxy:
            DPhi_next_B = DBw
        else:
            DPhi_next_B = apply_D(_phi_image(tangent, point.u_next, Bw, norm), ds)
        Sw = Kw - DPhi_next_B
        wf = w.reshape(-1).double()
        tr_K.append(float(torch.dot(wf, Kw.reshape(-1).double())))
        tr_DB.append(float(torch.dot(wf, DBw.reshape(-1).double())))
        tr_DPhi_next.append(float(torch.dot(wf, DPhi_next_B.reshape(-1).double())))
        tr_S.append(float(torch.dot(wf, Sw.reshape(-1).double())))

    sA, sFeed, sJ = _stats(tr_A), _stats(tr_feedback), _stats(tr_J)
    sK, sDB, sNext, sS = _stats(tr_K), _stats(tr_DB), _stats(tr_DPhi_next), _stats(tr_S)
    sAmp, sOff, sComp = _stats(amp), _stats(off), _stats(comp)
    gap = sFeed["mean"] - sDB["mean"]
    gap_ci = float(np.hypot(sFeed["ci95"], sDB["ci95"]))

    # Difference-quotient proxies. These are NOT a claim that HINE is Euler.
    # tr_DFz uses Tr(DB), not the physical feedback and not D Phi'_{n+1} B.
    def quot(tr_mean: float, dim: int) -> float:
        return (tr_mean - dim) / (dim * dt)

    add = fd_additivity(step_fn, point, eps_rel, seed)

    out = {
        "approximation": tag,
        "n_fine": n_fine,
        "n_coarse": n_coarse,
        "dt": float(dt),
        "tr_A": sA,
        "tr_B_DPhi_n": sFeed,
        "tr_J": sJ,
        "tr_K": sK,
        "tr_DB": sDB,
        "tr_DPhi_next_B": sNext,
        "tr_S": sS,
        "amp_J": sAmp,
        "offgraph_probe": sOff,
        "composition_rel": sComp,
        "tr_A_over_N": sA["mean"] / n_fine,
        "tr_J_over_N": sJ["mean"] / n_fine,
        "tr_K_over_M": sK["mean"] / n_coarse,
        "tr_S_over_M": sS["mean"] / n_coarse,
        "tr_DB_over_N": sDB["mean"] / n_fine,
        "tr_B_DPhi_n_over_N": sFeed["mean"] / n_fine,
        "tr_Fu_over_N": quot(sA["mean"], n_fine),
        "tr_DFz_over_N": sDB["mean"] / (n_fine * dt),
        "tr_Heff_over_N": quot(sJ["mean"], n_fine),
        "tr_Sigma_over_M": quot(sS["mean"], n_coarse),
        "composition_rel_mean": sComp["mean"],
        "composition_rel_max": float(np.max(sComp["probes"])) if sComp["probes"].size else float("nan"),
        "trace_estimator_gap": gap,
        "trace_estimator_gap_ci95": gap_ci,
        "trace_estimator_same_operator": bool(proxy),
        "fd_additivity_rel": add["rel"],
        "fd_additivity": add,
    }

    if spectral_radius:
        rad = rho_S(
            step_fn, point, ds, eps_rel, tangent, norm, ncv, maxiter, tol, seed
        )
        rad["fd_additivity_rel"] = add["rel"]
        out["rho_S"] = rad
    return out


def rho_S(step_fn, point, ds, eps_rel, tangent, norm, ncv, maxiter, tol, seed) -> dict:
    """Leading-magnitude eigenvalue of S. S uses D Phi' at u_{n+1}."""
    n = int(point.z.numel())
    proxy = is_proxy(tangent)
    tag = "proxy" if proxy else "py2d_fd"

    def matvec(v_np: np.ndarray) -> np.ndarray:
        w = _as_image(v_np, point.z)
        Bw, Kw = fd_both_z(step_fn, point, w, eps_rel)
        if proxy:
            DPhi_next_B = apply_D(Bw, ds)
        else:
            DPhi_next_B = apply_D(_phi_image(tangent, point.u_next, Bw, norm), ds)
        return (Kw - DPhi_next_B).detach().reshape(-1).float().cpu().numpy()

    ncv_eff = min(max(int(ncv), 2), n - 1)
    if ncv_eff < 2:
        return {
            "value_real": float("nan"),
            "value_imag": 0.0,
            "radius": float("nan"),
            "converged": False,
            "error": f"state dim {n} is too small for ARPACK",
            "approximation": tag,
        }
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float32)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    converged = True
    try:
        vals = spla.eigs(
            op, k=1, which="LM", ncv=ncv_eff, maxiter=maxiter, tol=tol, v0=v0,
            return_eigenvectors=False,
        )
        val = complex(vals[0])
    except Exception as exc:
        converged = False
        val = complex("nan")
        err = f"{type(exc).__name__}: {exc}"
    else:
        err = ""
    return {
        "value_real": float(np.real(val)),
        "value_imag": float(np.imag(val)),
        "radius": float(abs(val)) if np.isfinite(val) else float("nan"),
        "converged": converged,
        "error": err,
        "approximation": tag,
    }


def flat_diag(diag: dict) -> dict[str, np.ndarray]:
    """Scalar view of diagnose_point for np.savez."""
    keys = [
        "tr_A_over_N",
        "tr_J_over_N",
        "tr_K_over_M",
        "tr_S_over_M",
        "tr_DB_over_N",
        "tr_B_DPhi_n_over_N",
        "tr_Fu_over_N",
        "tr_DFz_over_N",
        "tr_Heff_over_N",
        "tr_Sigma_over_M",
        "composition_rel_mean",
        "composition_rel_max",
        "trace_estimator_gap",
        "trace_estimator_gap_ci95",
        "fd_additivity_rel",
        "n_fine",
        "n_coarse",
    ]
    out = {k: np.asarray(diag[k]) for k in keys}
    for name in (
        "tr_A",
        "tr_B_DPhi_n",
        "tr_J",
        "tr_K",
        "tr_DB",
        "tr_DPhi_next_B",
        "tr_S",
        "amp_J",
        "offgraph_probe",
    ):
        block = diag[name]
        out[f"{name}_mean"] = np.asarray(block["mean"])
        out[f"{name}_std"] = np.asarray(block["std"])
        out[f"{name}_ci95"] = np.asarray(block["ci95"])
        out[f"{name}_probes"] = np.asarray(block["probes"])
    if "rho_S" in diag:
        out["rho_S"] = np.asarray(diag["rho_S"]["radius"])
        out["rho_S_real"] = np.asarray(diag["rho_S"]["value_real"])
        out["rho_S_imag"] = np.asarray(diag["rho_S"]["value_imag"])
        out["rho_S_converged"] = np.asarray(diag["rho_S"]["converged"])
    out["approximation"] = np.asarray(diag["approximation"])
    return out
