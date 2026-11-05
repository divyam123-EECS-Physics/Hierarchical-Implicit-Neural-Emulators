"""Matrix-free L=2 blocks, Hutchinson traces, amplification, rho(S).

Proxy compositions use D Phi' ≈ D and are named *_proxy.
Exact compositions use PhysicalTangent.apply and are the Section 7 operators.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla
import torch

from l2_fd_jvp import (
    L2Point,
    apply_D,
    fd_both_u,
    fd_both_uz,
    fd_both_z,
)
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


def _phi_image(tangent, u_model: torch.Tensor, v_model: torch.Tensor, norm: float) -> torch.Tensor:
    """Phi'(u) v in model coordinates. Proxy returns v."""
    if not tangent.is_exact:
        return v_model
    u_phys = (u_model[0, 0].detach().float().cpu().numpy()) * norm
    v_phys = (v_model[0, 0].detach().float().cpu().numpy()) * norm
    jv = tangent.apply(u_phys, v_phys)
    out = torch.from_numpy(np.asarray(jv, dtype=np.float32)) / norm
    return out.to(device=v_model.device, dtype=v_model.dtype).view_as(v_model)


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

    Fine probes estimate Tr(A), Tr(BD), Tr(J_phys) and the amplification
    and off-graph probe norms. Coarse probes estimate Tr(K), Tr(DB), Tr(S).
    """
    proxy = not tangent.is_exact
    tag = "proxy" if proxy else "exact"
    rng = np.random.default_rng(seed)
    n_fine = int(point.u.numel())
    n_coarse = int(point.z.numel())

    tr_A, tr_BD, tr_J, amp, off = [], [], [], [], []
    for _ in range(n_probes):
        v = _as_image(_rademacher(n_fine, rng), point.u)
        Av, _Cv = fd_both_u(step_fn, point, v, eps_rel)
        v_for_z = _phi_image(tangent, point.u, v, norm)
        Jv, CKdv = fd_both_uz(step_fn, point, v, apply_D(v_for_z, ds), eps_rel)
        BDv = Jv - Av
        L = CKdv - apply_D(_phi_image(tangent, point.u_next, Jv, norm) if not proxy else Jv, ds)
        # Off-graph: (C + K D Phi') v - D Phi'_{n+1} (A + B D Phi') v
        # CKdv is (C v + K w) with w = D Phi' v, Jv is (A+BD-style) J v.
        vf = v.reshape(-1).double()
        tr_A.append(float(torch.dot(vf, Av.reshape(-1).double())))
        tr_BD.append(float(torch.dot(vf, BDv.reshape(-1).double())))
        tr_J.append(float(torch.dot(vf, Jv.reshape(-1).double())))
        amp.append(float(Jv.reshape(-1).double().pow(2).sum() / n_fine))
        off.append(float(L.reshape(-1).double().pow(2).sum() / n_fine))

    tr_K, tr_DB, tr_S = [], [], []
    for _ in range(n_probes):
        w = _as_image(_rademacher(n_coarse, rng), point.z)
        Bw, Kw = fd_both_z(step_fn, point, w, eps_rel)
        if proxy:
            DBw = apply_D(Bw, ds)
        else:
            img = _phi_image(tangent, point.u_next, Bw, norm)
            DBw = apply_D(img, ds)
        Sw = Kw - DBw
        wf = w.reshape(-1).double()
        tr_K.append(float(torch.dot(wf, Kw.reshape(-1).double())))
        tr_DB.append(float(torch.dot(wf, DBw.reshape(-1).double())))
        tr_S.append(float(torch.dot(wf, Sw.reshape(-1).double())))

    sA, sBD, sJ = _stats(tr_A), _stats(tr_BD), _stats(tr_J)
    sK, sDB, sS = _stats(tr_K), _stats(tr_DB), _stats(tr_S)
    sAmp, sOff = _stats(amp), _stats(off)

    # Difference-quotient proxies. These are NOT a claim that HINE is Euler.
    def quot(tr_mean: float, dim: int) -> float:
        return (tr_mean - dim) / (dim * dt)

    out = {
        "approximation": tag,
        "n_fine": n_fine,
        "n_coarse": n_coarse,
        "dt": float(dt),
        "tr_A": sA,
        "tr_BD_paired": sBD,
        "tr_J": sJ,
        "tr_K": sK,
        "tr_DB": sDB,
        "tr_S": sS,
        "amp_J": sAmp,
        "offgraph_probe": sOff,
        "tr_A_over_N": sA["mean"] / n_fine,
        "tr_J_over_N": sJ["mean"] / n_fine,
        "tr_K_over_M": sK["mean"] / n_coarse,
        "tr_S_over_M": sS["mean"] / n_coarse,
        "tr_DB_over_N": sDB["mean"] / n_fine,
        "tr_Fu_over_N": quot(sA["mean"], n_fine),
        "tr_DFz_over_N": sDB["mean"] / (n_fine * dt),
        "tr_Heff_over_N": quot(sJ["mean"], n_fine),
        "tr_Sigma_over_M": quot(sS["mean"], n_coarse),
        "identity_gap": sJ["mean"] - (sA["mean"] + sBD["mean"]),
    }

    if spectral_radius:
        rad = rho_S(
            step_fn, point, ds, eps_rel, tangent, norm, ncv, maxiter, tol, seed
        )
        out["rho_S"] = rad
    return out


def rho_S(step_fn, point, ds, eps_rel, tangent, norm, ncv, maxiter, tol, seed) -> dict:
    """Leading-magnitude eigenvalue of S (or S_proxy)."""
    n = int(point.z.numel())
    proxy = not tangent.is_exact

    def matvec(v_np: np.ndarray) -> np.ndarray:
        w = _as_image(v_np, point.z)
        Bw, Kw = fd_both_z(step_fn, point, w, eps_rel)
        if proxy:
            DBw = apply_D(Bw, ds)
        else:
            DBw = apply_D(_phi_image(tangent, point.u_next, Bw, norm), ds)
        return (Kw - DBw).detach().reshape(-1).float().cpu().numpy()

    ncv_eff = min(max(int(ncv), 2), n - 1)
    if ncv_eff < 2:
        return {
            "value_real": float("nan"),
            "value_imag": 0.0,
            "radius": float("nan"),
            "converged": False,
            "error": f"state dim {n} is too small for ARPACK",
            "approximation": "proxy" if proxy else "exact",
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
        "approximation": "proxy" if proxy else "exact",
    }


def flat_diag(diag: dict) -> dict[str, np.ndarray]:
    """Scalar view of diagnose_point for np.savez."""
    keys = [
        "tr_A_over_N",
        "tr_J_over_N",
        "tr_K_over_M",
        "tr_S_over_M",
        "tr_DB_over_N",
        "tr_Fu_over_N",
        "tr_DFz_over_N",
        "tr_Heff_over_N",
        "tr_Sigma_over_M",
        "identity_gap",
        "n_fine",
        "n_coarse",
    ]
    out = {k: np.asarray(diag[k]) for k in keys}
    for name in ("tr_A", "tr_BD_paired", "tr_J", "tr_K", "tr_DB", "tr_S", "amp_J", "offgraph_probe"):
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
