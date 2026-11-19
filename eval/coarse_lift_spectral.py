"""Matrix-free spectra for the coarse-lift operators T, U, C, and CQ.

T and U share nonzero eigenvalues. Singular values are computed separately:
they do not inherit that correspondence. Eigenvalues of T below one are not
a one-step contraction of HINE, and values at separate states are not the
spectrum of a time-varying product.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse.linalg as spla

from coarse_lift_ops import ZERO_FLOOR, D_dagger_numpy, D_numpy, Q_numpy, pool_shapes, rms_scale

def apply_complex(matvec, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    real = np.asarray(matvec(np.real(vector)), dtype=np.float64)
    imag = np.asarray(matvec(np.imag(vector)), dtype=np.float64)
    return real + 1j * imag


def eigen_residual(matvec, value: complex, vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    image = apply_complex(matvec, vector)
    numer = float(np.linalg.norm(image - value * vector))
    denom = max(float(np.linalg.norm(vector)), ZERO_FLOOR)
    return numer / denom


def singular_residual(matvec, rmatvec, sigma: float, u_vec: np.ndarray, v_vec: np.ndarray) -> dict:
    u_vec = np.asarray(u_vec, dtype=np.float64).reshape(-1)
    v_vec = np.asarray(v_vec, dtype=np.float64).reshape(-1)
    Av = np.asarray(matvec(v_vec), dtype=np.float64)
    Au = np.asarray(rmatvec(u_vec), dtype=np.float64)
    right = float(np.linalg.norm(Av - sigma * u_vec))
    left = float(np.linalg.norm(Au - sigma * v_vec))
    scale = max(abs(float(sigma)), ZERO_FLOOR)
    return {
        "right_abs": right,
        "left_abs": left,
        "right_rel": right / scale,
        "left_rel": left / scale,
    }


def _sort_eigen(values: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.abs(values))
    return values[order], vectors[:, order]


def leading_eigenpairs(matvec, n: int, k: int, ncv: int, maxiter: int, tol: float, seed: int, rmatvec=None) -> dict:
    """Leading-magnitude eigenpairs. Left eigenvectors use rmatvec when it is supplied."""
    k_req = int(k)
    if n < 2 or k_req < 1:
        return {
            "converged": False,
            "error": f"state dim {n} or k {k_req} is too small",
            "values": np.zeros(0, dtype=np.complex128),
            "vectors": np.zeros((n, 0), dtype=np.complex128),
            "residuals": np.zeros(0, dtype=np.float64),
            "n_matvec": 0,
        }
    k_eff = min(k_req, n - 2 if n > 2 else 1)
    ncv_eff = min(max(int(ncv), k_eff + 2), n - 1)
    if ncv_eff <= k_eff:
        ncv_eff = min(n - 1, k_eff + 1)
    if ncv_eff <= k_eff:
        return {
            "converged": False,
            "error": f"cannot choose ncv for n={n} k={k_eff}",
            "values": np.zeros(0, dtype=np.complex128),
            "vectors": np.zeros((n, 0), dtype=np.complex128),
            "residuals": np.zeros(0, dtype=np.float64),
            "n_matvec": 0,
        }
    calls = {"n": 0}

    def counted(v):
        calls["n"] += 1
        return np.asarray(matvec(v), dtype=np.float64)

    def counted_r(v):
        calls["n"] += 1
        return np.asarray(rmatvec(v), dtype=np.float64)

    op_kwargs = {"matvec": counted, "dtype": np.float64}
    if rmatvec is not None:
        op_kwargs["rmatvec"] = counted_r
    op = spla.LinearOperator((n, n), **op_kwargs)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    v0 /= max(np.linalg.norm(v0), ZERO_FLOOR)

    def pack(vals, vecs, ncv_used: int, error: str) -> dict:
        vals = np.asarray(vals, dtype=np.complex128).reshape(-1)
        vecs = np.asarray(vecs, dtype=np.complex128)
        if vecs.ndim == 1 and vals.size == 1:
            vecs = vecs.reshape(-1, 1)
        if vals.size == 0 or vecs.ndim != 2 or vecs.shape[1] != vals.size or vecs.shape[0] != n:
            vals = np.zeros(0, dtype=np.complex128)
            vecs = np.zeros((n, 0), dtype=np.complex128)
        else:
            vals, vecs = _sort_eigen(vals, vecs)
        residuals = np.array([eigen_residual(counted, vals[i], vecs[:, i]) for i in range(vals.size)])
        for i in range(vecs.shape[1]):
            scale = np.linalg.norm(vecs[:, i])
            if scale > 0:
                vecs[:, i] /= scale
        residual_tol = max(float(tol), 1e-4)
        mode_ok = np.isfinite(residuals) & (residuals <= residual_tol) if residuals.size else np.zeros(0, dtype=bool)
        full = bool(vals.size >= k_eff and mode_ok.size and np.all(mode_ok))
        return {
            "converged": full,
            "error": "" if full else error,
            "values": vals,
            "vectors": vecs,
            "residuals": residuals.astype(np.float64),
            "mode_converged": mode_ok,
            "n_matvec": int(calls["n"]),
            "k": k_eff,
            "ncv": int(ncv_used),
            "residual_tol": residual_tol,
        }

    def once(ncv_used: int, iters: int) -> dict:
        try:
            vals, vecs = spla.eigs(
                op,
                k=k_eff,
                which="LM",
                ncv=int(ncv_used),
                maxiter=int(iters),
                tol=float(tol),
                v0=v0,
                return_eigenvectors=True,
            )
            return pack(vals, vecs, ncv_used, "")
        except spla.ArpackNoConvergence as exc:
            return pack(
                getattr(exc, "eigenvalues", np.zeros(0)),
                getattr(exc, "eigenvectors", np.zeros((n, 0))),
                ncv_used,
                f"ArpackNoConvergence: {exc}",
            )
        except Exception as exc:
            return pack(
                np.zeros(0, dtype=np.complex128),
                np.zeros((n, 0), dtype=np.complex128),
                ncv_used,
                f"{type(exc).__name__}: {exc}",
            )

    first = once(ncv_eff, int(maxiter))
    if first["converged"]:
        return first
    ncv_retry = min(n - 1, max(ncv_eff * 2, k_eff + 10))
    if ncv_retry <= ncv_eff and int(maxiter) < 2:
        return first
    second = once(ncv_retry, max(int(maxiter) * 2, int(maxiter) + 1))
    if second["values"].size >= first["values"].size:
        return second
    return first


def match_left_eigenpairs(right_values: np.ndarray, left_values: np.ndarray, left_vectors: np.ndarray) -> dict:
    """Match adjoint eigenpairs to right eigenvalues.

    T* y = μ y means y^H T = conj(μ) y^H, so y is a left eigenvector for
    conj(μ). A real operator's Arnoldi run can return μ = λ instead of
    conj(λ) when a conjugate pair is split by the requested subspace size.
    In that case conj(y) is the left eigenvector for λ.
    """
    n = right_values.size
    matched_values = np.full(n, np.nan + 0j, dtype=np.complex128)
    matched_vectors = np.zeros((left_vectors.shape[0], n), dtype=np.complex128)
    distances = np.full(n, np.nan, dtype=np.float64)
    conjugated = np.zeros(n, dtype=bool)
    used: set[int] = set()
    for i, lam in enumerate(right_values):
        target = np.conj(lam)
        best_j = None
        best_d = float("inf")
        use_conjugate = False
        for j, mu in enumerate(left_values):
            if j in used:
                continue
            dist_target = abs(complex(mu) - complex(target))
            dist_same = abs(complex(mu) - complex(lam))
            if dist_target <= dist_same:
                dist, conjugate = dist_target, False
            else:
                dist, conjugate = dist_same, True
            if dist < best_d:
                best_d = dist
                best_j = j
                use_conjugate = conjugate
        if best_j is None:
            continue
        used.add(best_j)
        vector = left_vectors[:, best_j]
        value = left_values[best_j]
        if use_conjugate:
            vector = np.conj(vector)
            value = np.conj(value)
        matched_values[i] = value
        matched_vectors[:, i] = vector
        distances[i] = best_d
        conjugated[i] = use_conjugate
    return {
        "values": matched_values,
        "vectors": matched_vectors,
        "match_distance": distances,
        "conjugated": conjugated,
    }


def partial_svd(matvec, rmatvec, n_out: int, n_in: int, k: int, ncv: int, maxiter: int, tol: float, seed: int) -> dict:
    """Leading singular triplets. scipy returns ascending values; we sort descending."""
    k_req = int(k)
    limit = min(n_out, n_in)
    if limit < 3 or k_req < 1:
        return {
            "converged": False,
            "error": f"SVD dims {(n_out, n_in)} are too small",
            "values": np.zeros(0, dtype=np.float64),
            "u": np.zeros((n_out, 0), dtype=np.float64),
            "v": np.zeros((n_in, 0), dtype=np.float64),
            "residuals_right_rel": np.zeros(0),
            "residuals_left_rel": np.zeros(0),
        }
    k_eff = min(k_req, limit - 2)
    ncv_eff = min(max(int(ncv), k_eff + 2), limit - 1)
    if ncv_eff <= k_eff:
        return {
            "converged": False,
            "error": f"cannot choose SVD ncv for shape {(n_out, n_in)} k={k_eff}",
            "values": np.zeros(0, dtype=np.float64),
            "u": np.zeros((n_out, 0), dtype=np.float64),
            "v": np.zeros((n_in, 0), dtype=np.float64),
            "residuals_right_rel": np.zeros(0),
            "residuals_left_rel": np.zeros(0),
        }
    op = spla.LinearOperator(
        (n_out, n_in),
        matvec=lambda v: np.asarray(matvec(v), dtype=np.float64),
        rmatvec=lambda v: np.asarray(rmatvec(v), dtype=np.float64),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    n_v0 = min(n_out, n_in)
    v0 = rng.standard_normal(n_v0).astype(np.float64)
    v0 /= max(np.linalg.norm(v0), ZERO_FLOOR)
    kwargs = dict(k=k_eff, which="LM", ncv=ncv_eff, maxiter=int(maxiter), tol=float(tol), return_singular_vectors=True)
    try:
        try:
            u, s, vt = spla.svds(op, v0=v0, **kwargs)
        except TypeError:
            u, s, vt = spla.svds(op, **kwargs)
    except Exception as exc:
        return {
            "converged": False,
            "error": f"{type(exc).__name__}: {exc}",
            "values": np.zeros(0, dtype=np.float64),
            "u": np.zeros((n_out, 0), dtype=np.float64),
            "v": np.zeros((n_in, 0), dtype=np.float64),
            "residuals_right_rel": np.zeros(0),
            "residuals_left_rel": np.zeros(0),
            "k": k_eff,
            "ncv": ncv_eff,
        }
    s = np.asarray(s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.float64)
    order = np.argsort(-s)
    s = s[order]
    u = u[:, order]
    v = vt[order, :].T
    right_rel = []
    left_rel = []
    for i in range(s.size):
        stats = singular_residual(matvec, rmatvec, float(s[i]), u[:, i], v[:, i])
        right_rel.append(stats["right_rel"])
        left_rel.append(stats["left_rel"])
    right_rel = np.asarray(right_rel, dtype=np.float64)
    left_rel = np.asarray(left_rel, dtype=np.float64)
    residual_tol = max(float(tol), 1e-4)
    mode_ok = np.isfinite(right_rel) & np.isfinite(left_rel) & (right_rel <= residual_tol) & (left_rel <= residual_tol)
    return {
        "converged": bool(s.size > 0 and np.all(mode_ok)),
        "error": "",
        "values": s,
        "u": u,
        "v": v,
        "residuals_right_rel": right_rel,
        "residuals_left_rel": left_rel,
        "mode_converged": mode_ok,
        "k": k_eff,
        "ncv": ncv_eff,
        "residual_tol": residual_tol,
    }


def _pair_nonzero(left: np.ndarray, right: np.ndarray, tol: float) -> float:
    """Max distance after greedy matching of nonzero eigenvalues."""
    a = [complex(v) for v in np.asarray(left).ravel() if abs(v) > tol]
    b = [complex(v) for v in np.asarray(right).ravel() if abs(v) > tol]
    if len(a) != len(b):
        return float("inf")
    used = set()
    worst = 0.0
    for val in sorted(a, key=abs, reverse=True):
        best_j = None
        best_d = float("inf")
        for j, other in enumerate(b):
            if j in used:
                continue
            dist = abs(val - other)
            if dist < best_d:
                best_d = dist
                best_j = j
        used.add(best_j)
        worst = max(worst, best_d)
    return float(worst)


def dense_operator_correspondence(fine_n: int = 4, factor: int = 2, seed: int = 0) -> dict:
    """Dense T/U correspondence, including extra zero modes and unequal singular values.

    The structured C is large on an unresolved fine direction and small on a
    block-constant direction, so σ_max(U) exceeds σ_max(T).
    """
    fine_n, coarse_n, n_fine, n_coarse = pool_shapes(fine_n, factor)
    rng = np.random.default_rng(seed)

    def pack(field: np.ndarray) -> np.ndarray:
        return np.asarray(field, dtype=np.float64).reshape(-1)

    def unpack_fine(vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64).reshape(fine_n, fine_n)

    def unpack_coarse(vec: np.ndarray) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64).reshape(coarse_n, coarse_n)

    D = np.column_stack([
        pack(D_numpy(unpack_fine(np.eye(n_fine)[:, j]), factor)) for j in range(n_fine)
    ])
    eye_c = np.eye(n_coarse)
    D_dag = np.column_stack([
        pack(D_dagger_numpy(unpack_coarse(eye_c[:, j]), factor)) for j in range(n_coarse)
    ])
    C_random = rng.standard_normal((n_coarse, n_fine)) / math.sqrt(n_fine)

    q = rng.standard_normal((fine_n, fine_n))
    q = Q_numpy(q, factor)
    q /= max(np.linalg.norm(q), ZERO_FLOOR)
    p = D_dagger_numpy(rng.standard_normal((coarse_n, coarse_n)), factor)
    p /= max(np.linalg.norm(p), ZERO_FLOOR)
    e0 = np.zeros(n_coarse)
    e0[0] = 1.0
    e1 = np.zeros(n_coarse)
    e1[min(1, n_coarse - 1)] = 1.0
    C_structured = 10.0 * np.outer(e0, pack(q)) + 0.1 * np.outer(e1, pack(p))

    reports = {}
    for name, C in (("random", C_random), ("structured", C_structured)):
        T = C @ D_dag
        U = D_dag @ C
        eval_T = np.linalg.eigvals(T)
        eval_U = np.linalg.eigvals(U)
        zero_tol = 1e-8
        n_zero_T = int(np.sum(np.abs(eval_T) <= zero_tol))
        n_zero_U = int(np.sum(np.abs(eval_U) <= zero_tol))
        s_T = np.linalg.svd(T, compute_uv=False)
        s_U = np.linalg.svd(U, compute_uv=False)
        # Right eigenvectors of T, lifted by D†.
        eig_T, vec_T = np.linalg.eig(T)
        lift_residuals = []
        for i, lam in enumerate(eig_T):
            if abs(lam) <= zero_tol:
                continue
            lifted = D_dag @ vec_T[:, i]
            residual = np.linalg.norm(U @ lifted - lam * lifted) / max(np.linalg.norm(lifted), ZERO_FLOOR)
            lift_residuals.append(float(residual))
        # Left eigenvectors from the adjoint. T is real, so eig(T.T) gives conj(λ).
        eig_left, vec_left = np.linalg.eig(T.T)
        left_residuals = []
        for i, mu in enumerate(eig_left):
            lam = np.conj(mu)
            if abs(lam) <= zero_tol:
                continue
            z = C.conj().T @ vec_left[:, i]
            if np.linalg.norm(z) <= zero_tol:
                left_residuals.append(float("inf"))
                continue
            # U* z = conj(λ) z when z = C* y and y is a left eigenvector for λ.
            residual = np.linalg.norm(U.conj().T @ z - np.conj(lam) * z) / np.linalg.norm(z)
            left_residuals.append(float(np.real(residual)))
        reports[name] = {
            "nonzero_eigen_gap": _pair_nonzero(eval_T, eval_U, zero_tol),
            "n_zero_T": n_zero_T,
            "n_zero_U": n_zero_U,
            "extra_zeros": int(n_zero_U - n_zero_T),
            "expected_extra_zeros": int(n_fine - n_coarse),
            "right_lift_max_residual": float(max(lift_residuals) if lift_residuals else 0.0),
            "left_lift_max_residual": float(max(left_residuals) if left_residuals else 0.0),
            "sigma_max_T": float(s_T[0]),
            "sigma_max_U": float(s_U[0]),
            "sigma_max_gap": float(abs(s_T[0] - s_U[0])),
            "rms_scale_C": rms_scale(n_coarse, n_fine),
            "rms_scale_T": rms_scale(n_coarse, n_coarse),
        }
    structured_gap = reports["structured"]["sigma_max_gap"]
    ok = (
        reports["random"]["nonzero_eigen_gap"] < 1e-6
        and reports["random"]["extra_zeros"] == n_fine - n_coarse
        and reports["random"]["right_lift_max_residual"] < 1e-6
        and reports["random"]["left_lift_max_residual"] < 1e-6
        and reports["structured"]["sigma_max_U"] > reports["structured"]["sigma_max_T"] * 2
        and structured_gap > 1e-3
        and reports["random"]["rms_scale_T"] == 1.0
        and abs(reports["random"]["rms_scale_C"] - math.sqrt(n_fine / n_coarse)) < 1e-12
    )
    return {
        "ok": bool(ok),
        "fine_n": fine_n,
        "factor": factor,
        "N": n_fine,
        "M": n_coarse,
        "cases": reports,
        "note": (
            "Nonzero eigenvalues of T=C D† and U=D† C match. "
            "U has N-M additional zero eigenvalues. "
            "Singular values do not match; the structured case makes σ(U) larger "
            "because C sees unresolved fine directions that D† cannot feed into T."
        ),
    }


def sampled_directional_gains(action, seed: int, n_probes: int = 4) -> dict:
    """Rademacher directional gains. These are not maximum singular values."""
    rng = np.random.default_rng(seed)
    eucl = []
    rms = []
    for _ in range(n_probes):
        signs = rng.choice(np.array([-1.0, 1.0]), size=action.n_in)
        image = action.matvec(signs)
        base = float(np.linalg.norm(signs))
        out = float(np.linalg.norm(image))
        gain = out / max(base, ZERO_FLOOR)
        eucl.append(gain)
        rms.append(gain * math.sqrt(action.n_in / action.n_out))
    return {
        "euclidean_mean": float(np.mean(eucl)),
        "euclidean_max": float(np.max(eucl)),
        "rms_mean": float(np.mean(rms)),
        "rms_max": float(np.max(rms)),
        "n_probes": n_probes,
        "label": "sampled directional gain, not sigma_max",
    }
