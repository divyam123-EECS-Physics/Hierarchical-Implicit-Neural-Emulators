"""Pooling pseudoinverse and matrix-free fine-to-coarse blocks for L=2 HINE.

D is nonoverlapping uniform average pooling. D† is its Moore–Penrose
pseudoinverse: each coarse value is repeated over its r-by-r block.
D* is the Euclidean adjoint, which writes the coarse value divided by r².
These are not the same map. D† = r² D*.

C is dh/du at a teacher-forced latent input. The latent tensor is held
fixed. No latent initialization is differentiated through the fine input.

The network is evaluated on vorticity / norm. The physical map multiplies
the network output by norm, so its Jacobian equals the normalized Jacobian.
Operator methods below take and return physical fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from l2_fd_jvp import L2Point, jvp_both_u, jvp_both_uz, jvp_both_z

ZERO_FLOOR = 1e-12

# Deterministic subset of the corrected 20-window Task 1 manifest.
# Samples are positions in that manifest, which is already sorted by start.
SPECTRAL_SAMPLE_INDICES = (0, 10, 19)
SPECTRAL_STEPS = (0, 20, 39)
TWOSTEP_STEPS = (0, 20, 38)
BUDGET_LENGTH = 40


def pool_shapes(fine_n: int, factor: int) -> tuple[int, int, int, int]:
    """Return (fine side, coarse side, N, M) for a square grid."""
    if factor < 1 or fine_n % factor != 0:
        raise ValueError(f"factor {factor} does not divide fine side {fine_n}")
    coarse_n = fine_n // factor
    return fine_n, coarse_n, fine_n * fine_n, coarse_n * coarse_n


def rms_scale(n_out: int, n_in: int) -> float:
    """Factor converting a Euclidean singular value into an RMS gain.

    C_RMS = sqrt(N/M) C when the input is fine (N) and the output is coarse (M).
    Square maps have n_out == n_in, so the factor is 1.
    """
    if n_in < 1 or n_out < 1:
        raise ValueError(f"bad dimensions n_out={n_out} n_in={n_in}")
    return math.sqrt(float(n_in) / float(n_out))


def euclidean_norm(value) -> float:
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1).double()
        return float(torch.linalg.vector_norm(flat))
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(flat))


def field_rms(value) -> float:
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1).double()
        if flat.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(flat * flat)))
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat * flat)))


def cosine(a, b, floor: float = ZERO_FLOOR) -> float:
    """Cosine of two fields. Undefined (NaN) when either field is numerically zero."""
    na = euclidean_norm(a)
    nb = euclidean_norm(b)
    if not math.isfinite(na) or not math.isfinite(nb) or na <= floor or nb <= floor:
        return float("nan")
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        aa = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
        bb = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
        dot = torch.dot(aa.detach().reshape(-1).double(), bb.detach().reshape(-1).double())
        return float(dot) / (na * nb)
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(aa, bb) / (na * nb))


def directional_gains(output, input_field) -> dict[str, float]:
    """Sampled gain of one vector. This is not a singular value."""
    n_in = euclidean_norm(input_field)
    n_out = euclidean_norm(output)
    if n_in <= ZERO_FLOOR or not math.isfinite(n_in) or not math.isfinite(n_out):
        return {"euclidean": float("nan"), "rms": float("nan")}
    eucl = n_out / n_in
    nin = int(input_field.numel()) if isinstance(input_field, torch.Tensor) else int(np.asarray(input_field).size)
    nout = int(output.numel()) if isinstance(output, torch.Tensor) else int(np.asarray(output).size)
    return {"euclidean": float(eucl), "rms": float(eucl * math.sqrt(nin / nout))}


def D_numpy(field: np.ndarray, factor: int) -> np.ndarray:
    """Block mean. field is (..., n, n)."""
    arr = np.asarray(field, dtype=np.float64)
    n = int(arr.shape[-1])
    if arr.shape[-2] != n or n % factor:
        raise ValueError(f"cannot pool shape {arr.shape} by {factor}")
    m = n // factor
    lead = arr.shape[:-2]
    view = arr.reshape(lead + (m, factor, m, factor))
    return view.mean(axis=(-3, -1))


def D_dagger_numpy(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Repeat each coarse value across its r-by-r block. Not D*."""
    arr = np.asarray(coarse, dtype=np.float64)
    return np.repeat(np.repeat(arr, factor, axis=-2), factor, axis=-1)


def D_star_numpy(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Euclidean adjoint of D. Distributes the coarse value divided by r²."""
    return D_dagger_numpy(coarse, factor) / float(factor * factor)


def P_numpy(field: np.ndarray, factor: int) -> np.ndarray:
    return D_dagger_numpy(D_numpy(field, factor), factor)


def Q_numpy(field: np.ndarray, factor: int) -> np.ndarray:
    return np.asarray(field, dtype=np.float64) - P_numpy(field, factor)


def D_torch(field: torch.Tensor, factor: int) -> torch.Tensor:
    """Average pool. Same call as FutureINNPlus._pool_seq and l2_fd_jvp.apply_D."""
    return F.avg_pool2d(field, kernel_size=factor, stride=factor)


def D_dagger_torch(coarse: torch.Tensor, factor: int) -> torch.Tensor:
    """Moore–Penrose lift. Repeats coarse values. Does not divide by r²."""
    return coarse.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


def D_star_torch(coarse: torch.Tensor, factor: int) -> torch.Tensor:
    return D_dagger_torch(coarse, factor) / float(factor * factor)


def P_torch(field: torch.Tensor, factor: int) -> torch.Tensor:
    return D_dagger_torch(D_torch(field, factor), factor)


def Q_torch(field: torch.Tensor, factor: int) -> torch.Tensor:
    return field - P_torch(field, factor)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(np.asarray(b, dtype=np.float64).ravel())), ZERO_FLOOR)
    return float(np.linalg.norm(diff.ravel()) / denom)


def projector_identity_report(fine_n: int, factor: int, seed: int = 0) -> dict:
    """DD†=I, Penrose identities, and projector algebra on random fields."""
    rng = np.random.default_rng(seed)
    _, coarse_n, _, _ = pool_shapes(fine_n, factor)
    v = rng.standard_normal((fine_n, fine_n))
    w = rng.standard_normal((coarse_n, coarse_n))
    y = rng.standard_normal((fine_n, fine_n))
    z = rng.standard_normal((coarse_n, coarse_n))
    dv = D_numpy(v, factor)
    dw = D_dagger_numpy(w, factor)
    residuals = {
        "DD_dagger": _rel(D_numpy(dw, factor), w),
        "D_dagger_equals_r2_D_star": _rel(dw, (factor ** 2) * D_star_numpy(w, factor)),
        "D_star_is_not_D_dagger": 0.0 if factor == 1 else _rel(dw, D_star_numpy(w, factor)),
        "penrose_D_Ddagger_D": _rel(D_numpy(D_dagger_numpy(dv, factor), factor), dv),
        "penrose_Ddagger_D_Ddagger": _rel(D_dagger_numpy(D_numpy(dw, factor), factor), dw),
        "P_idempotent": _rel(P_numpy(P_numpy(v, factor), factor), P_numpy(v, factor)),
        "Q_idempotent": _rel(Q_numpy(Q_numpy(v, factor), factor), Q_numpy(v, factor)),
        "DQ": float(np.linalg.norm(D_numpy(Q_numpy(v, factor), factor))),
        "Q_Ddagger": float(np.linalg.norm(Q_numpy(dw, factor))),
        "P_symmetric": abs(
            float(np.dot(P_numpy(v, factor).ravel(), y.ravel()) - np.dot(v.ravel(), P_numpy(y, factor).ravel()))
        )
        / max(float(np.linalg.norm(v) * np.linalg.norm(y)), ZERO_FLOOR),
        "Q_symmetric": abs(
            float(np.dot(Q_numpy(v, factor).ravel(), y.ravel()) - np.dot(v.ravel(), Q_numpy(y, factor).ravel()))
        )
        / max(float(np.linalg.norm(v) * np.linalg.norm(y)), ZERO_FLOOR),
        "D_adjoint": abs(
            float(np.dot(dv.ravel(), z.ravel()) - np.dot(v.ravel(), D_star_numpy(z, factor).ravel()))
        )
        / max(float(np.linalg.norm(dv) * np.linalg.norm(z)), ZERO_FLOOR),
        "Ddagger_adjoint": abs(
            float(np.dot(dw.ravel(), y.ravel()) - np.dot(w.ravel(), (factor ** 2) * D_numpy(y, factor).ravel()))
        )
        / max(float(np.linalg.norm(dw) * np.linalg.norm(y)), ZERO_FLOOR),
    }
    # D* must fail DD*=I unless r=1. Record the gap; it is not a failure.
    residuals["DD_star_gap_from_I"] = _rel(D_numpy(D_star_numpy(w, factor), factor), w)
    structural = {k: v for k, v in residuals.items() if k not in ("D_star_is_not_D_dagger", "DD_star_gap_from_I")}
    star_separated = factor == 1 or residuals["D_star_is_not_D_dagger"] > 1e-6
    star_not_right_inverse = factor == 1 or residuals["DD_star_gap_from_I"] > 1e-6
    ok = max(structural.values()) <= 1e-10 and star_separated and star_not_right_inverse
    return {
        "ok": bool(ok),
        "fine_n": fine_n,
        "factor": factor,
        "coarse_n": coarse_n,
        "N": fine_n * fine_n,
        "M": coarse_n * coarse_n,
        "max_structural_residual": float(max(structural.values())),
        "residuals": residuals,
        "D_dagger": "repeat coarse value over r-by-r block",
        "D_star": "repeat coarse value divided by r^2",
    }


def torch_matches_numpy_pool(fine_n: int, factor: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((1, 1, fine_n, fine_n))
    coarse = rng.standard_normal((1, 1, fine_n // factor, fine_n // factor))
    ft = torch.from_numpy(field)
    ct = torch.from_numpy(coarse)
    d_err = _rel(D_torch(ft, factor).numpy(), D_numpy(field, factor))
    lift_err = _rel(D_dagger_torch(ct, factor).numpy(), D_dagger_numpy(coarse, factor))
    star_err = _rel(D_star_torch(ct, factor).numpy(), D_star_numpy(coarse, factor))
    return {
        "ok": max(d_err, lift_err, star_err) <= 1e-12,
        "D": d_err,
        "D_dagger": lift_err,
        "D_star": star_err,
    }


def _as_like(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=like.device, dtype=like.dtype)
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(value, dtype=np.float64)))
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor.reshape_as(like)


def _numpy_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.squeeze(arr)


@dataclass
class DownsamplerReport:
    ok: bool
    factor: int
    img_size: int
    latent_res: int
    kind: str
    pool_vs_avg: float
    pool_vs_block_mean: float
    message: str

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "factor": self.factor,
            "img_size": self.img_size,
            "latent_res": self.latent_res,
            "N": self.img_size ** 2,
            "M": self.latent_res ** 2,
            "kind": self.kind,
            "pool_vs_avg": self.pool_vs_avg,
            "pool_vs_block_mean": self.pool_vs_block_mean,
            "message": self.message,
            "D_dagger": "repeat" if self.ok else "not derived",
        }


def inspect_checkpoint_downsampler(model, field: torch.Tensor | None = None) -> DownsamplerReport:
    """Compare the checkpoint pool with nonoverlapping uniform averaging.

    A match means D† is block repetition. A mismatch does not guess another lift.
    """
    factor = int(model.ds_factors[0])
    img_size = int(model.img_size)
    latent = int(model.latent_res[0])
    if field is None:
        field = torch.randn(1, 1, img_size, img_size, dtype=torch.float64)
    else:
        field = field.detach().to(dtype=torch.float64).reshape(1, 1, img_size, img_size)
    with torch.no_grad():
        pooled = model._pool_seq(field)[0]
        avg = D_torch(field, factor)
        block = torch.from_numpy(D_numpy(field.cpu().numpy(), factor)).to(dtype=torch.float64)
    pool_vs_avg = _rel(pooled.cpu().numpy(), avg.cpu().numpy())
    pool_vs_block = _rel(pooled.cpu().numpy(), block.numpy())
    size_ok = latent == img_size // factor and img_size % factor == 0
    uniform = pool_vs_avg <= 1e-10 and pool_vs_block <= 1e-10 and size_ok
    if uniform:
        kind = "nonoverlapping_uniform_average"
        message = (
            f"Checkpoint pool matches avg_pool2d kernel=stride={factor} "
            f"and the r-by-r block mean. D† repeats coarse values."
        )
    else:
        kind = "unexpected"
        message = (
            f"Checkpoint pool does not match nonoverlapping uniform averaging "
            f"(avg residual {pool_vs_avg}, block-mean residual {pool_vs_block}, "
            f"img {img_size}, latent {latent}, factor {factor}). "
            "D† was not replaced by interpolation or D*."
        )
    return DownsamplerReport(
        ok=bool(uniform),
        factor=factor,
        img_size=img_size,
        latent_res=latent,
        kind=kind,
        pool_vs_avg=pool_vs_avg,
        pool_vs_block_mean=pool_vs_block,
        message=message,
    )


class LiftedBlocks:
    """Physical A, B, C, K and the lift-and-response maps T, U, CQ.

    Latent input of C is the point's z tensor, not a function of u.
    """

    def __init__(self, step_fn, point: L2Point, factor: int, norm: float):
        if norm == 0.0:
            raise ValueError("norm must be nonzero")
        self.step_fn = step_fn
        self.point = point
        self.factor = int(factor)
        self.norm = float(norm)
        self.n_fine = int(point.u.numel())
        self.n_coarse = int(point.z.numel())
        side = int(point.u.shape[-1])
        coarse_side = int(point.z.shape[-1])
        if side // self.factor != coarse_side:
            raise ValueError(
                f"factor {self.factor} does not map fine side {side} to coarse side {coarse_side}"
            )

    def _model(self, physical: torch.Tensor) -> torch.Tensor:
        return physical / self.norm

    def _physical(self, model_tangent: torch.Tensor) -> torch.Tensor:
        return model_tangent * self.norm

    def C(self, v_phys) -> torch.Tensor:
        v = self._model(_as_like(v_phys, self.point.u))
        with torch.enable_grad():
            _g, h = jvp_both_u(self.step_fn, self.point, v)
        return self._physical(h).detach()

    def C_star(self, w_phys) -> torch.Tensor:
        """Euclidean adjoint of C. z is closed over and is not an input."""
        w = self._model(_as_like(w_phys, self.point.z))

        def h_only(u):
            _g, h = self.step_fn(u, self.point.z)
            return h

        with torch.enable_grad():
            _out, vjp = torch.autograd.functional.vjp(h_only, self.point.u, w)
        if isinstance(vjp, tuple):
            vjp = vjp[0]
        return self._physical(vjp).detach()

    def A(self, v_phys) -> torch.Tensor:
        v = self._model(_as_like(v_phys, self.point.u))
        with torch.enable_grad():
            g, _h = jvp_both_u(self.step_fn, self.point, v)
        return self._physical(g).detach()

    def B(self, w_phys) -> torch.Tensor:
        w = self._model(_as_like(w_phys, self.point.z))
        with torch.enable_grad():
            g, _h = jvp_both_z(self.step_fn, self.point, w)
        return self._physical(g).detach()

    def K(self, w_phys) -> torch.Tensor:
        w = self._model(_as_like(w_phys, self.point.z))
        with torch.enable_grad():
            _g, h = jvp_both_z(self.step_fn, self.point, w)
        return self._physical(h).detach()

    def augmented(self, du_phys, dz_phys) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step tangent of (g, h) at the teacher-forced point."""
        du = self._model(_as_like(du_phys, self.point.u))
        dz = self._model(_as_like(dz_phys, self.point.z))
        with torch.enable_grad():
            g, h = jvp_both_uz(self.step_fn, self.point, du, dz)
        return self._physical(g).detach(), self._physical(h).detach()

    def P(self, v_phys) -> torch.Tensor:
        return P_torch(_as_like(v_phys, self.point.u), self.factor).detach()

    def Q(self, v_phys) -> torch.Tensor:
        return Q_torch(_as_like(v_phys, self.point.u), self.factor).detach()

    def D(self, v_phys) -> torch.Tensor:
        return D_torch(_as_like(v_phys, self.point.u), self.factor).detach()

    def D_dagger(self, w_phys) -> torch.Tensor:
        return D_dagger_torch(_as_like(w_phys, self.point.z), self.factor).detach()

    def D_star(self, w_phys) -> torch.Tensor:
        return D_star_torch(_as_like(w_phys, self.point.z), self.factor).detach()

    def T(self, w_phys) -> torch.Tensor:
        """C D†. Coarse to coarse. Lift-and-response, not the HINE rollout."""
        return self.C(self.D_dagger(w_phys))

    def T_star(self, y_phys) -> torch.Tensor:
        """(D†)* C* = r² D C*."""
        return (self.factor ** 2) * self.D(self.C_star(y_phys))

    def U(self, v_phys) -> torch.Tensor:
        """D† C. Fine to fine. Same nonzero eigenvalues as T, not the same singular values."""
        return self.D_dagger(self.C(v_phys))

    def U_star(self, v_phys) -> torch.Tensor:
        """C* (D†)* = r² C* D."""
        return (self.factor ** 2) * self.C_star(self.D(v_phys))

    def CQ(self, v_phys) -> torch.Tensor:
        """C restricted to the unresolved fine subspace."""
        return self.C(self.Q(v_phys))

    def CQ_star(self, w_phys) -> torch.Tensor:
        """Q C*, using Q* = Q."""
        return self.Q(self.C_star(w_phys))

    def matvec(self, name: str):
        """Flattened float64 action and its adjoint for a named operator."""
        table = {
            "T": (self.T, self.T_star, self.point.z, self.point.z),
            "U": (self.U, self.U_star, self.point.u, self.point.u),
            "C": (self.C, self.C_star, self.point.u, self.point.z),
            "CQ": (self.CQ, self.CQ_star, self.point.u, self.point.z),
        }
        if name not in table:
            raise KeyError(name)
        forward, adjoint, in_like, out_like = table[name]
        return FlatAction(name, forward, adjoint, in_like, out_like, self._rms_factor(name))

    def _rms_factor(self, name: str) -> float:
        if name in ("T", "U"):
            return 1.0
        if name in ("C", "CQ"):
            return rms_scale(self.n_coarse, self.n_fine)
        raise KeyError(name)


class FlatAction:
    """Numpy matvec / rmatvec around a physical torch operator."""

    def __init__(self, name, forward, adjoint, in_like, out_like, rms_factor: float):
        self.name = name
        self.forward = forward
        self.adjoint = adjoint
        self.in_like = in_like
        self.out_like = out_like
        self.n_in = int(in_like.numel())
        self.n_out = int(out_like.numel())
        self.shape_in = tuple(int(s) for s in in_like.shape)
        self.shape_out = tuple(int(s) for s in out_like.shape)
        self.rms_factor = float(rms_factor)

    def _pack_in(self, vec: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).reshape(-1))
        return torch.from_numpy(arr).to(device=self.in_like.device, dtype=self.in_like.dtype).reshape(self.shape_in)

    def _pack_out(self, vec: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).reshape(-1))
        return torch.from_numpy(arr).to(device=self.out_like.device, dtype=self.out_like.dtype).reshape(self.shape_out)

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        out = self.forward(self._pack_in(vec))
        return out.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)

    def rmatvec(self, vec: np.ndarray) -> np.ndarray:
        out = self.adjoint(self._pack_out(vec))
        return out.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)

    def image_shape(self, which: str) -> tuple[int, int]:
        like = self.out_like if which == "out" else self.in_like
        return int(like.shape[-2]), int(like.shape[-1])


def adjoint_dot_report(action: FlatAction, seed: int = 0, n_pairs: int = 3) -> dict:
    """Relative gap of <Av, w> - <v, A* w>."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_pairs):
        v = rng.standard_normal(action.n_in)
        w = rng.standard_normal(action.n_out)
        Av = action.matvec(v)
        Asw = action.rmatvec(w)
        left = float(np.dot(Av, w))
        right = float(np.dot(v, Asw))
        gaps.append(abs(left - right) / max(abs(left), abs(right), ZERO_FLOOR))
    return {"operator": action.name, "max_rel": float(max(gaps)), "gaps": gaps}


def centered_fd_C(step_fn, point: L2Point, v_model: torch.Tensor, eps: float) -> torch.Tensor:
    """Centered difference of h with z held fixed. Audit only, not the operator."""
    _, hp = step_fn(point.u + eps * v_model, point.z)
    _, hm = step_fn(point.u - eps * v_model, point.z)
    return ((hp - hm) / (2.0 * eps)).detach()


def jvp_fd_report(step_fn, point: L2Point, factor: int, norm: float, eps_list=(1e-4, 1e-5, 1e-6), seed: int = 0) -> dict:
    """Compare the autograd action of C with centered differences in model coordinates."""
    rng = np.random.default_rng(seed)
    v = torch.from_numpy(rng.standard_normal(point.u.shape)).to(device=point.u.device, dtype=point.u.dtype)
    v = v / max(euclidean_norm(v), ZERO_FLOOR)
    blocks = LiftedBlocks(step_fn, point, factor, norm)
    # Model-coordinate JVP: physical wrapper on (v * norm) equals this JVP.
    with torch.enable_grad():
        _g, jvp_h = jvp_both_u(step_fn, point, v)
    jvp_h = jvp_h.detach()
    rows = []
    best = float("inf")
    for eps in eps_list:
        fd = centered_fd_C(step_fn, point, v, float(eps))
        rel = _rel(fd.cpu().numpy(), jvp_h.cpu().numpy())
        rows.append({"eps": float(eps), "rel": rel})
        best = min(best, rel)
    physical = blocks.C(v * norm)
    phys_rel = _rel(physical.cpu().numpy(), (jvp_h * norm).cpu().numpy())
    return {
        "ok_physical_scale": phys_rel <= 1e-8,
        "physical_scale_rel": phys_rel,
        "best_rel": float(best),
        "rows": rows,
    }


def two_step_parts(op_n: LiftedBlocks, op_np1: LiftedBlocks, e_H) -> dict:
    """Fine pieces of the two-step augmented tangent at initial latent perturbation 0.

    The fine component is A_{n+1} A_n e_H + B_{n+1} C_n e_H.
    B_{n+1} C_n e_H alone is not that component.
    """
    e = _as_like(e_H, op_n.point.u)
    a_e = op_n.A(e)
    c_e = op_n.C(e)
    aa = op_np1.A(a_e)
    bc = op_np1.B(c_e)
    composed_fine, composed_latent = op_np1.augmented(a_e, c_e)
    total = aa + bc
    return {
        "A_A_e": aa.detach(),
        "B_C_e": bc.detach(),
        "sum": total.detach(),
        "composed_fine": composed_fine.detach(),
        "composed_latent": composed_latent.detach(),
        "sum_vs_composed_rel": _rel(total.detach().cpu().numpy(), composed_fine.detach().cpu().numpy()),
        "norm_AA": euclidean_norm(aa),
        "norm_BC": euclidean_norm(bc),
        "norm_sum": euclidean_norm(total),
        "rms_AA": field_rms(aa),
        "rms_BC": field_rms(bc),
        "cosine_AA_BC": cosine(aa, bc),
    }


def _flat_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.detach().reshape(-1).double(), right.detach().reshape(-1).double()))


def route_matrices(A_n, A_np1, B_np1, C_n, lift) -> dict:
    """Dense two-step split. lift is D+, not D*. B_np1 is the return at the next frame."""
    direct_map = np.asarray(A_np1, dtype=np.float64) @ np.asarray(A_n, dtype=np.float64)
    loop = np.asarray(B_np1, dtype=np.float64) @ np.asarray(C_n, dtype=np.float64)
    lifted = np.asarray(lift, dtype=np.float64) @ np.asarray(C_n, dtype=np.float64)
    revised = loop - lifted
    n = int(direct_map.shape[0])
    delta = (2.0 * np.sum(direct_map * loop) + np.sum(loop * loop)) / n
    return {
        "T": direct_map,
        "La": lifted,
        "Lb": revised,
        "L": loop,
        "tau_a": float(np.trace(lifted) / n),
        "tau_b": float(np.trace(revised) / n),
        "tau_loop": float(np.trace(loop) / n),
        "delta_loop": float(delta),
    }


def sign_probes(n: int) -> np.ndarray:
    from itertools import product

    return np.asarray(list(product([-1.0, 1.0], repeat=int(n))), dtype=np.float64)


def probe_quadratics(operator: np.ndarray, probes: np.ndarray) -> np.ndarray:
    width = int(operator.shape[0])
    return np.asarray([float(v @ operator @ v) / width for v in probes], dtype=np.float64)


def probe_energy(direct_map: np.ndarray, loop: np.ndarray, probes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = int(direct_map.shape[0])
    primary = []
    check = []
    for vector in probes:
        direct = direct_map @ vector
        image = loop @ vector
        primary.append((2.0 * float(np.dot(direct, image)) + float(np.dot(image, image))) / width)
        total = direct + image
        check.append((float(np.dot(total, total)) - float(np.dot(direct, direct))) / width)
    return np.asarray(primary), np.asarray(check)


def mc_mean_se(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    if finite.size < 2:
        return mean, float("nan")
    return mean, float(finite.std(ddof=1) / np.sqrt(finite.size))


def feedback_action(op_n: LiftedBlocks, op_np1: LiftedBlocks, v_phys) -> dict:
    """Physical split of one fine probe with zero latent perturbation.

    lift = D+ C_n v enters a_{n+2}. revise = (B_{n+1} - D+) C_n v enters b_{n+2}.
    direct + loop is the fine response and is checked against one joint JVP.
    """
    probe = _as_like(v_phys, op_n.point.u)
    with torch.enable_grad():
        fine_n, latent_n = jvp_both_u(op_n.step_fn, op_n.point, op_n._model(probe))
    direct_in = op_n._physical(fine_n).detach()
    coarse = op_n._physical(latent_n).detach()
    lift = op_n.D_dagger(coarse)
    loop = op_np1.B(coarse)
    revise = (loop - lift).detach()
    direct = op_np1.A(direct_in)
    composed_fine, _latent = op_np1.augmented(direct_in, coarse)
    total = (direct + loop).detach()
    width = op_n.n_fine
    tau_a = _flat_dot(probe, lift) / width
    tau_b = _flat_dot(probe, revise) / width
    tau_loop = _flat_dot(probe, loop) / width
    return {
        "lift": lift.detach(),
        "revise": revise,
        "loop": loop.detach(),
        "direct": direct.detach(),
        "tau_a": tau_a,
        "tau_b": tau_b,
        "tau_loop": tau_loop,
        "tau_A": _flat_dot(probe, direct_in) / width - 1.0,
        "delta_loop": (2.0 * _flat_dot(direct, loop) + _flat_dot(loop, loop)) / width,
        "delta_check": (_flat_dot(total, total) - _flat_dot(direct, direct)) / width,
        "residual": tau_a + tau_b - tau_loop,
        "sum_vs_composed_rel": _rel(total.detach().cpu().numpy(), composed_fine.detach().cpu().numpy()),
    }


def latent_shift(op: LiftedBlocks, w_phys) -> float:
    """tr estimator v^T (K - I) v / M for one unnormalized latent probe."""
    probe = _as_like(w_phys, op.point.z)
    image = op.K(probe)
    return _flat_dot(probe, image) / op.n_coarse - 1.0
