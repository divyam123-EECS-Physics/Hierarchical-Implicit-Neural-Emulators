"""Matrix-free trace and energy aggregates for the HINE comparison.

Probe summaries are Hutchinson estimates. They are not exact traces.
A negative normalized trace is a negative isotropic linear energy
contribution. It is not a contraction certificate.

Square-map probes are Rademacher vectors in the operator's input space.
For those vectors E[v^T M v] = Tr(M) and E[||M v||^2] = ||M||_F^2.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

NORM_FLOOR = 1e-12


def summarize(samples: list[float] | np.ndarray) -> dict:
    """Mean and Monte Carlo standard error of one probe scalar."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    count = int(x.size)
    if count == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "count": 0}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if count > 1 else 0.0
    se = float(std / np.sqrt(count)) if count > 1 else 0.0
    return {"mean": mean, "std": std, "se": se, "count": count}


def rademacher(dim: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Rows are independent Rademacher probes of length ``dim``."""
    if dim < 1 or count < 1:
        raise ValueError(f"dim and count must be positive, got {dim}, {count}")
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(count, dim))
    return np.ascontiguousarray(signs)


def enumerate_rademacher(dim: int) -> np.ndarray:
    """Every ±1 vector in dimension ``dim``. The group average is exact."""
    if dim < 1 or dim > 8:
        raise ValueError(f"enumeration is only for dim 1..8, got {dim}")
    grid = np.indices((2,) * dim).reshape(dim, -1).T
    return np.where(grid == 0, -1.0, 1.0).astype(np.float64)


def square_probe_metrics(dots: np.ndarray, images: np.ndarray, dim: int) -> dict:
    """Aggregate v·T v and ||T v||^2 into trace and energy diagnostics.

    ``dots`` is v·T v. ``images`` is ||T v||^2. Identity subtraction uses
    ||v||^2 = dim, which is exact for Rademacher probes. The energy
    identity then holds algebraically for these same probes.
    """
    dots = np.asarray(dots, dtype=np.float64).reshape(-1)
    images = np.asarray(images, dtype=np.float64).reshape(-1)
    if dots.shape != images.shape:
        raise ValueError(f"dot/image length mismatch {dots.shape} vs {images.shape}")
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    shifted = images - 2.0 * dots + float(dim)
    dot_stats = summarize(dots)
    image_stats = summarize(images)
    shift_stats = summarize(shifted)
    tau = (dot_stats["mean"] - float(dim)) / float(dim)
    tau_se = dot_stats["se"] / float(dim)
    energy = image_stats["mean"] / float(dim) - 1.0
    energy_se = image_stats["se"] / float(dim)
    two_tau = 2.0 * tau
    shift_over_d = shift_stats["mean"] / float(dim)
    residual = energy - (two_tau + shift_over_d)
    return {
        "trace": dot_stats["mean"],
        "trace_se": dot_stats["se"],
        "tau": float(tau),
        "tau_se": float(tau_se),
        "energy": float(energy),
        "energy_se": float(energy_se),
        "two_tau": float(two_tau),
        "shift_over_d": float(shift_over_d),
        "identity_residual": float(residual),
        "probes": int(dot_stats["count"]),
    }


def mean_over_dim(samples: np.ndarray, dim: int) -> dict:
    """Mean(samples) / dim, with the standard error scaled the same way."""
    stats = summarize(samples)
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    return {
        "mean": stats["mean"] / float(dim),
        "se": stats["se"] / float(dim),
        "raw_mean": stats["mean"],
        "raw_se": stats["se"],
        "probes": stats["count"],
    }


def two_step_energy(direct: np.ndarray, loop: np.ndarray, cross: np.ndarray, dim: int) -> dict:
    """Frobenius two-step split from shared probes.

    direct = ||F v||^2, loop = ||L v||^2, cross = (F v)·(L v).
    total = direct + 2 cross + loop holds probe by probe.
    """
    direct = np.asarray(direct, dtype=np.float64).reshape(-1)
    loop = np.asarray(loop, dtype=np.float64).reshape(-1)
    cross = np.asarray(cross, dtype=np.float64).reshape(-1)
    if not (direct.shape == loop.shape == cross.shape):
        raise ValueError("two-step probe arrays must share a shape")
    total = direct + 2.0 * cross + loop
    direct_m = mean_over_dim(direct, dim)
    loop_m = mean_over_dim(loop, dim)
    interference = mean_over_dim(2.0 * cross, dim)
    total_m = mean_over_dim(total, dim)
    net = interference["mean"] + loop_m["mean"]
    residual = total_m["mean"] - (direct_m["mean"] + interference["mean"] + loop_m["mean"])
    return {
        "direct": direct_m["mean"],
        "direct_se": direct_m["se"],
        "loop": loop_m["mean"],
        "loop_se": loop_m["se"],
        "interference": interference["mean"],
        "interference_se": interference["se"],
        "total": total_m["mean"],
        "total_se": total_m["se"],
        "loop_net": float(net),
        "identity_residual": float(residual),
        "probes": direct_m["probes"],
    }


def directional_quadratic(image: np.ndarray, vector: np.ndarray, subtract_identity: bool) -> float:
    """v^T T v / ||v||^2, or v^T (T-I) v / ||v||^2. NaN if ||v|| is negligible."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    image = np.asarray(image, dtype=np.float64).reshape(-1)
    if vector.shape != image.shape:
        raise ValueError("directional vector and image lengths differ")
    norm_sq = float(np.dot(vector, vector))
    if not np.isfinite(norm_sq) or norm_sq <= NORM_FLOOR:
        return float("nan")
    quad = float(np.dot(vector, image))
    if subtract_identity:
        quad -= norm_sq
    return quad / norm_sq


def directional_energy(image: np.ndarray, vector: np.ndarray) -> float:
    """||T v||^2 / ||v||^2 - 1. NaN if ||v|| is negligible."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    image = np.asarray(image, dtype=np.float64).reshape(-1)
    norm_sq = float(np.dot(vector, vector))
    if not np.isfinite(norm_sq) or norm_sq <= NORM_FLOOR:
        return float("nan")
    return float(np.dot(image, image) / norm_sq - 1.0)


def directional_two_step(
    direct_image: np.ndarray,
    loop_image: np.ndarray,
    vector: np.ndarray,
) -> dict:
    """Two-step energy split along one error vector. NaNs if ||v|| is negligible."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    direct_image = np.asarray(direct_image, dtype=np.float64).reshape(-1)
    loop_image = np.asarray(loop_image, dtype=np.float64).reshape(-1)
    norm_sq = float(np.dot(vector, vector))
    empty = {
        "direct": float("nan"),
        "loop": float("nan"),
        "interference": float("nan"),
        "total": float("nan"),
        "loop_net": float("nan"),
    }
    if not np.isfinite(norm_sq) or norm_sq <= NORM_FLOOR:
        return empty
    direct = float(np.dot(direct_image, direct_image) / norm_sq)
    loop = float(np.dot(loop_image, loop_image) / norm_sq)
    interference = float(2.0 * np.dot(direct_image, loop_image) / norm_sq)
    total = float(np.dot(direct_image + loop_image, direct_image + loop_image) / norm_sq)
    return {
        "direct": direct,
        "loop": loop,
        "interference": interference,
        "total": total,
        "loop_net": interference + loop,
    }


def project_blocks(flat: np.ndarray, side: int, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Return P v and Q v for nonoverlapping average pooling.

    P = D† D repeats each block average. It is the Euclidean orthogonal
    projector onto block-constant fields. Q = I - P.
    """
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    if side < 1 or factor < 1 or side % factor != 0:
        raise ValueError(f"factor {factor} does not divide side {side}")
    if flat.size != side * side:
        raise ValueError(f"expected {side * side} entries, got {flat.size}")
    image = torch.from_numpy(np.ascontiguousarray(flat)).reshape(1, 1, side, side)
    pooled = F.avg_pool2d(image, kernel_size=factor, stride=factor)
    lifted = pooled.repeat_interleave(factor, dim=-1).repeat_interleave(factor, dim=-2)
    projected = lifted.reshape(-1).numpy()
    return projected, flat - projected


def dense_square_reference(matrix: np.ndarray) -> dict:
    """Exact trace and energy quantities for a square dense matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    dim = int(matrix.shape[0])
    if matrix.shape != (dim, dim):
        raise ValueError("reference matrix must be square")
    eye = np.eye(dim, dtype=np.float64)
    shift = matrix - eye
    trace = float(np.trace(matrix))
    fro_sq = float(np.sum(matrix * matrix))
    shift_sq = float(np.sum(shift * shift))
    tau = (trace - dim) / dim
    energy = fro_sq / dim - 1.0
    return {
        "trace": trace,
        "tau": tau,
        "energy": energy,
        "two_tau": 2.0 * tau,
        "shift_over_d": shift_sq / dim,
        "identity_residual": energy - (2.0 * tau + shift_sq / dim),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(matrix)))),
    }


def expanding_negative_trace() -> np.ndarray:
    """Tr(T-I) < 0 while the spectral radius is greater than one."""
    return np.diag(np.array([-2.0, 0.5], dtype=np.float64))


def loop_increases_despite_negative_trace() -> dict:
    """Tr(B C) < 0, but the two-step loop raises energy above the direct path.

    N=2, M=1, F=I, L=B C = diag(-3, 0). Interference is negative and the
    loop's own energy is larger, so loop_net_effect > 0.
    """
    direct_map = np.eye(2, dtype=np.float64)
    loop_map = np.diag(np.array([-3.0, 0.0], dtype=np.float64))
    probes = enumerate_rademacher(2)
    direct, loop, cross = [], [], []
    for vector in probes:
        direct_image = direct_map @ vector
        loop_image = loop_map @ vector
        direct.append(float(np.dot(direct_image, direct_image)))
        loop.append(float(np.dot(loop_image, loop_image)))
        cross.append(float(np.dot(direct_image, loop_image)))
    metrics = two_step_energy(np.asarray(direct), np.asarray(loop), np.asarray(cross), 2)
    return {
        "trace_loop": float(np.trace(loop_map)),
        "tau_loop_over_n": float(np.trace(loop_map) / 2.0),
        "metrics": metrics,
        "direct_map": direct_map,
        "loop_map": loop_map,
    }


def finite_difference_check(
    forward,
    state: torch.Tensor,
    direction: torch.Tensor,
    jvp_image: torch.Tensor,
    epsilons: tuple[float, ...] = (1e-2, 1e-3),
) -> dict:
    """Relative agreement of a JVP with centered differences.

    The step is ``eps * ||state|| / ||direction||``. The best epsilon is
    reported; callers decide whether that residual is acceptable.
    """
    state_norm = float(torch.linalg.vector_norm(state.detach().double()))
    direction_norm = float(torch.linalg.vector_norm(direction.detach().double()))
    jvp_norm = float(torch.linalg.vector_norm(jvp_image.detach().double()))
    denom = max(jvp_norm, NORM_FLOOR)
    records = []
    for eps in epsilons:
        step = float(eps) * max(state_norm, NORM_FLOOR) / max(direction_norm, NORM_FLOOR)
        plus = forward(state + step * direction)
        minus = forward(state - step * direction)
        estimate = (plus - minus) / (2.0 * step)
        gap = float(torch.linalg.vector_norm((estimate - jvp_image).detach().double()))
        records.append({"eps": float(eps), "step": step, "relative": gap / denom})
    best = min(records, key=lambda item: item["relative"])
    return {"checks": records, "best_relative": float(best["relative"]), "best_eps": best["eps"]}
