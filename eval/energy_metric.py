"""Kinetic-energy inner product for L=2 HINE diagnostics.

E(u) = 0.5 <u, G u> with the quadratic form used by kinetic_energy_torch:
streamfunction psi_hat = invKsq * omega_hat, velocity from the spectral
derivatives, then an inverse real FFT. That inverse FFT drops the imaginary
part of the Nyquist column, so <f, G g> = mean(u_f u_g + v_f v_g) after that
projection. It is not the raw sum of invKsq |omega_hat|^2, which is larger
by exactly the discarded Nyquist piece.

<f, g> in the definition of G is this velocity inner product. The spatial
average is the domain mean on the 2π × 2π box. invKsq[0, 0] = 0, so a
constant vorticity has zero energy. The Nyquist column of invKsq is kept;
only the non-real part of the derived velocity is removed by irfft.

The source energy note was not in the repository. See ENERGY_L2_README.md.
"""

from __future__ import annotations

import numpy as np
import torch

from util.py2d import initialize_wavenumbers_rfft2

LX = LY = 2.0 * np.pi
LOW_K_MAX = 16
MID_K_MAX = 64
D_E_THRESHOLD = 5.0
REF_STEM_MIN = 10000
REF_STEM_MAX = 105000


class EnergyMetric:
    def __init__(self, nx: int = 256, ny: int = 256, lx: float = LX, ly: float = LY):
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.n = self.nx * self.ny
        Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(
            self.nx, self.ny, self.lx, self.ly, INDEXING="ij"
        )
        self.Kx = np.asarray(Kx, dtype=np.float64)
        self.Ky = np.asarray(Ky, dtype=np.float64)
        self.Kabs = np.asarray(Kabs, dtype=np.float64)
        self.Ksq = np.asarray(Ksq, dtype=np.float64)
        self.invKsq = np.asarray(invKsq, dtype=np.float64)
        if self.invKsq.shape != (self.ny, self.nx // 2 + 1):
            raise RuntimeError(f"unexpected rFFT shape {self.invKsq.shape}")
        if self.invKsq[0, 0] != 0.0:
            raise RuntimeError("mean mode of invKsq must stay 0")
        if self.nx % 2 == 0 and not np.isfinite(self.invKsq[:, -1]).all():
            raise RuntimeError("Nyquist invKsq is not finite")
        self.weight = np.ones_like(self.Kabs, dtype=np.float64)
        self.weight[:, 1:-1] = 2.0
        self.k_bin = np.rint(self.Kabs).astype(np.int64)
        self.n_shells = int(self.k_bin.max()) + 1
        self.shell_k = np.arange(self.n_shells, dtype=np.int64)
        self.parseval_scale = np.sqrt(self.weight) / float(self.n)
        self.hat_shape = self.invKsq.shape
        # Real and imaginary parts of every rFFT coefficient, including the
        # Hermitian kx=0 and Nyquist columns. Dropping those imaginary parts
        # under-counts the velocity energy.
        self.n_real = int(np.prod(self.hat_shape))
        self.n_imag = self.n_real
        self.n_scalar_coord = self.n_real + self.n_imag
        # Two velocity components. ||pack(y)||^2 = <y, G y> = mean(|velocity|^2).
        self.n_coord = 2 * self.n_scalar_coord
        self._scale_t = torch.from_numpy(np.ascontiguousarray(self.parseval_scale))
        self._inv_t = torch.from_numpy(np.ascontiguousarray(self.invKsq))
        self._kx_t = torch.from_numpy(np.ascontiguousarray(self.Kx))
        self._ky_t = torch.from_numpy(np.ascontiguousarray(self.Ky))

    def velocities(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src = np.asarray(field, dtype=np.float64)
        hat = np.fft.rfft2(src)
        psi = hat * self.invKsq
        u_hat = 1j * self.Ky * psi
        v_hat = -1j * self.Kx * psi
        shape = (self.ny, self.nx)
        return np.fft.irfft2(u_hat, s=shape), np.fft.irfft2(v_hat, s=shape)

    def inner_G(self, left: np.ndarray, right: np.ndarray) -> float:
        """<left, G right> after the same Nyquist projection as the KE statistic."""
        u_left, v_left = self.velocities(left)
        u_right, v_right = self.velocities(right)
        return float(np.mean(u_left * u_right + v_left * v_right))

    def energy(self, field: np.ndarray) -> float:
        return 0.5 * self.inner_G(field, field)

    def energy_split(self, reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        ref = np.asarray(reference, dtype=np.float64)
        pred = np.asarray(predicted, dtype=np.float64)
        err = pred - ref
        e_ref = self.energy(ref)
        e_pred = self.energy(pred)
        linear = self.inner_G(ref, err)
        quad = 0.5 * self.inner_G(err, err)
        return {
            "E_ref": e_ref,
            "E_pred": e_pred,
            "signed_energy_error": e_pred - e_ref,
            "L": linear,
            "Q": quad,
            "closure": (e_pred - e_ref) - (linear + quad),
            "vorticity_mse": float(np.mean(err * err)),
        }

    def _velocity_modal(self, left_vel: np.ndarray, right_vel: np.ndarray) -> np.ndarray:
        left_hat = np.fft.rfft2(np.asarray(left_vel, dtype=np.float64))
        right_hat = np.fft.rfft2(np.asarray(right_vel, dtype=np.float64))
        return self.weight * np.real(left_hat * np.conj(right_hat)) / float(self.n) ** 2

    def cross_shells(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Shell contributions to <left, G right>. Their sum is the scalar."""
        u_left, v_left = self.velocities(left)
        u_right, v_right = self.velocities(right)
        modal = self._velocity_modal(u_left, u_right) + self._velocity_modal(v_left, v_right)
        return np.bincount(
            self.k_bin.ravel(), weights=modal.ravel(), minlength=self.n_shells
        ).astype(np.float64)

    def quad_shells(self, field: np.ndarray) -> np.ndarray:
        """Shell contributions to 0.5 <field, G field>."""
        return 0.5 * self.cross_shells(field, field)

    def band_mask(self, which: str) -> np.ndarray:
        k = self.shell_k
        if which == "low":
            return (k > 0) & (k <= LOW_K_MAX)
        if which == "middle":
            return (k > LOW_K_MAX) & (k <= MID_K_MAX)
        if which == "high":
            return k > MID_K_MAX
        raise ValueError(which)

    def band_sums(self, shells: np.ndarray) -> dict[str, float]:
        arr = np.asarray(shells, dtype=np.float64)
        return {name: float(arr[self.band_mask(name)].sum()) for name in ("low", "middle", "high")}

    def _pack_scalar(self, field: np.ndarray) -> np.ndarray:
        hat = np.fft.rfft2(np.asarray(field, dtype=np.float64)) * self.parseval_scale
        return np.concatenate(
            [np.ascontiguousarray(hat.real).ravel(), np.ascontiguousarray(hat.imag).ravel()]
        ).astype(np.float64)

    def _unpack_scalar(self, coord: np.ndarray) -> np.ndarray:
        vec = np.asarray(coord, dtype=np.float64).reshape(-1)
        real = vec[: self.n_real].reshape(self.hat_shape)
        imag = vec[self.n_real : self.n_scalar_coord].reshape(self.hat_shape)
        scale = np.where(self.parseval_scale > 0.0, self.parseval_scale, 1.0)
        return np.fft.irfft2((real + 1j * imag) / scale, s=(self.ny, self.nx))

    def pack(self, field: np.ndarray) -> np.ndarray:
        """Real coordinate with ||coord||^2 = <field, G field>."""
        u_vel, v_vel = self.velocities(field)
        return np.concatenate([self._pack_scalar(u_vel), self._pack_scalar(v_vel)])

    def velocities_from_coord(self, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of pack on each velocity component. This round-trips exactly."""
        vec = np.asarray(coord, dtype=np.float64).reshape(-1)
        if vec.size != self.n_coord:
            raise ValueError(f"coord length {vec.size} != {self.n_coord}")
        n = self.n_scalar_coord
        return self._unpack_scalar(vec[:n]), self._unpack_scalar(vec[n:])

    def pack_adjoint(self, coord: np.ndarray) -> np.ndarray:
        """Sum-inner-product adjoint of pack. dot(pack(y), c) = sum(y * adj)."""
        vec = np.asarray(coord, dtype=np.float64).reshape(-1).copy()
        field = torch.zeros((self.ny, self.nx), dtype=torch.float64, requires_grad=True)
        packed = self.pack_torch(field)
        cot = torch.from_numpy(vec)
        (grad,) = torch.autograd.grad(packed, field, grad_outputs=cot)
        return grad.detach().cpu().numpy()

    def _pack_scalar_torch(self, field: torch.Tensor) -> torch.Tensor:
        hat = torch.fft.rfft2(field) * self._scale_t.to(device=field.device, dtype=field.dtype)
        return torch.cat([hat.real.reshape(-1), hat.imag.reshape(-1)], dim=0)

    def pack_torch(self, field: torch.Tensor) -> torch.Tensor:
        hat = torch.fft.rfft2(field)
        inv = self._inv_t.to(device=field.device, dtype=field.dtype)
        kx = self._kx_t.to(device=field.device, dtype=field.dtype)
        ky = self._ky_t.to(device=field.device, dtype=field.dtype)
        psi = hat * inv
        u_vel = torch.fft.irfft2(1j * ky * psi, s=(self.ny, self.nx))
        v_vel = torch.fft.irfft2(-1j * kx * psi, s=(self.ny, self.nx))
        return torch.cat([self._pack_scalar_torch(u_vel), self._pack_scalar_torch(v_vel)], dim=0)

    def rms_coord_from_physical(self, field: np.ndarray) -> np.ndarray:
        src = np.asarray(field, dtype=np.float64).reshape(-1)
        return src / np.sqrt(float(src.size))

    def physical_from_rms_coord(self, coord: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        vec = np.asarray(coord, dtype=np.float64).reshape(shape)
        return vec * np.sqrt(float(vec.size))


def d_E(energy: float, mu_E: float, sigma_E: float) -> float:
    if not np.isfinite(sigma_E) or sigma_E <= 0.0:
        raise ValueError(f"sigma_E must be positive, got {sigma_E}")
    return abs(float(energy) - float(mu_E)) / float(sigma_E)


def first_exit(d_values: np.ndarray, threshold: float = D_E_THRESHOLD) -> int:
    """First lead with d_E > threshold, or -1 if the series stays inside."""
    arr = np.asarray(d_values, dtype=np.float64)
    hits = np.flatnonzero(np.isfinite(arr) & (arr > threshold))
    if hits.size == 0:
        return -1
    return int(hits[0])


def ratio_or_nan(numer: float, denom: float, floor: float = 0.0) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) <= floor:
        return float("nan")
    return float(numer) / float(denom)


def sort_singular_triplets(
    left: np.ndarray, values: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort singular values descending and make the first large right entry positive."""
    order = np.argsort(-np.asarray(values, dtype=np.float64))
    u = np.array(left, dtype=np.float64, copy=True)[:, order]
    s = np.asarray(values, dtype=np.float64)[order]
    v = np.array(right, dtype=np.float64, copy=True)[:, order]
    for i in range(s.size):
        col = v[:, i]
        idx = int(np.argmax(np.abs(col))) if col.size else 0
        if col.size and col[idx] < 0.0:
            v[:, i] *= -1.0
            u[:, i] *= -1.0
    return u, s, v


def degeneracy_flags(values: np.ndarray, rel_gap: float = 1e-3) -> np.ndarray:
    """True when value i is tied with value i+1 relative to |value i|."""
    s = np.asarray(values, dtype=np.float64)
    flags = np.zeros(s.size, dtype=bool)
    for i in range(s.size - 1):
        scale = max(abs(float(s[i])), 1e-30)
        if abs(float(s[i]) - float(s[i + 1])) / scale < rel_gap:
            flags[i] = True
            flags[i + 1] = True
    return flags


def principal_angles(left_a: np.ndarray, left_b: np.ndarray) -> np.ndarray:
    """Principal angles, in radians, between column-spaces."""
    mat = np.asarray(left_a, dtype=np.float64).T @ np.asarray(left_b, dtype=np.float64)
    sv = np.linalg.svd(mat, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return np.arccos(sv)


def subspace_fraction(basis: np.ndarray, vector: np.ndarray) -> float:
    """||P v||^2 / ||v||^2 for an orthonormal column basis. NaN if v is zero."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):
        return float("nan")
    coeff = np.asarray(basis, dtype=np.float64).T @ vec
    return float(np.dot(coeff, coeff) / (norm * norm))


def orthonormality_error(basis: np.ndarray) -> float:
    gram = np.asarray(basis, dtype=np.float64).T @ np.asarray(basis, dtype=np.float64)
    eye = np.eye(gram.shape[0], dtype=np.float64)
    return float(np.linalg.norm(gram - eye))
