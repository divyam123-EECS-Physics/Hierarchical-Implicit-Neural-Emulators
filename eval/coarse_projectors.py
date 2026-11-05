"""Coarse projectors for Hypothesis H.

Sharp spectral projectors keep modes with a strict inequality at `kc`.
Wavenumbers are integers on the 2-pi box (cycles per domain). A square
cutoff is |kx| < kc and |ky| < kc. A radial cutoff is |k| < kc. They are
never mixed in one run.

The pooling projector matches HINE: truncate the solver grid to the model
grid, then average-pool by `r`. Dplus embeds the repeated coarse blocks.
Nyquist modes of the model grid are dropped, matching truncate_field.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from py2d_inmem import pool_avg


class SharpProjector:
    def __init__(self, solver, kc: int, shape: str):
        if kc < 1:
            raise ValueError("kc must be >= 1; use projector=None for the unclamped control")
        self.kc = int(kc)
        self.shape = shape
        self.nx = int(solver.cfg.nx)
        jnp = solver.jnp
        if shape == "square":
            self.mask = (jnp.abs(solver.Kx) < self.kc) & (jnp.abs(solver.Ky) < self.kc)
            self.inequality = "|kx| < kc and |ky| < kc"
        elif shape == "radial":
            self.mask = solver.Kabs < self.kc
            self.inequality = "|k| < kc"
        else:
            raise ValueError(f"unknown cutoff shape {shape!r}")
        self.units = "integer wavenumber, cycles per 2-pi domain, mean mode retained"
        self.jnp = jnp
        self._solver = solver

    def coarse_hat(self, hat):
        return self.jnp.where(self.mask, hat, 0)

    def impose_hat(self, hat, coarse):
        return self.jnp.where(self.mask, coarse, hat)

    def Q_hat(self, hat):
        return self.jnp.where(self.mask, 0, hat)

    def Q_phys_hat(self, hat):
        return self._solver.to_hat_phys(self.Q_hat(hat))

    def as_path(self, path: np.ndarray):
        return self.jnp.asarray(path)

    def describe(self) -> dict:
        return {
            "kind": "sharp",
            "shape": self.shape,
            "kc": self.kc,
            "inequality": self.inequality,
            "units": self.units,
            "nx": self.nx,
        }


class PoolingProjector:
    """P = Dplus D with D = pool_r o truncate(nx_solver -> nx_model)."""

    def __init__(self, nx_solver: int, nx_model: int, factor: int):
        if nx_solver % nx_model or nx_model % factor:
            raise ValueError(
                f"bad pooling sizes solver={nx_solver} model={nx_model} r={factor}"
            )
        self.nx_solver = int(nx_solver)
        self.nx_model = int(nx_model)
        self.factor = int(factor)
        self.coarse_n = self.nx_model // self.factor
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jnp = jnp

    def _truncate(self, phys):
        jnp = self.jnp
        n_out = self.nx_model
        n = int(phys.shape[0])
        if n == n_out:
            return phys
        h = n_out // 2
        w_hat = jnp.fft.rfft2(phys)
        w_hat = jnp.concatenate([w_hat[:h, : h + 1], w_hat[-h:, : h + 1]], axis=0)
        w_hat = w_hat.at[h, :].set(0)
        w_hat = w_hat.at[:, h].set(0)
        return jnp.fft.irfft2(w_hat, s=(n_out, n_out)) * (n_out / n) ** 2

    def _embed_hat(self, phys_lo):
        jnp = self.jnp
        n_lo = self.nx_model
        n_hi = self.nx_solver
        if n_hi == n_lo:
            return jnp.fft.rfft2(phys_lo)
        h = n_lo // 2
        hat = jnp.fft.rfft2(phys_lo) * (n_hi / n_lo) ** 2
        hat = hat.at[h, :].set(0)
        hat = hat.at[:, h].set(0)
        out = jnp.zeros((n_hi, n_hi // 2 + 1), dtype=jnp.complex128)
        out = out.at[:h, : h + 1].set(hat[:h, :])
        out = out.at[-h:, : h + 1].set(hat[h:, :])
        return out

    def _pool(self, phys_model):
        r = self.factor
        m = self.coarse_n
        return phys_model.reshape(m, r, m, r).mean(axis=(1, 3))

    def _repeat(self, coarse):
        r = self.factor
        return self.jnp.repeat(self.jnp.repeat(coarse, r, axis=0), r, axis=1)

    def D_phys(self, phys_solver):
        return self._pool(self._truncate(phys_solver))

    def Dplus_phys(self, coarse):
        return self.jnp.fft.irfft2(self._embed_hat(self._repeat(coarse)), s=(self.nx_solver, self.nx_solver))

    def coarse_hat(self, hat):
        phys = self.jnp.fft.irfft2(hat, s=(self.nx_solver, self.nx_solver))
        return self.D_phys(phys)

    def impose_hat(self, hat, coarse):
        phys = self.jnp.fft.irfft2(hat, s=(self.nx_solver, self.nx_solver))
        updated = self.Dplus_phys(coarse) + phys - self.Dplus_phys(self.D_phys(phys))
        return self.jnp.fft.rfft2(updated)

    def Q_hat(self, hat):
        phys = self.jnp.fft.irfft2(hat, s=(self.nx_solver, self.nx_solver))
        q = phys - self.Dplus_phys(self.D_phys(phys))
        return self.jnp.fft.rfft2(q)

    def Q_phys_hat(self, hat):
        return self.jnp.fft.irfft2(self.Q_hat(hat), s=(self.nx_solver, self.nx_solver))

    def as_path(self, path: np.ndarray):
        return self.jnp.asarray(np.asarray(path, dtype=np.float64))

    def describe(self) -> dict:
        return {
            "kind": "pooling",
            "nx_solver": self.nx_solver,
            "nx_model": self.nx_model,
            "factor": self.factor,
            "coarse_n": self.coarse_n,
            "D": "avg_pool(truncate(solver -> model), factor)",
            "Dplus": "embed(repeat blocks)",
            "inequality": "not a sharp wavenumber cutoff",
        }

    def right_inverse_error(self, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        z = self.jnp.asarray(rng.standard_normal((self.coarse_n, self.coarse_n)))
        back = self.D_phys(self.Dplus_phys(z))
        err = back - z
        return float(self.jnp.sqrt(self.jnp.mean(err ** 2)))


def pool_model(field_256: np.ndarray, factor: int) -> np.ndarray:
    return pool_avg(field_256, factor)


def q_model(field: np.ndarray, factor: int) -> np.ndarray:
    """Fine part on the model grid: field - repeat(pool(field))."""
    coarse = pool_avg(field, factor)
    repeated = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
    return np.asarray(field, dtype=np.float64) - repeated


def p_model(field: np.ndarray, factor: int) -> np.ndarray:
    coarse = pool_avg(field, factor)
    return np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
