"""Physical tangent map Phi' for one emulator step.

ProxyTangent implements D Phi' ≈ D (the small-step proxy). It is not the
Section 7 operator.

Py2DFDTangent is a centered finite difference through a py2d-equivalent
AB2/Crank-Nicolson step (NoSGS, 2/3 dealias) over dt_emulator. It needs jax
and is not the hine conda env. Import of jax is lazy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ProxyTangent:
    """D Phi'_n ≈ identity on the fine field, so D Phi' v = D v.

    is_exact is False. Callers must label results *_proxy.
    """

    name: str = "proxy"
    is_exact: bool = False

    def flow(self, u_phys: np.ndarray) -> np.ndarray:
        return np.asarray(u_phys, dtype=np.float64)

    def apply(self, u_phys: np.ndarray, v_phys: np.ndarray) -> np.ndarray:
        return np.asarray(v_phys, dtype=np.float64)


class Py2DFDTangent:
    """Centered FD of one py2d emulator step. is_exact=True for this solver.

    Defaults match the HINE data provenance: Re=1e4, fkx=fky=4, alpha=0.1,
    beta=20, solver dt=1e-4, emulator step=0.25.
    """

    name = "py2d_fd"
    is_exact = True

    def __init__(
        self,
        nx: int = 256,
        Re: float = 1e4,
        fkx: int = 4,
        fky: int = 4,
        alpha: float = 0.1,
        beta: float = 20.0,
        solver_dt: float = 1e-4,
        emulator_dt: float = 0.25,
        fd_eps_rel: float = 1e-3,
    ):
        self.nx = int(nx)
        self.Re = float(Re)
        self.fkx = int(fkx)
        self.fky = int(fky)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.solver_dt = float(solver_dt)
        self.emulator_dt = float(emulator_dt)
        self.fd_eps_rel = float(fd_eps_rel)
        self._step = None
        self._n_sub = int(round(self.emulator_dt / self.solver_dt))
        if self._n_sub < 1:
            raise ValueError("emulator_dt must be >= solver_dt")

    def _ensure(self):
        if self._step is not None:
            return
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:
            raise RuntimeError(
                "py2d physical JVP needs jax. Create the separate env with "
                "nersc_job_launchers/make_hine_jax_env.sh "
                "(prefix /pscratch/sd/d/divyam/conda2/envs/hine-jax). "
                "The hine env is intentionally unchanged."
            ) from exc

        jax.config.update("jax_enable_x64", True)
        nx = self.nx
        dx = 2.0 * np.pi / nx
        kx = jnp.fft.fftfreq(nx, d=dx) * (2.0 * np.pi)
        kr = jnp.fft.rfftfreq(nx, d=dx) * (2.0 * np.pi)
        Ky, Kx = jnp.meshgrid(kx, kr, indexing="ij")
        Ksq = Kx ** 2 + Ky ** 2
        inv = jnp.where(Ksq == 0, 0.0, 1.0 / Ksq)
        y = jnp.arange(nx) * dx
        x = jnp.arange(nx) * dx
        Y, X = jnp.meshgrid(y, x, indexing="ij")
        force = self.fky * jnp.cos(self.fky * Y) + self.fkx * jnp.cos(self.fkx * X)
        Fk = jnp.fft.rfft2(force)
        nu = 1.0 / self.Re
        dt = self.solver_dt
        alpha = self.alpha
        beta = self.beta
        cutoff = nx / 3.0

        def _dealias(hat):
            mask = (jnp.abs(Kx) <= cutoff) & (jnp.abs(Ky) <= cutoff)
            return jnp.where(mask, hat, 0.0)

        def _psi_uv(omega_hat):
            psi = -inv * omega_hat
            u = jnp.fft.irfft2(1j * Ky * psi, s=(nx, nx))
            v = jnp.fft.irfft2(-1j * Kx * psi, s=(nx, nx))
            return u, v

        def _convection(omega_hat, u, v):
            omega = jnp.fft.irfft2(omega_hat, s=(nx, nx))
            u_hat = _dealias(jnp.fft.rfft2(u))
            v_hat = _dealias(jnp.fft.rfft2(v))
            o_hat = _dealias(omega_hat)
            # conservative
            uo = jnp.fft.rfft2(jnp.fft.irfft2(u_hat, s=(nx, nx)) * omega)
            vo = jnp.fft.rfft2(jnp.fft.irfft2(v_hat, s=(nx, nx)) * omega)
            cons = 1j * Kx * uo + 1j * Ky * vo
            # non-conservative
            ox = jnp.fft.irfft2(1j * Kx * o_hat, s=(nx, nx))
            oy = jnp.fft.irfft2(1j * Ky * o_hat, s=(nx, nx))
            nonc = jnp.fft.rfft2(u * ox) + jnp.fft.rfft2(v * oy)
            return _dealias(0.5 * (cons + nonc))

        def step(omega0, omega1, conv0):
            u, v = _psi_uv(omega1)
            conv1 = _convection(omega1, u, v)
            conv = 1.5 * conv1 - 0.5 * conv0
            diffu = -Ksq * omega1
            rhs = (
                omega1
                - dt * conv
                + dt * 0.5 * nu * diffu
                - dt * Fk
                + dt * beta * jnp.fft.rfft2(v)
            )
            denom = 1.0 + dt * alpha + 0.5 * dt * nu * Ksq
            omega_new = rhs / denom
            return omega1, omega_new, conv1

        step = jax.jit(step)
        # warmup compile on a tiny call shape
        z = jnp.zeros((nx, nx // 2 + 1), dtype=jnp.complex128)
        step(z, z, z)

        def integrate(omega_phys: np.ndarray) -> np.ndarray:
            w = jnp.asarray(omega_phys, dtype=jnp.float64)
            hat = jnp.fft.rfft2(w)
            o0, o1, c0 = hat, hat, _convection(hat, *_psi_uv(hat))
            for _ in range(self._n_sub):
                o0, o1, c0 = step(o0, o1, c0)
            return np.asarray(jnp.fft.irfft2(o1, s=(nx, nx)))

        self._integrate = integrate
        self._step = step

    def flow(self, u_phys: np.ndarray) -> np.ndarray:
        self._ensure()
        field = np.asarray(u_phys, dtype=np.float64)
        if field.shape != (self.nx, self.nx):
            raise ValueError(f"expected {(self.nx, self.nx)}, got {field.shape}")
        return self._integrate(field)

    def apply(self, u_phys: np.ndarray, v_phys: np.ndarray) -> np.ndarray:
        """Centered FD of Phi at u_phys in direction v_phys. Physical units."""
        self._ensure()
        u = np.asarray(u_phys, dtype=np.float64)
        v = np.asarray(v_phys, dtype=np.float64)
        vn = np.linalg.norm(v)
        un = np.linalg.norm(u)
        if vn == 0.0:
            return np.zeros_like(u)
        delta = self.fd_eps_rel * max(un, 1e-8) / vn
        plus = self._integrate(u + delta * v)
        minus = self._integrate(u - delta * v)
        return (plus - minus) / (2.0 * delta)


def make_tangent(mode: str, **py2d_kwargs):
    if mode == "proxy":
        return ProxyTangent()
    if mode == "py2d":
        return Py2DFDTangent(**py2d_kwargs)
    raise ValueError(f"unknown physical-jvp mode {mode!r}")
