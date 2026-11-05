"""Physical tangent map Phi' for one emulator step.

ProxyTangent implements D Phi' ≈ D (the small-step proxy). It is not the
Section 7 operator.

Py2DFDTangent is a centered finite difference through the in-memory py2d
stepper in eval/py2d_inmem.py. That stepper calls the kernels in
/global/homes/d/divyam/py2d. The finite difference is not an exact tangent.
Jax is imported only when the solver is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PY2D_ROOT = "/global/homes/d/divyam/py2d"


def is_proxy(tangent) -> bool:
    return getattr(tangent, "name", "") == "proxy"


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
    """Centered FD through one py2d emulator step. Not an exact tangent.

    Defaults match the HINE data provenance: Re=1e4, fkx=fky=4, alpha=0.1,
    beta=20, solver dt=1e-4, emulator step=0.25. nx follows the field.
    """

    name = "py2d_fd"
    is_exact = False

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
        py2d_root: str = PY2D_ROOT,
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
        self.py2d_root = str(py2d_root)
        self._solver = None
        self._fd = None
        if self.emulator_dt < self.solver_dt:
            raise ValueError("emulator_dt must be >= solver_dt")

    def _ensure(self):
        if self._solver is not None:
            return
        try:
            from py2d_inmem import Py2DConfig, Py2DInMemory, fd_tangent
        except ImportError as exc:
            raise RuntimeError(
                "py2d physical JVP needs jax and the py2d checkout at "
                f"{self.py2d_root}. Create the separate env with "
                "nersc_job_launchers/make_hine_jax_env.sh "
                "(prefix /pscratch/sd/d/divyam/conda2/envs/hine-jax). "
                "The hine env is intentionally unchanged."
            ) from exc
        self._fd = fd_tangent
        self._solver = Py2DInMemory(
            Py2DConfig(
                nx=self.nx,
                Re=self.Re,
                fkx=self.fkx,
                fky=self.fky,
                alpha=self.alpha,
                beta=self.beta,
                dt_dns=self.solver_dt,
            )
        )

    def flow(self, u_phys: np.ndarray) -> np.ndarray:
        self._ensure()
        return self._solver.advance_physical(np.asarray(u_phys, dtype=np.float64), self.emulator_dt)

    def apply(self, u_phys: np.ndarray, v_phys: np.ndarray) -> np.ndarray:
        """Centered FD of Phi at u_phys in direction v_phys. Physical units."""
        self._ensure()
        return self._fd(
            self.flow,
            np.asarray(u_phys, dtype=np.float64),
            np.asarray(v_phys, dtype=np.float64),
            self.fd_eps_rel,
            1e-8,
        )


def make_tangent(mode: str, **py2d_kwargs):
    if mode == "proxy":
        return ProxyTangent()
    if mode == "py2d":
        return Py2DFDTangent(**py2d_kwargs)
    raise ValueError(f"unknown physical-jvp mode {mode!r}")
