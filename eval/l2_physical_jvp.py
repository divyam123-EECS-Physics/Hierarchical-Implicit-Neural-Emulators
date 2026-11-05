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

    def apply(self, u_phys: np.ndarray, v_phys: np.ndarray, eps_rel: float | None = None) -> np.ndarray:
        """Centered FD of Phi at u_phys in direction v_phys. Physical units."""
        self._ensure()
        eps = self.fd_eps_rel if eps_rel is None else float(eps_rel)
        return self._fd(
            self.flow,
            np.asarray(u_phys, dtype=np.float64),
            np.asarray(v_phys, dtype=np.float64),
            eps,
            1e-8,
        )


def make_tangent(mode: str, **py2d_kwargs):
    if mode == "proxy":
        return ProxyTangent()
    if mode == "py2d":
        return Py2DFDTangent(**py2d_kwargs)
    raise ValueError(f"unknown physical-jvp mode {mode!r}")


def validate_physical_reference(tangent, u_now, u_next, eps: float, tol: float, seed: int, dt: float, stems) -> dict:
    """Flow and finite-difference checks on one Task 1 reference pair.

    Compares flow(u_now) with the supplied next frame at that frame's shape
    and dt. Does not read a standalone replay of another dataset.
    """
    u_now = np.asarray(u_now, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)
    if u_now.shape != u_next.shape:
        raise ValueError(f"reference shapes differ: {u_now.shape} vs {u_next.shape}")
    pred = np.asarray(tangent.flow(u_now), dtype=np.float64)
    denom = max(float(np.linalg.norm(u_next)), 1e-12)
    flow_rel = float(np.linalg.norm(pred - u_next) / denom)
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(u_now.shape)
    base = np.asarray(tangent.apply(u_now, direction, eps), dtype=np.float64)
    checks = []
    for factor, label in ((1.0 / 3.0, "eps/3"), (0.1, "eps/10")):
        other = np.asarray(tangent.apply(u_now, direction, eps * factor), dtype=np.float64)
        rel = float(np.linalg.norm(base - other) / max(float(np.linalg.norm(other)), 1e-12))
        checks.append({"against": label, "rel": rel, "ok": bool(rel <= tol)})
    fd_ok = bool(checks) and all(c["ok"] for c in checks)
    flow_ok = bool(flow_rel <= float(tol))
    return {
        "ok": bool(flow_ok and fd_ok),
        "flow_ok": flow_ok,
        "fd_ok": fd_ok,
        "flow_rel": flow_rel,
        "nx": int(u_now.shape[-1]),
        "dt": float(dt),
        "stems": [int(s) for s in stems],
        "source": "task1_window",
        "checks": checks,
        "tol": float(tol),
        "eps": float(eps),
    }
