"""Physical tangent map Phi' for one emulator step.

ProxyTangent implements D Phi' ≈ D (the small-step proxy). It is not the
Section 7 operator.

Py2DJaxTangent differentiates the 512-grid py2d advance with jax.jvp.
A model-grid direction is embedded to 512 and the tangent is spectrally
truncated back. Py2DFDTangent remains for callers that still use a centered
difference. Jax is imported only when the solver is constructed.
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


class Py2DJaxTangent:
    """jax.jvp of one NoSGS advance on the 512 DNS grid.

    is_exact is False: this is the derivative of the discrete startup map,
    not an identity proxy. The solver grid is 512 even when the network
    field is 256.
    """

    name = "py2d_jvp"
    is_exact = False

    def __init__(
        self,
        nx: int = 512,
        Re: float = 1e4,
        fkx: int = 4,
        fky: int = 4,
        alpha: float = 0.1,
        beta: float = 20.0,
        solver_dt: float = 1e-4,
        emulator_dt: float = 0.05,
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
        self.py2d_root = str(py2d_root)
        self._solver = None
        if self.nx != 512:
            raise ValueError(f"Task 1 physical tangent integrates the 512 DNS grid, got nx={self.nx}")
        if self.emulator_dt < self.solver_dt:
            raise ValueError("emulator_dt must be >= solver_dt")

    def _ensure(self):
        if self._solver is not None:
            return
        try:
            from py2d_inmem import Py2DConfig, Py2DInMemory
        except ImportError as exc:
            raise RuntimeError(
                "py2d physical JVP needs jax and the py2d checkout at "
                f"{self.py2d_root}. Create the separate env with "
                "nersc_job_launchers/make_hine_jax_env.sh "
                "(prefix /pscratch/sd/d/divyam/conda2/envs/hine-jax). "
                "The hine env is intentionally unchanged."
            ) from exc
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

    def flow(self, dns_phys: np.ndarray, hats=None) -> np.ndarray:
        self._ensure()
        field = np.asarray(dns_phys, dtype=np.float64)
        if hats is None:
            return self._solver.advance_physical(field, self.emulator_dt)
        from py2d_inmem import steps_for

        state = self._solver.from_hats(hats[0], hats[1])
        state = self._solver.advance_state(state, steps_for(self.solver_dt, self.emulator_dt))
        return self._solver.to_phys(state)

    def apply(self, u_phys: np.ndarray, v_phys: np.ndarray, dns=None, hats=None, eps_rel=None):
        """Tangent at the 512 field `dns` in the embedded direction v_phys.

        eps_rel is accepted and ignored. The derivative is a jax.jvp.
        """
        del eps_rel
        self._ensure()
        from py2d_inmem import embed_field, truncate_field

        if dns is None:
            raise ValueError("py2d jax tangent requires the 512 DNS field as dns=")
        base = np.asarray(dns, dtype=np.float64)
        direction = np.asarray(v_phys, dtype=np.float64)
        if direction.shape[-1] != self.nx:
            direction = embed_field(direction, self.nx)
        tang = self._solver.jvp_physical(base, direction, self.emulator_dt, hats=hats)
        n_out = int(np.asarray(v_phys).shape[-1])
        if tang.shape[-1] != n_out:
            tang = truncate_field(tang, n_out)
        return np.asarray(tang, dtype=np.float64)


def make_tangent(mode: str, **py2d_kwargs):
    if mode == "proxy":
        return ProxyTangent()
    if mode == "py2d":
        py2d_kwargs.pop("fd_eps_rel", None)
        py2d_kwargs.pop("nx", None)
        return Py2DJaxTangent(**py2d_kwargs)
    raise ValueError(f"unknown physical-jvp mode {mode!r}")


def _rel(a, b) -> float:
    denom = max(float(np.linalg.norm(np.asarray(b, dtype=np.float64))), 1e-12)
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / denom)


def validate_physical_reference(
    tangent,
    u_now,
    u_next,
    eps: float,
    tol: float,
    seed: int,
    dt: float,
    stems,
    dns_now=None,
    dns_next=None,
    hats=None,
) -> dict:
    """Replay one Task 1 step.

    Without DNS arrays, compare tangent.flow(u_now) with u_next. That path
    is the unit-test contract. With DNS arrays, advance the 512 field and
    compare both the 512 successor and its spectral truncation with the
    Task 1 frame. Saved AB2 hats are used only when supplied.
    """
    del seed
    u_now = np.asarray(u_now, dtype=np.float64)
    u_next = np.asarray(u_next, dtype=np.float64)
    if dns_now is None:
        pred = np.asarray(tangent.flow(u_now), dtype=np.float64)
        flow_rel = _rel(pred, u_next)
        return {
            "ok": bool(flow_rel <= float(tol)),
            "flow_ok": bool(flow_rel <= float(tol)),
            "fd_ok": True,
            "flow_rel": flow_rel,
            "nx": int(u_now.shape[-1]),
            "dt": float(dt),
            "stems": [int(s) for s in stems],
            "source": "task1_window",
            "checks": [],
            "tol": float(tol),
            "eps": float(eps),
            "startup": "caller_supplied_flow",
        }
    dns_now = np.asarray(dns_now, dtype=np.float64)
    dns_next = np.asarray(dns_next, dtype=np.float64)
    from py2d_inmem import truncate_field

    pred = np.asarray(tangent.flow(dns_now, hats=hats), dtype=np.float64)
    dns_rel = _rel(pred, dns_next)
    task_rel = _rel(truncate_field(pred, u_next.shape[-1]), u_next)
    base_rel = _rel(truncate_field(dns_now, u_now.shape[-1]), u_now)
    flow_ok = bool(dns_rel <= float(tol) and task_rel <= float(tol) and base_rel <= float(tol))
    return {
        "ok": flow_ok,
        "flow_ok": flow_ok,
        "fd_ok": True,
        "flow_rel": task_rel,
        "dns_flow_rel": dns_rel,
        "task1_replay_rel": task_rel,
        "truncate_base_rel": base_rel,
        "nx": int(dns_now.shape[-1]),
        "model_nx": int(u_now.shape[-1]),
        "dt": float(dt),
        "stems": [int(s) for s in stems],
        "source": "task1_dns512",
        "startup": "saved_ab2_hats" if hats is not None else "Omega0=Omega1",
        "checks": [],
        "tol": float(tol),
        "eps": float(eps),
        "derivative": "jax.jvp",
    }


def load_dns_frame(stem: int) -> dict:
    """512 DNS vorticity at one stem, plus AB2 hats when that file stores them."""
    import scipy.io
    from py2d_inmem import DNS_DATA

    path = DNS_DATA / f"{int(stem)}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"missing 512 DNS frame {path}")
    raw = scipy.io.loadmat(path)
    omega = np.asarray(raw["Omega"], dtype=np.float64)
    time = float(np.asarray(raw["time"]).reshape(-1)[0]) if "time" in raw else float("nan")
    hats = None
    if "Omega0_hat" in raw and "Omega1_hat" in raw:
        hats = (np.asarray(raw["Omega0_hat"]), np.asarray(raw["Omega1_hat"]))
    return {"omega": omega, "time": time, "hats": hats, "path": str(path)}
