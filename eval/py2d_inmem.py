#!/usr/bin/env python3
"""In-memory 512^2 py2d stepper for Tasks 3 and 5.

The update is the NoSGS Adams-Bashforth / Crank-Nicolson step in
py2d/Py2D_solver.py (convection AB2, diffusion and drag Crank-Nicolson,
beta from the spectral meridional velocity, no forcing filter). Kernels
are imported from the local py2d checkout. Nothing is written to the
solver's results directory.

Startup from a single physical field sets Omega0 = Omega1 and builds
convec0 from that field. That is the solver's own first-iteration
policy, and it is not the same as restarting from a saved IC pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

PY2D_ROOT = Path("/global/homes/d/divyam/py2d/py2d")
if str(PY2D_ROOT) not in sys.path:
    sys.path.insert(0, str(PY2D_ROOT))

DNS_ROOT = Path(
    "/pscratch/sd/d/divyam/py2d_runs/paper/results/"
    "Re10000_fkx4fky4_r0.1_b20/NoSGS/NX512/dt0.0001_IC1"
)
DNS_DATA = DNS_ROOT / "data"
DNS_IC = DNS_ROOT / "IC"


def steps_for(dt_dns: float, interval: float) -> int:
    """Number of microsteps. Rejects intervals that are not an integer count."""
    if dt_dns <= 0:
        raise ValueError(f"dt_dns must be positive, got {dt_dns}")
    ratio = float(interval) / float(dt_dns)
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-9 * max(1.0, abs(ratio)):
        raise ValueError(
            f"interval {interval} is not an integer multiple of dt_dns {dt_dns} "
            f"(ratio={ratio})"
        )
    return n


def rms(field: np.ndarray) -> float:
    x = np.asarray(field)
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def fd_tangent(flow, x: np.ndarray, v: np.ndarray, eps_rel: float, ref_scale: float) -> np.ndarray:
    """Central directional derivative.

    q = v / rms(v), h = eps_rel * max(rms(x), ref_scale),
    Jv = rms(v) * (F(x + h q) - F(x - h q)) / (2 h).
    A zero direction returns zeros.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scale = rms(v)
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(x)
    q = v / scale
    h = float(eps_rel) * max(rms(x), float(ref_scale))
    if h == 0.0:
        raise ValueError("finite-difference amplitude is zero")
    plus = np.asarray(flow(x + h * q), dtype=np.float64)
    minus = np.asarray(flow(x - h * q), dtype=np.float64)
    return scale * (plus - minus) / (2.0 * h)


def truncate_field(omega: np.ndarray, n_out: int) -> np.ndarray:
    """Spectral truncation, same scaling as py2d/scripts/downsample_paper_256.py."""
    omega = np.asarray(omega, dtype=np.float64)
    n = int(omega.shape[0])
    if n == n_out:
        return omega.copy()
    h = n_out // 2
    w_hat = np.fft.rfft2(omega)
    w_hat = np.concatenate([w_hat[:h, : h + 1], w_hat[-h:, : h + 1]], axis=0)
    w_hat[h, :] = 0.0
    w_hat[:, h] = 0.0
    return np.fft.irfft2(w_hat, s=(n_out, n_out)) * (n_out / n) ** 2


def embed_field(omega_lo: np.ndarray, n_hi: int) -> np.ndarray:
    """Band-limited embedding, inverse of truncate_field on retained modes."""
    omega_lo = np.asarray(omega_lo, dtype=np.float64)
    n_lo = int(omega_lo.shape[0])
    if n_hi == n_lo:
        return omega_lo.copy()
    if n_hi < n_lo or n_hi % 2 or n_lo % 2:
        raise ValueError(f"cannot embed {n_lo} into {n_hi}")
    h = n_lo // 2
    hat = np.fft.rfft2(omega_lo) * (n_hi / n_lo) ** 2
    hat[h, :] = 0.0
    hat[:, h] = 0.0
    out = np.zeros((n_hi, n_hi // 2 + 1), dtype=np.complex128)
    out[:h, : h + 1] = hat[:h, :]
    out[-h:, : h + 1] = hat[h:, :]
    return np.fft.irfft2(out, s=(n_hi, n_hi))


def pool_avg(field: np.ndarray, factor: int) -> np.ndarray:
    n = int(field.shape[0])
    if n % factor:
        raise ValueError(f"field side {n} is not divisible by {factor}")
    m = n // factor
    x = np.asarray(field, dtype=np.float64)
    return x.reshape(m, factor, m, factor).mean(axis=(1, 3))


@dataclass
class Py2DConfig:
    nx: int = 512
    Re: float = 1e4
    fkx: int = 4
    fky: int = 4
    alpha: float = 0.1
    beta: float = 20.0
    dt_dns: float = 1e-4

    def as_dict(self) -> dict:
        return {
            "nx": self.nx,
            "Re": self.Re,
            "fkx": self.fkx,
            "fky": self.fky,
            "alpha": self.alpha,
            "beta": self.beta,
            "dt_dns": self.dt_dns,
            "scheme": "AB2 convection, Crank-Nicolson diffusion and drag, NoSGS, 3/2 dealias",
            "startup": "Omega0=Omega1 and convec0 from that field (solver first iteration)",
        }


class Py2DInMemory:
    """JAX float64 stepper. State is (omega0_hat, omega1_hat, convec0_hat)."""

    def __init__(self, cfg: Py2DConfig | None = None):
        self.cfg = cfg or Py2DConfig()
        import jax
        import jax.numpy as jnp
        from py2d.convection_conserved import convection_conserved_dealias
        from py2d.convert import Omega2Psi_spectral, Psi2UV_spectral
        from py2d.initialize import gridgen, initialize_wavenumbers_rfft2

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self._conv = convection_conserved_dealias
        self._psi = Omega2Psi_spectral
        self._uv = Psi2UV_spectral

        nx = int(self.cfg.nx)
        lx = 2.0 * np.pi
        _, _, x, y, _, _ = gridgen(lx, lx, nx, nx)
        kx, ky, _, ksq, inv = initialize_wavenumbers_rfft2(nx, nx, lx, lx)
        self.X = jnp.asarray(x)
        self.Y = jnp.asarray(y)
        self.Kx = jnp.asarray(kx)
        self.Ky = jnp.asarray(ky)
        self.Ksq = jnp.asarray(ksq)
        self.invKsq = jnp.asarray(inv)
        self.Kabs = jnp.sqrt(self.Kx ** 2 + self.Ky ** 2)
        force = self.cfg.fky * jnp.cos(self.cfg.fky * self.Y) + self.cfg.fkx * jnp.cos(
            self.cfg.fkx * self.X
        )
        self.Fk = jnp.fft.rfft2(force)
        self.nu = 1.0 / float(self.cfg.Re)
        self._step_jit = jax.jit(self._step_pure)

    def _step_pure(self, state):
        """One AB2/CN microstep. Mirrors Py2D_solver.py lines 244-313 with NoSGS."""
        jnp = self.jnp
        o0, o1, c0 = state
        psi1 = self._psi(o1, self.invKsq)
        _u1, v1 = self._uv(psi1, self.Kx, self.Ky)
        c1 = self._conv(o1, _u1, v1, self.Kx, self.Ky)
        conv = 1.5 * c1 - 0.5 * c0
        diffu = -self.Ksq * o1
        dt = float(self.cfg.dt_dns)
        rhs = (
            o1
            - dt * conv
            + dt * 0.5 * self.nu * diffu
            - dt * self.Fk
            + dt * float(self.cfg.beta) * v1
        )
        denom = 1.0 + dt * float(self.cfg.alpha) + 0.5 * dt * self.nu * self.Ksq
        o_new = rhs / denom
        return o1, o_new, c1

    def _hat(self, omega_phys: np.ndarray):
        field = np.asarray(omega_phys, dtype=np.float64)
        if field.shape != (self.cfg.nx, self.cfg.nx):
            raise ValueError(f"expected {(self.cfg.nx, self.cfg.nx)}, got {field.shape}")
        return self.jnp.asarray(np.fft.rfft2(field))

    def startup(self, omega_phys: np.ndarray):
        """Single-field start: previous and current vorticity are the same field."""
        hat = self._hat(omega_phys)
        psi = self._psi(hat, self.invKsq)
        u, v = self._uv(psi, self.Kx, self.Ky)
        convec = self._conv(hat, u, v, self.Kx, self.Ky)
        return hat, hat, convec

    def from_hats(self, omega0_hat: np.ndarray, omega1_hat: np.ndarray):
        """Restart from a saved IC pair. convec0 is recomputed from omega0."""
        o0 = self.jnp.asarray(np.asarray(omega0_hat))
        o1 = self.jnp.asarray(np.asarray(omega1_hat))
        psi0 = self._psi(o0, self.invKsq)
        u0, v0 = self._uv(psi0, self.Kx, self.Ky)
        c0 = self._conv(o0, u0, v0, self.Kx, self.Ky)
        return o0, o1, c0

    def to_phys(self, state) -> np.ndarray:
        _o0, o1, _c0 = state
        return np.asarray(self.jnp.fft.irfft2(o1, s=(self.cfg.nx, self.cfg.nx)), dtype=np.float64)

    def step_state(self, state):
        return self._step_jit(state)

    def advance_state(self, state, n_steps: int):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        def body(s, _):
            return self._step_pure(s), None

        new_state, _ = self.jax.lax.scan(body, state, None, length=int(n_steps))
        return new_state

    def advance_physical(self, omega_phys: np.ndarray, interval: float) -> np.ndarray:
        n = steps_for(self.cfg.dt_dns, interval)
        state = self.advance_state(self.startup(omega_phys), n)
        return self.to_phys(state)

    def advance_conditional(self, omega_phys, coarse_path, projector, interval: float) -> np.ndarray:
        """Unrestricted microsteps with the coarse part replaced after every step.

        coarse_path[i] is the reference coarse state after microstep i.
        """
        n = steps_for(self.cfg.dt_dns, interval)
        path = projector.as_path(coarse_path)
        if int(path.shape[0]) != n:
            raise ValueError(f"coarse path length {path.shape[0]} != {n} microsteps")
        state = self.startup(omega_phys)

        def body(s, z):
            o0, o1, c0 = self._step_pure(s)
            o1 = projector.impose_hat(o1, z)
            return (o0, o1, c0), None

        state, _ = self.jax.lax.scan(body, state, path)
        return self.to_phys(state)

    def record_coarse_path(self, omega_phys, projector, interval: float):
        """Coarse coefficient of an unrestricted run after every microstep."""
        n = steps_for(self.cfg.dt_dns, interval)
        state = self.startup(omega_phys)

        def body(s, _):
            s2 = self._step_pure(s)
            return s2, projector.coarse_hat(s2[1])

        _state, path = self.jax.lax.scan(body, state, None, length=n)
        return np.asarray(path)

    def clamped_block(self, ref, fol, n_steps: int, projector):
        """Advance reference and follower together, clamping after each microstep.

        Returns new states and the RMS of Q(follower - reference) at every step.
        projector=None leaves the follower unclamped and uses the full field.
        """
        jnp = self.jnp

        def q_rms(ref_s, fol_s):
            if projector is None:
                diff = self.to_hat_phys(fol_s[1]) - self.to_hat_phys(ref_s[1])
            else:
                diff = projector.Q_phys_hat(fol_s[1] - ref_s[1])
            return jnp.sqrt(jnp.mean(diff ** 2))

        def body(carry, _):
            ref_s, fol_s = carry
            ref_s = self._step_pure(ref_s)
            fol_s = self._step_pure(fol_s)
            if projector is not None:
                z = projector.coarse_hat(ref_s[1])
                fol_s = (fol_s[0], projector.impose_hat(fol_s[1], z), fol_s[2])
            return (ref_s, fol_s), q_rms(ref_s, fol_s)

        (ref, fol), series = self.jax.lax.scan(body, (ref, fol), None, length=int(n_steps))
        return ref, fol, np.asarray(series, dtype=np.float64)

    def to_hat_phys(self, hat):
        return self.jnp.fft.irfft2(hat, s=(self.cfg.nx, self.cfg.nx))

    def renormalize_with(self, ref, fol, eps: float, projector):
        """Scale the fine difference at both AB2 levels back to RMS eps.

        convec0 is recomputed from the renormalized previous vorticity.
        Returns (new follower state or None, RMS before scaling).
        """
        jnp = self.jnp
        if projector is None:
            d1 = fol[1] - ref[1]
            d0 = fol[0] - ref[0]
            diff_phys = self.to_hat_phys(d1)
        else:
            d1 = projector.Q_hat(fol[1] - ref[1])
            d0 = projector.Q_hat(fol[0] - ref[0])
            diff_phys = projector.Q_phys_hat(fol[1] - ref[1])
        amp = jnp.sqrt(jnp.mean(self.to_hat_phys(d1) ** 2))
        amp_f = float(amp)
        if not np.isfinite(amp_f) or amp_f < 1e-30:
            return None, amp_f
        scale = float(eps) / amp_f
        o1 = ref[1] + scale * d1
        o0 = ref[0] + scale * d0
        return self.from_hats(np.asarray(o0), np.asarray(o1)), amp_f


def _load_hat(path: Path, key: str) -> np.ndarray:
    import scipy.io

    raw = scipy.io.loadmat(path)[key]
    hat = np.asarray(raw)
    if hat.ndim == 2 and hat.shape[0] == hat.shape[1]:
        from py2d.util import fft2_to_rfft2

        hat = fft2_to_rfft2(hat)
    return np.asarray(hat)


def replay_check(out_path: Path, stem: int = 91836) -> dict:
    """Compare the in-memory stepper with saved DNS frames.

    Exact check: IC/{stem_ic} advanced by tSAVE matches data/{stem_ic+1},
    using the file pair IC/104999 -> data/105000 (100 steps of 1e-4).
    Startup check: physical data/{stem} advanced 0.01, versus data/{stem+1}.
    Half-step check: the same interval at dt_dns=5e-5.
    """
    import scipy.io

    solver = Py2DInMemory(Py2DConfig())
    ic_stem = 104999
    o0 = _load_hat(DNS_IC / f"{ic_stem}.mat", "Omega0_hat")
    o1 = _load_hat(DNS_IC / f"{ic_stem}.mat", "Omega1_hat")
    state = solver.from_hats(o0, o1)
    state = solver.advance_state(state, 100)
    target = np.asarray(scipy.io.loadmat(DNS_DATA / f"{ic_stem + 1}.mat")["Omega"], dtype=np.float64)
    got = solver.to_phys(state)
    exact_rel = rms(got - target) / rms(target)

    omega = np.asarray(scipy.io.loadmat(DNS_DATA / f"{stem}.mat")["Omega"], dtype=np.float64)
    nxt = np.asarray(scipy.io.loadmat(DNS_DATA / f"{stem + 1}.mat")["Omega"], dtype=np.float64)
    pred = solver.advance_physical(omega, 0.01)
    startup_rel = rms(pred - nxt) / rms(nxt)

    fine = Py2DInMemory(Py2DConfig(dt_dns=5e-5))
    pred_fine = fine.advance_physical(omega, 0.01)
    half_rel = rms(pred_fine - nxt) / rms(nxt)
    report = {
        "exact_ic_pair": f"IC/{ic_stem} -> data/{ic_stem + 1}",
        "exact_rel_rms": exact_rel,
        "startup_stem": stem,
        "startup_rel_rms": startup_rel,
        "half_dt_rel_rms": half_rel,
        "note": (
            "exact_rel_rms uses the saved AB2 pair. startup_rel_rms drops that "
            "history on purpose. half_dt_rel_rms is a time-step check of the "
            "single-field startup, not an exact replay."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-check", action="store_true")
    p.add_argument(
        "--out",
        default="/pscratch/sd/d/divyam/hine_runs/py2d_sync/debug_test/replay_check.json",
    )
    p.add_argument("--stem", type=int, default=91836)
    args = p.parse_args()
    if not args.replay_check:
        p.error("pass --replay-check")
    replay_check(Path(args.out), stem=args.stem)


if __name__ == "__main__":
    main()
