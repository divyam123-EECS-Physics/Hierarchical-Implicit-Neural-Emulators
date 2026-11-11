#!/usr/bin/env python3
"""Coupled homogeneous perturbation growth for L=2 HINE.

p is a fine-state perturbation and q is a latent-input perturbation.
They are differences between nearby model trajectories, not prediction
errors against the truth. There is no eps_u, eps_z, or xi injection.

    p_next = A_n p_n + B_n q_n
    q_next = C_n p_n + K_n q_n
    w_n = q_n - M_n p_n
    M_n = D Phi'_n

An initially pure off-graph disturbance is p_0 = 0, q_0 = v, w_0 = v.
Coupling is kept: p is allowed to become nonzero and then affect q.

The control, on the same base and the same seed, is
    w_next = S_n w,  S_n = K_n - M_{n+1} B_n.

Reference mode linearizes at (u_true[n], D u_true[n+1]) with Phi' on the
512-grid DNS field at time n. That is a prescribed-reference tangent,
not a free model trajectory.

Autonomous mode linearizes on the model's own rollout. A 256-grid state
does not determine a unique 512-grid field. The lift used here is
    U_base[n] = U_DNS[n] + E(u_hat[n] - R U_DNS[n]),
with R spectral truncation and E band-limited embedding. Results in that
mode are conditional on this choice. Embedding drops the Nyquist row and
column. The solver's single-field startup (Omega0 = Omega1) is the
existing first-iteration approximation, not a saved IC-pair restart.

Decaying w with growing p means the off-graph inconsistency can shrink
while fine-state sensitivity grows. A damped isolated-S curve beside a
growing coupled curve implicates the fine/latent feedback for these
seeds. Decay on the sampled directions and windows is evidence, not a
global stability guarantee.

The physical stepper imports JAX. Here that is the hine-jax environment
at /pscratch/sd/d/divyam/conda2/envs/hine-jax. The model-only coarse
response script does not need it.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

_argv = sys.argv[1:]
if "--worker" not in _argv and "--n-gpus" in _argv:
    _n = int(_argv[_argv.index("--n-gpus") + 1])
    if _n > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from l2_fd_jvp import (
    L2Point,
    apply_D,
    fd_both_uz,
    fd_both_z,
    load_ema_float64,
    lock_precision,
    make_point,
    model_step,
)
from l2_retention import apply_shared_rescale
from py2d_inmem import (
    DNS_DATA,
    Py2DConfig,
    Py2DInMemory,
    embed_field,
    fd_tangent,
    pool_avg,
    rms,
    steps_for,
    truncate_field,
)
from task35_common import base_metadata, load_omega, rel_rms, sample_windows, sha256_file, write_json

LIFTING_NOTE = (
    "Autonomous 512-grid base U_base[n] = U_DNS[n] + E(u_hat[n] - R U_DNS[n]). "
    "R is spectral truncation to the model grid and E is band-limited embedding. "
    "Embedding zeros the Nyquist modes, so only retained wavenumbers are replaced. "
    "A 256-grid predicted state does not uniquely determine a 512-grid physical "
    "state. Tangents in autonomous mode are conditional on this lifting choice."
)
STARTUP_NOTE = (
    "The in-memory solver starts from one physical field by setting Omega0 = Omega1 "
    "and building convec0 from that field. That is the solver's own first-iteration "
    "policy, not a restart from a saved IC pair."
)
REFERENCE_NOTE = (
    "Reference mode linearizes at consecutive true pairs (u_true[n], D u_true[n+1]) "
    "and applies Phi' at the matching 512-grid DNS field at time n. This is a "
    "prescribed-reference tangent product, not a free model trajectory."
)
NOTES = f"""\
Coupled homogeneous perturbation growth (L=2).

p_n is the fine-state perturbation and q_n is the latent-input perturbation.
Both are differences between nearby model trajectories. No prediction-error
injection is added.

w_n = q_n - M_n p_n, with M_n = D Phi'_n.
Initial data: p_0 = 0, q_0 = v, w_0 = v, v a unit-RMS latent seed.
The coupled recurrence keeps A, B, C, and K, so p can leave zero and feed back.

Isolated control, same base sequence and same v:
    w_next = S_n w,  S_n = K_n - M_{{n+1}} B_n.
M_n multiplies p_n. M_{{n+1}} multiplies the fine image inside S_n.
Autonomous mode uses the lifted physical base at both times, not the raw DNS
field, so the control is not hine_S_action on a true-reference base.

{REFERENCE_NOTE}

{LIFTING_NOTE}

{STARTUP_NOTE}

Curves are RMS(signal_n) / RMS(w_0). The combined gain is
sqrt(RMS(p_n)^2 + RMS(q_n)^2) / RMS(w_0). p and q share one rescale factor.
The cumulative log of that factor is stored so the unscaled curve is
stored * exp(log_scale). They are not renormalized separately.
The isolated chain has its own cumulative log.

endpoint_log_gain = log(final ratio) / (n_steps * delta_t).
fitted_slope is a least-squares fit of log(ratio) against time.
Those two numbers are different. Neither is a spectral abscissa.

Decaying w with growing p means inconsistency can shrink while fine-state
sensitivity grows. A damped isolated-S curve with a growing coupled curve
implicates feedback for these tested disturbances. Decay on the sampled
directions and windows is not a global stability guarantee.
"""


def augmented_step_dense(A, B, C, K, p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return A @ p + B @ q, C @ p + K @ q


def pw_step_dense(A, B, C, K, M, M_next, p, w):
    """(p, w) block form. w_next uses M_next, not M."""
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    p_next = (A + B @ M) @ p + B @ w
    w_next = (C + K @ M - M_next @ (A + B @ M)) @ p + (K - M_next @ B) @ w
    return p_next, w_next


def isolated_s_dense(B, K, M_next, w):
    w = np.asarray(w, dtype=np.float64)
    return (K - M_next @ B) @ w


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--mode", choices=["reference", "autonomous", "both"], default="both")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/coupled_growth/n20",
    )
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--n-dirs", type=int, default=2)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--delta-t", type=float, default=0.05)
    p.add_argument("--fd-eps", type=float, default=3e-3, help="Model centered-difference size")
    p.add_argument("--fd-rtol", type=float, default=1e-2)
    p.add_argument("--fd-atol", type=float, default=1e-10)
    p.add_argument("--phys-fd-eps", type=float, default=1e-3)
    p.add_argument("--phys-rtol", type=float, default=5e-2)
    p.add_argument("--phys-atol", type=float, default=1e-4)
    p.add_argument("--fd-ref-scale", type=float, default=1.0)
    p.add_argument("--solver-dt", type=float, default=1e-4)
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--pw-steps", type=int, default=3)
    p.add_argument("--paired-steps", type=int, default=3)
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args(argv)
    if min(args.n_windows, args.n_steps, args.n_dirs) < 1:
        p.error("n-windows, n-steps, and n-dirs must be positive")
    return args


def mix_seed(seed: int, window: int, direction: int) -> int:
    return int((int(seed) * 1000003 + int(window) * 10007 + int(direction) * 97) % (2**31 - 1))


def numpy_field(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().double().cpu().numpy().reshape(tensor.shape[-2], tensor.shape[-1])


def as_tensor(field: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    src = np.ascontiguousarray(field, dtype=np.float64)
    return torch.from_numpy(src).to(device=like.device, dtype=like.dtype).view(1, 1, *src.shape)


def t_rms(x: torch.Tensor) -> float:
    flat = x.detach().reshape(-1).double()
    return float(torch.sqrt(torch.mean(flat * flat)))


def unit_gaussian(side: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((side, side))
    scale = rms(field)
    if scale == 0.0:
        raise RuntimeError("direction was identically zero")
    return field / scale


def endpoint_log_gain(final_ratio: float, n_steps: int, dt: float) -> float:
    if final_ratio <= 0.0 or not np.isfinite(final_ratio) or n_steps < 1 or dt <= 0.0:
        return float("nan")
    return float(math.log(final_ratio) / (n_steps * dt))


def fitted_slope(ratios: np.ndarray, dt: float) -> float:
    """Least-squares slope of log(ratio) versus time. Not the endpoint rate."""
    ratios = np.asarray(ratios, dtype=np.float64)
    time = np.arange(ratios.shape[0], dtype=np.float64) * float(dt)
    mask = np.isfinite(ratios) & (ratios > 0.0)
    if int(mask.sum()) < 2:
        return float("nan")
    tt = time[mask]
    yy = np.log(ratios[mask])
    tt = tt - tt.mean()
    denom = float(np.dot(tt, tt))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(tt, yy - yy.mean()) / denom)


def vector_ok(work: np.ndarray, ref: np.ndarray, rtol: float, atol: float) -> dict:
    diff = np.asarray(work, dtype=np.float64).reshape(-1) - np.asarray(ref, dtype=np.float64).reshape(-1)
    base = np.asarray(ref, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(diff))
    den = float(np.linalg.norm(base))
    if not np.isfinite(num) or not np.isfinite(den):
        return {"ok": False, "abs": num, "rel": float("nan"), "limit": float("nan")}
    limit = float(rtol) * den + float(atol) * float(diff.size) ** 0.5
    rel = float(num / den) if den > 0.0 else float("nan")
    return {"ok": bool(num <= limit), "abs": num, "rel": rel, "limit": limit}


def tensor_ok(work: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float, norm: float) -> dict:
    return vector_ok(
        numpy_field(work) * float(norm),
        numpy_field(ref) * float(norm),
        rtol,
        atol,
    )


def lift_base(dns: np.ndarray, u_hat_phys: np.ndarray) -> np.ndarray:
    """U_DNS + E(u_hat - R U_DNS). See LIFTING_NOTE."""
    reduced = truncate_field(dns, int(u_hat_phys.shape[0]))
    return np.asarray(dns, dtype=np.float64) + embed_field(
        np.asarray(u_hat_phys, dtype=np.float64) - reduced, int(dns.shape[0])
    )


def make_solver(args) -> Py2DInMemory:
    return Py2DInMemory(
        Py2DConfig(
            nx=args.nx,
            Re=args.re,
            fkx=args.fkx,
            fky=args.fky,
            alpha=args.alpha,
            beta=args.beta,
            dt_dns=args.solver_dt,
        )
    )


def apply_M(solver, base512, p_model: torch.Tensor, args, factor: int) -> np.ndarray:
    """M p in model coordinates. Zero directions return a zero latent field."""
    p_phys = numpy_field(p_model) * float(args.norm)
    side = int(p_phys.shape[0]) // int(factor)
    if rms(p_phys) == 0.0 or not np.isfinite(rms(p_phys)):
        return np.zeros((side, side), dtype=np.float64)
    direction = embed_field(p_phys, solver.cfg.nx)

    def flow(field):
        return solver.advance_physical(field, args.delta_t)

    phi = fd_tangent(flow, base512, direction, args.phys_fd_eps, args.fd_ref_scale)
    if phi.shape[0] != p_phys.shape[0]:
        phi = truncate_field(phi, int(p_phys.shape[0]))
    return pool_avg(phi, factor) / float(args.norm)


def check_window_frames(sample: dict, args) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    steps_for(args.solver_dt, args.delta_t)
    frames = []
    dns = []
    times = []
    align = []
    for frame in sample["frames"]:
        omega = np.asarray(frame["omega"], dtype=np.float64)
        path = DNS_DATA / f"{int(frame['stem'])}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"missing 512-grid frame {path}")
        dns_omega, dns_time = load_omega(path)
        if abs(float(dns_time) - float(frame["time"])) > 1e-6:
            raise RuntimeError(
                f"stem {frame['stem']} eval time {frame['time']} != DNS time {dns_time}"
            )
        rel = rel_rms(truncate_field(dns_omega, omega.shape[0]), omega)
        if not np.isfinite(rel) or rel > 1e-4:
            raise RuntimeError(f"stem {frame['stem']} truncation mismatch rel_rms={rel}")
        align.append(float(rel))
        frames.append(omega)
        dns.append(np.asarray(dns_omega, dtype=np.float64))
        times.append(float(frame["time"]))
    for left, right in zip(times, times[1:]):
        if abs((right - left) - float(args.delta_t)) > 1e-6:
            raise RuntimeError(f"spacing {right - left} != --delta-t {args.delta_t}")
    return frames, dns, align


def field_tensor(frame: np.ndarray, like: torch.Tensor, norm: float) -> torch.Tensor:
    src = np.ascontiguousarray(frame, dtype=np.float64)
    return torch.from_numpy(src).to(device=like.device, dtype=like.dtype).view(1, 1, *src.shape) / float(norm)


def build_bases(mode, ema, model, frames, dns, args):
    device = next(ema.parameters()).device
    dtype = next(ema.parameters()).dtype
    n_steps = int(args.n_steps)
    ds = int(model.ds_factors[0])
    teacher = [make_point(frames[n : n + 3], model, device, dtype, args.norm) for n in range(n_steps)]
    if mode == "reference":
        return teacher, [np.asarray(item, dtype=np.float64) for item in dns[: n_steps + 1]]
    first = teacher[0]
    u = first.u.detach().clone()
    z = first.z.detach().clone()
    points = []
    u_hats = []
    for n in range(n_steps):
        t = torch.ones(u.shape[0], device=device, dtype=dtype)
        true_next = field_tensor(frames[n + 1], u, args.norm)
        true_after = field_tensor(frames[n + 2], u, args.norm)
        points.append(
            L2Point(
                u=u.detach().clone(),
                z=z.detach().clone(),
                t=t,
                u_next=true_next.detach(),
                z_next=apply_D(true_after, ds).detach(),
                stems=(0, 0, 0),
            )
        )
        u_hats.append(numpy_field(u) * float(args.norm))
        g, h = model_step(ema, u, z, t)
        u, z = g.detach(), h.detach()
    u_hats.append(numpy_field(u) * float(args.norm))
    lifted = [lift_base(dns[n], u_hats[n]) for n in range(n_steps + 1)]
    return points, lifted


def step_fn(ema, point: L2Point):
    return lambda u, z, _t=point.t: model_step(ema, u, z, _t)


def propagate_direction(ema, solver, points, bases512, v_np: np.ndarray, args, factor: int) -> dict:
    like = points[0].z
    p = torch.zeros_like(points[0].u)
    q = as_tensor(v_np, like)
    w_iso = q.detach().clone()
    cum = 0.0
    cum_iso = 0.0
    rms0 = t_rms(q)
    n_steps = len(points)
    ratio_w, ratio_p, ratio_q, ratio_c, ratio_s = [], [], [], [], []
    log_c, log_s = [], []

    def push(stored_p, stored_q, stored_iso, c_log, s_log, base_now):
        m_np = apply_M(solver, base_now, stored_p, args, factor)
        w = stored_q - as_tensor(m_np, stored_q)
        scale = math.exp(c_log)
        scale_s = math.exp(s_log)
        rp = t_rms(stored_p) * scale / rms0
        rq = t_rms(stored_q) * scale / rms0
        rw = t_rms(w) * scale / rms0
        rs = t_rms(stored_iso) * scale_s / rms0
        ratio_p.append(rp)
        ratio_q.append(rq)
        ratio_w.append(rw)
        ratio_c.append(math.sqrt(rp ** 2 + rq ** 2))
        ratio_s.append(rs)
        log_c.append(c_log)
        log_s.append(s_log)

    for n in range(n_steps + 1):
        push(p, q, w_iso, cum, cum_iso, bases512[n])
        if n == n_steps:
            break
        fn = step_fn(ema, points[n])
        if t_rms(p) == 0.0 and t_rms(q) == 0.0:
            dg = torch.zeros_like(p)
            dh = torch.zeros_like(q)
        else:
            dg, dh = fd_both_uz(fn, points[n], p, q, args.fd_eps)
        p, q = dg.detach(), dh.detach()
        p, q, cum, _flag = apply_shared_rescale(p, q, cum)
        bw, kw = fd_both_z(fn, points[n], w_iso, args.fd_eps)
        m_next = apply_M(solver, bases512[n + 1], bw, args, factor)
        w_iso = (kw - as_tensor(m_next, kw)).detach()
        dummy = torch.zeros_like(w_iso)
        _dummy, w_iso, cum_iso, _flag = apply_shared_rescale(dummy, w_iso, cum_iso)
    ratios = {
        "w": np.asarray(ratio_w),
        "p": np.asarray(ratio_p),
        "q": np.asarray(ratio_q),
        "combined": np.asarray(ratio_c),
        "isolated": np.asarray(ratio_s),
    }
    out = {
        "ratio_w": ratios["w"],
        "ratio_p": ratios["p"],
        "ratio_q": ratios["q"],
        "ratio_combined": ratios["combined"],
        "ratio_isolated": ratios["isolated"],
        "log_scale": np.asarray(log_c),
        "log_scale_isolated": np.asarray(log_s),
    }
    for name, series in ratios.items():
        out[f"endpoint_log_gain_{name}"] = endpoint_log_gain(float(series[-1]), n_steps, args.delta_t)
        out[f"fitted_slope_{name}"] = fitted_slope(series, args.delta_t)
        out[f"max_transient_{name}"] = float(np.nanmax(series))
        out[f"final_{name}"] = float(series[-1])
    return out


def compare_pw(ema, solver, points, bases512, v_np, coupled_w, args, factor: int) -> list[dict]:
    """Short unscaled (p, w) product versus the coupled off-graph gain."""
    n_check = min(int(args.pw_steps), len(points))
    like = points[0].z
    p = torch.zeros_like(points[0].u)
    w = as_tensor(v_np, like)
    report = []
    for n in range(n_check):
        fn = step_fn(ema, points[n])
        m_now = apply_M(solver, bases512[n], p, args, factor)
        m_tensor = as_tensor(m_now, w)
        if t_rms(p) == 0.0 and t_rms(m_tensor) == 0.0:
            dg = torch.zeros_like(p)
            dh = torch.zeros_like(w)
        else:
            dg, dh = fd_both_uz(fn, points[n], p, m_tensor, args.fd_eps)
        bw, kw = fd_both_z(fn, points[n], w, args.fd_eps)
        p_next = (dg + bw).detach()
        q_next = (dh + kw).detach()
        m_b = apply_M(solver, bases512[n + 1], p_next, args, factor)
        w_next = (q_next - as_tensor(m_b, q_next)).detach()
        ref = coupled_w[n + 1].detach().to(device=w_next.device, dtype=w_next.dtype)
        cmp = tensor_ok(w_next, ref, args.fd_rtol, args.fd_atol, args.norm)
        report.append({"step": n + 1, **cmp})
        p, w = p_next, w_next
    return report


def unscaled_direction(ema, points, v_np, args) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Homogeneous product with rescaling left off, for the paired-rollout check."""
    p = torch.zeros_like(points[0].u)
    q = as_tensor(v_np, points[0].z)
    ps = [p.detach().clone()]
    qs = [q.detach().clone()]
    for n, point in enumerate(points):
        fn = step_fn(ema, point)
        dg, dh = fd_both_uz(fn, point, p, q, args.fd_eps)
        p, q = dg.detach(), dh.detach()
        ps.append(p.detach().clone())
        qs.append(q.detach().clone())
    return ps, qs


def paired_rollout(ema, points, v_np, tangent_p, tangent_q, args) -> list[dict]:
    """Centered nonlinear rollouts from z +/- h v. No solver correction."""
    n_check = min(int(args.paired_steps), len(points))
    u0 = points[0].u.detach().clone()
    z0 = points[0].z.detach().clone()
    v = as_tensor(v_np, z0)
    z_rms = max(t_rms(z0), 1e-12)
    report = []
    for h_rel in (1e-3, 3e-3, 1e-2):
        amp = float(h_rel) * z_rms
        u_plus, z_plus = u0.clone(), z0 + amp * v
        u_minus, z_minus = u0.clone(), z0 - amp * v
        step_rows = []
        for n in range(n_check + 1):
            dp = (u_plus - u_minus) / (2.0 * amp)
            dq = (z_plus - z_minus) / (2.0 * amp)
            cmp_p = tensor_ok(dp, tangent_p[n], args.fd_rtol, args.fd_atol, args.norm)
            cmp_q = tensor_ok(dq, tangent_q[n], args.fd_rtol, args.fd_atol, args.norm)
            step_rows.append({"step": n, "p": cmp_p, "q": cmp_q})
            if n == n_check:
                break
            t = points[n].t
            u_plus, z_plus = model_step(ema, u_plus, z_plus, t)
            u_minus, z_minus = model_step(ema, u_minus, z_minus, t)
        holds = all(row["p"]["ok"] and row["q"]["ok"] for row in step_rows)
        report.append({"h_rel": h_rel, "holds": bool(holds), "steps": step_rows})
    return report


def fd_checks(ema, solver, points, bases512, p_series, q_series, args, factor: int) -> dict:
    n_steps = len(points)
    indices = sorted({0, n_steps // 2, n_steps})
    model_rows = []
    phys_rows = []
    for n in indices:
        point = points[min(n, n_steps - 1)]
        # The state index n lives at the base used to advance into it when n>0,
        # and at the initial base when n=0. Derivative checks use that base.
        base_index = min(n, n_steps - 1)
        p = p_series[n]
        q = q_series[n]
        fn = step_fn(ema, points[base_index])
        if t_rms(p) == 0.0 and t_rms(q) == 0.0:
            model_rows.append({"step": n, "status": "skipped_zero"})
        else:
            g, h = fd_both_uz(fn, points[base_index], p, q, args.fd_eps)
            g3, h3 = fd_both_uz(fn, points[base_index], p, q, args.fd_eps / 3.0)
            g10, h10 = fd_both_uz(fn, points[base_index], p, q, args.fd_eps / 10.0)
            model_rows.append(
                {
                    "step": n,
                    "status": "checked",
                    "g_eps3": tensor_ok(g, g3, args.fd_rtol, args.fd_atol, args.norm),
                    "g_eps10": tensor_ok(g, g10, args.fd_rtol, args.fd_atol, args.norm),
                    "h_eps3": tensor_ok(h, h3, args.fd_rtol, args.fd_atol, args.norm),
                    "h_eps10": tensor_ok(h, h10, args.fd_rtol, args.fd_atol, args.norm),
                }
            )
        p_phys = numpy_field(p) * float(args.norm)
        if rms(p_phys) == 0.0:
            phys_rows.append({"step": n, "status": "skipped_zero_fine_perturbation"})
            continue
        direction = embed_field(p_phys, solver.cfg.nx)
        base = bases512[min(n, len(bases512) - 1)]

        def flow(field, _solver=solver):
            return _solver.advance_physical(field, args.delta_t)

        t1 = fd_tangent(flow, base, direction, args.phys_fd_eps, args.fd_ref_scale)
        t3 = fd_tangent(flow, base, direction, args.phys_fd_eps / 3.0, args.fd_ref_scale)
        t10 = fd_tangent(flow, base, direction, args.phys_fd_eps / 10.0, args.fd_ref_scale)
        phys_rows.append(
            {
                "step": n,
                "status": "checked",
                "eps3": vector_ok(t1, t3, args.phys_rtol, args.phys_atol),
                "eps10": vector_ok(t1, t10, args.phys_rtol, args.phys_atol),
            }
        )
    return {"model_fd": model_rows, "physical_fd": phys_rows}


def replay_dns(solver, dns, args) -> list[float]:
    out = []
    for n in range(args.n_steps):
        pred = solver.advance_physical(dns[n], args.delta_t)
        out.append(float(rel_rms(pred, dns[n + 1])))
    return out


def run_mode_window(args, ema, model, solver, sample, mode: str) -> dict:
    frames, dns, align = check_window_frames(sample, args)
    points, bases512 = build_bases(mode, ema, model, frames, dns, args)
    factor = int(model.ds_factors[0])
    curves = []
    first_unscaled = None
    for direction in range(args.n_dirs):
        seed = mix_seed(args.seed, int(sample["sample"]), direction)
        v = unit_gaussian(int(points[0].z.shape[-1]), seed)
        curve = propagate_direction(ema, solver, points, bases512, v, args, factor)
        curve["seed"] = seed
        curve["direction"] = direction
        curves.append(curve)
        if direction == 0:
            first_unscaled = (v, *unscaled_direction(ema, points, v, args))
        print(
            f"[window {int(sample['sample']):02d} {mode} dir {direction}] "
            f"final_w={curve['final_w']:.3e} final_isolated={curve['final_isolated']:.3e} "
            f"final_combined={curve['final_combined']:.3e}",
            flush=True,
        )
    validation = None
    if int(sample["sample"]) == 0 and not args.skip_validate and first_unscaled is not None:
        v0, p_series, q_series = first_unscaled
        # Coupled w along the unscaled product, for the (p, w) agreement check.
        coupled_w = []
        for n, (p_n, q_n) in enumerate(zip(p_series, q_series)):
            m_np = apply_M(solver, bases512[n], p_n, args, factor)
            coupled_w.append((q_n - as_tensor(m_np, q_n)).detach())
        validation = {
            "fd": fd_checks(ema, solver, points, bases512, p_series, q_series, args, factor),
            "pw_agreement": compare_pw(ema, solver, points, bases512, v0, coupled_w, args, factor),
            "replay_rel_rms": replay_dns(solver, dns, args),
            "alignment_rel_rms": align,
        }
        if mode == "autonomous":
            validation["paired_rollout"] = paired_rollout(ema, points, v0, p_series, q_series, args)
            validation["lifting"] = LIFTING_NOTE
        else:
            validation["reference"] = REFERENCE_NOTE
    return {
        "sample": int(sample["sample"]),
        "start_index": int(sample["start_index"]),
        "stems": [int(frame["stem"]) for frame in sample["frames"]],
        "times": [float(frame["time"]) for frame in sample["frames"]],
        "alignment_rel_rms": align,
        "replay_rel_rms": replay_dns(solver, dns, args) if validation is None else validation["replay_rel_rms"],
        "curves": curves,
        "validation": validation,
        "mode": mode,
    }


def save_mode_window(path: Path, payload: dict) -> None:
    curves = payload["curves"]
    data = {
        "sample": np.asarray(payload["sample"]),
        "start_index": np.asarray(payload["start_index"]),
        "stems": np.asarray(payload["stems"], dtype=np.int64),
        "times": np.asarray(payload["times"], dtype=np.float64),
        "alignment_rel_rms": np.asarray(payload["alignment_rel_rms"], dtype=np.float64),
        "replay_rel_rms": np.asarray(payload["replay_rel_rms"], dtype=np.float64),
        "direction": np.asarray([c["direction"] for c in curves], dtype=np.int64),
        "seed": np.asarray([c["seed"] for c in curves], dtype=np.int64),
    }
    for key in (
        "ratio_w",
        "ratio_p",
        "ratio_q",
        "ratio_combined",
        "ratio_isolated",
        "log_scale",
        "log_scale_isolated",
    ):
        data[key] = np.stack([c[key] for c in curves])
    for name in ("w", "p", "q", "combined", "isolated"):
        for stat in ("endpoint_log_gain", "fitted_slope", "max_transient", "final"):
            data[f"{stat}_{name}"] = np.asarray([c[f"{stat}_{name}"] for c in curves], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stack is (n_windows, n_steps). Non-positive ratios are already NaN."""
    med = np.full(stack.shape[1], np.nan)
    lo = np.full(stack.shape[1], np.nan)
    hi = np.full(stack.shape[1], np.nan)
    for index in range(stack.shape[1]):
        column = stack[:, index]
        column = column[np.isfinite(column)]
        if column.size == 0:
            continue
        med[index] = float(np.median(column))
        lo[index] = float(np.quantile(column, 0.10))
        hi[index] = float(np.quantile(column, 0.90))
    return med, lo, hi


def plot_mode(mode_dir: Path) -> None:
    files = sorted(mode_dir.glob("window_*.npz"))
    if not files:
        return
    series = {
        "w": [],
        "p": [],
        "q": [],
        "combined": [],
        "isolated_S": [],
    }
    endpoints = []
    slopes = []
    for path in files:
        data = np.load(path)
        series["w"].append(np.median(data["ratio_w"], axis=0))
        series["p"].append(np.median(data["ratio_p"], axis=0))
        series["q"].append(np.median(data["ratio_q"], axis=0))
        series["combined"].append(np.median(data["ratio_combined"], axis=0))
        series["isolated_S"].append(np.median(data["ratio_isolated"], axis=0))
        endpoints.append(float(np.median(data["endpoint_log_gain_w"])))
        slopes.append(float(np.median(data["fitted_slope_w"])))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = np.arange(series["w"][0].shape[0])
    for name, rows in series.items():
        stack = np.vstack(rows)
        for row in stack:
            y = np.where(row > 0.0, row, np.nan)
            ax.semilogy(steps, y, color="0.75", lw=0.8)
        med, lo, hi = _band(np.where(stack > 0.0, stack, np.nan))
        ax.semilogy(steps, med, lw=2, label=name)
        ax.fill_between(steps, lo, hi, alpha=0.12)
    ax.set_xlabel("step")
    ax.set_ylabel("RMS / RMS(w_0)")
    ax.set_title(f"{mode_dir.name}: coupled growth (log scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(mode_dir / "growth.png", dpi=140)
    plt.close(fig)
    write_json(
        mode_dir / "endpoint_vs_slope.json",
        {
            "endpoint_log_gain_w_median_over_windows": float(np.nanmedian(endpoints)),
            "fitted_slope_w_median_over_windows": float(np.nanmedian(slopes)),
            "endpoint_definition": "log(final RMS(w)/RMS(w_0)) / (n_steps * delta_t)",
            "slope_definition": "least-squares slope of log(ratio) versus time; not the endpoint rate",
        },
    )


def summarize_mode(mode_dir: Path) -> dict:
    files = sorted(mode_dir.glob("window_*.npz"))
    per_window = []
    for path in files:
        data = np.load(path)
        per_window.append(
            {
                "window": int(data["sample"]),
                "median_final_w": float(np.median(data["final_w"])),
                "median_final_p": float(np.median(data["final_p"])),
                "median_final_combined": float(np.median(data["final_combined"])),
                "median_final_isolated": float(np.median(data["final_isolated"])),
                "median_max_transient_w": float(np.median(data["max_transient_w"])),
                "median_endpoint_log_gain_w": float(np.median(data["endpoint_log_gain_w"])),
                "median_fitted_slope_w": float(np.median(data["fitted_slope_w"])),
                "replay_rel_rms_max": float(np.max(data["replay_rel_rms"])) if data["replay_rel_rms"].size else None,
            }
        )
    plot_mode(mode_dir)
    summary = {
        "n_windows": len(per_window),
        "per_window": per_window,
        "across_windows_median_final_w": _median_key(per_window, "median_final_w"),
        "across_windows_q10_final_w": _q_key(per_window, "median_final_w", 0.10),
        "across_windows_q90_final_w": _q_key(per_window, "median_final_w", 0.90),
        "endpoint_and_slope_are_separate": True,
    }
    write_json(mode_dir / "summary.json", summary)
    return summary


def _median_key(rows, key):
    x = np.asarray([row[key] for row in rows], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _q_key(rows, key, q):
    x = np.asarray([row[key] for row in rows], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def run_rank(args) -> None:
    precision = lock_precision()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested and is not available")
    device = torch.device(args.device)
    before = sha256_file(args.checkpoint)
    model, ema = load_ema_float64(args.checkpoint, device)
    after = sha256_file(args.checkpoint)
    solver = make_solver(args)
    samples = sample_windows(
        args.eval_dir,
        args.every_nth_field,
        args.n_windows,
        lookahead=args.n_steps + 1,
        seed=args.seed,
    )
    modes = ["reference", "autonomous"] if args.mode == "both" else [args.mode]
    save_dir = Path(args.save_dir)
    mine = [sample for sample in samples if int(sample["sample"]) % args.n_gpus == args.gpu_rank]
    print(f"[rank {args.gpu_rank}] windows {[s['sample'] for s in mine]} modes={modes}", flush=True)
    for mode in modes:
        mode_dir = save_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for sample in mine:
            payload = run_mode_window(args, ema, model, solver, sample, mode)
            save_mode_window(mode_dir / f"window_{int(sample['sample']):02d}.npz", payload)
            if payload["validation"] is not None:
                write_json(mode_dir / "validation_window00.json", payload["validation"])
        if args.gpu_rank == 0:
            meta = base_metadata(
                args,
                {
                    "task": "L2 coupled homogeneous perturbation growth",
                    "mode": mode,
                    "model_dtype": "float64",
                    "precision": precision,
                    "checkpoint_sha256_after_load": after,
                    "weights_unchanged": before == after,
                    "ds_factors": [int(v) for v in model.ds_factors],
                    "pooling": f"avg_pool2d kernel=stride={int(model.ds_factors[0])}",
                    "norm": float(args.norm),
                    "delta_t_checked_against_timestamps": True,
                    "physical_grid": 512,
                    "M": "D truncate Phi' embed, Phi' centered FD of advance_physical",
                    "indexing": "M_n acts on p_n; S_n uses M_{n+1}",
                    "lifting": LIFTING_NOTE if mode == "autonomous" else "not used",
                    "startup": STARTUP_NOTE,
                    "reference": REFERENCE_NOTE if mode == "reference" else "not this output group",
                    "solver": solver.cfg.as_dict(),
                    "rescale": "shared factor on p and q when combined RMS leaves [1e-6, 1e6]",
                },
            )
            write_json(mode_dir / "metadata.json", meta)
    if args.n_gpus == 1:
        for mode in modes:
            summarize_mode(save_dir / mode)
        (save_dir / "notes.txt").write_text(NOTES)


def orchestrate(args) -> None:
    script = str(Path(__file__).resolve())
    procs = []
    for rank in range(args.n_gpus):
        cmd = [sys.executable, "-u", script, "--worker", "--gpu-rank", str(rank)]
        for key, value in vars(args).items():
            if key in ("worker", "gpu_rank"):
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[orchestrator] rank {rank}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [proc.wait() for proc in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"worker exit codes {codes}")
    modes = ["reference", "autonomous"] if args.mode == "both" else [args.mode]
    save_dir = Path(args.save_dir)
    for mode in modes:
        summarize_mode(save_dir / mode)
    (save_dir / "notes.txt").write_text(NOTES)


def main() -> None:
    args = parse_args()
    if args.n_gpus > 1 and not args.worker:
        orchestrate(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
