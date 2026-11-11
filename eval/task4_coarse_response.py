#!/usr/bin/env python3
"""Coarse-input copying and disturbance attenuation for L=2 HINE.

Question: does D g(u, z) reproduce z, in value and in its local response,
and how much of a latent-input disturbance reaches that coarse output?

DB v = D (dg/dz v). This is not K, S, or D Phi' B. No py2d solve.

Teacher-forced bases are (u_n, D u_{n+1}). Autonomous bases are the
model's own (u_hat_n, z_in_n), stored before the step. The latent state
is not reset during that rollout.

A small teacher-forced value residual is what a good one-step fit
produces. It does not, by itself, show that the fine head copies z away
from the training graph. Gain near one is not enough either: copying is
gain one, alignment one, and copy defect zero together.

Gain below one means the tested latent disturbance is attenuated in the
coarse part of g. It is not a proof of latent-memory stability or of
damping over a whole rollout.
"""

from __future__ import annotations

import argparse
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
    fd_both_z,
    load_ema_float64,
    lock_precision,
    make_point,
    model_step,
)
from task35_common import base_metadata, sample_windows, sha256_file, write_json

H_REL = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
FAMILIES = ("random", "constant", "band_low", "band_mid", "band_high", "eta")
NOTES = """\
Coarse-input copying and disturbance attenuation (L=2).

DB v = D (partial g / partial z) v, with the fine state held fixed.
This is not K, not S, and not D Phi' B.

Value residual c = D g(u, z) - z.
Local response, for a unit-RMS latent direction v:
  copy_defect = RMS(b - v) / RMS(v)
  gain = RMS(b) / RMS(v)
  energy_gain = gain^2
  alignment = cosine(b, v)
Exact copying is defect 0, gain 1, and alignment 1 together.
Gain near one alone does not identify copying.

Teacher-forced bases are (u_n, D u_{n+1}) from the evaluation frames.
A small value residual there is expected from a good one-step fit and
does not by itself show copying away from the training graph.

Autonomous bases are (u_hat_n, z_in_n) taken before each model step.
The rollout starts at (u_0, D u_1) and then uses the model's own latent
output. The latent state is not reset.

eta_n = z_in_n - D u_true[n+1] is the latent-input discrepancy along that
rollout. It is a latent-error direction, not an off-graph error. Its mean
is kept. The constant mode is its own probe.

Random probes are Gaussian and are not mean-removed. Their mean is stored.
Band probes are real fields whose Fourier support lies in a radial shell
of integer wavenumbers. The constant mode is excluded from every band.

Gain < 1 means this tested disturbance is locally attenuated in the coarse
part of g. It is not a certificate of latent-memory stability, and it is
not a statement about damping of a full rollout.

Window-level medians are the unit of variation. Probes inside one window
are not independent trajectories. The maximum gain is the largest value
among the sampled probes, not an operator norm and not the largest
singular value of DB.

Finite differences are centered, float64, eval mode, autocast off, TF32 off.
A probe is validated only when the working step agrees with eps/3 and
eps/10 under the recorded absolute-plus-relative tolerance. Failed probes
stay in the raw tables with validated=false and are left out of the
validated summaries.
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task4_coarse_response/n20",
    )
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--n-random", type=int, default=32)
    p.add_argument("--n-band", type=int, default=1)
    p.add_argument("--diag-stride", type=int, default=1)
    p.add_argument("--bands", default="1:4,5:10,11:16")
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--delta-t", type=float, default=0.05)
    p.add_argument("--fd-eps", type=float, default=3e-3)
    p.add_argument("--fd-rtol", type=float, default=1e-2)
    p.add_argument("--fd-atol", type=float, default=1e-10)
    p.add_argument("--skip-linear", action="store_true")
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args(argv)
    if args.n_windows < 1 or args.n_steps < 1 or args.diag_stride < 1:
        p.error("n-windows, n-steps, and diag-stride must be positive")
    if args.n_random < 1 or args.n_band < 1:
        p.error("n-random and n-band must be positive")
    parse_bands(args.bands)
    return args


def parse_bands(text: str) -> list[tuple[str, int, int]]:
    names = ("band_low", "band_mid", "band_high")
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"--bands needs three lo:hi shells, got {text!r}")
    out = []
    for name, part in zip(names, parts):
        lo_s, hi_s = part.split(":")
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi or lo < 0:
            raise ValueError(f"bad band {part}")
        out.append((name, lo, hi))
    return out


def mix_seed(seed: int, window: int, step: int, family_id: int, index: int) -> int:
    value = (
        int(seed) * 1000003
        + int(window) * 10007
        + int(step) * 97
        + int(family_id) * 13
        + int(index)
    )
    return int(value % (2**31 - 1))


def rms(x) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def t_rms(x: torch.Tensor) -> float:
    flat = x.detach().reshape(-1).double()
    if flat.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(flat * flat)))


def t_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().reshape(-1).double()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = t_norm(a)
    nb = t_norm(b)
    if na == 0.0 or nb == 0.0 or not np.isfinite(na) or not np.isfinite(nb):
        return float("nan")
    return float(torch.dot(a.reshape(-1).double(), b.reshape(-1).double()) / (na * nb))


def unit_rms_np(field: np.ndarray) -> np.ndarray | None:
    scale = rms(field)
    if scale == 0.0 or not np.isfinite(scale):
        return None
    return np.asarray(field, dtype=np.float64) / scale


def wavenumber_grid(side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(side, d=1.0 / side)
    kx = np.fft.rfftfreq(side, d=1.0 / side)
    radius = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    return ky, kx, radius


def band_field(side: int, kmin: int, kmax: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Real unit-RMS field supported on kmin <= |k| <= kmax, excluding |k|=0."""
    ky, kx, radius = wavenumber_grid(side)
    mask = (radius >= float(kmin)) & (radius <= float(kmax)) & (radius > 0.0)
    if not np.any(mask):
        raise ValueError(f"band [{kmin}, {kmax}] is empty on a {side} grid")
    rng = np.random.default_rng(seed)
    hat = rng.normal(size=mask.shape) + 1j * rng.normal(size=mask.shape)
    hat = np.asarray(hat, dtype=np.complex128)
    hat[~mask] = 0.0
    hat[:, 0] = np.real(hat[:, 0])
    if side % 2 == 0:
        hat[:, -1] = np.real(hat[:, -1])
    field = np.fft.irfft2(hat, s=(side, side)).real
    scaled = unit_rms_np(field)
    if scaled is None:
        raise RuntimeError(f"band [{kmin}, {kmax}] produced a zero field")
    definition = {
        "grid": int(side),
        "kmin": int(kmin),
        "kmax": int(kmax),
        "radius": "sqrt(kx^2 + ky^2) with integer fftfreq * grid",
        "excluded": "k = 0, and every mode outside [kmin, kmax]",
        "reality": "imaginary part of the kx=0 and Nyquist columns is set to 0 before irfft2",
        "n_modes": int(np.count_nonzero(mask)),
        "mean_removed": False,
    }
    return scaled, mask, definition


def random_field(side: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = unit_rms_np(rng.standard_normal((side, side)))
    if field is None:
        raise RuntimeError("gaussian probe was identically zero")
    return field


def constant_field(side: int) -> np.ndarray:
    return np.ones((side, side), dtype=np.float64)


def as_latent(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    src = np.ascontiguousarray(arr, dtype=np.float64)
    return torch.from_numpy(src).to(device=like.device, dtype=like.dtype).view(1, 1, *src.shape)


def value_agreement(coarse: torch.Tensor, z: torch.Tensor, norm: float) -> dict:
    """c = Dg - z. RMS values are physical (model coordinates times norm)."""
    residual = coarse - z
    rms_c = t_rms(residual) * float(norm)
    rms_z = t_rms(z) * float(norm)
    undefined = not np.isfinite(rms_z) or rms_z == 0.0
    rel = float("nan") if undefined else float(rms_c / rms_z)
    return {
        "rms_c": float(rms_c),
        "rms_z": float(rms_z),
        "rel_c": rel,
        "rel_undefined": bool(undefined),
        "residual": residual.detach(),
    }


def response_metrics(b: torch.Tensor, v: torch.Tensor) -> dict:
    rv = t_rms(v)
    rb = t_rms(b)
    if rv == 0.0 or not np.isfinite(rv):
        gain = float("nan")
        defect = float("nan")
    else:
        gain = float(rb / rv)
        defect = float(t_rms(b - v) / rv)
    return {
        "copy_defect": defect,
        "gain": gain,
        "energy_gain": float(gain ** 2) if np.isfinite(gain) else float("nan"),
        "alignment": cosine(b, v),
        "rms_v": float(rv),
        "probe_mean": float(v.detach().double().mean()),
    }


def passes_fd(work: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float, norm: float) -> dict:
    """||b_work - b_ref|| <= rtol ||b_ref|| + atol sqrt(numel), in physical units."""
    diff = (work - ref).detach().reshape(-1).double() * float(norm)
    base = ref.detach().reshape(-1).double() * float(norm)
    num = float(torch.linalg.vector_norm(diff))
    den = float(torch.linalg.vector_norm(base))
    if not np.isfinite(num) or not np.isfinite(den):
        return {"ok": False, "abs": num, "limit": float("nan"), "rel": float("nan")}
    limit = float(rtol) * den + float(atol) * float(diff.numel()) ** 0.5
    rel = float(num / den) if den > 0.0 else float("nan")
    return {"ok": bool(num <= limit), "abs": num, "limit": limit, "rel": rel}


def db_at(step_fn, point: L2Point, v: torch.Tensor, eps: float, ds: int) -> torch.Tensor:
    fine, _latent = fd_both_z(step_fn, point, v, eps)
    return apply_D(fine, ds)


def eta_direction(eta: torch.Tensor) -> torch.Tensor | None:
    """Unit-RMS latent-error direction. Zero fields are skipped, not mean-removed."""
    scale = t_rms(eta)
    if scale == 0.0 or not np.isfinite(scale):
        return None
    return eta / scale


def build_probe_arrays(
    side: int,
    seed: int,
    window: int,
    step: int,
    n_random: int,
    n_band: int,
    bands: list[tuple[str, int, int]],
    eta: torch.Tensor | None,
) -> list[dict]:
    probes = []
    for index in range(n_random):
        probes.append(
            {
                "family": "random",
                "index": index,
                "seed": mix_seed(seed, window, step, 0, index),
                "field": random_field(side, mix_seed(seed, window, step, 0, index)),
            }
        )
    probes.append(
        {
            "family": "constant",
            "index": 0,
            "seed": mix_seed(seed, window, step, 1, 0),
            "field": constant_field(side),
        }
    )
    for fam_i, (name, lo, hi) in enumerate(bands, start=2):
        for index in range(n_band):
            this_seed = mix_seed(seed, window, step, fam_i, index)
            field, _mask, _defn = band_field(side, lo, hi, this_seed)
            probes.append(
                {"family": name, "index": index, "seed": this_seed, "field": field}
            )
    if eta is not None:
        probes.append(
            {
                "family": "eta",
                "index": 0,
                "seed": -1,
                "field": eta.detach().double().cpu().numpy().reshape(side, side),
            }
        )
    return probes


def measure_step(step_fn, point, probes, like, ds, norm, eps, rtol, atol) -> tuple[dict, list[dict]]:
    g, _h = step_fn(point.u, point.z)
    value = value_agreement(apply_D(g, ds), point.z, norm)
    rows = []
    for probe in probes:
        v = as_latent(probe["field"], like)
        b = db_at(step_fn, point, v, eps, ds)
        b3 = db_at(step_fn, point, v, eps / 3.0, ds)
        b10 = db_at(step_fn, point, v, eps / 10.0, ds)
        cmp3 = passes_fd(b, b3, rtol, atol, norm)
        cmp10 = passes_fd(b, b10, rtol, atol, norm)
        row = response_metrics(b, v)
        row.update(
            {
                "family": probe["family"],
                "index": int(probe["index"]),
                "seed": int(probe["seed"]),
                "validated": bool(cmp3["ok"] and cmp10["ok"]),
                "fd_rel_eps3": cmp3["rel"],
                "fd_rel_eps10": cmp10["rel"],
                "fd_abs_eps3": cmp3["abs"],
                "fd_abs_eps10": cmp10["abs"],
            }
        )
        rows.append(row)
    return value, rows


def linearity_sweep(step_fn, point, v, b, ds, norm, rtol, atol) -> list[dict]:
    """One-sided Dg(u, z + h v) - Dg(u, z) against h DB v."""
    g0, _h0 = step_fn(point.u, point.z)
    coarse0 = apply_D(g0, ds)
    z_rms = max(t_rms(point.z), 1e-12)
    out = []
    for h_rel in H_REL:
        amp = float(h_rel) * z_rms
        gp, _hp = step_fn(point.u, point.z + amp * v)
        actual = apply_D(gp, ds) - coarse0
        pred = amp * b
        cmp = passes_fd(actual, pred, rtol, atol, norm)
        out.append(
            {
                "h_rel": float(h_rel),
                "amplitude_model": amp,
                "rel": cmp["rel"],
                "abs": cmp["abs"],
                "limit": cmp["limit"],
                "ok": cmp["ok"],
            }
        )
    return out


def _finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x)]


def window_probe_stats(gains, defects, alignments) -> dict:
    g = _finite(gains)
    d = _finite(defects)
    a = _finite(alignments)
    mean_sq = float(np.mean(np.square(g))) if g.size else float("nan")
    return {
        "n": int(g.size),
        "median_gain": float(np.median(g)) if g.size else float("nan"),
        "mean_squared_gain": mean_sq,
        "sqrt_mean_squared_gain": float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) else float("nan"),
        "frac_gain_lt_1": float(np.mean(g < 1.0)) if g.size else float("nan"),
        "sampled_maximum_gain": float(np.max(g)) if g.size else float("nan"),
        "median_defect": float(np.median(d)) if d.size else float("nan"),
        "median_alignment": float(np.median(a)) if a.size else float("nan"),
    }


def across_windows(per_window: list[dict]) -> dict:
    def col(key):
        return _finite([row[key] for row in per_window])

    def pack(key):
        x = col(key)
        if x.size == 0:
            return {"median": float("nan"), "q10": float("nan"), "q90": float("nan")}
        return {
            "median": float(np.median(x)),
            "q10": float(np.quantile(x, 0.10)),
            "q90": float(np.quantile(x, 0.90)),
        }

    maxima = col("sampled_maximum_gain")
    mean_sq = pack("mean_squared_gain")
    return {
        "n_windows": len(per_window),
        "median_gain": pack("median_gain"),
        "mean_squared_gain": mean_sq,
        "sqrt_of_median_mean_squared_gain": (
            float(np.sqrt(mean_sq["median"])) if np.isfinite(mean_sq["median"]) else float("nan")
        ),
        "frac_gain_lt_1": pack("frac_gain_lt_1"),
        "copy_defect": pack("median_defect"),
        "alignment": pack("median_alignment"),
        "sampled_maximum_gain": float(np.max(maxima)) if maxima.size else float("nan"),
        "sampled_maximum_is": (
            "maximum over tested probes and windows, not an operator norm "
            "and not the largest singular value of DB"
        ),
    }


def aggregate_rows(rows: list[dict], validated_only: bool) -> dict:
    """Window medians first, then quantiles of those medians across windows."""
    chosen = [row for row in rows if (row["validated"] or not validated_only)]
    grouped: dict[tuple, list] = {}
    for row in chosen:
        grouped.setdefault((row["mode"], row["family"], int(row["window"])), []).append(row)
    families: dict[tuple, list] = {}
    for (mode, family, _window), items in grouped.items():
        stats = window_probe_stats(
            [item["gain"] for item in items],
            [item["copy_defect"] for item in items],
            [item["alignment"] for item in items],
        )
        stats["window"] = int(items[0]["window"])
        stats["n_failed_in_window_raw"] = int(sum(1 for item in items if not item["validated"]))
        families.setdefault((mode, family), []).append(stats)
    out = {}
    for key, per_window in families.items():
        per_window = sorted(per_window, key=lambda item: item["window"])
        mode, family = key
        out[f"{mode}|{family}"] = {
            "mode": mode,
            "family": family,
            "validated_only": bool(validated_only),
            "per_window": per_window,
            "across_windows": across_windows(per_window),
        }
    return out


def measured_steps(n_steps: int, stride: int) -> list[int]:
    return list(range(0, n_steps, stride))


def check_spacing(sample: dict, delta_t: float) -> None:
    times = [float(frame["time"]) for frame in sample["frames"]]
    for left, right in zip(times, times[1:]):
        if abs((right - left) - float(delta_t)) > 1e-6:
            raise RuntimeError(
                f"window {sample['sample']} spacing {right - left} != --delta-t {delta_t}"
            )


def field_tensor(frame: np.ndarray, like: torch.Tensor, norm: float) -> torch.Tensor:
    src = np.ascontiguousarray(np.asarray(frame, dtype=np.float64))
    return torch.from_numpy(src).to(device=like.device, dtype=like.dtype).view(1, 1, *src.shape) / float(norm)


def autonomous_rollout(ema, frames, model, device, dtype, norm, n_steps: int):
    """Free rollout. Bases and eta are stored before each advance."""
    ds = int(model.ds_factors[0])
    first = make_point(frames[:3], model, device, dtype, norm)
    u = first.u.detach().clone()
    z = first.z.detach().clone()
    bases = []
    etas = []
    for n in range(n_steps):
        true_next = field_tensor(frames[n + 1], u, norm)
        true_after = field_tensor(frames[n + 2], u, norm)
        t = torch.ones(u.shape[0], device=device, dtype=dtype)
        eta = z - apply_D(true_next, ds)
        bases.append(
            L2Point(
                u=u.detach().clone(),
                z=z.detach().clone(),
                t=t,
                u_next=true_next.detach(),
                z_next=apply_D(true_after, ds).detach(),
                stems=(0, 0, 0),
            )
        )
        etas.append(eta.detach())
        g, h = model_step(ema, u, z, t)
        u, z = g.detach(), h.detach()
    return bases, etas


def teacher_points(frames, model, device, dtype, norm, n_steps: int) -> list[L2Point]:
    return [
        make_point(frames[n : n + 3], model, device, dtype, norm) for n in range(n_steps)
    ]


def _step_fn(ema, point: L2Point):
    return lambda u, z, _t=point.t: model_step(ema, u, z, _t)


def run_window(args, ema, model, sample, bands) -> dict:
    device = next(ema.parameters()).device
    dtype = next(ema.parameters()).dtype
    ds = int(model.ds_factors[0])
    frames = [frame["omega"] for frame in sample["frames"]]
    check_spacing(sample, args.delta_t)
    n_steps = int(args.n_steps)
    if len(frames) < n_steps + 2:
        raise RuntimeError(f"window {sample['sample']} has {len(frames)} frames, need {n_steps + 2}")
    teacher = teacher_points(frames, model, device, dtype, args.norm, n_steps)
    auto_bases, auto_eta = autonomous_rollout(
        ema, frames, model, device, dtype, args.norm, n_steps
    )
    steps = measured_steps(n_steps, args.diag_stride)
    side = int(teacher[0].z.shape[-1])
    modes = {
        "teacher_forced": teacher,
        "autonomous": auto_bases,
    }
    values = {name: [] for name in modes}
    rows = []
    linear = []
    skipped_eta = []
    mid = steps[len(steps) // 2]
    linear_steps = {steps[0], mid, steps[-1]}
    for step in steps:
        eta_unit = eta_direction(auto_eta[step])
        if eta_unit is None:
            skipped_eta.append(int(step))
        probes = build_probe_arrays(
            side,
            args.seed,
            int(sample["sample"]),
            step,
            args.n_random,
            args.n_band,
            bands,
            eta_unit,
        )
        for mode, points in modes.items():
            point = points[step]
            value, measured = measure_step(
                _step_fn(ema, point),
                point,
                probes,
                point.z,
                ds,
                args.norm,
                args.fd_eps,
                args.fd_rtol,
                args.fd_atol,
            )
            if args.save_fields:
                value["residual_field"] = (
                    value["residual"].detach().double().cpu().numpy().reshape(side, side)
                )
            else:
                value.pop("residual", None)
            value["step"] = int(step)
            value["mode"] = mode
            values[mode].append(value)
            for row in measured:
                row["mode"] = mode
                row["window"] = int(sample["sample"])
                row["step"] = int(step)
                rows.append(row)
            do_linear = (
                not args.skip_linear
                and int(sample["sample"]) == 0
                and step in linear_steps
            )
            if do_linear:
                seen = set()
                for probe, row in zip(probes, measured):
                    if probe["family"] in seen:
                        continue
                    seen.add(probe["family"])
                    v = as_latent(probe["field"], point.z)
                    b = db_at(_step_fn(ema, point), point, v, args.fd_eps, ds)
                    for item in linearity_sweep(
                        _step_fn(ema, point),
                        point,
                        v,
                        b,
                        ds,
                        args.norm,
                        args.fd_rtol,
                        args.fd_atol,
                    ):
                        item.update(
                            {
                                "mode": mode,
                                "step": int(step),
                                "family": probe["family"],
                                "window": int(sample["sample"]),
                            }
                        )
                        linear.append(item)
    failed = sum(1 for row in rows if not row["validated"])
    print(
        f"[window {int(sample['sample']):02d}] steps={steps} probes={len(rows)} "
        f"fd_failed={failed} eta_skipped={skipped_eta}",
        flush=True,
    )
    return {
        "sample": int(sample["sample"]),
        "start_index": int(sample["start_index"]),
        "stems": [int(frame["stem"]) for frame in sample["frames"]],
        "times": [float(frame["time"]) for frame in sample["frames"]],
        "steps": steps,
        "values": values,
        "rows": rows,
        "linear": linear,
        "skipped_eta": skipped_eta,
        "latent_side": side,
        "pool_factor": ds,
    }


def _column(rows, key, dtype=np.float64):
    if not rows:
        return np.zeros(0, dtype=dtype)
    return np.asarray([row[key] for row in rows], dtype=dtype)


def save_window(path: Path, payload: dict) -> None:
    rows = payload["rows"]
    data = {
        "sample": np.asarray(payload["sample"]),
        "start_index": np.asarray(payload["start_index"]),
        "stems": np.asarray(payload["stems"], dtype=np.int64),
        "times": np.asarray(payload["times"], dtype=np.float64),
        "measured_steps": np.asarray(payload["steps"], dtype=np.int64),
        "skipped_eta": np.asarray(payload["skipped_eta"], dtype=np.int64),
        "latent_side": np.asarray(payload["latent_side"]),
        "pool_factor": np.asarray(payload["pool_factor"]),
        "family": np.asarray([row["family"] for row in rows], dtype="U32"),
        "mode": np.asarray([row["mode"] for row in rows], dtype="U32"),
        "probe_index": _column(rows, "index", np.int64),
        "probe_seed": _column(rows, "seed", np.int64),
        "step": _column(rows, "step", np.int64),
        "gain": _column(rows, "gain"),
        "energy_gain": _column(rows, "energy_gain"),
        "copy_defect": _column(rows, "copy_defect"),
        "alignment": _column(rows, "alignment"),
        "probe_mean": _column(rows, "probe_mean"),
        "rms_v": _column(rows, "rms_v"),
        "validated": np.asarray([bool(row["validated"]) for row in rows], dtype=bool),
        "fd_rel_eps3": _column(rows, "fd_rel_eps3"),
        "fd_rel_eps10": _column(rows, "fd_rel_eps10"),
    }
    for mode, items in payload["values"].items():
        prefix = "tf" if mode == "teacher_forced" else "ar"
        data[f"{prefix}_step"] = np.asarray([item["step"] for item in items], dtype=np.int64)
        data[f"{prefix}_rms_c"] = np.asarray([item["rms_c"] for item in items], dtype=np.float64)
        data[f"{prefix}_rms_z"] = np.asarray([item["rms_z"] for item in items], dtype=np.float64)
        data[f"{prefix}_rel_c"] = np.asarray([item["rel_c"] for item in items], dtype=np.float64)
        data[f"{prefix}_rel_undefined"] = np.asarray(
            [bool(item["rel_undefined"]) for item in items], dtype=bool
        )
        if items and "residual_field" in items[0]:
            data[f"{prefix}_residual"] = np.stack([item["residual_field"] for item in items])
    if payload["linear"]:
        lin = payload["linear"]
        data["lin_mode"] = np.asarray([item["mode"] for item in lin], dtype="U32")
        data["lin_family"] = np.asarray([item["family"] for item in lin], dtype="U32")
        data["lin_step"] = _column(lin, "step", np.int64)
        data["lin_h_rel"] = _column(lin, "h_rel")
        data["lin_rel"] = _column(lin, "rel")
        data["lin_ok"] = np.asarray([bool(item["ok"]) for item in lin], dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_rows(save_dir: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    values = []
    for path in sorted(save_dir.glob("window_*.npz")):
        data = np.load(path)
        window = int(data["sample"])
        for i in range(data["gain"].shape[0]):
            rows.append(
                {
                    "window": window,
                    "mode": str(data["mode"][i]),
                    "family": str(data["family"][i]),
                    "step": int(data["step"][i]),
                    "gain": float(data["gain"][i]),
                    "energy_gain": float(data["energy_gain"][i]),
                    "copy_defect": float(data["copy_defect"][i]),
                    "alignment": float(data["alignment"][i]),
                    "validated": bool(data["validated"][i]),
                }
            )
        for mode, prefix in (("teacher_forced", "tf"), ("autonomous", "ar")):
            if f"{prefix}_step" not in data:
                continue
            for i, step in enumerate(data[f"{prefix}_step"]):
                values.append(
                    {
                        "window": window,
                        "mode": mode,
                        "step": int(step),
                        "rel_c": float(data[f"{prefix}_rel_c"][i]),
                        "rms_c": float(data[f"{prefix}_rms_c"][i]),
                        "rms_z": float(data[f"{prefix}_rms_z"][i]),
                    }
                )
    return rows, values


def _quantile_band(step_to_values: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = sorted(step_to_values)
    med, lo, hi = [], [], []
    for step in steps:
        x = _finite(step_to_values[step])
        if x.size == 0:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            med.append(float(np.median(x)))
            lo.append(float(np.quantile(x, 0.10)))
            hi.append(float(np.quantile(x, 0.90)))
    return np.asarray(steps), np.asarray(med), np.asarray(lo), np.asarray(hi)


def plot_results(save_dir: Path, rows: list[dict], values: list[dict]) -> None:
    if not rows and not values:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, color in (("teacher_forced", "C0"), ("autonomous", "C1")):
        by_window: dict[int, list] = {}
        for item in values:
            if item["mode"] == mode:
                by_window.setdefault(item["window"], []).append(item)
        for window, items in by_window.items():
            order = sorted(items, key=lambda rec: rec["step"])
            ax.plot(
                [rec["step"] for rec in order],
                [rec["rel_c"] for rec in order],
                color=color,
                alpha=0.35,
                lw=1,
            )
        bucket: dict[int, list] = {}
        for item in values:
            if item["mode"] == mode and np.isfinite(item["rel_c"]):
                bucket.setdefault(item["step"], []).append(item["rel_c"])
        steps, med, lo, hi = _quantile_band(bucket)
        ax.plot(steps, med, color=color, lw=2, label=f"{mode} median")
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("RMS(Dg - z) / RMS(z)")
    ax.set_title("value-copy residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "value_residual.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, mode in zip(axes, ("teacher_forced", "autonomous")):
        mids = []
        for window in sorted({row["window"] for row in rows}):
            defects = [
                row["copy_defect"]
                for row in rows
                if row["window"] == window and row["mode"] == mode and np.isfinite(row["copy_defect"])
            ]
            if defects:
                mids.append(float(np.median(defects)))
        if mids:
            ax.hist(mids, bins=min(10, max(len(mids), 1)), color="C0")
        ax.set_title(mode)
        ax.set_xlabel("window-median copy defect")
    axes[0].set_ylabel("windows")
    fig.suptitle("local copy defect, aggregated within each window")
    fig.tight_layout()
    fig.savefig(save_dir / "copy_defect_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for family in FAMILIES:
        xs, ys = [], []
        for row in rows:
            if row["family"] == family and np.isfinite(row["gain"]) and np.isfinite(row["alignment"]):
                xs.append(row["alignment"])
                ys.append(row["gain"])
        if xs:
            ax.scatter(xs, ys, s=12, alpha=0.7, label=family)
    ax.axhline(1.0, color="0.4", lw=0.8)
    ax.axvline(1.0, color="0.4", lw=0.8)
    ax.set_xlabel("alignment cosine(b, v)")
    ax.set_ylabel("gain RMS(b)/RMS(v)")
    ax.set_title("gain versus alignment")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "gain_vs_alignment.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, mode in zip(axes, ("teacher_forced", "autonomous")):
        positions = []
        data = []
        labels = []
        for pos, family in enumerate(FAMILIES):
            mids = []
            windows = sorted({row["window"] for row in rows if row["mode"] == mode and row["family"] == family})
            for window in windows:
                gains = [
                    row["gain"]
                    for row in rows
                    if row["window"] == window
                    and row["mode"] == mode
                    and row["family"] == family
                    and row["validated"]
                    and np.isfinite(row["gain"])
                ]
                if gains:
                    mids.append(float(np.median(gains)))
            if mids:
                positions.append(pos)
                data.append(mids)
                labels.append(family)
        if data:
            ax.boxplot(data, positions=positions, widths=0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(1.0, color="C3", lw=1, label="gain = 1")
        ax.set_title(mode)
    axes[0].set_ylabel("window-median gain")
    axes[0].legend(fontsize=8)
    fig.suptitle("validated gain by probe family")
    fig.tight_layout()
    fig.savefig(save_dir / "gain_by_family.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, ("teacher_forced", "autonomous")):
        for family in FAMILIES:
            bucket: dict[int, list] = {}
            windows = sorted({row["window"] for row in rows if row["mode"] == mode and row["family"] == family})
            for window in windows:
                by_step: dict[int, list] = {}
                for row in rows:
                    if (
                        row["window"] == window
                        and row["mode"] == mode
                        and row["family"] == family
                        and np.isfinite(row["gain"])
                    ):
                        by_step.setdefault(row["step"], []).append(row["gain"])
                for step, gains in by_step.items():
                    bucket.setdefault(step, []).append(float(np.median(gains)))
            steps, med, lo, hi = _quantile_band(bucket)
            if steps.size:
                ax.plot(steps, med, label=family)
                ax.fill_between(steps, lo, hi, alpha=0.12)
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_title(mode)
        ax.set_xlabel("rollout step")
    axes[0].set_ylabel("window-median gain")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("gain versus rollout step")
    fig.tight_layout()
    fig.savefig(save_dir / "gain_vs_step.png", dpi=140)
    plt.close(fig)


def write_summary(save_dir: Path, args, bands, precision: dict | None) -> dict:
    rows, values = load_rows(save_dir)
    band_def = []
    side = None
    files = sorted(save_dir.glob("window_*.npz"))
    if files:
        side = int(np.load(files[0])["latent_side"])
    for name, lo, hi in bands:
        if side is not None:
            _field, mask, definition = band_field(side, lo, hi, seed=0)
            definition["family"] = name
            definition["mask_true"] = int(np.count_nonzero(mask))
            band_def.append(definition)
    n_failed = int(sum(1 for row in rows if not row["validated"]))
    summary = {
        "n_probe_rows": len(rows),
        "n_failed_validation": n_failed,
        "failed_probes_are_not_validated": True,
        "aggregation": "median inside each window, then median and 10/90 quantiles across windows",
        "bands": band_def,
        "all_probes": aggregate_rows(rows, validated_only=False),
        "validated_only": aggregate_rows(rows, validated_only=True),
        "precision": precision,
        "fd_eps": float(args.fd_eps),
        "fd_rtol": float(args.fd_rtol),
        "fd_atol": float(args.fd_atol),
        "atol_units": "physical vorticity, Euclidean norm plus atol * sqrt(numel)",
        "norm": float(args.norm),
        "diag_stride": int(args.diag_stride),
        "n_steps": int(args.n_steps),
        "n_random": int(args.n_random),
        "notes_file": "notes.txt",
    }
    write_json(save_dir / "summary.json", summary)
    (save_dir / "notes.txt").write_text(NOTES)
    plot_results(save_dir, rows, values)
    return summary


def run_rank(args) -> None:
    precision = lock_precision()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested and is not available")
    device = torch.device(args.device)
    bands = parse_bands(args.bands)
    before = sha256_file(args.checkpoint)
    model, ema = load_ema_float64(args.checkpoint, device)
    after = sha256_file(args.checkpoint)
    samples = sample_windows(
        args.eval_dir,
        args.every_nth_field,
        args.n_windows,
        lookahead=args.n_steps + 1,
        seed=args.seed,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mine = [sample for sample in samples if int(sample["sample"]) % args.n_gpus == args.gpu_rank]
    print(
        f"[rank {args.gpu_rank}] windows {[s['sample'] for s in mine]} "
        f"device={device} dtype={next(ema.parameters()).dtype}",
        flush=True,
    )
    for sample in mine:
        payload = run_window(args, ema, model, sample, bands)
        save_window(save_dir / f"window_{int(sample['sample']):02d}.npz", payload)
    if args.gpu_rank == 0:
        meta = base_metadata(
            args,
            {
                "task": "L2 coarse-input copying and DB disturbance gain",
                "operator": "DB = D dg/dz",
                "py2d_solver": "not used",
                "model_dtype": "float64",
                "precision": precision,
                "checkpoint_sha256_after_load": after,
                "weights_unchanged": before == after,
                "ds_factors": [int(v) for v in model.ds_factors],
                "latent_res": [int(v) for v in getattr(model, "latent_res", [])],
                "pooling": f"avg_pool2d kernel=stride={int(model.ds_factors[0])}",
                "norm": float(args.norm),
                "bands": args.bands,
                "base_points": ["teacher_forced (u_n, D u_{n+1})", "autonomous (u_hat_n, z_in_n)"],
                "eta": "z_in_n - D u_true[n+1], mean kept, zero directions skipped",
                "constant_mode": "separate probe of ones; not removed from random or eta probes",
            },
        )
        write_json(save_dir / "metadata.json", meta)
    if args.n_gpus == 1:
        write_summary(save_dir, args, bands, precision)


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
    write_summary(Path(args.save_dir), args, parse_bands(args.bands), None)


def main() -> None:
    args = parse_args()
    if args.n_gpus > 1 and not args.worker:
        orchestrate(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
