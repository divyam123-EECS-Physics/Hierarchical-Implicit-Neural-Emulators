#!/usr/bin/env python3
"""Arrival-step cosines against the four (T_j - I) channels.

Each channel is the sum of (T_j - I) on the realized (e_j, nu_j).
The three reference rows are the single arriving injection, the present
rollout error, and the sum of signed injections. Section 10 r_H and r_z
are not these channels. No traces and no eigenvalues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_EVAL = Path(__file__).resolve().parent
_REPO = _EVAL.parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

KEYS = (
    "cos_eu_fine_self",
    "cos_eu_fine_cross",
    "cos_xi_off_cross",
    "cos_xi_off_self",
    "cos_e_fine_self",
    "cos_e_fine_cross",
    "cos_nu_off_cross",
    "cos_nu_off_self",
    "cos_sum_eu_fine_self",
    "cos_sum_eu_fine_cross",
    "cos_sum_xi_off_cross",
    "cos_sum_xi_off_self",
)

OFF_KEYS = (
    "cos_xi_off_cross_S",
    "cos_xi_off_self_Z",
    "cos_nu_off_cross_S",
    "cos_nu_off_self_Z",
)


def _frames(entry: dict) -> tuple[list[np.ndarray], list[int]]:
    import scipy.io

    frames = []
    for path in entry["paths"]:
        raw = scipy.io.loadmat(path)
        frames.append(np.asarray(raw["Omega"]))
    stems = [int(s) for s in entry["stems"]]
    return frames, stems


def _attach(point, stem: int, nxt: int | None) -> None:
    from l2_physical_jvp import load_dns_frame

    now = load_dns_frame(stem)
    point.dns = now["omega"]
    point.hats = now["hats"]
    if nxt is None:
        return
    nxt_frame = load_dns_frame(nxt)
    point.dns_next = nxt_frame["omega"]
    point.hats_next = nxt_frame["hats"]


def _offgraph_one_factor(ema, points, tail, e, eta, eps_u, xi, tangent, norm, ds, n_steps):
    """One-factor off-graph channels, the lower-row analogues of Eq. (10.4).

    The fine vector at step k is S_k = sum_{i<k} epsilon^u_i.
    The latent vector at step k is Z_k = sum_{i<k} xi_i.
    r_H and r_z stay on the fine row and are not rebuilt here.
    """
    import torch

    from l2_fd_jvp import apply_D, jvp_both_uz, jvp_both_z, model_step
    from l2_operators import _phi_image
    from l2_retention import _cos, _norm

    t_window = time.time()
    off_cross = torch.zeros_like(eta[0])
    off_self = torch.zeros_like(eta[0])
    sum_s = torch.zeros_like(e[0])
    sum_z = torch.zeros_like(eta[0])
    series = {key: np.full(n_steps + 1, np.nan) for key in OFF_KEYS}
    lifts_e: dict[int, torch.Tensor] = {}
    lifts_s: dict[int, torch.Tensor] = {}

    def _point(n):
        return tail if n == n_steps else points[n]

    def _lift(n, err, cache):
        if _norm(err) == 0.0:
            return torch.zeros_like(eta[0])
        cached = cache.get(n)
        if cached is not None:
            return cached
        point = _point(n)
        if getattr(point, "dns", None) is None:
            raise RuntimeError(f"physical tangent missing DNS at step {n}")
        image = _phi_image(
            tangent, point.u, err, norm, dns=point.dns, hats=point.hats,
        )
        pooled = apply_D(image, ds).detach()
        cache[n] = pooled
        return pooled

    def _nu(n):
        return eta[n] - _lift(n, e[n], lifts_e)

    if _norm(e[0]) != 0.0 or _norm(_nu(0)) != 0.0:
        raise RuntimeError("e_0 and nu_0 must be zero")

    def _step(k):
        return lambda u, z, _t=points[k].t: model_step(ema, u, z, _t)

    def _cross(k, vec):
        if _norm(vec) == 0.0:
            return torch.zeros_like(eta[0])
        lift = _lift(k, vec, lifts_s)
        _dg, dh = jvp_both_uz(_step(k), points[k], vec, lift, 0.0)
        dg = _dg.detach()
        dh = dh.detach()
        pooled = apply_D(
            _phi_image(
                tangent, points[k].u_next, dg, norm,
                dns=points[k].dns_next, hats=points[k].hats_next,
            ),
            ds,
        )
        return (dh - pooled).detach()

    for n in range(n_steps + 1):
        nu_n = _nu(n)
        if n == 1 and max(_norm(off_cross), _norm(off_self)) > 1e-8:
            raise RuntimeError("off-graph channels at n=1 must be zero")
        if n >= 1:
            series["cos_xi_off_cross_S"][n] = _cos(sum_z, off_cross)
            series["cos_xi_off_self_Z"][n] = _cos(sum_z, off_self)
            series["cos_nu_off_cross_S"][n] = _cos(nu_n, off_cross)
            series["cos_nu_off_self_Z"][n] = _cos(nu_n, off_self)
        if n == n_steps:
            break
        if _norm(sum_s) != 0.0:
            off_cross = off_cross + _cross(n, sum_s)
        if n >= 1 and _norm(sum_z) != 0.0:
            bw, kw = jvp_both_z(_step(n), points[n], sum_z, 0.0)
            bw = bw.detach()
            kw = kw.detach()
            correction = apply_D(
                _phi_image(
                    tangent, points[n].u_next, bw, norm,
                    dns=points[n].dns_next, hats=points[n].hats_next,
                ),
                ds,
            )
            off_self = off_self + (kw - correction - sum_z)
        sum_s = sum_s + eps_u[n]
        sum_z = sum_z + xi[n]
        if n in (0, 1, 9, 19, 29, 39):
            print(f"step {n} sec={time.time() - t_window:.1f}", flush=True)
    return series


def _one_window(ema, model, frames, stems, tangent, device, dtype, norm: float, n_steps: int, channels: str = "realized") -> dict:
    import torch

    from l2_fd_jvp import apply_D, jvp_both_uz, jvp_both_z, make_point, model_step
    from l2_operators import _phi_image
    from l2_retention import _cos, _norm

    t_window = time.time()
    ds = int(model.ds_factors[0])
    points = []
    for k in range(n_steps):
        pt = make_point(frames[k : k + 3], model, device, dtype, norm)
        _attach(pt, stems[k], stems[k + 1])
        points.append(pt)
    eps_u = []
    eps_z = []
    xi = []
    for pt in points:
        _g, h = model_step(ema, pt.u, pt.z, pt.t)
        eu = (_g - pt.u_next).detach()
        ez = (h - pt.z_next).detach()
        image = _phi_image(
            tangent, pt.u_next, eu, norm,
            dns=pt.dns_next, hats=pt.hats_next,
        )
        eps_u.append(eu)
        eps_z.append(ez)
        xi.append((ez - apply_D(image, ds)).detach())

    high = points[0].u.clone()
    latent = points[0].z.clone()
    e = [torch.zeros_like(high)]
    eta = [torch.zeros_like(latent)]
    for k in range(n_steps):
        high, latent = model_step(ema, high, latent, points[k].t)
        high = high.detach()
        latent = latent.detach()
        e.append(high - points[k].u_next)
        eta.append(latent - points[k].z_next)

    tail = make_point(
        [frames[n_steps], frames[n_steps + 1], frames[n_steps + 1]],
        model, device, dtype, norm,
    )
    _attach(tail, stems[n_steps], None)
    if channels == "offgraph":
        return _offgraph_one_factor(
            ema, points, tail, e, eta, eps_u, xi, tangent, norm, ds, n_steps,
        )

    fine_self = torch.zeros_like(e[0])
    fine_cross = torch.zeros_like(e[0])
    off_cross = torch.zeros_like(eta[0])
    off_self = torch.zeros_like(eta[0])
    sum_eu = torch.zeros_like(e[0])
    sum_xi = torch.zeros_like(eta[0])
    series = {key: np.full(n_steps + 1, np.nan) for key in KEYS}
    lifts: dict[int, torch.Tensor] = {}

    def _point(n):
        return tail if n == n_steps else points[n]

    def _lift(n, err):
        if _norm(err) == 0.0:
            return torch.zeros_like(eta[0])
        cached = lifts.get(n)
        if cached is not None:
            return cached
        point = _point(n)
        if getattr(point, "dns", None) is None:
            raise RuntimeError(f"physical tangent missing DNS at step {n}")
        image = _phi_image(
            tangent, point.u, err, norm, dns=point.dns, hats=point.hats,
        )
        pooled = apply_D(image, ds).detach()
        lifts[n] = pooled
        return pooled

    def _nu(n):
        return eta[n] - _lift(n, e[n])

    if _norm(e[0]) != 0.0 or _norm(_nu(0)) != 0.0:
        raise RuntimeError("e_0 and nu_0 must be zero")

    def _step(k):
        return lambda u, z, _t=points[k].t: model_step(ema, u, z, _t)

    def _offgraph_from_fine(k, err, lift):
        if _norm(err) == 0.0:
            return torch.zeros_like(err), torch.zeros_like(eta[0])
        dg, dh = jvp_both_uz(_step(k), points[k], err, lift, 0.0)
        dg = dg.detach()
        dh = dh.detach()
        pooled = apply_D(
            _phi_image(
                tangent, points[k].u_next, dg, norm,
                dns=points[k].dns_next, hats=points[k].hats_next,
            ),
            ds,
        )
        return dg - err, (dh - pooled).detach()

    for n in range(n_steps + 1):
        nu_n = _nu(n)
        if n == 1:
            norms = (
                _norm(fine_self), _norm(fine_cross),
                _norm(off_cross), _norm(off_self),
            )
            if max(norms) > 1e-8:
                raise RuntimeError(f"channels at n=1 are {norms}, expected 0")
        if n >= 1:
            eu = eps_u[n - 1]
            xinj = xi[n - 1]
            series["cos_eu_fine_self"][n] = _cos(eu, fine_self)
            series["cos_eu_fine_cross"][n] = _cos(eu, fine_cross)
            series["cos_xi_off_cross"][n] = _cos(xinj, off_cross)
            series["cos_xi_off_self"][n] = _cos(xinj, off_self)
            series["cos_e_fine_self"][n] = _cos(e[n], fine_self)
            series["cos_e_fine_cross"][n] = _cos(e[n], fine_cross)
            series["cos_nu_off_cross"][n] = _cos(nu_n, off_cross)
            series["cos_nu_off_self"][n] = _cos(nu_n, off_self)
            series["cos_sum_eu_fine_self"][n] = _cos(sum_eu, fine_self)
            series["cos_sum_eu_fine_cross"][n] = _cos(sum_eu, fine_cross)
            series["cos_sum_xi_off_cross"][n] = _cos(sum_xi, off_cross)
            series["cos_sum_xi_off_self"][n] = _cos(sum_xi, off_self)
        if n == n_steps:
            break
        d_fine, d_off = _offgraph_from_fine(n, e[n], _lift(n, e[n]))
        fine_self = fine_self + d_fine
        off_cross = off_cross + d_off
        if _norm(nu_n) != 0.0:
            bw, kw = jvp_both_z(_step(n), points[n], nu_n, 0.0)
            bw = bw.detach()
            kw = kw.detach()
            correction = apply_D(
                _phi_image(
                    tangent, points[n].u_next, bw, norm,
                    dns=points[n].dns_next, hats=points[n].hats_next,
                ),
                ds,
            )
            fine_cross = fine_cross + bw
            off_self = off_self + (kw - correction - nu_n)
        sum_eu = sum_eu + eps_u[n]
        sum_xi = sum_xi + xi[n]
        if n in (0, 1, 9, 19, 29, 39):
            print(f"step {n} sec={time.time() - t_window:.1f}", flush=True)
    return series


def _compute(args) -> None:
    import torch

    os.environ["HINE_DNS_DATA"] = str(Path(args.dns_data))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    from eval_jacobian_manual_mse_plus import load_model
    from l2_physical_jvp import make_tangent

    payload = json.loads(Path(args.sample_index).read_text())
    mine = [s for s in payload["samples"] if int(s["sample"]) % args.n_gpus == args.rank]
    model, ema = load_model(args.checkpoint, device)
    ema.double().eval()
    tangent = make_tangent(
        "py2d",
        Re=args.re,
        fkx=args.fkx,
        fky=args.fky,
        alpha=args.alpha,
        beta=args.beta,
        solver_dt=args.solver_dt,
        emulator_dt=args.dt,
    )
    active = OFF_KEYS if args.channels == "offgraph" else KEYS
    dest = Path(args.save_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rows = {key: [] for key in active}
    indices = []
    for sample in mine:
        t0 = time.time()
        frames, stems = _frames(sample)
        if len(frames) < args.rollout + 2:
            raise RuntimeError(f"window {sample['sample']} has {len(frames)} frames")
        out = _one_window(
            ema, model, frames, stems, tangent, device, torch.float64, args.norm, args.rollout,
            channels=args.channels,
        )
        for key in active:
            rows[key].append(out[key])
        indices.append(int(sample["sample"]))
        probe = out[active[0]]
        finite = probe[np.isfinite(probe)]
        print(
            f"rank {args.rank} window {sample['sample']} "
            f"{active[0]}_mean={float(np.mean(finite)) if finite.size else float('nan'):.4f} "
            f"sec={time.time() - t0:.1f}",
            flush=True,
        )
        np.savez_compressed(
            dest / f"window_{int(sample['sample']):02d}.npz",
            **{key: out[key] for key in active},
        )
    path = dest / f"t_channel_rank{args.rank}.npz"
    np.savez_compressed(
        path,
        indices=np.asarray(indices, dtype=np.int64),
        **{key: np.stack(rows[key]) for key in active},
    )
    print(f"wrote {path}", flush=True)


def merge_ranks(save_dir: Path, n_gpus: int) -> Path:
    windows = sorted(save_dir.glob("window_*.npz"))
    if windows:
        indices = np.asarray([int(path.stem.split("_")[1]) for path in windows], dtype=np.int64)
        order = np.argsort(indices)
        if indices[order].tolist() != list(range(int(indices.size))):
            raise RuntimeError(f"windows {indices.tolist()} do not cover 0..{indices.size - 1}")
        merged = {"indices": indices[order]}
        loaded = [np.load(path) for path in windows]
        keys = [key for key in loaded[0].files if key != "indices"]
        for key in keys:
            stacked = np.stack([np.asarray(part[key]) for part in loaded])
            merged[key] = stacked[order]
    else:
        parts = [np.load(save_dir / f"t_channel_rank{rank}.npz") for rank in range(n_gpus)]
        indices = np.concatenate([np.asarray(part["indices"]) for part in parts])
        order = np.argsort(indices)
        if indices[order].tolist() != list(range(int(indices.size))):
            raise RuntimeError(f"windows {indices.tolist()} do not cover 0..{indices.size - 1}")
        merged = {"indices": indices[order]}
        for key in KEYS:
            stacked = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
            merged[key] = stacked[order]
    dest = save_dir / "cosines.npz"
    np.savez_compressed(dest, **merged)
    return dest


def _band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.full(arr.shape[1], np.nan)
    lo = np.full(arr.shape[1], np.nan)
    hi = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        good = arr[:, j][np.isfinite(arr[:, j])]
        if good.size:
            mean[j] = float(np.mean(good))
            lo[j] = float(np.percentile(good, 16))
            hi[j] = float(np.percentile(good, 84))
    return mean, lo, hi


PLOTS = (
    (
        "01_eu_fine_self.png",
        "Fine injection vs fine self-channel",
        "cos_eu_fine_self",
        r"$\cos\!\left(\varepsilon^u_{n-1},"
        r"\sum_{j=0}^{n-1}(\Phi'_j+\Delta^g_j-I_N)e_j\right)$",
    ),
    (
        "02_eu_fine_cross.png",
        "Fine injection vs fine cross-channel",
        "cos_eu_fine_cross",
        r"$\cos\!\left(\varepsilon^u_{n-1},"
        r"\sum_{j=0}^{n-1} B_j\nu_j\right)$",
    ),
    (
        "03_xi_off_cross.png",
        "Off-graph injection vs off-graph cross-channel",
        "cos_xi_off_cross",
        r"$\cos\!\left(\xi_{n-1},"
        r"\sum_{j=0}^{n-1}(\Delta^h_j-D\Phi'_{j+1}\Delta^g_j)e_j\right)$",
    ),
    (
        "04_xi_off_self.png",
        "Off-graph injection vs off-graph self-channel",
        "cos_xi_off_self",
        r"$\cos\!\left(\xi_{n-1},"
        r"\sum_{j=0}^{n-1}(S_j-I_M)\nu_j\right)$",
    ),
    (
        "05_e_fine_self.png",
        "Fine error vs fine self-channel",
        "cos_e_fine_self",
        r"$\cos\!\left(e_n,"
        r"\sum_{j=0}^{n-1}(\Phi'_j+\Delta^g_j-I_N)e_j\right)$",
    ),
    (
        "06_e_fine_cross.png",
        "Fine error vs fine cross-channel",
        "cos_e_fine_cross",
        r"$\cos\!\left(e_n,"
        r"\sum_{j=0}^{n-1} B_j\nu_j\right)$",
    ),
    (
        "07_nu_off_cross.png",
        "Off-graph error vs off-graph cross-channel",
        "cos_nu_off_cross",
        r"$\cos\!\left(\nu_n,"
        r"\sum_{j=0}^{n-1}(\Delta^h_j-D\Phi'_{j+1}\Delta^g_j)e_j\right)$",
    ),
    (
        "08_nu_off_self.png",
        "Off-graph error vs off-graph self-channel",
        "cos_nu_off_self",
        r"$\cos\!\left(\nu_n,"
        r"\sum_{j=0}^{n-1}(S_j-I_M)\nu_j\right)$",
    ),
    (
        "09_sum_eu_fine_self.png",
        "Accumulated fine injection vs fine self-channel",
        "cos_sum_eu_fine_self",
        r"$\cos\!\left(\sum_{i=0}^{n-1}\varepsilon^u_i,"
        r"\sum_{j=0}^{n-1}(\Phi'_j+\Delta^g_j-I_N)e_j\right)$",
    ),
    (
        "10_sum_eu_fine_cross.png",
        "Accumulated fine injection vs fine cross-channel",
        "cos_sum_eu_fine_cross",
        r"$\cos\!\left(\sum_{i=0}^{n-1}\varepsilon^u_i,"
        r"\sum_{j=0}^{n-1} B_j\nu_j\right)$",
    ),
    (
        "11_sum_xi_off_cross.png",
        "Accumulated off-graph injection vs off-graph cross-channel",
        "cos_sum_xi_off_cross",
        r"$\cos\!\left(\sum_{i=0}^{n-1}\xi_i,"
        r"\sum_{j=0}^{n-1}(\Delta^h_j-D\Phi'_{j+1}\Delta^g_j)e_j\right)$",
    ),
    (
        "12_sum_xi_off_self.png",
        "Accumulated off-graph injection vs off-graph self-channel",
        "cos_sum_xi_off_self",
        r"$\cos\!\left(\sum_{i=0}^{n-1}\xi_i,"
        r"\sum_{j=0}^{n-1}(S_j-I_M)\nu_j\right)$",
    ),
)


def draw(k4: Path, k5: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = np.load(k4)
    right = np.load(k5)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {}
    for key in KEYS:
        combined[f"k4_{key}"] = np.asarray(left[key])
        combined[f"k5_{key}"] = np.asarray(right[key])
    np.savez_compressed(out_dir / "per_window_cosines.npz", **combined)
    step = np.arange(combined["k4_cos_e_fine_self"].shape[1])
    for filename, title, key, caption in PLOTS:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for label, color, values in (
            ("k = 4", "C0", combined[f"k4_{key}"]),
            ("k = 5", "C3", combined[f"k5_{key}"]),
        ):
            mean, lo, hi = _band(values)
            ax.plot(step, mean, color=color, label=label)
            ax.fill_between(step, lo, hi, color=color, alpha=0.18)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlim(0, step[-1])
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("rollout step n")
        ax.set_ylabel("cosine")
        ax.set_title(title)
        ax.legend(title="wavenumber")
        fig.text(0.01, 0.01, caption, fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out_dir / filename, dpi=140)
        plt.close(fig)
        print(out_dir / filename, flush=True)
    lines = [
        "All twelve cosines were backfilled. None were in the saved Task 1 series.",
        "Saved roll_cos_e_rH, roll_cos_rH_inj, roll_cos_e_rZ, and roll_cos_nu_xi",
        "use Section 10 r_H and r_z, or cos(nu_n, sum xi). Those names stay on",
        "those objects and were not used as these channels.",
        "Reused: the same 20 windows, checkpoints, norm=10, rollout 40, float64,",
        "and the py2d jax tangent on each run's own DNS tree.",
        "Each xi_i was formed with its own D Phi'_{i+1} before the sum.",
        "e_0 = nu_0 = 0, so every channel is zero at n=1 and those cosines are undefined.",
        "",
        "plot  reference                         channel                                          k=4         k=5",
    ]
    labels = (
        r"epsilon^u_{n-1}",
        r"epsilon^u_{n-1}",
        r"xi_{n-1}",
        r"xi_{n-1}",
        "e_n",
        "e_n",
        "nu_n",
        "nu_n",
        r"sum_{i<n} epsilon^u_i",
        r"sum_{i<n} epsilon^u_i",
        r"sum_{i<n} xi_i",
        r"sum_{i<n} xi_i",
    )
    channels = (
        r"sum (Phi'+Delta^g-I_N) e",
        r"sum B nu",
        r"sum (Delta^h-D Phi' Delta^g) e",
        r"sum (S-I_M) nu",
        r"sum (Phi'+Delta^g-I_N) e",
        r"sum B nu",
        r"sum (Delta^h-D Phi' Delta^g) e",
        r"sum (S-I_M) nu",
        r"sum (Phi'+Delta^g-I_N) e",
        r"sum B nu",
        r"sum (Delta^h-D Phi' Delta^g) e",
        r"sum (S-I_M) nu",
    )
    for i, ((filename, _title, key, _caption), reference, channel) in enumerate(
        zip(PLOTS, labels, channels), start=1,
    ):
        lines.append(
            f"{i:<5} {reference:<32} {channel:<48} backfilled   backfilled   {filename}"
        )
    (out_dir / "STATUS.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--sample-index", required=True)
    run.add_argument("--save-dir", required=True)
    run.add_argument("--dns-data", required=True)
    run.add_argument("--fkx", type=int, required=True)
    run.add_argument("--fky", type=int, required=True)
    run.add_argument("--rank", type=int, required=True)
    run.add_argument("--n-gpus", type=int, required=True)
    run.add_argument("--norm", type=float, default=10.0)
    run.add_argument("--rollout", type=int, default=40)
    run.add_argument("--dt", type=float, default=0.05)
    run.add_argument("--re", type=float, default=1e4)
    run.add_argument("--alpha", type=float, default=0.1)
    run.add_argument("--beta", type=float, default=20.0)
    run.add_argument("--solver-dt", type=float, default=1e-4)
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--channels", choices=["realized", "offgraph"], default="realized")
    merge = sub.add_parser("merge")
    merge.add_argument("--save-dir", required=True)
    merge.add_argument("--n-gpus", type=int, required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--k4", required=True)
    plot.add_argument("--k5", required=True)
    plot.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "run":
        _compute(args)
    elif args.cmd == "merge":
        print(merge_ranks(Path(args.save_dir), args.n_gpus), flush=True)
    else:
        draw(Path(args.k4), Path(args.k5), Path(args.out))


if __name__ == "__main__":
    main()
