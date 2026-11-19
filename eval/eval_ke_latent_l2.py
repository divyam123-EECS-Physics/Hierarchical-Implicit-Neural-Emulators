#!/usr/bin/env python3
"""Wavenumber-5 L=2 kinetic energy: high field versus the latent grid.

The L=2 coarse state is a gridded vorticity. FutureINNPlus._pool_seq restricts
the high field with average pooling (kernel = stride = img_size / latent_res),
and the coarse head is trained to match that pooled vorticity. Kinetic energy
of the latent is the same stream-function energy on that coarser grid.

This script does not import or edit the running 100k rollout. It writes only
under the save directory passed on the command line.

Time index. One emulator step is 0.05 and files are 0.01 apart, so lead n is
stem ic + 5*n. A full L=2 step emits the high field at lead n and the coarse
field at lead n+1 (the coarse input is already one step ahead). The arrays
below are aligned to one physical time: ar_high[n] is the high prediction of
that stem, and ar_latent[n] is the coarse prediction of that same stem
(the t=0 warmup for lead 1, otherwise the coarse head from the previous step).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch.nn import functional as F

from eval.eval_jacobian_manual_mse import load_kept_paths
from eval.eval_jacobian_manual_mse_plus import load_model as load_l2
from eval.eval_ke_long_rollout import choose_ics
from eval.eval_kinetic_energy_survival import (
    SIGMA_MULT,
    kinetic_energy_torch,
    make_wavenumber_tensors,
)

IMG_SIZE = 256
REF_STEM_MIN = 10000
REF_STEM_MAX = 105000
TRAIN_MAX = 91015
VAL_MAX = 91620
EXPECTED_ICS = (
    91625,
    93110,
    94595,
    96085,
    97570,
    99055,
    100540,
    102030,
    103515,
    105000,
)
ALIGNMENT = (
    "lead n = 1..N is the physical time ic_stem + every_nth*n. "
    "ar_high[:, n-1] is the high field after n autoregressive steps (indicator t=K). "
    "ar_latent[:, n-1, i] is the level-i coarse field trained against avg-pooled "
    "vorticity at that same stem: warmup indicator t=i when n=i+1, otherwise the "
    "coarse head emitted by autoregressive step n-2-i. "
    "gt_high and gt_downsampled are the true Omega and its avg-pool at that stem."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/k5/l2/checkpoint-last.pth",
    )
    p.add_argument(
        "--eval-dir",
        default="/global/homes/d/divyam/TACSLAB/k5_data/eval",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/global/homes/d/divyam/TACSLAB/k5_data"),
    )
    p.add_argument(
        "--high-ref",
        type=Path,
        default=Path("/pscratch/sd/d/divyam/hine_runs/k5/reference_kinetic_energy.npz"),
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/pscratch/sd/d/divyam/hine_runs/k5/l2/ke_latent_100"),
    )
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ics", type=int, default=10)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--ref-batch", type=int, default=64)
    p.add_argument("--load-workers", type=int, default=8)
    return p.parse_args()


def mat_path_for_stem(root: Path, stem: int) -> Path:
    if stem <= TRAIN_MAX:
        split = "train"
    elif stem <= VAL_MAX:
        split = "validation"
    else:
        split = "eval"
    return root / split / f"{stem}.mat"


def load_omega(root: Path, stem: int) -> np.ndarray:
    path = mat_path_for_stem(root, int(stem))
    if not path.is_file():
        raise FileNotFoundError(path)
    omega = np.asarray(scipy.io.loadmat(path)["Omega"], dtype=np.float32)
    if omega.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"{path}: expected {(IMG_SIZE, IMG_SIZE)}, got {omega.shape}")
    return omega


def load_high_reference(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        energies = np.asarray(z["energies"], dtype=np.float64)
        stems = np.asarray(z["stems"], dtype=np.int64)
        mu = float(np.asarray(z["mu_E"]))
        sigma = float(np.asarray(z["sigma_E"]))
    expected = np.arange(REF_STEM_MIN, REF_STEM_MAX + 1, dtype=np.int64)
    if stems.shape != expected.shape or not np.array_equal(stems, expected):
        raise ValueError(f"{path} stems are not {REF_STEM_MIN}..{REF_STEM_MAX} inclusive")
    if energies.shape != expected.shape:
        raise ValueError(f"{path} energies shape {energies.shape}")
    return energies, stems, mu, sigma


def select_ics(
    eval_dir: str, every: int, n_ics: int, n_steps: int
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Same choose_ics / load_kept_paths(mod) path as the 100k rollout.

    An initial condition is kept only when every future lead stem exists in the
    eval set. Frames are never invented.
    """
    kept = load_kept_paths(eval_dir, every, "mod")
    ic_index_all = choose_ics(len(kept), n_ics)
    stems_all = [int(kept[int(i)][0]) for i in ic_index_all]
    if stems_all != list(EXPECTED_ICS):
        raise RuntimeError(
            f"choose_ics stems {stems_all} do not match the 100k set {list(EXPECTED_ICS)}"
        )
    kept_stems = {int(stem) for stem, _ in kept}
    paths = {int(stem): path for stem, path in kept}
    usable_index: list[int] = []
    usable_stems: list[int] = []
    usable_paths: list[str] = []
    dropped: list[int] = []
    for index, stem in zip(ic_index_all, stems_all):
        future = [stem + every * lead for lead in range(1, n_steps + 1)]
        n_ok = sum(s in kept_stems for s in future)
        if n_ok == n_steps:
            usable_index.append(int(index))
            usable_stems.append(stem)
            usable_paths.append(paths[stem])
        else:
            dropped.append(stem)
            print(
                f"drop IC stem {stem}: {n_ok}/{n_steps} future frames "
                f"(last needed {future[-1]}, eval ends at {max(kept_stems)})",
                flush=True,
            )
    if not usable_stems:
        raise RuntimeError("no initial condition can supply the requested leads")
    return (
        np.asarray(usable_index, dtype=np.int64),
        np.asarray(usable_stems, dtype=np.int64),
        usable_paths,
        dropped,
    )


def energies_physical(field: torch.Tensor, waves) -> np.ndarray:
    """field is physical vorticity, shape (B, H, W) or (B, 1, H, W)."""
    if field.ndim == 4:
        field = field[:, 0]
    kx, ky, inv = waves
    return kinetic_energy_torch(field, kx, ky, inv).double().cpu().numpy()


@torch.no_grad()
def pooled_physical(omega_bhw: torch.Tensor, ds: int) -> torch.Tensor:
    """Average-pool restriction. Same operator as FutureINNPlus._pool_seq."""
    return F.avg_pool2d(omega_bhw.unsqueeze(1), kernel_size=ds, stride=ds)


def build_latent_reference(
    root: Path,
    save_path: Path,
    latent_res: list[int],
    ds_factors: list[int],
    device: torch.device,
    batch: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stems = np.arange(REF_STEM_MIN, REF_STEM_MAX + 1, dtype=np.int64)
    n_levels = len(latent_res)
    if save_path.is_file():
        with np.load(save_path) as z:
            cached_stems = np.asarray(z["stems"], dtype=np.int64)
            cached_res = np.asarray(z["latent_res"], dtype=np.int64)
            energies = np.asarray(z["energies"], dtype=np.float64)
            mu = np.asarray(z["mu_E"], dtype=np.float64)
            sigma = np.asarray(z["sigma_E"], dtype=np.float64)
        if (
            np.array_equal(cached_stems, stems)
            and np.array_equal(cached_res, np.asarray(latent_res, dtype=np.int64))
            and energies.shape == (stems.size, n_levels)
        ):
            print(
                f"loaded latent reference {save_path} mu={mu.tolist()} sigma={sigma.tolist()}",
                flush=True,
            )
            return energies, stems, mu, sigma
        raise RuntimeError(f"existing {save_path} does not match this L=2 grid; refusing to overwrite blindly")

    waves = [make_wavenumber_tensors(int(r), int(r), device)[:3] for r in latent_res]
    energies = np.empty((stems.size, n_levels), dtype=np.float64)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, stems.size, batch):
            chunk = [int(s) for s in stems[start : start + batch]]
            omegas = np.stack(list(pool.map(lambda s: load_omega(root, s), chunk)), axis=0)
            batch_t = torch.from_numpy(omegas).to(device)
            for level, ds in enumerate(ds_factors):
                pooled = pooled_physical(batch_t, int(ds))
                if tuple(pooled.shape[-2:]) != (latent_res[level], latent_res[level]):
                    raise RuntimeError(
                        f"pool ds={ds} produced {tuple(pooled.shape)} "
                        f"expected latent {latent_res[level]}"
                    )
                energies[start : start + pooled.shape[0], level] = energies_physical(pooled, waves[level])
            done = start + len(chunk)
            if done == stems.size or done % (batch * 32) == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"latent reference {done}/{stems.size}  {rate:.1f} files/s", flush=True)
    mu = energies.mean(axis=0)
    sigma = energies.std(axis=0, ddof=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_name(save_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        energies=energies,
        stems=stems,
        mu_E=mu,
        sigma_E=sigma,
        latent_res=np.asarray(latent_res, dtype=np.int64),
        ds_factors=np.asarray(ds_factors, dtype=np.int64),
        img_size=np.asarray(IMG_SIZE, dtype=np.int64),
        stem_min=np.asarray(REF_STEM_MIN, dtype=np.int64),
        stem_max=np.asarray(REF_STEM_MAX, dtype=np.int64),
        downsample=np.asarray(
            "F.avg_pool2d kernel_size=stride=img_size/latent_res (FutureINNPlus._pool_seq)"
        ),
        physical_vorticity=np.asarray(True),
    )
    os.replace(tmp, save_path)
    print(
        f"wrote {save_path} n={energies.shape[0]} levels={n_levels} "
        f"mu={mu.tolist()} sigma={sigma.tolist()}",
        flush=True,
    )
    return energies, stems, mu, sigma


def assert_latent_is_vorticity(model, ema, high: torch.Tensor) -> None:
    """One warmup step. Stop if the coarse tensor is not a spatial vorticity grid."""
    b = high.shape[0]
    k = model.num_coarse
    coarses = [
        torch.zeros(
            b, 1, model.latent_res[i], model.latent_res[i], device=high.device, dtype=high.dtype
        )
        for i in range(k)
    ]
    tt = torch.zeros(b, device=high.device, dtype=high.dtype)
    _, outs = ema(high, coarses, tt)
    out = outs[0]
    expected = (b, 1, model.latent_res[0], model.latent_res[0])
    if tuple(out.shape) != expected:
        raise RuntimeError(
            "Latent kinetic energy is not defined. "
            f"The L=2 coarse head returned shape {tuple(out.shape)}, not a gridded "
            f"vorticity of shape {expected}. No energy was invented and the reference "
            "scan was not started."
        )
    print(
        f"coarse head is gridded vorticity shape={tuple(out.shape)} "
        f"latent_res={list(model.latent_res)} ds_factors={list(model.ds_factors)}",
        flush=True,
    )


@torch.no_grad()
def rollout_aligned(
    ema,
    model,
    high: torch.Tensor,
    n_steps: int,
    norm: float,
    waves_high,
    waves_lat: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time-aligned high and latent KE, plus the latent emitted with each high.

    ar_high[:, n] and ar_latent[:, n] are predictions of the same physical lead
    n+1. ar_latent_same_forward[:, n] is the coarse head from the forward that
    produced that high field; the model trains that head against lead n+2+i.
    """
    b = high.shape[0]
    k = model.num_coarse
    ar_high = np.full((b, n_steps), np.nan, dtype=np.float64)
    ar_lat = np.full((b, n_steps, k), np.nan, dtype=np.float64)
    ar_lat_same = np.full((b, n_steps, k), np.nan, dtype=np.float64)
    coarses = [
        torch.zeros(
            b, 1, model.latent_res[i], model.latent_res[i], device=high.device, dtype=high.dtype
        )
        for i in range(k)
    ]
    for c in range(k):
        tt = float(c) * torch.ones(b, device=high.device, dtype=high.dtype)
        _, outs = ema(high, coarses, tt)
        coarses[c] = outs[c]
        if c < n_steps:
            ar_lat[:, c, c] = energies_physical(coarses[c] * float(norm), waves_lat[c])
    t_step = float(k) * torch.ones(b, device=high.device, dtype=high.dtype)
    for step in range(n_steps):
        high, coarses = ema(high, coarses, t_step)
        ar_high[:, step] = energies_physical(high * float(norm), waves_high)
        for i in range(k):
            emitted = energies_physical(coarses[i] * float(norm), waves_lat[i])
            ar_lat_same[:, step, i] = emitted
            dest = step + 1 + i
            if dest < n_steps:
                ar_lat[:, dest, i] = emitted
    return ar_high, ar_lat, ar_lat_same


def lookup(energies: np.ndarray, stems: np.ndarray, query: np.ndarray) -> np.ndarray:
    if int(stems[0]) != REF_STEM_MIN or int(np.diff(stems).min()) != 1:
        raise ValueError("reference stems are not a unit grid starting at 10000")
    pos = query.astype(np.int64) - int(stems[0])
    if np.any(pos < 0) or np.any(pos >= stems.size):
        raise ValueError("a target stem falls outside the reference span")
    return energies[pos]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def plot_comparison(
    path: Path,
    lead: np.ndarray,
    ar_high: np.ndarray,
    ar_lat: np.ndarray,
    gt_high: np.ndarray,
    gt_down: np.ndarray,
    ic_stems: np.ndarray,
    latent_res: list[int],
    ds_factors: list[int],
    high_mu: float,
    high_sigma: float,
    lat_mu: np.ndarray,
    lat_sigma: np.ndarray,
    dropped: list[int],
) -> None:
    n_levels = ar_lat.shape[-1]
    fig, axes = plt.subplots(
        1 + n_levels, 1, figsize=(10, 3.6 * (1 + n_levels)), sharex=True, constrained_layout=True
    )
    if n_levels == 0:
        raise RuntimeError("no latent level to plot")
    dropped_txt = (
        f"Dropped IC stem {dropped[0]} (last eval frame, no future steps)."
        if dropped
        else ""
    )
    fig.suptitle(
        f"Wavenumber-5 L=2 kinetic energy, 100 autoregressive steps\n"
        f"{ar_high.shape[0]} eval initial conditions. {dropped_txt}"
    )
    _draw_panel(
        axes[0],
        lead,
        ar_high,
        gt_high,
        high_mu,
        high_sigma,
        f"High resolution {IMG_SIZE}²",
        "high-res reference",
    )
    for level in range(n_levels):
        _draw_panel(
            axes[1 + level],
            lead,
            ar_lat[:, :, level],
            gt_down[:, :, level],
            float(lat_mu[level]),
            float(lat_sigma[level]),
            f"Latent {latent_res[level]}², average-pool stride {ds_factors[level]}",
            "latent reference",
        )
    axes[-1].set_xlabel("lead (autoregressive step, one step = 0.05)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path} ics={ic_stems.tolist()}", flush=True)


def _draw_panel(ax, lead, ar, gt, mu, sigma, title, band_name) -> None:
    lo, hi = mu - SIGMA_MULT * sigma, mu + SIGMA_MULT * sigma
    for i in range(ar.shape[0]):
        ax.plot(lead, ar[i], color="C0", alpha=0.35, lw=0.9, label="AR initial conditions" if i == 0 else None)
    ax.plot(lead, ar.mean(axis=0), color="C0", lw=2.2, label="AR mean")
    ax.plot(lead, gt.mean(axis=0), color="C1", lw=2.0, label="true mean")
    ax.axhline(mu, color="k", lw=1.1, label=rf"{band_name} $\mu$")
    ax.axhline(lo, color="k", ls="--", lw=1.0, label=rf"$\mu \pm {SIGMA_MULT:g}\sigma$")
    ax.axhline(hi, color="k", ls="--", lw=1.0)
    finite = np.concatenate([ar[np.isfinite(ar)], gt[np.isfinite(gt)], np.array([lo, hi])])
    pad = 0.08 * max(float(finite.max() - finite.min()), 1e-3)
    ax.set_ylim(float(finite.min()) - pad, float(finite.max()) + pad)
    ax.set_ylabel("kinetic energy $E$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def in_band(values: np.ndarray, mu: float, sigma: float) -> float:
    ok = np.isfinite(values) & (np.abs(values - mu) <= SIGMA_MULT * sigma)
    return float(ok.mean())


def main() -> None:
    args = parse_args()
    if args.n_steps < 1:
        raise ValueError(f"n-steps must be >= 1, got {args.n_steps}")
    save_dir = args.save_dir
    if "ke_100k" in save_dir.parts:
        raise RuntimeError(f"refusing to write into a ke_100k directory: {save_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latent reference and the rollout")
    device = torch.device("cuda")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"host={os.uname().nodename} device={device} save_dir={save_dir} "
        f"job={os.environ.get('SLURM_JOB_ID', 'none')}",
        flush=True,
    )

    high_energies, high_stems, high_mu, high_sigma = load_high_reference(args.high_ref)
    print(
        f"high-res reference mu_E={high_mu:.8g} sigma_E={high_sigma:.8g} n={high_energies.size}",
        flush=True,
    )

    model, ema = load_l2(args.checkpoint, device)
    latent_res = [int(r) for r in model.latent_res]
    ds_factors = [int(d) for d in model.ds_factors]
    if len(latent_res) != len(ds_factors) or any(IMG_SIZE // r != d for r, d in zip(latent_res, ds_factors)):
        raise RuntimeError(f"latent_res {latent_res} does not match ds_factors {ds_factors}")
    print(
        f"checkpoint={args.checkpoint} num_coarse={model.num_coarse} "
        f"latent_res={latent_res} ds_factors={ds_factors} norm={args.norm}",
        flush=True,
    )

    ic_index, ic_stems, ic_paths, dropped = select_ics(
        args.eval_dir, args.every_nth_field, args.n_ics, args.n_steps
    )
    print(
        f"using {ic_stems.tolist()} dropped={dropped} "
        f"leads={args.n_steps} every={args.every_nth_field}",
        flush=True,
    )

    probe = torch.from_numpy(load_omega(args.data_root, int(ic_stems[0]))).to(device)
    probe = probe.div(float(args.norm)).view(1, 1, IMG_SIZE, IMG_SIZE)
    assert_latent_is_vorticity(model, ema, probe)
    del probe

    waves_high = make_wavenumber_tensors(IMG_SIZE, IMG_SIZE, device)[:3]
    check = torch.from_numpy(load_omega(args.data_root, REF_STEM_MIN)).to(device)
    check_e = float(energies_physical(check.view(1, IMG_SIZE, IMG_SIZE), waves_high)[0])
    rel = abs(check_e - float(high_energies[0])) / max(abs(float(high_energies[0])), 1e-12)
    print(
        f"stem {REF_STEM_MIN} high KE recomputed={check_e:.8g} "
        f"reference={float(high_energies[0]):.8g} rel={rel:.3e}",
        flush=True,
    )
    if rel > 1e-4:
        raise RuntimeError(f"high-res KE does not match the k5 reference (rel={rel})")

    lat_energies, lat_stems, lat_mu, lat_sigma = build_latent_reference(
        args.data_root,
        save_dir / "latent_reference_kinetic_energy.npz",
        latent_res,
        ds_factors,
        device,
        args.ref_batch,
        args.load_workers,
    )
    for level, (res, ds) in enumerate(zip(latent_res, ds_factors)):
        print(
            f"latent level {level} res={res} ds={ds} "
            f"mu_E={float(lat_mu[level]):.8g} sigma_E={float(lat_sigma[level]):.8g}",
            flush=True,
        )

    high0 = torch.cat(
        [
            torch.from_numpy(load_omega(args.data_root, int(stem)))
            .div(float(args.norm))
            .view(1, 1, IMG_SIZE, IMG_SIZE)
            for stem in ic_stems
        ],
        dim=0,
    ).to(device)
    waves_lat = [make_wavenumber_tensors(r, r, device)[:3] for r in latent_res]
    ar_high, ar_lat, ar_lat_same = rollout_aligned(
        ema, model, high0, args.n_steps, args.norm, waves_high, waves_lat
    )
    if not np.isfinite(ar_high).all() or not np.isfinite(ar_lat).all():
        raise RuntimeError(
            f"rollout left non-finite energies high_nan={int((~np.isfinite(ar_high)).sum())} "
            f"lat_nan={int((~np.isfinite(ar_lat)).sum())}"
        )

    lead = np.arange(1, args.n_steps + 1, dtype=np.int64)
    target_stem = ic_stems[:, None] + args.every_nth_field * lead[None, :]
    gt_high = lookup(high_energies, high_stems, target_stem)
    gt_down = lookup(lat_energies, lat_stems, target_stem)

    out_npz = save_dir / "kinetic_energy_high_vs_latent.npz"
    np.savez_compressed(
        out_npz,
        ar_high_E=ar_high,
        ar_latent_E=ar_lat,
        ar_latent_same_forward_E=ar_lat_same,
        gt_high_E=gt_high,
        gt_downsampled_E=gt_down,
        ic_stem=ic_stems,
        ic_index=ic_index,
        lead=lead,
        target_stem=target_stem,
        dropped_ic_stem=np.asarray(dropped, dtype=np.int64),
        latent_res=np.asarray(latent_res, dtype=np.int64),
        ds_factors=np.asarray(ds_factors, dtype=np.int64),
        norm=np.asarray(args.norm),
        every_nth_field=np.asarray(args.every_nth_field, dtype=np.int64),
        checkpoint=np.asarray(args.checkpoint),
        alignment=np.asarray(ALIGNMENT),
        high_mu_E=np.asarray(high_mu),
        high_sigma_E=np.asarray(high_sigma),
        latent_mu_E=lat_mu,
        latent_sigma_E=lat_sigma,
        sigma_mult=np.asarray(SIGMA_MULT),
    )
    print(f"wrote {out_npz}", flush=True)

    plot_comparison(
        save_dir / "high_vs_latent_ke.png",
        lead,
        ar_high,
        ar_lat,
        gt_high,
        gt_down,
        ic_stems,
        latent_res,
        ds_factors,
        high_mu,
        high_sigma,
        lat_mu,
        lat_sigma,
        dropped,
    )

    print("--- summary ---", flush=True)
    print(f"high-res reference mu_E={high_mu:.8g} sigma_E={high_sigma:.8g}", flush=True)
    for level, res in enumerate(latent_res):
        mean_ar_h = float(ar_high.mean())
        mean_ar_l = float(ar_lat[:, :, level].mean())
        mean_gt_h = float(gt_high.mean())
        mean_gt_l = float(gt_down[:, :, level].mean())
        curve_ar_h = ar_high.mean(axis=0)
        curve_ar_l = ar_lat[:, :, level].mean(axis=0)
        curve_gt_h = gt_high.mean(axis=0)
        curve_gt_l = gt_down[:, :, level].mean(axis=0)
        per_ic = [
            pearson(ar_high[i], ar_lat[i, :, level]) for i in range(ar_high.shape[0])
        ]
        print(
            f"level {level} res={res} latent_mu={float(lat_mu[level]):.8g} "
            f"latent_sigma={float(lat_sigma[level]):.8g}",
            flush=True,
        )
        print(
            f"means over ICs and 100 steps: ar_high={mean_ar_h:.6g} ar_latent={mean_ar_l:.6g} "
            f"gt_high={mean_gt_h:.6g} gt_down={mean_gt_l:.6g}",
            flush=True,
        )
        print(
            "correlation of IC-mean curves across steps: "
            f"ar_high_vs_ar_latent={pearson(curve_ar_h, curve_ar_l):.4f} "
            f"ar_high_vs_gt_high={pearson(curve_ar_h, curve_gt_h):.4f} "
            f"ar_latent_vs_gt_down={pearson(curve_ar_l, curve_gt_l):.4f} "
            f"gt_high_vs_gt_down={pearson(curve_gt_h, curve_gt_l):.4f} "
            f"mean_per_ic_ar_high_vs_ar_latent={float(np.mean(per_ic)):.4f}",
            flush=True,
        )
        frac_ar = in_band(ar_lat[:, :, level], float(lat_mu[level]), float(lat_sigma[level]))
        frac_gt = in_band(gt_down[:, :, level], float(lat_mu[level]), float(lat_sigma[level]))
        z_ar = np.abs(ar_lat[:, :, level] - float(lat_mu[level])) / float(lat_sigma[level])
        print(
            f"latent band |E-mu|<={SIGMA_MULT:g} sigma: "
            f"AR fraction={frac_ar:.4f} GT-down fraction={frac_gt:.4f} "
            f"AR mean |z|={float(z_ar.mean()):.3f} AR max |z|={float(z_ar.max()):.3f}",
            flush=True,
        )
    print("KE_LATENT_L2_DONE", flush=True)


if __name__ == "__main__":
    main()
