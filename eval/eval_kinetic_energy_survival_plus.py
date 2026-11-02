#!/usr/bin/env python3
"""Kinetic-energy survival from a FutureINNPlus rollout_mse.npz.

Does not modify eval_kinetic_energy_survival.py. Reads ar_pred_high / gt_high
from --rollout-npz produced by eval_jacobian_manual_mse_plus.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from eval_kinetic_energy_survival import (
    EVAL_DIR,
    IMG_SIZE,
    PRED_BATCH,
    REF_STEM_MAX,
    REF_STEM_MIN,
    SIGMA_MULT,
    energies_from_high_field,
    enstrophy_half_mean_sq,
    kinetic_energy_numpy,
    kinetic_energy_torch,
    make_wavenumber_tensors,
    mat_path_for_stem,
    survival_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rollout-npz",
        type=Path,
        required=True,
        help="rollout_mse.npz from eval_jacobian_manual_mse_plus.py",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Defaults to the parent of --rollout-npz",
    )
    p.add_argument(
        "--ref-cache",
        type=Path,
        default=None,
        help="Defaults to <save-dir>/reference_kinetic_energy.npz",
    )
    return p.parse_args()


def load_or_build_reference(ref_cache: Path, Kx_t, Ky_t, invKsq_t, device):
    if ref_cache.is_file():
        cache = np.load(ref_cache)
        energies = np.asarray(cache["energies"], dtype=np.float64)
        stems = np.asarray(cache["stems"], dtype=np.int64)
        mu = float(cache["mu_E"])
        sigma = float(cache["sigma_E"])
        print(f"loaded cache {ref_cache}  n={energies.size}  mu={mu:.6g} sigma={sigma:.6g}")
        expected = np.arange(REF_STEM_MIN, REF_STEM_MAX + 1, dtype=np.int64)
        assert stems.shape == expected.shape and np.array_equal(stems, expected)
        return energies, mu, sigma

    stems = np.arange(REF_STEM_MIN, REF_STEM_MAX + 1, dtype=np.int64)
    energies = np.empty(stems.size, dtype=np.float64)
    buf: list[np.ndarray] = []
    idx0 = 0
    with torch.no_grad():
        for i, stem in enumerate(tqdm(stems, desc="reference KE")):
            path = mat_path_for_stem(int(stem))
            omega = np.asarray(scipy.io.loadmat(path)["Omega"], dtype=np.float32)
            if omega.shape != (IMG_SIZE, IMG_SIZE):
                raise ValueError(f"{path}: expected {(IMG_SIZE, IMG_SIZE)}, got {omega.shape}")
            buf.append(omega)
            if len(buf) == 64 or i == stems.size - 1:
                batch = torch.from_numpy(np.stack(buf, axis=0)).to(device)
                e = kinetic_energy_torch(batch, Kx_t, Ky_t, invKsq_t).double().cpu().numpy()
                energies[idx0 : idx0 + e.size] = e
                idx0 += e.size
                buf.clear()

    mu = float(energies.mean())
    sigma = float(energies.std(ddof=0))
    os.makedirs(ref_cache.parent, exist_ok=True)
    np.savez_compressed(
        ref_cache,
        energies=energies,
        stems=stems,
        mu_E=np.asarray(mu),
        sigma_E=np.asarray(sigma),
    )
    print(f"wrote {ref_cache}  n={energies.size}  mu={mu:.6g} sigma={sigma:.6g}")
    return energies, mu, sigma


def main() -> None:
    args = parse_args()
    rollout_npz = args.rollout_npz
    save_dir = args.save_dir if args.save_dir is not None else rollout_npz.parent
    ref_cache = (
        args.ref_cache
        if args.ref_cache is not None
        else save_dir / "reference_kinetic_energy.npz"
    )
    out_npz = save_dir / "kinetic_energy_survival.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"host={os.uname().nodename} device={device} save_dir={save_dir}")
    assert rollout_npz.is_file(), f"missing {rollout_npz}"
    os.makedirs(save_dir, exist_ok=True)

    meta = np.load(rollout_npz, allow_pickle=True)
    lead = np.asarray(meta["lead"], dtype=np.int64)
    norm = float(np.asarray(meta["norm"]))
    every_nth_field = int(np.asarray(meta["every_nth_field"]))
    horizon = int(np.asarray(meta["horizon"]))
    file_stem = np.asarray(meta["file_stem"])
    target_stem = np.asarray(meta["target_stem"])
    chosen = np.asarray(meta["chosen"])
    ar_shape = meta["ar_pred_high"].shape
    gt_shape = meta["gt_high"].shape
    n_trials, n_leads = ar_shape[0], ar_shape[1]
    assert ar_shape == gt_shape
    assert ar_shape[1] == lead.shape[0] == n_leads
    print(
        f"ar_pred_high={ar_shape} trials={n_trials} leads={n_leads} "
        f"norm={norm} every_nth={every_nth_field} horizon={horizon}"
    )
    meta.close()

    Kx_t, Ky_t, invKsq_t, Kx_np, Ky_np, invKsq_np = make_wavenumber_tensors(
        IMG_SIZE, IMG_SIZE, device
    )

    probe_path = EVAL_DIR / "91621.mat"
    omega_phys = np.asarray(scipy.io.loadmat(probe_path)["Omega"], dtype=np.float64)
    e_np = kinetic_energy_numpy(omega_phys, Kx_np, Ky_np, invKsq_np)
    with torch.no_grad():
        e_torch = float(
            kinetic_energy_torch(
                torch.from_numpy(omega_phys.astype(np.float32)).to(device),
                Kx_t,
                Ky_t,
                invKsq_t,
            ).cpu()
        )
    rel = abs(e_torch - e_np) / max(abs(e_np), 1e-12)
    print(f"physical field KE: numpy={e_np:.8g} torch={e_torch:.8g} rel_err={rel:.3e}")
    assert rel < 1e-5

    ref_energies, mu_E, sigma_E = load_or_build_reference(
        ref_cache, Kx_t, Ky_t, invKsq_t, device
    )
    lo = mu_E - SIGMA_MULT * sigma_E
    hi = mu_E + SIGMA_MULT * sigma_E

    with np.load(rollout_npz) as z:
        ar_pred_high = np.asarray(z["ar_pred_high"])
        gt_high = np.asarray(z["gt_high"])
    ar_E = energies_from_high_field(ar_pred_high, norm, Kx_t, Ky_t, invKsq_t, device)
    gt_E = energies_from_high_field(gt_high, norm, Kx_t, Ky_t, invKsq_t, device)
    del ar_pred_high, gt_high

    S_ar = survival_curve(ar_E, mu_E, sigma_E)
    S_gt = survival_curve(gt_E, mu_E, sigma_E)
    print(f"AR S(1)={S_ar[0]:.4%}  S(final)={S_ar[-1]:.4%}")
    print(f"GT S(1)={S_gt[0]:.4%}  S(final)={S_gt[-1]:.4%}")

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i in range(n_trials):
        ax.plot(lead, ar_E[i], color="C0", alpha=0.15, lw=0.8)
        ax.plot(lead, gt_E[i], color="C1", alpha=0.10, lw=0.7)
    ax.plot(lead, ar_E.mean(axis=0), color="C0", lw=2.5, label="AR mean")
    ax.plot(lead, gt_E.mean(axis=0), color="C1", lw=2.5, label="GT mean")
    ax.axhline(mu_E, color="k", ls="-", lw=1.2, label=r"$\mu_E$")
    ax.axhline(lo, color="k", ls="--", lw=1.0, label=fr"$\mu_E \pm {SIGMA_MULT:g}\sigma_E$")
    ax.axhline(hi, color="k", ls="--", lw=1.0)
    ax.set_xlabel("lead")
    ax.set_ylabel(r"kinetic energy $E$")
    ax.set_title("Plus kinetic energy vs lead")
    ax.legend(loc="best")
    ke_fig = save_dir / "kinetic_energy_vs_lead.png"
    fig.savefig(ke_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {ke_fig}")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(lead, S_ar, color="C0", lw=2.5, marker="o", markersize=4, label="AR")
    ax.plot(lead, S_gt, color="C1", lw=2.0, ls="--", marker="s", markersize=3, label="GT")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("lead")
    ax.set_ylabel(r"stability rate $S(n)$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    s_fig = save_dir / "stability_rate_vs_lead.png"
    fig.savefig(s_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {s_fig}")

    np.savez_compressed(
        out_npz,
        lead=lead,
        ar_E=ar_E,
        gt_E=gt_E,
        S_ar=S_ar,
        S_gt=S_gt,
        mu_E=np.asarray(mu_E),
        sigma_E=np.asarray(sigma_E),
        band_lo=np.asarray(lo),
        band_hi=np.asarray(hi),
        norm=np.asarray(norm),
        every_nth_field=np.asarray(every_nth_field, dtype=np.int64),
        sigma_mult=np.asarray(SIGMA_MULT),
        rollout_npz=np.asarray(str(rollout_npz)),
        chosen=chosen,
        file_stem=file_stem,
        target_stem=target_stem,
    )
    print(f"wrote {out_npz}")
    print("KINETIC_ENERGY_SURVIVAL_PLUS_DONE")


if __name__ == "__main__":
    main()
