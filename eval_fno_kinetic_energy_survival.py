#!/usr/bin/env python3
"""Kinetic-energy survival from the FNO rollout_mse.npz.

Same energy formula and 5-sigma survival rule as eval_kinetic_energy_survival.py.
Reads the shared HINE reference cache; writes outputs next to the FNO rollout.
Leaves the original kinetic-energy script and notebook untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from util.py2d import Omega2Psi, Psi2UV, initialize_wavenumbers_rfft2

ROLLOUT_NPZ = Path(
    "/pscratch/sd/d/divyam/hine_runs/fno_baseline_debug/jacobian_manual_mse/rollout_mse.npz"
)
SAVE_DIR = ROLLOUT_NPZ.parent
DATA_ROOT = Path("/global/homes/d/divyam/TACSLAB")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"
EVAL_DIR = DATA_ROOT / "eval"
REF_STEM_MIN = 10000
REF_STEM_MAX = 105000
# Shared with the HINE run; do not write next to HINE outputs.
REF_CACHE = Path(
    "/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_manual_mse/"
    "reference_kinetic_energy.npz"
)
OUT_NPZ = SAVE_DIR / "kinetic_energy_survival.npz"
LX = LY = 2.0 * np.pi
IMG_SIZE = 256
SIGMA_MULT = 5.0
REF_BATCH = 64
PRED_BATCH = 32


def make_wavenumber_tensors(nx: int, ny: int, device: torch.device, dtype=torch.float32):
    Kx_np, Ky_np, _, _, invKsq_np = initialize_wavenumbers_rfft2(
        nx, ny, LX, LY, INDEXING="ij"
    )
    Kx = torch.from_numpy(np.ascontiguousarray(Kx_np)).to(device=device, dtype=dtype)
    Ky = torch.from_numpy(np.ascontiguousarray(Ky_np)).to(device=device, dtype=dtype)
    invKsq = torch.from_numpy(np.ascontiguousarray(invKsq_np)).to(
        device=device, dtype=dtype
    )
    return Kx, Ky, invKsq, Kx_np, Ky_np, invKsq_np


def kinetic_energy_torch(
    omega: torch.Tensor,
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    invKsq: torch.Tensor,
) -> torch.Tensor:
    H, W = omega.shape[-2:]
    omega_hat = torch.fft.rfft2(omega)
    psi_hat = omega_hat * invKsq
    u = torch.fft.irfft2(1j * Ky * psi_hat, s=(H, W))
    v = torch.fft.irfft2(-1j * Kx * psi_hat, s=(H, W))
    return 0.5 * (u.square() + v.square()).mean(dim=(-2, -1))


def kinetic_energy_numpy(omega: np.ndarray, Kx_np, Ky_np, invKsq_np) -> float:
    psi = Omega2Psi(omega, invKsq_np)
    u, v = Psi2UV(psi, Kx_np, Ky_np)
    return float(0.5 * np.mean(u * u + v * v))


def enstrophy_half_mean_sq(omega: np.ndarray) -> float:
    return float(0.5 * np.mean(np.asarray(omega, dtype=np.float64) ** 2))


def mat_path_for_stem(stem: int) -> Path:
    if stem <= 91015:
        return TRAIN_DIR / f"{stem}.mat"
    if stem <= 91620:
        return VAL_DIR / f"{stem}.mat"
    return EVAL_DIR / f"{stem}.mat"


def load_or_build_reference(
    Kx_t, Ky_t, invKsq_t, device: torch.device
) -> tuple[np.ndarray, float, float]:
    if REF_CACHE.is_file():
        cache = np.load(REF_CACHE)
        energies = np.asarray(cache["energies"], dtype=np.float64)
        stems = np.asarray(cache["stems"], dtype=np.int64)
        mu = float(cache["mu_E"])
        sigma = float(cache["sigma_E"])
        print(f"loaded cache {REF_CACHE}  n={energies.size}  mu={mu:.6g} sigma={sigma:.6g}")
        expected = np.arange(REF_STEM_MIN, REF_STEM_MAX + 1, dtype=np.int64)
        assert stems.shape == expected.shape and np.array_equal(stems, expected)
        return energies, mu, sigma

    # Fallback: build once under the FNO save dir without touching the HINE path.
    local_cache = SAVE_DIR / "reference_kinetic_energy.npz"
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
            if len(buf) == REF_BATCH or i == stems.size - 1:
                batch = torch.from_numpy(np.stack(buf, axis=0)).to(device)
                e = kinetic_energy_torch(batch, Kx_t, Ky_t, invKsq_t).double().cpu().numpy()
                energies[idx0 : idx0 + e.size] = e
                idx0 += e.size
                buf.clear()

    mu = float(energies.mean())
    sigma = float(energies.std(ddof=0))
    np.savez_compressed(
        local_cache,
        energies=energies,
        stems=stems,
        mu_E=np.asarray(mu),
        sigma_E=np.asarray(sigma),
        stem_min=np.asarray(REF_STEM_MIN, dtype=np.int64),
        stem_max=np.asarray(REF_STEM_MAX, dtype=np.int64),
    )
    print(f"wrote {local_cache}  n={energies.size}  mu={mu:.6g} sigma={sigma:.6g}")
    return energies, mu, sigma


def energies_from_high_field(
    arr: np.ndarray, scale: float, Kx_t, Ky_t, invKsq_t, device: torch.device
) -> np.ndarray:
    b, t = arr.shape[:2]
    flat = arr.reshape(b * t, arr.shape[-2], arr.shape[-1])
    out = np.empty(b * t, dtype=np.float64)
    with torch.no_grad():
        for start in tqdm(range(0, flat.shape[0], PRED_BATCH), desc="rollout KE"):
            chunk = flat[start : start + PRED_BATCH].astype(np.float32, copy=False)
            omega = torch.from_numpy(chunk).to(device) * scale
            e = kinetic_energy_torch(omega, Kx_t, Ky_t, invKsq_t).double().cpu().numpy()
            out[start : start + e.size] = e
    return out.reshape(b, t)


def survival_curve(
    energies: np.ndarray, mu: float, sigma: float, k: float = SIGMA_MULT
) -> np.ndarray:
    ok = np.isfinite(energies) & (np.abs(energies - mu) <= k * sigma)
    survived = np.cumprod(ok.astype(np.float64), axis=1)
    s = survived.mean(axis=0)
    assert np.all(s[1:] <= s[:-1] + 1e-15), "stability rate must be non-increasing"
    return s


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"host={os.uname().nodename} device={device} save_dir={SAVE_DIR}")
    assert ROLLOUT_NPZ.is_file(), f"missing {ROLLOUT_NPZ}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    meta = np.load(ROLLOUT_NPZ, allow_pickle=True)
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
    assert norm == 10.0, f"expected norm=10, got {norm}"
    assert every_nth_field == 5, f"expected every_nth_field=5, got {every_nth_field}"
    assert ar_shape == gt_shape
    assert ar_shape[1] == lead.shape[0] == n_leads
    assert list(lead) == list(range(1, n_leads + 1))
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
    assert omega_phys.shape == (IMG_SIZE, IMG_SIZE)
    print(
        f"{probe_path.name}: mean={omega_phys.mean():.4g} std={omega_phys.std():.4g} "
        f"std/norm={omega_phys.std() / norm:.4g}"
    )
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
    assert rel < 1e-5, f"torch vs numpy KE mismatch: {rel}"
    ens = enstrophy_half_mean_sq(omega_phys)
    print(f"enstrophy (1/2 mean omega^2)={ens:.8g}  KE={e_np:.8g}")
    assert abs(ens - e_np) / max(abs(e_np), abs(ens), 1e-12) > 1e-3

    with np.load(ROLLOUT_NPZ) as z:
        sample_norm = np.asarray(z["ar_pred_high"][0, 0, 0], dtype=np.float64)
    sample_phys = sample_norm * norm
    e_np_s = kinetic_energy_numpy(sample_phys, Kx_np, Ky_np, invKsq_np)
    with torch.no_grad():
        e_torch_s = float(
            kinetic_energy_torch(
                torch.from_numpy(sample_phys.astype(np.float32)).to(device),
                Kx_t,
                Ky_t,
                invKsq_t,
            ).cpu()
        )
    rel_s = abs(e_torch_s - e_np_s) / max(abs(e_np_s), 1e-12)
    print(
        f"saved ar_pred_high[0,0]*norm KE: numpy={e_np_s:.8g} torch={e_torch_s:.8g} "
        f"rel_err={rel_s:.3e}"
    )
    assert rel_s < 1e-5
    print("cross-check OK")

    ref_energies, mu_E, sigma_E = load_or_build_reference(Kx_t, Ky_t, invKsq_t, device)
    lo = mu_E - SIGMA_MULT * sigma_E
    hi = mu_E + SIGMA_MULT * sigma_E
    z_max = float((ref_energies.max() - mu_E) / sigma_E)
    z_min = float((ref_energies.min() - mu_E) / sigma_E)
    outside_4 = int(np.sum(np.abs(ref_energies - mu_E) > 4.0 * sigma_E))
    outside_5 = int(np.sum(np.abs(ref_energies - mu_E) > 5.0 * sigma_E))
    print(
        f"ref n={ref_energies.size}  mu_E={mu_E:.8g}  sigma_E={sigma_E:.8g}\n"
        f"band [{lo:.8g}, {hi:.8g}]\n"
        f"(max-mu)/sigma={z_max:.4g}  (min-mu)/sigma={z_min:.4g}\n"
        f"outside 4 sigma: {outside_4}  outside 5 sigma: {outside_5}"
    )

    with np.load(ROLLOUT_NPZ) as z:
        ar_pred_high = np.asarray(z["ar_pred_high"])
        gt_high = np.asarray(z["gt_high"])
    ar_E = energies_from_high_field(ar_pred_high, norm, Kx_t, Ky_t, invKsq_t, device)
    gt_E = energies_from_high_field(gt_high, norm, Kx_t, Ky_t, invKsq_t, device)
    del ar_pred_high, gt_high
    print(
        f"AR KE: mean={ar_E.mean():.6g} min={ar_E.min():.6g} max={ar_E.max():.6g} "
        f"finite={np.isfinite(ar_E).mean():.4%}"
    )
    print(
        f"GT KE: mean={gt_E.mean():.6g} min={gt_E.min():.6g} max={gt_E.max():.6g} "
        f"finite={np.isfinite(gt_E).mean():.4%}"
    )

    S_ar = survival_curve(ar_E, mu_E, sigma_E)
    S_gt = survival_curve(gt_E, mu_E, sigma_E)
    print(f"AR S(1)={S_ar[0]:.4%}  S(final)={S_ar[-1]:.4%}")
    print(f"GT S(1)={S_gt[0]:.4%}  S(final)={S_gt[-1]:.4%}")
    print(f"AR trials still alive at end: {int(round(S_ar[-1] * n_trials))} / {n_trials}")

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i in range(n_trials):
        ax.plot(lead, ar_E[i], color="C0", alpha=0.15, lw=0.8)
        ax.plot(lead, gt_E[i], color="C1", alpha=0.10, lw=0.7)
    ax.plot(lead, ar_E.mean(axis=0), color="C0", lw=2.5, label="AR mean")
    ax.plot(lead, gt_E.mean(axis=0), color="C1", lw=2.5, label="GT mean (every 5th)")
    ax.axhline(mu_E, color="k", ls="-", lw=1.2, label=r"$\mu_E$")
    ax.axhline(lo, color="k", ls="--", lw=1.0, label=fr"$\mu_E \pm {SIGMA_MULT:g}\sigma_E$")
    ax.axhline(hi, color="k", ls="--", lw=1.0)
    ax.set_xlabel("lead (every-5th-frame autoregressive step)")
    ax.set_ylabel(r"kinetic energy $E$")
    ax.set_title("FNO kinetic energy vs lead (saved rollout)")
    ax.legend(loc="best")
    ke_fig = SAVE_DIR / "kinetic_energy_vs_lead.png"
    fig.savefig(ke_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {ke_fig}")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(lead, S_ar, color="C0", lw=2.5, marker="o", markersize=4, label="AR")
    ax.plot(lead, S_gt, color="C1", lw=2.0, ls="--", marker="s", markersize=3, label="GT (sanity)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("lead")
    ax.set_ylabel(r"stability rate $S(n)$")
    ax.set_title(
        fr"FNO stability rate ($|E-\mu_E|\leq {SIGMA_MULT:g}\sigma_E$; once failed stays failed)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    s_fig = SAVE_DIR / "stability_rate_vs_lead.png"
    fig.savefig(s_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {s_fig}")

    np.savez_compressed(
        OUT_NPZ,
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
        ref_stem_min=np.asarray(REF_STEM_MIN, dtype=np.int64),
        ref_stem_max=np.asarray(REF_STEM_MAX, dtype=np.int64),
        ref_n=np.asarray(ref_energies.size, dtype=np.int64),
        ref_outside_4sigma=np.asarray(outside_4, dtype=np.int64),
        ref_outside_5sigma=np.asarray(outside_5, dtype=np.int64),
        rollout_npz=np.asarray(str(ROLLOUT_NPZ)),
        chosen=chosen,
        file_stem=file_stem,
        target_stem=target_stem,
        model_type=np.asarray("fno"),
    )
    print(f"wrote {OUT_NPZ}")
    print("FNO_KINETIC_ENERGY_SURVIVAL_DONE")


if __name__ == "__main__":
    main()
