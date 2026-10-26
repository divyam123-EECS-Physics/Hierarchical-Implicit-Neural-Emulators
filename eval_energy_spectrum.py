#!/usr/bin/env python3
"""Average kinetic-energy spectrum E(k) over 200 consecutive eval frames (Figure 5(c) reference)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm

from util.py2d import Omega2Psi, Psi2UV, initialize_wavenumbers_rfft2

REPO_DIR = Path(__file__).resolve().parent
EVAL_DIR = Path("/global/homes/d/divyam/TACSLAB/eval")
OUT_NPZ = REPO_DIR / "energy_spectrum_eval200.npz"
OUT_PNG = REPO_DIR / "energy_spectrum_eval200.png"

N = 256
LX = LY = 2.0 * np.pi
STEM_START = 91621  # first eval stem
STRIDE = 1  # no skipped frames
N_STEPS = 200


def kinetic_spectrum(
    omega: np.ndarray,
    invKsq: np.ndarray,
    Ky: np.ndarray,
    Kx: np.ndarray,
    weight: np.ndarray,
    k_bin: np.ndarray,
) -> np.ndarray:
    """Shell-averaged kinetic energy including k=0 and corner shells (|k|>N/2)."""
    omega = np.asarray(omega, dtype=np.float64)
    omega_hat = np.fft.rfft2(omega)
    psi_hat = omega_hat * invKsq
    u_hat = 1j * Ky * psi_hat
    v_hat = -1j * Kx * psi_hat
    modal = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) * weight / (N * N) ** 2
    return np.bincount(k_bin.ravel(), weights=modal.ravel()).astype(np.float64)


def scalar_kinetic_energy(omega: np.ndarray, Kx, Ky, invKsq) -> float:
    psi = Omega2Psi(omega, invKsq)
    u, v = Psi2UV(psi, Kx, Ky)
    return float(0.5 * np.mean(u * u + v * v))


def main() -> None:
    stems = STEM_START + STRIDE * np.arange(N_STEPS, dtype=np.int64)
    assert stems[-1] == STEM_START + N_STEPS - 1

    Kx, Ky, Kabs, _, invKsq = initialize_wavenumbers_rfft2(N, N, LX, LY, INDEXING="ij")
    weight = np.ones_like(Kabs, dtype=np.float64)
    weight[:, 1:-1] = 2.0
    k_bin = np.rint(Kabs).astype(np.int64)

    acc = None
    first_Ek = None
    first_omega = None

    for i, stem in enumerate(tqdm(stems, desc="E(k) eval")):
        path = EVAL_DIR / f"{stem}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        omega = np.asarray(scipy.io.loadmat(path)["Omega"], dtype=np.float64)
        if omega.shape != (N, N):
            raise ValueError(f"{path}: expected {(N, N)}, got {omega.shape}")
        Ek = kinetic_spectrum(omega, invKsq, Ky, Kx, weight, k_bin)
        if i == 0:
            first_Ek = Ek
            first_omega = omega
            acc = np.zeros_like(Ek)
        if Ek.size > acc.size:
            acc = np.pad(acc, (0, Ek.size - acc.size))
        acc[: Ek.size] += Ek

    # Plot / save Nyquist-limited shells k = 1 .. N/2 (corner |k|>N/2 dropped from figure).
    k_max = N // 2
    Ek_mean_full = acc / N_STEPS
    Ek_mean = Ek_mean_full[1 : k_max + 1]
    k = np.arange(1, k_max + 1, dtype=np.int64)

    e_scalar = scalar_kinetic_energy(first_omega, Kx, Ky, invKsq)
    e_sum = float(first_Ek.sum())
    rel = abs(e_sum - e_scalar) / max(abs(e_scalar), 1e-30)
    print(f"first-frame check: sum(E(k))={e_sum:.8g}  scalar KE={e_scalar:.8g}  rel={rel:.3e}")
    if rel > 1e-6:
        raise RuntimeError(f"Parseval check failed: rel err {rel}")

    np.savez_compressed(OUT_NPZ, k=k, Ek_mean=Ek_mean, stems=stems)
    print(f"wrote {OUT_NPZ}")

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.loglog(k, Ek_mean, color="0.4", linestyle="--", label="Reference Truth")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Energy Spectrum")
    ax.set_xlim(1, 128)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"stems {stems[0]}..{stems[-1]}  Ek_mean[k=1]={Ek_mean[0]:.6g}  Ek_mean[k=100]={Ek_mean[99]:.6g}")


if __name__ == "__main__":
    main()
