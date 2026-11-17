#!/usr/bin/env python3
"""Reference kinetic-energy spectrum on the 512^2 py2d DNS (Figure 5(c)).

The 256^2 training files are a spectral truncation, so they stop at k=128
while the viscous tail in the paper continues past that. This script averages
the shell-sum spectrum of the original 512^2 snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm

from util.py2d import Omega2Psi, Psi2UV, initialize_wavenumbers_rfft2

OUT_DIR = _REPO / "output"
REF_FKX4 = OUT_DIR / "energy_spectrum_dns512_200.npz"

N = 512
LX = LY = 2.0 * np.pi
STEM_START = 91621
STRIDE = 5
N_STEPS = 200

DEFAULT_DNS = Path(
    "/pscratch/sd/d/divyam/py2d_runs/paper_fkx5/results/"
    "Re10000_fkx5fky5_r0.1_b20/NoSGS/NX512/dt0.0001_IC1/data"
)


def kinetic_spectrum(omega, invKsq, Ky, Kx, weight, k_bin) -> np.ndarray:
    """Shell-summed kinetic energy. sum_k E(k) = 1/2 mean(u^2+v^2)."""
    n = omega.shape[0]
    omega_hat = np.fft.rfft2(np.asarray(omega, dtype=np.float64))
    psi_hat = omega_hat * invKsq
    u_hat = 1j * Ky * psi_hat
    v_hat = -1j * Kx * psi_hat
    modal = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) * weight / (n * n) ** 2
    return np.bincount(k_bin.ravel(), weights=modal.ravel()).astype(np.float64)


def scalar_kinetic_energy(omega, Kx, Ky, invKsq) -> float:
    psi = Omega2Psi(omega, invKsq)
    u, v = Psi2UV(psi, Kx, Ky)
    return float(0.5 * np.mean(u * u + v * v))


def log_bin(k: np.ndarray, ek: np.ndarray, bins_per_decade: float = 6.0):
    """Average E(k) in logarithmic bins. Width-normalized so the level matches the shells."""
    edges = np.logspace(np.log10(k[0]), np.log10(k[-1] + 0.5), int(np.log10(k[-1]) * bins_per_decade) + 1)
    idx = np.digitize(k, edges) - 1
    kc, ec = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not np.any(m):
            continue
        width = edges[b + 1] - edges[b]
        ec.append(float(ek[m].sum()) / width)
        kc.append(float(np.sqrt(edges[b] * edges[b + 1])))
    return np.asarray(kc), np.asarray(ec)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="512^2 shell kinetic-energy spectrum")
    p.add_argument("--dns-dir", type=Path, default=DEFAULT_DNS)
    p.add_argument("--stem-start", type=int, default=STEM_START)
    p.add_argument("--out-prefix", default="energy_spectrum_dns512_fkx5_200")
    p.add_argument("--label", default="fkx=fky=5")
    args = p.parse_args()
    dns_dir = args.dns_dir
    out_npz = OUT_DIR / f"{args.out_prefix}.npz"
    out_png = OUT_DIR / f"{args.out_prefix}.png"

    stems = args.stem_start + STRIDE * np.arange(N_STEPS, dtype=np.int64)
    missing = [int(s) for s in stems if not (dns_dir / f"{int(s)}.mat").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing DNS frames, first {missing[:5]}")

    Kx, Ky, Kabs, _, invKsq = initialize_wavenumbers_rfft2(N, N, LX, LY, INDEXING="ij")
    weight = np.ones_like(Kabs, dtype=np.float64)
    weight[:, 1:-1] = 2.0
    k_bin = np.rint(Kabs).astype(np.int64)

    acc = None
    first_ek = None
    first_omega = None
    for i, stem in enumerate(tqdm(stems, desc="E(k) 512")):
        omega = np.asarray(scipy.io.loadmat(dns_dir / f"{int(stem)}.mat")["Omega"], dtype=np.float64)
        if omega.shape != (N, N):
            raise ValueError(f"{stem}.mat: expected {(N, N)}, got {omega.shape}")
        ek = kinetic_spectrum(omega, invKsq, Ky, Kx, weight, k_bin)
        if i == 0:
            first_ek = ek
            first_omega = omega
            acc = np.zeros_like(ek)
        if ek.size > acc.size:
            acc = np.pad(acc, (0, ek.size - acc.size))
            first_ek = np.pad(first_ek, (0, ek.size - first_ek.size))
        acc[: ek.size] += ek

    ek_mean_full = acc / N_STEPS
    k_all = np.arange(ek_mean_full.size, dtype=np.int64)
    # Nyquist of the square grid. Corner shells |k|>N/2 are kept in the sum, not the figure.
    k_max = N // 2
    k = k_all[1 : k_max + 1]
    ek = ek_mean_full[1 : k_max + 1]
    kc, ec = log_bin(k.astype(np.float64), ek)

    e_scalar = scalar_kinetic_energy(first_omega, Kx, Ky, invKsq)
    e_sum = float(first_ek.sum())
    rel = abs(e_sum - e_scalar) / max(abs(e_scalar), 1e-30)
    print(f"first-frame check: sum(E(k))={e_sum:.8g}  scalar KE={e_scalar:.8g}  rel={rel:.3e}")
    if rel > 1e-6:
        raise RuntimeError(f"Parseval check failed: rel err {rel}")
    print(f"mean KE over {N_STEPS} frames: {float(ek_mean_full.sum()):.6g}")
    for kk in (1, 2, 3, 4, 5, 10, 20, 50, 100, 128, 170, 200, 256):
        if kk < ek_mean_full.size:
            print(f"  E({kk})={ek_mean_full[kk]:.6e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        k=k,
        Ek_mean=ek,
        k_log=kc,
        Ek_log=ec,
        stems=stems,
        k_full=k_all,
        Ek_full=ek_mean_full,
    )
    print(f"wrote {out_npz}")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    positive = ek > 0
    ax.loglog(k[positive], ek[positive], color="0.75", lw=0.8, label=f"{args.label}, integer shells")
    pos_b = ec > 0
    ax.loglog(kc[pos_b], ec[pos_b], color="black", lw=2.0, label=args.label)
    if REF_FKX4.is_file() and "fkx5" in args.out_prefix:
        ref4 = np.load(REF_FKX4)
        m4 = ref4["Ek_log"] > 0
        ax.loglog(ref4["k_log"][m4], ref4["Ek_log"][m4], color="0.35", ls="--", lw=1.2, label="fkx=fky=4")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Energy Spectrum")
    ax.set_xlim(1, k_max)
    ax.set_ylim(1e-11, 1e1)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.7)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
