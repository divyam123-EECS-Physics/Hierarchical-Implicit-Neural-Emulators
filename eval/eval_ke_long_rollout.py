#!/usr/bin/env python3
"""Long autoregressive rollout scored only by kinetic energy.

Ten eval initial conditions (stem % 5 == 0), batched together. No future
ground truth and no saved fields. Energy is compared to the fixed reference
mean and ±5σ band. A trial that leaves the band stays failed, but the rollout
continues so a later return is still visible in E(t).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval.eval_ablation_mse import load_ablation_model
from eval.eval_jacobian_manual_mse import load_kept_paths, load_model as load_l3
from eval.eval_jacobian_manual_mse_plus import load_model as load_l2
from eval.eval_kinetic_energy_survival import (
    SIGMA_MULT,
    kinetic_energy_torch,
    make_wavenumber_tensors,
    survival_curve,
)
from eval.eval_long_ar_ke_evalspan import load_high

DEFAULT_REF = Path(
    "/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_manual_mse/reference_kinetic_energy.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["l3", "l2", "l1"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--ref-cache", type=Path, default=DEFAULT_REF)
    p.add_argument("--n-steps", type=int, default=100_000)
    p.add_argument("--n-ics", type=int, default=10)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--flush-every", type=int, default=1000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p.parse_args()


def choose_ics(n_kept: int, n_ics: int) -> np.ndarray:
    if n_ics < 1:
        raise ValueError(f"n-ics must be >= 1, got {n_ics}")
    if n_ics > n_kept:
        raise ValueError(f"asked for {n_ics} initial conditions, only {n_kept} eval frames")
    idx = np.rint(np.linspace(0, n_kept - 1, n_ics)).astype(np.int64)
    if len(np.unique(idx)) != n_ics:
        raise ValueError(f"even spacing collapsed to {len(np.unique(idx))} indices")
    return idx


def load_reference(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"missing reference cache {path}")
    with np.load(path) as z:
        mu = float(np.asarray(z["mu_E"]))
        sigma = float(np.asarray(z["sigma_E"]))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"bad reference mu={mu} sigma={sigma}")
    return mu, sigma


def energies_of(high: torch.Tensor, norm: float, Kx, Ky, invKsq) -> np.ndarray:
    omega = high[:, 0] * float(norm)
    return kinetic_energy_torch(omega, Kx, Ky, invKsq).double().cpu().numpy()


@torch.no_grad()
def rollout(kind, ema, model, high, n_steps, norm, Kx, Ky, invKsq, flush) -> np.ndarray:
    b = high.shape[0]
    energies = np.full((b, n_steps), np.nan, dtype=np.float64)
    if kind == "l3":
        mid0 = torch.zeros(b, 1, model.ds_res, model.ds_res, device=high.device, dtype=high.dtype)
        low0 = torch.zeros(b, 1, model.low_res, model.low_res, device=high.device, dtype=high.dtype)
        _, mid, _ = ema(high, mid0, low0, torch.zeros(b, device=high.device, dtype=high.dtype))
        _, _, low = ema(high, mid, low0, torch.ones(b, device=high.device, dtype=high.dtype))
        t_step = 2 * torch.ones(b, device=high.device, dtype=high.dtype)
    elif kind == "l2":
        k = model.num_coarse
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
        t_step = float(k) * torch.ones(b, device=high.device, dtype=high.dtype)
    else:
        aux = getattr(ema, "num_aux", 0) > 0

    for step in tqdm(range(n_steps), desc=f"AR {kind}"):
        if kind == "l3":
            high, mid, low = ema(high, mid, low, t_step)
        elif kind == "l2":
            high, coarses = ema(high, coarses, t_step)
        else:
            out = ema(high)
            high = out[0] if aux else out
        energies[:, step] = energies_of(high, norm, Kx, Ky, invKsq)
        done = step + 1
        if done == n_steps or done % flush == 0:
            yield energies[:, :done], done
    return energies


def save_state(save_dir: Path, energies: np.ndarray, meta: dict, mu: float, sigma: float) -> None:
    n_steps = energies.shape[1]
    lead = np.arange(1, n_steps + 1, dtype=np.int64)
    s_ar = survival_curve(energies, mu, sigma, SIGMA_MULT)
    np.savez_compressed(
        save_dir / "kinetic_energy_survival.npz",
        lead=lead,
        ar_E=energies,
        S_ar=s_ar,
        mu_E=np.asarray(mu),
        sigma_E=np.asarray(sigma),
        sigma_mult=np.asarray(SIGMA_MULT),
        band_lo=np.asarray(mu - SIGMA_MULT * sigma),
        band_hi=np.asarray(mu + SIGMA_MULT * sigma),
        ic_index=meta["ic_index"],
        ic_stem=meta["ic_stem"],
        kind=np.asarray(meta["kind"]),
        checkpoint=np.asarray(meta["checkpoint"]),
        n_steps_requested=np.asarray(meta["n_steps"], dtype=np.int64),
    )
    plot_one(save_dir, lead, energies, s_ar, mu, sigma, meta["kind"])


def _plot_xy(lead: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, lead.shape[0] // 2000)
    return lead[::stride], values[..., ::stride]


def plot_one(save_dir, lead, energies, s_ar, mu, sigma, kind) -> None:
    lo, hi = mu - SIGMA_MULT * sigma, mu + SIGMA_MULT * sigma
    x, traces = _plot_xy(lead, energies)
    _, s_plot = _plot_xy(lead, s_ar)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i in range(traces.shape[0]):
        ax.plot(x, traces[i], color="C0", alpha=0.35, lw=0.8)
    ax.plot(x, np.nanmean(traces, axis=0), color="C0", lw=2.2, label=f"{kind} mean")
    ax.axhline(mu, color="k", lw=1.2, label=r"$\mu_E$")
    ax.axhline(lo, color="k", ls="--", lw=1.0, label=fr"$\mu_E \pm {SIGMA_MULT:g}\sigma_E$")
    ax.axhline(hi, color="k", ls="--", lw=1.0)
    finite = energies[np.isfinite(energies)]
    if finite.size and finite.max() < 20 and finite.min() > 0:
        pad = 0.4
        ax.set_ylim(min(lo, finite.min()) - pad, max(hi, finite.max()) + pad)
    else:
        ax.set_ylim(lo - 0.4, hi + 3.0)
    ax.set_xlabel("lead (autoregressive step)")
    ax.set_ylabel("kinetic energy $E$")
    ax.set_title(f"{kind} kinetic energy, {energies.shape[0]} eval initial conditions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / "kinetic_energy_vs_lead.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, s_plot, color="C0", lw=2.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("lead (autoregressive step)")
    ax.set_ylabel(r"stability rate $S(n)$")
    ax.set_title(fr"{kind} stability ($|E-\mu_E|\leq {SIGMA_MULT:g}\sigma_E$; once failed stays failed)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / "stability_rate_vs_lead.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_steps < 1:
        raise ValueError(f"n-steps must be >= 1, got {args.n_steps}")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mu, sigma = load_reference(args.ref_cache)
    kept = load_kept_paths(args.eval_dir, args.every_nth_field, "mod")
    ic_index = choose_ics(len(kept), args.n_ics)
    ic_stem = np.asarray([kept[int(i)][0] for i in ic_index], dtype=np.int64)
    print(
        f"kind={args.kind} n_steps={args.n_steps} n_ics={args.n_ics} "
        f"eval_kept={len(kept)} stems={ic_stem.tolist()} "
        f"mu={mu:.6g} sigma={sigma:.6g} device={device}"
    )
    high = torch.cat(
        [load_high(kept[int(i)][1], args.norm, device) for i in ic_index], dim=0
    )
    if args.kind == "l3":
        model, ema = load_l3(args.checkpoint, device)
    elif args.kind == "l2":
        model, ema = load_l2(args.checkpoint, device)
    else:
        model, ema = load_ablation_model(args.checkpoint, device)
        model = None
    Kx, Ky, invKsq, _, _, _ = make_wavenumber_tensors(high.shape[-1], high.shape[-2], device)
    meta = {
        "ic_index": ic_index,
        "ic_stem": ic_stem,
        "kind": args.kind,
        "checkpoint": args.checkpoint,
        "n_steps": args.n_steps,
    }
    latest = None
    for energies, done in rollout(
        args.kind, ema, model, high, args.n_steps, args.norm, Kx, Ky, invKsq, args.flush_every
    ):
        latest = energies
        save_state(save_dir, energies, meta, mu, sigma)
        alive = float(survival_curve(energies, mu, sigma)[-1])
        print(f"flushed step {done}/{args.n_steps} S(end)={alive:.3%} Emean={np.nanmean(energies[:, -1]):.6g}")
    if latest is None:
        raise RuntimeError("rollout produced no energies")
    print(f"KE_LONG_ROLLOUT_DONE kind={args.kind} steps={latest.shape[1]} ics={latest.shape[0]}")


if __name__ == "__main__":
    main()
