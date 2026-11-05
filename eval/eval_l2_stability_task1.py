#!/usr/bin/env python3
"""Task 1 diagnostics for a trained direct-step L=2 HINE checkpoint.

Matrix-free centered finite differences only. Does not retrain, does not
edit weights, and does not treat HINE as an Euler integrator.

Proxy mode (default) uses D Phi' ≈ D and labels every derived operator
*_proxy / approximation=proxy. py2d mode uses a finite-difference physical
tangent and requires the hine-jax env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_jacobian_blocks import choose_sample_starts
from eval_jacobian_manual_mse import load_kept_paths
from eval_jacobian_manual_mse_plus import load_model
from l2_fd_jvp import (
    DEFAULT_EPS_LIST,
    assert_l2,
    eps_sweep,
    make_point,
    model_step,
)
from l2_operators import diagnose_point, flat_diag
from l2_physical_jvp import make_tangent
from l2_retention import rollout_diagnostics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_debug_test",
    )
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--n-snapshots", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=20)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument(
        "--fd-eps-list",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPS_LIST),
    )
    p.add_argument("--fd-tol", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--physical-jvp", choices=["proxy", "py2d"], default="proxy")
    p.add_argument("--spectral-radius", action="store_true")
    p.add_argument("--dt", type=float, default=0.25, help="Emulator step in time units")
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--model-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--solver-dt", type=float, default=1e-4)
    return p.parse_args(argv)


def _hash_module(mod: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for tensor in mod.state_dict().values():
        arr = tensor.detach().float().cpu().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def _load_window(kept, start: int, length: int) -> tuple[list[np.ndarray], list[int]]:
    frames = []
    stems = []
    for j in range(length):
        stem, path = kept[start + j]
        frames.append(scipy.io.loadmat(path)["Omega"])
        stems.append(int(stem))
    return frames, stems


def _errbar(ax, x, y, yerr, label):
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)


def write_plots(save_dir: Path, diag: dict, rolls: dict) -> None:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(diag["tr_Fu_over_N"].shape[0])

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(ax, idx, diag["tr_Fu_over_N"], diag["tr_A_ci95"] / diag["n_fine"], r"Tr$(F_u)/N$")
    _errbar(ax, idx, diag["tr_DFz_over_N"], diag["tr_DB_ci95"] / (diag["n_fine"] * diag["dt"]), r"Tr$(DF_z)/N$")
    ax.plot(idx, diag["tr_Heff_over_N"], "k--", label=r"sum $\widetilde H_{\mathrm{eff}}$")
    ax.set_xlabel("snapshot")
    ax.set_title("Difference-quotient trace split (proxy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "trace_split.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(
        ax,
        idx,
        diag["tr_Sigma_over_M"],
        diag["tr_S_ci95"] / (diag["n_coarse"] * diag["dt"]),
        r"Tr$(\widetilde\Sigma)/M$",
    )
    ax.set_xlabel("snapshot")
    ax.set_title("Parasitic trace, difference-quotient proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "parasitic_trace.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(ax, idx, diag["amp_J_mean"], diag["amp_J_ci95"], r"$\|J v\|^2/N$")
    ax.set_xlabel("snapshot")
    ax.set_title("Direct-step amplification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "amplification.png", dpi=140)
    plt.close(fig)

    if "rho_S" in diag:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(idx, diag["rho_S"], "o-")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("snapshot")
        ax.set_ylabel(r"$\rho(S)$")
        ax.set_title("Parasitic spectral radius")
        fig.tight_layout()
        fig.savefig(plot_dir / "rho_S.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(ax, idx, diag["offgraph_probe_mean"], diag["offgraph_probe_ci95"], r"$\|L^{\mathrm{off}} v\|^2/N$")
    ax.set_xlabel("snapshot")
    ax.set_title("Off-graph coupling probe norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "offgraph_norm.png", dpi=140)
    plt.close(fig)

    lead = np.arange(rolls["I"].shape[1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["I"].mean(0), label=r"$I_n$")
    ax.plot(lead, rolls["R"].mean(0), label=r"$R_n=\|e_n\|-I_n$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title("Injection and scalar retention")
    fig.tight_layout()
    fig.savefig(plot_dir / "injection_retention.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["rH_norm"].mean(0), label=r"$\|\tilde r_H\|$")
    ax.plot(lead, rolls["rZ_norm"].mean(0), label=r"$\|\tilde r_z\|$")
    ax.plot(lead, rolls["cos_rH_rZ"].mean(0), label=r"$\cos(r_H,r_z)$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title("First-factor retention channels")
    fig.tight_layout()
    fig.savefig(plot_dir / "retention_channels.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["recon_norm"].mean(0), label=r"$\|\rho-\tilde r_H-\tilde r_z\|$")
    ax.plot(lead, rolls["rho_norm"].mean(0), label=r"$\|\rho\|$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title("First-factor reconstruction residual")
    fig.tight_layout()
    fig.savefig(plot_dir / "reconstruction_residual.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float64 if args.model_dtype == "float64" else torch.float32

    model, ema_vit = load_model(args.checkpoint, device)
    assert_l2(model)
    if dtype == torch.float64:
        ema_vit.double()
    weight_hash = _hash_module(ema_vit)
    failed: list[str] = []

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    lookahead = args.rollout_length + 1
    starts = choose_sample_starts(len(kept), lookahead, args.n_snapshots, args.seed)
    windows = []
    for s in starts:
        frames, stems = _load_window(kept, int(s), args.rollout_length + 2)
        windows.append({"start": int(s), "stems": stems, "frames": frames})

    tangent = make_tangent(
        args.physical_jvp,
        nx=int(model.img_size),
        Re=args.re,
        fkx=args.fkx,
        fky=args.fky,
        alpha=args.alpha,
        beta=args.beta,
        solver_dt=args.solver_dt,
        emulator_dt=args.dt,
    )

    first = make_point(windows[0]["frames"][:3], model, device, dtype, args.norm)
    step0 = lambda u, z, _t=first.t: model_step(ema_vit, u, z, _t)
    sweep = eps_sweep(step0, first, args.fd_eps_list, args.fd_tol)
    if not sweep["ok"]:
        failed.append(
            "fd eps sweep: no consecutive eps pair within "
            f"tol={args.fd_tol}; rels={sweep['consecutive_rel']}"
        )
    print(f"[fd] eps sweep ok={sweep['ok']} rels={sweep['consecutive_rel']}", flush=True)

    diags = []
    rolls = []
    ds = int(model.ds_factors[0])
    for i, window in enumerate(windows):
        point = make_point(window["frames"][:3], model, device, dtype, args.norm)
        point.stems = tuple(window["stems"][:3])  # type: ignore[assignment]
        step_fn = lambda u, z, _t=point.t: model_step(ema_vit, u, z, _t)
        print(f"[snap {i}] stems={window['stems'][:3]} diagnose", flush=True)
        diag = diagnose_point(
            step_fn,
            point,
            ds,
            args.fd_eps,
            tangent,
            args.dt,
            args.norm,
            args.n_probes,
            args.seed + i,
            args.spectral_radius,
            args.ncv,
            args.maxiter,
            args.tol,
        )
        diags.append(diag)
        print(
            f"[snap {i}] trA/N={diag['tr_A_over_N']:.4e} "
            f"trJ/N={diag['tr_J_over_N']:.4e} "
            f"amp={diag['amp_J']['mean']:.4e}",
            flush=True,
        )
        print(f"[roll {i}] length={args.rollout_length}", flush=True)
        roll = rollout_diagnostics(
            ema_vit,
            window["frames"],
            model,
            device,
            dtype,
            args.norm,
            args.fd_eps,
            tangent,
            args.rollout_length,
        )
        rolls.append(roll)

    after = _hash_module(ema_vit)
    if after != weight_hash:
        failed.append("checkpoint weights changed during the diagnostic")

    def stack_diag(key):
        return np.stack([flat_diag(d)[key] for d in diags])

    diag_np = {}
    probe_keys = [
        "tr_A_over_N", "tr_J_over_N", "tr_K_over_M", "tr_S_over_M", "tr_DB_over_N",
        "tr_Fu_over_N", "tr_DFz_over_N", "tr_Heff_over_N", "tr_Sigma_over_M",
        "identity_gap", "n_fine", "n_coarse",
        "tr_A_mean", "tr_A_std", "tr_A_ci95",
        "tr_J_mean", "tr_J_std", "tr_J_ci95",
        "tr_S_mean", "tr_S_std", "tr_S_ci95",
        "tr_DB_mean", "tr_DB_std", "tr_DB_ci95",
        "amp_J_mean", "amp_J_std", "amp_J_ci95",
        "offgraph_probe_mean", "offgraph_probe_std", "offgraph_probe_ci95",
        "tr_A_probes", "tr_J_probes", "tr_S_probes", "amp_J_probes",
    ]
    if args.spectral_radius:
        probe_keys += ["rho_S", "rho_S_real", "rho_S_imag", "rho_S_converged"]
    for key in probe_keys:
        diag_np[key] = stack_diag(key)
    diag_np["dt"] = np.asarray(args.dt)
    diag_np["approximation"] = np.asarray(diags[0]["approximation"])

    roll_np = {}
    for key in ("I", "R", "e_norm", "rho_norm", "rH_norm", "rZ_norm", "recon_norm", "cos_rH_rZ", "eps_u_norm", "eps_z_norm"):
        roll_np[key] = np.stack([r[key] for r in rolls])

    np.savez_compressed(save_dir / "results.npz", **diag_np, **{f"roll_{k}": v for k, v in roll_np.items()})
    write_plots(save_dir, diag_np, roll_np)

    meta = {
        "task": "L2 Task 1 direct-step diagnostics",
        "checkpoint": args.checkpoint,
        "weight_sha256": weight_hash,
        "weights_unchanged": after == weight_hash,
        "eval_dir": args.eval_dir,
        "every_nth_field": args.every_nth_field,
        "windows": [{"start": w["start"], "stems": w["stems"]} for w in windows],
        "norm": args.norm,
        "dt_emulator": args.dt,
        "dt_meaning": "one emulator step; default 0.25 = five 0.05 snapshots",
        "downsampling": f"avg_pool2d kernel=stride={ds}",
        "physical_jvp": tangent.name,
        "is_exact_physical_jvp": bool(tangent.is_exact),
        "difference_quotient": "F=(g-u)/dt and E=(h-z)/dt are labels only; the network is direct-step",
        "first_factor": "r_H and r_z are one-Jacobian-factor approximations of rho",
        "seed": args.seed,
        "fd_eps": args.fd_eps,
        "fd_sweep": sweep,
        "n_probes": args.n_probes,
        "model_dtype": args.model_dtype,
        "py2d": {
            "Re": args.re, "fkx": args.fkx, "fky": args.fky,
            "alpha": args.alpha, "beta": args.beta, "solver_dt": args.solver_dt,
        },
        "failed_tests": failed,
        "num_coarse": int(model.num_coarse),
        "latent_res": list(model.latent_res),
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[done] save_dir={save_dir} failed={len(failed)}", flush=True)
    if failed:
        for item in failed:
            print(f"  FAIL {item}", flush=True)


if __name__ == "__main__":
    main()
