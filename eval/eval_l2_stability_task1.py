#!/usr/bin/env python3
"""Task 1 diagnostics for a trained direct-step L=2 HINE checkpoint.

Matrix-free centered finite differences only. Does not retrain, does not
edit weights, and does not treat HINE as an Euler integrator.

Proxy mode (default) uses D Phi' ≈ D and labels results approximation=proxy.
py2d mode uses a finite-difference physical tangent through the py2d
checkout. That finite difference is not an exact tangent. It requires the
hine-jax env.

With --n-gpus N>1 the parent shards the N_snapshots windows across GPUs
(round-robin) and merges shard_*.npz into results.npz + plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

# Parent orchestrator must not bind GPUs; workers set CUDA_VISIBLE_DEVICES.
if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    FD_OUTPUT_FLOOR,
    FD_SCALING,
    apply_D,
    assert_l2,
    make_point,
    model_step,
    validate_fd_eps,
)
from l2_operators import _phi_image, diagnose_point, flat_diag, quotient_halfwidth
from l2_physical_jvp import make_tangent
from l2_retention import relative_residual, rollout_diagnostics


DIAG_KEYS = [
    "tr_A_over_N",
    "tr_J_over_N",
    "tr_K_over_M",
    "tr_S_over_M",
    "tr_DB_over_N",
    "tr_B_DPhi_n_over_N",
    "tr_Fu_over_N",
    "tr_DFz_over_N",
    "tr_Heff_over_N",
    "tr_Sigma_over_M",
    "composition_rel_mean",
    "composition_rel_max",
    "trace_estimator_gap",
    "trace_estimator_gap_ci95",
    "fd_additivity_rel",
    "n_fine",
    "n_coarse",
    "tr_A_mean",
    "tr_A_std",
    "tr_A_ci95",
    "tr_J_mean",
    "tr_J_std",
    "tr_J_ci95",
    "tr_S_mean",
    "tr_S_std",
    "tr_S_ci95",
    "tr_DB_mean",
    "tr_DB_std",
    "tr_DB_ci95",
    "tr_B_DPhi_n_mean",
    "tr_B_DPhi_n_ci95",
    "tr_DPhi_next_B_mean",
    "tr_DPhi_next_B_ci95",
    "amp_J_mean",
    "amp_J_std",
    "amp_J_ci95",
    "offgraph_probe_mean",
    "offgraph_probe_std",
    "offgraph_probe_ci95",
    "tr_A_probes",
    "tr_J_probes",
    "tr_S_probes",
    "amp_J_probes",
]
RHO_KEYS = ["rho_S", "rho_S_real", "rho_S_imag", "rho_S_converged"]
ROLL_KEYS = (
    "I",
    "R",
    "e_norm",
    "rho_norm",
    "rH_norm",
    "rZ_norm",
    "recon_norm",
    "cos_rH_rZ",
    "eps_u_norm",
    "eps_z_norm",
    "e_lin_norm",
    "eta_norm",
    "eta_lin_norm",
    "e_lin_err",
    "eta_lin_err",
    "recon_lin_norm",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_n20",
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
    p.add_argument(
        "--allow-unvalidated-fd",
        action="store_true",
        help="Write results even when the selected fd-eps fails validation. Recorded in metadata.",
    )
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
    p.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Shard snapshots across this many GPUs. 1 runs in-process.",
    )
    p.add_argument(
        "--worker",
        action="store_true",
        help="Internal: process one GPU shard of windows",
    )
    p.add_argument("--gpu-rank", type=int, default=None)
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


def _load_window_from_entry(entry: dict) -> tuple[list[np.ndarray], list[int]]:
    frames = [scipy.io.loadmat(p)["Omega"] for p in entry["paths"]]
    stems = [int(s) for s in entry["stems"]]
    return frames, stems


def sample_index_path(save_dir: Path) -> Path:
    return save_dir / "sample_index.json"


def write_sample_index(
    save_dir: Path,
    kept: list[tuple[int, str]],
    starts: np.ndarray,
    window_len: int,
) -> list[dict]:
    samples = []
    for i, start in enumerate(starts):
        start = int(start)
        stems = [int(kept[start + j][0]) for j in range(window_len)]
        paths = [kept[start + j][1] for j in range(window_len)]
        samples.append(
            {
                "sample": i,
                "start_index": start,
                "stems": stems,
                "paths": paths,
            }
        )
    payload = {
        "n_samples": len(samples),
        "window_len": window_len,
        "samples": samples,
    }
    sample_index_path(save_dir).write_text(json.dumps(payload, indent=2) + "\n")
    return samples


def _errbar(ax, x, y, yerr, label):
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)


def _merge_fd_reports(base: dict, extra: dict) -> dict:
    if not extra.get("checks"):
        return base
    checks = list(base["checks"]) + list(extra["checks"])
    out = dict(base)
    out["checks"] = checks
    out["ok"] = all(c["ok"] for c in checks)
    out["n_checks"] = len(checks)
    out["n_failed"] = sum(1 for c in checks if not c["ok"])
    out["worst"] = max(checks, key=lambda c: c["rel"])
    return out


def _title(text: str, validated: bool) -> str:
    return text if validated else f"UNVALIDATED {text}"


def write_plots(save_dir: Path, diag: dict, rolls: dict, validated: bool = True) -> None:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(diag["tr_Fu_over_N"].shape[0])
    kind = str(np.asarray(diag["approximation"]).item())
    dt = float(np.asarray(diag["dt"]))
    fu_err = quotient_halfwidth(diag["tr_A_ci95"], diag["n_fine"], dt)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(ax, idx, diag["tr_Fu_over_N"], fu_err, r"Tr$(F_u)/N$")
    _errbar(
        ax,
        idx,
        diag["tr_DFz_over_N"],
        quotient_halfwidth(diag["tr_DB_ci95"], diag["n_fine"], dt),
        r"Tr$(DB)/(N\,dt)$ proxy",
    )
    ax.plot(idx, diag["tr_Heff_over_N"], "k--", label=r"$(Tr J - N)/(N dt)$")
    _errbar(
        ax,
        idx,
        diag["tr_J_over_N"],
        diag["tr_J_ci95"] / diag["n_fine"],
        r"Tr$(J)/N$",
    )
    ax.set_xlabel("snapshot")
    ax.set_title(_title(f"Direct-step and difference-quotient traces ({kind})", validated))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "trace_split.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(
        ax,
        idx,
        diag["tr_Sigma_over_M"],
        quotient_halfwidth(diag["tr_S_ci95"], diag["n_coarse"], dt),
        r"$(Tr S - M)/(M dt)$",
    )
    _errbar(
        ax,
        idx,
        diag["tr_S_over_M"],
        diag["tr_S_ci95"] / diag["n_coarse"],
        r"Tr$(S)/M$",
    )
    ax.set_xlabel("snapshot")
    ax.set_title(_title(f"Parasitic trace ({kind})", validated))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "parasitic_trace.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(
        ax,
        idx,
        diag["amp_J_mean"],
        diag["amp_J_ci95"],
        r"mean $\|J v\|^2/N$",
    )
    ax.set_xlabel("snapshot")
    ax.set_title(_title("Mean-square directional amplification", validated))
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
        ax.set_title(_title(f"Parasitic spectral radius ({kind})", validated))
        fig.tight_layout()
        fig.savefig(plot_dir / "rho_S.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _errbar(
        ax,
        idx,
        diag["offgraph_probe_mean"],
        diag["offgraph_probe_ci95"],
        r"$\|L^{\mathrm{off}} v\|^2/N$",
    )
    ax.set_xlabel("snapshot")
    ax.set_title(_title(f"Off-graph coupling probe norm ({kind})", validated))
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
    ax.set_title(_title("Injection and scalar retention", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "injection_retention.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["rH_norm"].mean(0), label=r"$\|\tilde r_H\|$ truncated")
    ax.plot(lead, rolls["rZ_norm"].mean(0), label=r"$\|\tilde r_z\|$ truncated")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Truncated first-factor channels", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "retention_channels.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, np.nanmean(rolls["cos_rH_rZ"], axis=0), label=r"$\cos(r_H,r_z)$")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Channel alignment; undefined cosines are gaps", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "retention_alignment.png", dpi=140)
    plt.close(fig)

    rel = relative_residual(rolls["recon_norm"], rolls["rho_norm"], FD_OUTPUT_FLOOR)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["recon_norm"].mean(0), label=r"$\|\rho-\tilde r_H-\tilde r_z\|$")
    ax.plot(lead, rolls["rho_norm"].mean(0), label=r"$\|\rho\|$")
    ax.plot(lead, np.nanmean(rel, axis=0), label="mean of per-window ratios")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Truncated first-factor residual", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "reconstruction_residual.png", dpi=140)
    plt.close(fig)

    if "e_lin_norm" in rolls:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes[0, 0].plot(lead, rolls["e_norm"].mean(0), label="actual")
        axes[0, 0].plot(lead, rolls["e_lin_norm"].mean(0), label="linearized")
        axes[0, 0].set_title(r"fine $\|e\|$")
        axes[0, 0].legend()
        axes[0, 1].plot(lead, rolls["eta_norm"].mean(0), label="actual")
        axes[0, 1].plot(lead, rolls["eta_lin_norm"].mean(0), label="linearized")
        axes[0, 1].set_title(r"latent $\|\eta\|$")
        axes[0, 1].legend()
        axes[1, 0].plot(lead, rolls["e_lin_err"].mean(0), label=r"$\|e-e_{lin}\|$")
        axes[1, 0].plot(lead, rolls["eta_lin_err"].mean(0), label=r"$\|\eta-\eta_{lin}\|$")
        axes[1, 0].set_title("vector linearization error")
        axes[1, 0].legend()
        axes[1, 1].plot(lead, rolls["recon_norm"].mean(0), label=r"$\|\rho-r_H-r_z\|$")
        axes[1, 1].plot(lead, rolls["recon_lin_norm"].mean(0), label=r"$\|\rho_{lin}-r_H-r_z\|$")
        axes[1, 1].set_title("truncation versus linearization")
        axes[1, 1].legend()
        for ax in axes.ravel():
            ax.set_xlabel("step")
        fig.suptitle(_title("Full linearized error versus free rollout", validated))
        fig.tight_layout()
        fig.savefig(plot_dir / "linearized_error.png", dpi=140)
        plt.close(fig)


def _stack_results(
    diags: list[dict], rolls: list[dict], spectral_radius: bool, dt: float
) -> tuple[dict, dict]:
    probe_keys = list(DIAG_KEYS)
    if spectral_radius:
        probe_keys += RHO_KEYS

    def stack_diag(key: str) -> np.ndarray:
        return np.stack([flat_diag(d)[key] for d in diags])

    diag_np = {key: stack_diag(key) for key in probe_keys}
    diag_np["dt"] = np.asarray(dt)
    diag_np["approximation"] = np.asarray(diags[0]["approximation"])
    roll_np = {key: np.stack([r[key] for r in rolls]) for key in ROLL_KEYS}
    return diag_np, roll_np


def run_windows(
    args: argparse.Namespace,
    window_entries: list[dict],
    global_indices: list[int],
) -> tuple[list[dict], list[dict], dict]:
    """Diagnose and roll out a list of windows. Returns diags, rolls, meta extras."""
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(0)
    dtype = torch.float64 if args.model_dtype == "float64" else torch.float32

    model, ema_vit = load_model(args.checkpoint, device)
    assert_l2(model)
    if dtype == torch.float64:
        ema_vit.double()
    weight_hash = _hash_module(ema_vit)
    failed: list[str] = []

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
    ds = int(model.ds_factors[0])

    first_frames, _ = _load_window_from_entry(window_entries[0])
    named_points = [("step0", make_point(first_frames[:3], model, device, dtype, args.norm))]
    later_k = args.rollout_length - 1
    if later_k > 0 and len(first_frames) >= later_k + 3:
        named_points.append(
            (f"step{later_k}", make_point(first_frames[later_k : later_k + 3], model, device, dtype, args.norm))
        )
    first = named_points[0][1]
    step0 = lambda u, z, _t=first.t: model_step(ema_vit, u, z, _t)

    def graph_w(point, vec, _tangent=tangent, _ds=ds, _norm=args.norm):
        return apply_D(_phi_image(_tangent, point.u, vec, _norm), _ds)

    sweep = validate_fd_eps(
        step0,
        named_points,
        args.fd_eps,
        args.fd_tol,
        args.seed,
        ds,
        graph_w=graph_w,
    )
    print(
        f"[fd] working eps={args.fd_eps} ok={sweep['ok']} "
        f"failed={sweep['n_failed']}/{sweep['n_checks']}",
        flush=True,
    )

    diags: list[dict] = []
    rolls: list[dict] = []
    for local_i, (entry, gidx) in enumerate(zip(window_entries, global_indices)):
        frames, stems = _load_window_from_entry(entry)
        point = make_point(frames[:3], model, device, dtype, args.norm)
        point.stems = tuple(stems[:3])  # type: ignore[assignment]
        step_fn = lambda u, z, _t=point.t: model_step(ema_vit, u, z, _t)
        print(
            f"[snap {gidx}] stems={stems[:3]} diagnose",
            flush=True,
        )
        diag = diagnose_point(
            step_fn,
            point,
            ds,
            args.fd_eps,
            tangent,
            args.dt,
            args.norm,
            args.n_probes,
            args.seed + int(gidx),
            args.spectral_radius,
            args.ncv,
            args.maxiter,
            args.tol,
        )
        diags.append(diag)
        print(
            f"[snap {gidx}] trA/N={diag['tr_A_over_N']:.4e} "
            f"trJ/N={diag['tr_J_over_N']:.4e} "
            f"amp={diag['amp_J']['mean']:.4e}",
            flush=True,
        )
        print(f"[roll {gidx}] length={args.rollout_length}", flush=True)
        sink: list | None = [] if local_i == 0 else None
        roll = rollout_diagnostics(
            ema_vit,
            frames,
            model,
            device,
            dtype,
            args.norm,
            args.fd_eps,
            tangent,
            args.rollout_length,
            direction_sink=sink,
        )
        if sink:
            extra = validate_fd_eps(
                step0, [], args.fd_eps, args.fd_tol, args.seed, ds,
                graph_w=graph_w, extra_directions=sink,
            )
            sweep = _merge_fd_reports(sweep, extra)
            print(
                f"[fd] with retention directions ok={sweep['ok']} "
                f"failed={sweep['n_failed']}/{sweep['n_checks']}",
                flush=True,
            )
        rolls.append(roll)

    if not sweep["ok"]:
        worst = sweep.get("worst") or {}
        failed.append(
            "fd validation failed for the selected eps "
            f"{args.fd_eps}: {sweep['n_failed']}/{sweep['n_checks']} checks, "
            f"worst {worst.get('name')}/{worst.get('block')} "
            f"rel={worst.get('rel')}"
        )

    after = _hash_module(ema_vit)
    if after != weight_hash:
        failed.append("checkpoint weights changed during the diagnostic")

    extras = {
        "weight_sha256": weight_hash,
        "weights_unchanged": after == weight_hash,
        "fd_validated": bool(sweep["ok"]),
        "fd_scaling": FD_SCALING,
        "fd_sweep": sweep,
        "allow_unvalidated_fd": bool(args.allow_unvalidated_fd),
        "failed_tests": failed,
        "downsampling": f"avg_pool2d kernel=stride={ds}",
        "physical_jvp": tangent.name,
        "is_exact_physical_jvp": False,
        "py2d_root": getattr(tangent, "py2d_root", None),
        "physical_jvp_note": "finite-difference estimates are not exact tangents",
        "num_coarse": int(model.num_coarse),
        "latent_res": list(model.latent_res),
        "global_indices": [int(g) for g in global_indices],
        "windows": [
            {"start": e["start_index"], "stems": e["stems"]} for e in window_entries
        ],
    }
    return diags, rolls, extras


def save_shard(
    save_dir: Path,
    rank: int,
    diags: list[dict],
    rolls: list[dict],
    extras: dict,
    args: argparse.Namespace,
) -> Path:
    diag_np, roll_np = _stack_results(
        diags, rolls, args.spectral_radius, args.dt
    )
    out = {
        **diag_np,
        **{f"roll_{k}": v for k, v in roll_np.items()},
        "global_indices": np.asarray(extras["global_indices"], dtype=np.int64),
    }
    path = save_dir / f"shard_{rank}.npz"
    np.savez_compressed(path, **out)
    (save_dir / f"shard_{rank}_meta.json").write_text(
        json.dumps(extras, indent=2) + "\n"
    )
    return path


def merge_shards(
    save_dir: Path, n_gpus: int, n_snapshots: int, spectral_radius: bool, dt: float
) -> tuple[dict, dict, list[dict]]:
    by_index: dict[int, dict] = {}
    metas: list[dict] = []
    for rank in range(n_gpus):
        shard_path = save_dir / f"shard_{rank}.npz"
        meta_path = save_dir / f"shard_{rank}_meta.json"
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing shard: {shard_path}")
        data = np.load(shard_path, allow_pickle=True)
        meta = json.loads(meta_path.read_text())
        metas.append(meta)
        indices = np.asarray(data["global_indices"]).astype(int)
        for local, gidx in enumerate(indices):
            entry = {"gidx": int(gidx)}
            for key in DIAG_KEYS:
                entry[key] = data[key][local]
            if spectral_radius:
                for key in RHO_KEYS:
                    entry[key] = data[key][local]
            for key in ROLL_KEYS:
                entry[f"roll_{key}"] = data[f"roll_{key}"][local]
            entry["approximation"] = np.asarray(data["approximation"]).item()
            by_index[int(gidx)] = entry

    missing = [i for i in range(n_snapshots) if i not in by_index]
    if missing:
        raise RuntimeError(f"missing snapshot indices after merge: {missing}")

    ordered = [by_index[i] for i in range(n_snapshots)]
    diag_np = {key: np.stack([e[key] for e in ordered]) for key in DIAG_KEYS}
    if spectral_radius:
        for key in RHO_KEYS:
            diag_np[key] = np.stack([e[key] for e in ordered])
    diag_np["dt"] = np.asarray(dt)
    diag_np["approximation"] = np.asarray(ordered[0]["approximation"])
    roll_np = {
        key: np.stack([e[f"roll_{key}"] for e in ordered]) for key in ROLL_KEYS
    }
    return diag_np, roll_np, metas


def write_final(
    save_dir: Path,
    args: argparse.Namespace,
    diag_np: dict,
    roll_np: dict,
    samples: list[dict],
    metas: list[dict],
) -> None:
    np.savez_compressed(
        save_dir / "results.npz",
        **diag_np,
        **{f"roll_{k}": v for k, v in roll_np.items()},
    )
    validated = all(bool(m.get("fd_validated", False)) for m in metas)
    write_plots(save_dir, diag_np, roll_np, validated=validated)

    failed: list[str] = []
    for m in metas:
        failed.extend(m.get("failed_tests", []))
    primary = metas[0]
    meta = {
        "task": "L2 Task 1 direct-step diagnostics",
        "checkpoint": args.checkpoint,
        "weight_sha256": primary["weight_sha256"],
        "weights_unchanged": all(m.get("weights_unchanged", False) for m in metas),
        "eval_dir": args.eval_dir,
        "every_nth_field": args.every_nth_field,
        "windows": [
            {"start": s["start_index"], "stems": s["stems"]} for s in samples
        ],
        "norm": args.norm,
        "dt_emulator": args.dt,
        "dt_meaning": "one emulator step; default 0.25 = five 0.05 snapshots",
        "downsampling": primary["downsampling"],
        "physical_jvp": primary["physical_jvp"],
        "is_exact_physical_jvp": False,
        "py2d_root": primary.get("py2d_root"),
        "physical_jvp_note": primary.get("physical_jvp_note"),
        "difference_quotient": (
            "F=(g-u)/dt and Tr(DB)/(N dt) are labels only; the network is direct-step. "
            "tr_B_DPhi_n is the physical feedback trace and is not Tr(DB)."
        ),
        "first_factor": (
            "r_H and r_z are a truncated one-factor approximation. "
            "A large reconstruction residual is not automatically an implementation bug."
        ),
        "amplification": (
            "amp_J is the mean of ||Jv||^2/N over Rademacher probes. "
            "It is mean-square directional amplification, not contraction in every direction."
        ),
        "seed": args.seed,
        "fd_eps": args.fd_eps,
        "fd_validated": validated,
        "allow_unvalidated_fd": bool(args.allow_unvalidated_fd),
        "fd_scaling": primary.get("fd_scaling"),
        "fd_sweep": primary.get("fd_sweep"),
        "n_probes": args.n_probes,
        "n_snapshots": args.n_snapshots,
        "rollout_length": args.rollout_length,
        "n_gpus": args.n_gpus,
        "spectral_radius": bool(args.spectral_radius),
        "model_dtype": args.model_dtype,
        "py2d": {
            "Re": args.re,
            "fkx": args.fkx,
            "fky": args.fky,
            "alpha": args.alpha,
            "beta": args.beta,
            "solver_dt": args.solver_dt,
        },
        "failed_tests": failed,
        "num_coarse": primary["num_coarse"],
        "latent_res": primary["latent_res"],
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[done] save_dir={save_dir} failed={len(failed)} fd_validated={validated}",
        flush=True,
    )
    if failed:
        for item in failed:
            print(f"  FAIL {item}", flush=True)
    if not validated and not args.allow_unvalidated_fd:
        raise SystemExit(2)


def run_single(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Restore visibility for in-process single-GPU path.
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    window_len = args.rollout_length + 2
    starts = choose_sample_starts(
        len(kept), args.rollout_length + 1, args.n_snapshots, args.seed
    )
    samples = write_sample_index(save_dir, kept, starts, window_len)
    diags, rolls, extras = run_windows(
        args, samples, list(range(len(samples)))
    )
    diag_np, roll_np = _stack_results(
        diags, rolls, args.spectral_radius, args.dt
    )
    write_final(save_dir, args, diag_np, roll_np, samples, [extras])


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu_rank is None:
        raise ValueError("--gpu-rank is required with --worker")
    save_dir = Path(args.save_dir)
    payload = json.loads(sample_index_path(save_dir).read_text())
    samples = payload["samples"]
    mine = [
        (s, int(s["sample"]))
        for s in samples
        if int(s["sample"]) % args.n_gpus == args.gpu_rank
    ]
    if not mine:
        print(f"[worker rank={args.gpu_rank}] no windows assigned", flush=True)
        # Still write an empty-compatible stub? Prefer error if n_snapshots < n_gpus
        raise RuntimeError(
            f"rank {args.gpu_rank} got zero windows "
            f"(n_snapshots={len(samples)} n_gpus={args.n_gpus})"
        )
    entries = [m[0] for m in mine]
    indices = [m[1] for m in mine]
    print(
        f"[worker rank={args.gpu_rank}] windows={indices}",
        flush=True,
    )
    diags, rolls, extras = run_windows(args, entries, indices)
    path = save_shard(save_dir, args.gpu_rank, diags, rolls, extras, args)
    print(
        f"[worker rank={args.gpu_rank}] wrote {path}",
        flush=True,
    )


def _pump(stream, prefix: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    stream.close()


def _common_worker_cmd(args: argparse.Namespace, script: str, rank: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        script,
        "--worker",
        "--gpu-rank",
        str(rank),
        "--n-gpus",
        str(args.n_gpus),
        "--checkpoint",
        args.checkpoint,
        "--eval-dir",
        args.eval_dir,
        "--save-dir",
        str(args.save_dir),
        "--n-probes",
        str(args.n_probes),
        "--n-snapshots",
        str(args.n_snapshots),
        "--rollout-length",
        str(args.rollout_length),
        "--fd-eps",
        str(args.fd_eps),
        "--fd-tol",
        str(args.fd_tol),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--physical-jvp",
        args.physical_jvp,
        "--dt",
        str(args.dt),
        "--norm",
        str(args.norm),
        "--every-nth-field",
        str(args.every_nth_field),
        "--model-dtype",
        args.model_dtype,
        "--ncv",
        str(args.ncv),
        "--maxiter",
        str(args.maxiter),
        "--tol",
        str(args.tol),
        "--re",
        str(args.re),
        "--fkx",
        str(args.fkx),
        "--fky",
        str(args.fky),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--solver-dt",
        str(args.solver_dt),
    ]
    cmd.append("--fd-eps-list")
    cmd.extend(str(eps) for eps in args.fd_eps_list)
    if args.spectral_radius:
        cmd.append("--spectral-radius")
    if args.allow_unvalidated_fd:
        cmd.append("--allow-unvalidated-fd")
    return cmd


def run_orchestrator(args: argparse.Namespace) -> None:
    if args.n_gpus < 1:
        raise ValueError("--n-gpus must be >= 1")
    if args.n_snapshots < args.n_gpus:
        raise ValueError(
            f"--n-snapshots ({args.n_snapshots}) must be >= --n-gpus ({args.n_gpus})"
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    window_len = args.rollout_length + 2
    starts = choose_sample_starts(
        len(kept), args.rollout_length + 1, args.n_snapshots, args.seed
    )
    samples = write_sample_index(save_dir, kept, starts, window_len)
    print(
        f"[orchestrator] n_snapshots={len(samples)} n_gpus={args.n_gpus} "
        f"save_dir={save_dir}",
        flush=True,
    )

    script = str(Path(__file__).resolve())
    procs: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []
    for rank in range(args.n_gpus):
        cmd = _common_worker_cmd(args, script, rank)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(
            f"[orchestrator] launching rank {rank} CUDA_VISIBLE_DEVICES={rank}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(
            target=_pump,
            args=(proc.stdout, f"[rank{rank}] "),
            daemon=True,
        )
        t.start()
        pumps.append(t)

    codes = [proc.wait() for proc in procs]
    for t in pumps:
        t.join(timeout=5)
    if any(c != 0 for c in codes):
        raise RuntimeError(f"worker exit codes: {codes}")

    diag_np, roll_np, metas = merge_shards(
        save_dir, args.n_gpus, args.n_snapshots, args.spectral_radius, args.dt
    )
    write_final(save_dir, args, diag_np, roll_np, samples, metas)
    print("[orchestrator] merge complete", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    elif args.n_gpus <= 1:
        run_single(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"FATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
