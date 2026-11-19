#!/usr/bin/env python3
"""Task 1 diagnostics for a trained direct-step L=2 HINE checkpoint.

Network blocks are torch.autograd.functional.jvp. The physical tangent is
jax.jvp of the 512 DNS startup advance. A failed JVP is an error. It is not
replaced by a centered difference. Py2DFDTangent remains available for other
tasks and for explicitly labeled validation checks.

Proxy mode (the argparse default, not the production run) uses D Phi' ≈ D.
--production selects float64, py2d, 20 windows, rollout 40, 32 probes,
dt=0.05, and the corrected sample index unless those flags are also passed.

Does not retrain, does not edit weights, and does not treat HINE as Euler.

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

def _parent_hides_gpus(argv: list[str]) -> bool:
    """Multi-GPU parent must not bind devices. A 1-GPU run keeps them visible."""
    if "--worker" in argv:
        return False
    if "--n-gpus" not in argv:
        return False
    idx = argv.index("--n-gpus")
    if idx + 1 >= len(argv):
        return False
    try:
        return int(argv[idx + 1]) > 1
    except ValueError:
        return False


# Parent orchestrator must not bind GPUs; workers set CUDA_VISIBLE_DEVICES.
if _parent_hides_gpus(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

# Match the float64 finite-difference audit: no autocast and no TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "ieee"
if hasattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision"):
    torch.backends.cudnn.conv.fp32_precision = "ieee"

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_jacobian_blocks import choose_sample_starts

# That import clears CUDA_VISIBLE_DEVICES unless --worker is set. The 4-GPU
# parent must stay unbound. A one-GPU run must keep the GPU srun assigned.
if (
    not _parent_hides_gpus(sys.argv)
    and "--worker" not in sys.argv
    and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
):
    del os.environ["CUDA_VISIBLE_DEVICES"]

from eval_jacobian_manual_mse import load_kept_paths
from eval_jacobian_manual_mse_plus import load_model
from l2_fd_jvp import (
    DEFAULT_EPS_LIST,
    FD_OUTPUT_FLOOR,
    FD_SCALING,
    apply_D,
    assert_l2,
    jvp_direction_report,
    make_point,
    model_step,
)
from l2_operators import _phi_image, diagnose_point, flat_diag, gates_pass, quotient_halfwidth
from l2_physical_jvp import load_dns_frame, make_tangent, validate_physical_reference
from l2_retention import (
    LATENT_CHANNEL_KEYS,
    LATENT_LIN_KEYS,
    relative_residual,
    rollout_diagnostics,
)


DIAG_KEYS = [
    "tr_A_over_N",
    "tr_J_over_N",
    "tr_K_over_M",
    "tr_S_over_M",
    "tr_DB_over_N",
    "tr_B_DPhi_n_over_N",
    "tr_Fu_over_N",
        "tr_DFz_over_N",
        "tr_B_DPhi_n_quot",
        "tr_Heff_over_N",
    "tr_Sigma_over_M",
    "composition_rel_mean",
    "composition_rel_max",
    "trace_estimator_gap",
    "trace_estimator_gap_ci95",
    "fd_additivity_rel",
    "s_additivity_rel",
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
    "tr_DPhi_next_B_std",
    "tr_DPhi_next_B_ci95",
    "tr_DPhi_next_B_probes",
    "tr_K_mean",
    "tr_K_std",
    "tr_K_ci95",
    "tr_K_probes",
    "tr_DB_probes",
    "tr_B_DPhi_n_std",
    "tr_B_DPhi_n_probes",
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
    "tr_Ez_over_M",
    "tr_minus_Qnext_B_over_Mdt",
    "tr_S_split_probe_max",
    "tr_C_Dplus_quot",
    "gain_C_over_N",
    "gain_Eu_sq_over_N",
    "tr_C_Dplus_mean",
    "tr_C_Dplus_std",
    "tr_C_Dplus_ci95",
    "tr_C_Dplus_probes",
    "gain_C_mean",
    "gain_C_std",
    "gain_C_ci95",
    "gain_C_probes",
]
RHO_KEYS = ["rho_S", "rho_S_real", "rho_S_imag", "rho_S_converged", "rho_S_residual"]
ROLL_KEYS = (
    "I",
    "R",
    "e_norm",
    "rho_norm",
    "rH_norm",
    "rZ_norm",
    "recon_norm",
    "cos_rH_rZ",
    "cos_e_rho",
    "cos_e_rH",
    "cos_e_rZ",
    "cos_rho_rH",
    "cos_rho_rZ",
    "cos_e_inj",
    "cos_rho_inj",
    "cos_rH_inj",
    "cos_rZ_inj",
    "cos_eta_inj",
    "cos_eta_eta_lin",
    "eps_u_norm",
    "eps_z_norm",
    "e_lin_norm",
    "eta_norm",
    "eta_lin_norm",
    "e_lin_err",
    "eta_lin_err",
    "e_lin_rel",
    "eta_lin_rel",
    "recon_lin_norm",
    "cos_e_e_lin",
    "eps_z_sum_norm",
    "xi_sum_norm",
    "xi_norm",
    "nu_norm",
    "cos_eta_eps_z",
    "cos_nu_xi",
    "nu_available",
    *LATENT_CHANNEL_KEYS,
    *LATENT_LIN_KEYS,
)

PRODUCTION_SAMPLE_INDEX = (
    "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/"
    "task1_l2_n20_float64_py2d_dt05/sample_index.json"
)
FLOW_REPLAY_TOL = 1e-3
JVP_TOL = 1e-6
EIGEN_RESIDUAL_TOL = 1e-3


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
    p.add_argument("--dt", type=float, default=0.05, help="Emulator step in time units")
    p.add_argument(
        "--sample-index",
        default="",
        help="Reuse windows from this sample_index.json instead of drawing new ones",
    )
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--model-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--re", type=float, default=1e4)
    p.add_argument("--fkx", type=int, default=4)
    p.add_argument("--fky", type=int, default=4)
    p.add_argument(
        "--dns-data",
        default="",
        help="512 DNS directory. Empty keeps the fkx=4 paper data. Sets HINE_DNS_DATA.",
    )
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
    p.add_argument(
        "--production",
        action="store_true",
        help=(
            "Select the completed-run configuration: float64, py2d, 20 windows, "
            "rollout 40, 32 probes, dt=0.05, spectral radius, corrected sample index. "
            "An explicit flag on the command line still wins."
        ),
    )
    args = p.parse_args(argv)
    return apply_production(args, argv)


def apply_production(args: argparse.Namespace, argv=None) -> argparse.Namespace:
    """Fill production settings only for flags the user did not pass."""
    if not getattr(args, "production", False):
        return args
    seen = list(sys.argv[1:] if argv is None else argv)

    def absent(flag: str) -> bool:
        return flag not in seen

    if absent("--model-dtype"):
        args.model_dtype = "float64"
    if absent("--physical-jvp"):
        args.physical_jvp = "py2d"
    if absent("--n-snapshots"):
        args.n_snapshots = 20
    if absent("--rollout-length"):
        args.rollout_length = 40
    if absent("--n-probes"):
        args.n_probes = 32
    if absent("--dt"):
        args.dt = 0.05
    if absent("--spectral-radius"):
        args.spectral_radius = True
    if absent("--sample-index"):
        args.sample_index = PRODUCTION_SAMPLE_INDEX
    return args


def _hash_module(mod: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for tensor in mod.state_dict().values():
        arr = tensor.detach().float().cpu().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def _mat_frame(path: str) -> tuple[np.ndarray, float]:
    raw = scipy.io.loadmat(path)
    omega = raw["Omega"]
    if "time" not in raw:
        raise RuntimeError(f"{path} has no time value; refusing to infer spacing from the filename")
    return omega, float(np.asarray(raw["time"]).reshape(-1)[0])


def _check_spacing(times: list[float], dt: float, label: str) -> None:
    for left, right in zip(times, times[1:]):
        gap = float(right) - float(left)
        if abs(gap - float(dt)) > 1e-6:
            raise RuntimeError(f"{label} frame spacing {gap} != --dt {dt}")


def _load_window(kept, start: int, length: int) -> tuple[list[np.ndarray], list[int]]:
    frames = []
    stems = []
    for j in range(length):
        stem, path = kept[start + j]
        omega, _time = _mat_frame(path)
        frames.append(omega)
        stems.append(int(stem))
    return frames, stems


def _load_window_from_entry(entry: dict) -> tuple[list[np.ndarray], list[int], list[float]]:
    frames = []
    times = []
    for path in entry["paths"]:
        omega, time = _mat_frame(path)
        frames.append(omega)
        times.append(time)
    stems = [int(s) for s in entry["stems"]]
    return frames, stems, times


def _attach_dns(point, stem_now: int, stem_next: int) -> None:
    now = load_dns_frame(stem_now)
    nxt = load_dns_frame(stem_next)
    point.dns = now["omega"]
    point.hats = now["hats"]
    point.dns_next = nxt["omega"]
    point.hats_next = nxt["hats"]


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


def _select_samples(args, kept, save_dir: Path) -> list[dict]:
    """Draw windows, or copy a previous sample index when one is given."""
    window_len = args.rollout_length + 2
    if args.sample_index:
        payload = json.loads(Path(args.sample_index).read_text())
        samples = list(payload["samples"])[: int(args.n_snapshots)]
        if len(samples) < int(args.n_snapshots):
            raise ValueError(
                f"sample index has {len(payload['samples'])} windows, "
                f"asked for {args.n_snapshots}"
            )
        sample_index_path(save_dir).write_text(
            json.dumps(
                {"n_samples": len(samples), "window_len": window_len, "samples": samples, "reused_from": args.sample_index},
                indent=2,
            )
            + "\n"
        )
        return samples
    starts = choose_sample_starts(
        len(kept), args.rollout_length + 1, args.n_snapshots, args.seed
    )
    return write_sample_index(save_dir, kept, starts, window_len)


class DiagnosticsAborted(RuntimeError):
    """Initial epsilon check failed and the override is off."""

    def __init__(self, extras: dict):
        super().__init__("fd validation failed before diagnostics")
        self.extras = extras


def _errbar(ax, x, y, yerr, label):
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)


def _run_extras(
    args,
    tangent,
    ds,
    weight_before,
    weight_after,
    sweep,
    failed,
    window_entries,
    global_indices,
    *,
    model_fd_validated,
    physical_tangent_validated,
    spectral_validated,
    fd_validated,
    physical_flow,
    aborted_before_diagnostics,
    num_coarse,
    latent_res,
) -> dict:
    return {
        "weight_sha256": weight_before,
        "weights_unchanged": weight_after == weight_before,
        "model_fd_validated": bool(model_fd_validated),
        "physical_tangent_validated": physical_tangent_validated,
        "spectral_validated": spectral_validated,
        "fd_validated": bool(fd_validated),
        "fd_scaling": FD_SCALING,
        "fd_sweep": sweep,
        "physical_flow": physical_flow,
        "allow_unvalidated_fd": bool(args.allow_unvalidated_fd),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "aborted_before_diagnostics": bool(aborted_before_diagnostics),
        "failed_tests": failed,
        "downsampling": f"avg_pool2d kernel=stride={ds}",
        "physical_jvp": tangent.name,
        "is_exact_physical_jvp": False,
        "py2d_root": getattr(tangent, "py2d_root", None),
        "physical_jvp_note": (
            "Network blocks use torch.autograd.functional.jvp. "
            "The physical tangent is jax.jvp of the 512 single-field startup "
            "advance (Omega0 and convec0 are functions of that same field; no "
            "previous snapshot is held fixed). Finite differences are not a "
            "production fallback. Py2DFDTangent is validation-only or for other tasks."
        ),
        "tolerances": {
            "flow_replay": FLOW_REPLAY_TOL,
            "jvp_additivity": JVP_TOL,
            "eigen_residual": EIGEN_RESIDUAL_TOL,
            "legacy_fd_tol": float(args.fd_tol),
            "eigen_note": (
                "The eigenpair residual checks the pair ARPACK returned. "
                "A small residual does not prove that the value is dominant."
            ),
        },
        "num_coarse": num_coarse,
        "latent_res": latent_res,
        "global_indices": [int(g) for g in global_indices],
        "windows": [
            {"start": e["start_index"], "stems": e["stems"]} for e in window_entries
        ],
    }


def _status_all(values: list):
    if not values:
        return False
    labels = {"not_applicable", "not_requested", "not_checked"}
    if all(v in labels for v in values) and len(set(values)) == 1:
        return values[0]
    bools = [v for v in values if isinstance(v, bool)]
    return all(bools) if bools else False


def run_succeeded(fd_validated: bool, spectral_validated, spectral_requested: bool) -> bool:
    """Finite-difference success, and the spectral gate when Arnoldi was requested.

    fd_validated stays the finite-difference flag. A failed spectral radius
    does not change it, but it does fail the job.
    """
    if not bool(fd_validated):
        return False
    if not spectral_requested:
        return True
    return spectral_validated is True


def plot_validation_flags(fd_validated: bool, spectral_validated, spectral_requested: bool) -> tuple[bool, bool]:
    """The spectral plot is validated only when the operator gate passes too.

    fd_validated is the operator-validation flag (JVP additivity and composition
    on the current path, plus the physical replay when that tangent is on).
    A converged eigenpair does not validate rho_S if the operator gate failed.
    """
    operator_ok = bool(fd_validated)
    if spectral_requested:
        spectral_plot = operator_ok and spectral_validated is True
    else:
        spectral_plot = operator_ok
    return operator_ok, spectral_plot


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


def percentile_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean of defined values and the empirical 16th–84th percentile.

    The band is the spread across windows. It is not a confidence interval.
    An all-undefined step stays NaN. n_valid counts finite windows at that step.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mean = np.full(arr.shape[1], np.nan, dtype=np.float64)
    lo = np.full(arr.shape[1], np.nan, dtype=np.float64)
    hi = np.full(arr.shape[1], np.nan, dtype=np.float64)
    n_valid = np.zeros(arr.shape[1], dtype=np.int64)
    for j in range(arr.shape[1]):
        good = arr[:, j][np.isfinite(arr[:, j])]
        n_valid[j] = int(good.size)
        if good.size:
            mean[j] = float(np.mean(good))
            lo[j] = float(np.percentile(good, 16))
            hi[j] = float(np.percentile(good, 84))
    return mean, lo, hi, n_valid


def _defined_mean(values: np.ndarray) -> np.ndarray:
    """Mean across windows. A step that is undefined in every window stays NaN."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        good = col[np.isfinite(col)]
        if good.size:
            out[j] = float(good.mean())
    return out


def _optional(diag: dict, key: str):
    if key not in diag:
        return None
    return np.asarray(diag[key], dtype=np.float64)


def trace_rows(diag: dict) -> tuple[tuple, tuple]:
    """Raw traces and difference quotients with probe-uncertainty bars."""
    dt = float(np.asarray(diag["dt"]))
    n_fine = np.asarray(diag["n_fine"], dtype=np.float64)
    n_coarse = np.asarray(diag["n_coarse"], dtype=np.float64)
    tr_k = np.asarray(diag["tr_K_over_M"], dtype=np.float64) * n_coarse
    k_ci = _optional(diag, "tr_K_ci95")
    raw = (
        (r"$\mathrm{Tr}\,A$", diag["tr_A_mean"], diag["tr_A_ci95"]),
        (r"$\mathrm{Tr}\,K$", tr_k, k_ci),
        (r"$\mathrm{Tr}\,J_{\mathrm{graph}}$", diag["tr_J_mean"], diag["tr_J_ci95"]),
        (r"$\mathrm{Tr}\,S$", diag["tr_S_mean"], diag["tr_S_ci95"]),
        (r"$\mathrm{Tr}(DB)$ proxy", diag["tr_DB_mean"], diag["tr_DB_ci95"]),
        (r"$\mathrm{Tr}(B D\Phi'_n)$", diag["tr_B_DPhi_n_mean"], diag["tr_B_DPhi_n_ci95"]),
        (
            r"$\mathrm{Tr}(D\Phi'_{n+1} B)$",
            diag["tr_DPhi_next_B_mean"],
            diag["tr_DPhi_next_B_ci95"],
        ),
    )
    # Titles are the HTML Euler names. They equal the y-axis quotients only
    # for a forward-Euler step. The trained map is direct-step, so the titles
    # name these quotients. "proxy" is DB with Phi' replaced by I.
    quotients = (
        (
            r"$(\mathrm{Tr}\,A_n-N)/(N\Delta t)$",
            diag["tr_Fu_over_N"],
            quotient_halfwidth(diag["tr_A_ci95"], n_fine, dt),
            r"$\mathrm{Tr}(F_u)/N$",
        ),
        (
            r"$(\mathrm{Tr}\,K_n-M)/(M\Delta t)$",
            (np.asarray(diag["tr_K_over_M"], dtype=np.float64) - 1.0) / dt,
            None if k_ci is None else k_ci / (n_coarse * dt),
            r"$\mathrm{Tr}(E_z)/M$",
        ),
        (
            r"$(\mathrm{Tr}(A_n+B_n D\Phi'_n)-N)/(N\Delta t)$",
            diag["tr_Heff_over_N"],
            quotient_halfwidth(diag["tr_J_ci95"], n_fine, dt),
            r"$\mathrm{Tr}(H_{\mathrm{eff}})/N$",
        ),
        (
            r"$(\mathrm{Tr}\,S_n-M)/(M\Delta t)$",
            diag["tr_Sigma_over_M"],
            quotient_halfwidth(diag["tr_S_ci95"], n_coarse, dt),
            r"$\mathrm{Tr}(\Sigma)/M$",
        ),
        (
            r"$\mathrm{Tr}(DB_n)/(N\Delta t)$",
            diag["tr_DFz_over_N"],
            quotient_halfwidth(diag["tr_DB_ci95"], n_fine, dt),
            r"$\mathrm{Tr}(DF_z)/N$ proxy",
        ),
        (
            r"$\mathrm{Tr}(B_n D\Phi'_n)/(N\Delta t)$",
            diag["tr_B_DPhi_n_quot"],
            quotient_halfwidth(diag["tr_B_DPhi_n_ci95"], n_fine, dt),
            r"$\mathrm{Tr}(F_z D\Phi'_n)/N$",
        ),
        (
            r"$\mathrm{Tr}(D\Phi'_{n+1} B_n)/(M\Delta t)$",
            np.asarray(diag["tr_DPhi_next_B_mean"], dtype=np.float64) / (n_coarse * dt),
            np.asarray(diag["tr_DPhi_next_B_ci95"], dtype=np.float64) / (n_coarse * dt),
            r"$\mathrm{Tr}(D\Phi'_{n+1} F_z)/M$",
        ),
    )
    return raw, quotients


def _write_trace_panels(plot_dir: Path, diag: dict, validated: bool) -> None:
    """Raw traces and difference quotients, one scale each.

    Error bars are Hutchinson probe uncertainty at that state. They are not
    the spread across windows, and a negative trace is not a contraction.
    """
    idx = np.arange(np.asarray(diag["tr_A_over_N"]).shape[0])
    raw, quotients = trace_rows(diag)

    def _draw(path: Path, rows, title: str, *, footnote: str | None = None) -> None:
        titled = any(len(row) > 3 for row in rows)
        height = (2.45 if titled else 2.1) * len(rows)
        fig, axes = plt.subplots(len(rows), 1, figsize=(8.2, height), sharex=True)
        for ax, row in zip(axes, rows):
            label, y, err = row[0], row[1], row[2]
            _errbar(ax, idx, y, err, label)
            ax.set_ylabel(label, fontsize=9)
            if len(row) > 3:
                ax.set_title(row[3], fontsize=10)
        axes[-1].set_xlabel("window")
        fig.suptitle(_title(title, validated))
        if footnote:
            fig.text(0.5, 0.005, footnote, ha="center", va="bottom", fontsize=8)
            fig.tight_layout(rect=(0, 0.07, 1, 0.98))
        else:
            fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    _draw(
        plot_dir / "traces_raw.png",
        raw,
        "Hutchinson traces. Bars: probe uncertainty, not a contraction test",
    )
    quotient_note = (
        "Titles are the Euler tendency names. They equal the y-axis quotients\n"
        "only for a forward-Euler step. This checkpoint is direct-step, so the titles name the quotients.\n"
        r"$\mathrm{Tr}(DF_z)/N$ uses $DB$ with $\Phi'\approx I$ and is not the physical feedback."
        "\n"
        r"$\mathrm{Tr}(D\Phi'_{n+1} F_z)/M$ is the coarse feedback inside $S$, divided by $M$, and is not $\mathrm{Tr}(DF_z)/N$."
    )
    _draw(
        plot_dir / "traces_quotients.png",
        quotients,
        "Difference quotients. Bars: probe uncertainty, not a contraction test",
        footnote=quotient_note,
    )
    stems = ("Fu", "Ez", "Heff", "Sigma", "DFz", "FzDPhi", "DPhiFz")
    for stem, row in zip(stems, quotients):
        label, y, err, panel_title = row
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _errbar(ax, idx, y, err, label)
        ax.set_xlabel("window")
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(_title(panel_title, validated))
        fig.tight_layout()
        fig.savefig(plot_dir / f"traces_quotient_{stem}.png", dpi=140)
        plt.close(fig)
    # Seven quotients: two rows of four, last cell unused. A 4x4 would leave nine blanks.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True)
    flat = axes.ravel()
    for ax, row in zip(flat, quotients):
        label, y, err, panel_title = row
        _errbar(ax, idx, y, err, label)
        ax.set_title(panel_title, fontsize=10)
        ax.set_ylabel(label, fontsize=8)
    flat[-1].axis("off")
    for ax in axes[-1, :3]:
        ax.set_xlabel("window")
    fig.suptitle(_title(
        "Difference quotients. Bars: probe uncertainty, not a contraction test",
        validated,
    ))
    fig.text(0.5, 0.005, quotient_note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(plot_dir / "traces_quotients_grid.png", dpi=140)
    plt.close(fig)


def _write_s_argand(plot_dir: Path, diag: dict, spectral_validated: bool) -> None:
    """Leading eigenvalue of the parasitic block on the complex plane.

    Task 1 asks ARPACK for one largest-magnitude value. S is real, so the
    conjugate is an eigenvalue as well and is drawn with it. This is not a
    partial Ritz cloud.
    """
    re = np.asarray(diag["rho_S_real"], dtype=np.float64).reshape(-1)
    im = np.asarray(diag["rho_S_imag"], dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.45", lw=1.0, label=r"$|\lambda|=1$")
    ax.axhline(0.0, color="0.85", lw=0.6)
    ax.axvline(0.0, color="0.85", lw=0.6)
    ax.scatter(re, im, s=28, zorder=3, label=r"leading eigenvalue of $S$")
    conj = np.abs(im) > 1e-12
    if np.any(conj):
        ax.scatter(re[conj], -im[conj], s=28, zorder=3, marker="o", facecolors="none", edgecolors="C0", label="conjugate")
    ax.scatter([0.0], [0.0], marker="x", s=70, c="k", zorder=4, label="0")
    ax.scatter([1.0], [0.0], marker="+", s=120, c="C3", zorder=4, label="1")
    radius = float(max(np.max(np.hypot(re, im)), 1.0) * 1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(_title(r"Leading eigenvalue of parasitic $S$", spectral_validated))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "s_argand.png", dpi=140)
    plt.close(fig)


def write_plots(
    save_dir: Path,
    diag: dict,
    rolls: dict,
    validated: bool = True,
    spectral_validated: bool = True,
) -> None:
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(diag["tr_Fu_over_N"].shape[0])
    kind = str(np.asarray(diag["approximation"]).item())
    dt = float(np.asarray(diag["dt"]))
    fu_err = quotient_halfwidth(diag["tr_A_ci95"], diag["n_fine"], dt)

    direct_panels = (
        (diag["tr_Fu_over_N"], fu_err, r"$(\mathrm{Tr}\,A_n-N)/(N\Delta t)$"),
        (
            diag["tr_DFz_over_N"],
            quotient_halfwidth(diag["tr_DB_ci95"], diag["n_fine"], dt),
            r"$\mathrm{Tr}(DB_n)/(N\Delta t)$",
        ),
        (
            diag["tr_B_DPhi_n_quot"],
            quotient_halfwidth(diag["tr_B_DPhi_n_ci95"], diag["n_fine"], dt),
            r"$\mathrm{Tr}(B_n D\Phi'_n)/(N\Delta t)$",
        ),
        (
            diag["tr_Heff_over_N"],
            quotient_halfwidth(diag["tr_J_ci95"], diag["n_fine"], dt),
            r"$(\mathrm{Tr}(A_n+B_n D\Phi'_n)-N)/(N\Delta t)$",
        ),
        (diag["tr_J_over_N"], diag["tr_J_ci95"] / diag["n_fine"], r"$\mathrm{Tr}(J_{\mathrm{graph}})/N$"),
    )
    fig, axes = plt.subplots(len(direct_panels), 1, figsize=(8.2, 12), sharex=True)
    for ax, (y, yerr, label) in zip(axes, direct_panels):
        _errbar(ax, idx, y, yerr, label)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("snapshot")
    fig.suptitle(_title(f"Direct-step and difference-quotient traces ({kind})", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "trace_split.png", dpi=140)
    plt.close(fig)
    _write_trace_panels(plot_dir, diag, validated)

    parasitic_panels = (
        (
            diag["tr_Sigma_over_M"],
            quotient_halfwidth(diag["tr_S_ci95"], diag["n_coarse"], dt),
            r"$(\mathrm{Tr}\,S_n-M)/(M\Delta t)$",
        ),
        (diag["tr_S_over_M"], diag["tr_S_ci95"] / diag["n_coarse"], r"$\mathrm{Tr}(S_n)/M$"),
    )
    fig, axes = plt.subplots(len(parasitic_panels), 1, figsize=(8.2, 6), sharex=True)
    for ax, (y, yerr, label) in zip(axes, parasitic_panels):
        _errbar(ax, idx, y, yerr, label)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("snapshot")
    fig.suptitle(_title(f"Parasitic trace ({kind})", validated))
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
        ax.set_title(_title(f"Parasitic spectral radius ({kind})", spectral_validated))
        fig.tight_layout()
        fig.savefig(plot_dir / "rho_S.png", dpi=140)
        plt.close(fig)
        _write_s_argand(plot_dir, diag, spectral_validated)

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
    ax.plot(lead, rolls["rH_norm"].mean(0), label=r"$\|r_H\|$ truncated first-factor")
    ax.plot(lead, rolls["rZ_norm"].mean(0), label=r"$\|r_z\|$ truncated first-factor")
    ax.plot(lead, rolls["rho_norm"].mean(0), label=r"$\|\rho_n\|$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Channel norms", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "channel_norms.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["e_norm"].mean(0), label=r"$\|e\|$")
    ax.plot(lead, rolls["rho_norm"].mean(0), label=r"$\|\rho_n\|$")
    ax.plot(lead, rolls["eta_norm"].mean(0), label=r"$\|\eta\|$")
    ax.plot(lead, rolls["R"].mean(0), label=r"$R_n=\|e_n\|-I_n$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Error norms", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "error_norms.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lead, rolls["e_norm"].mean(0), label=r"$\|e_n\|$")
    ax.plot(lead, rolls["I"].mean(0), label=r"$I_n=\|\sum_{j<n}\epsilon^{u}_j\|$")
    ax.plot(lead, rolls["rho_norm"].mean(0), label=r"$\|\rho_n\|$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title(r"Fine error, accumulated injection, and $\rho$", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "fine_injection_rho.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    alignment = (
        ("cos_e_rho", r"$\cos(e_n,\rho_n)$"),
        ("cos_e_rH", r"$\cos(e_n,r_H)$"),
        ("cos_e_rZ", r"$\cos(e_n,r_z)$"),
        ("cos_rho_rH", r"$\cos(\rho_n,r_H)$"),
        ("cos_rho_rZ", r"$\cos(\rho_n,r_z)$"),
    )
    for key, label in alignment:
        ax.plot(lead, _defined_mean(rolls[key]), label=label)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.set_title(_title("Cosine alignment; undefined cosines are gaps", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "retention_alignment.png", dpi=140)
    plt.close(fig)

    def _bundle(ax, series, label):
        """Aggregate only. The shade is the 16th–84th percentile, not a CI."""
        mean, lo, hi, _n_valid = percentile_band(series)
        ax.plot(lead, mean, label=label)
        ax.fill_between(lead, lo, hi, alpha=0.15)

    fig, ax = plt.subplots(figsize=(6, 4))
    _bundle(ax, rolls["rH_norm"], r"$\|r_H\|$ truncated")
    _bundle(ax, rolls["rZ_norm"], r"$\|r_z\|$ truncated")
    _bundle(ax, rolls["rho_norm"], r"$\|\rho\|$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Truncated first-factor channels", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "retention_channels.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _bundle(ax, rolls["recon_norm"], r"$\|\rho-r_H-r_z\|$")
    if "recon_lin_norm" in rolls:
        _bundle(ax, rolls["recon_lin_norm"], r"$\|\rho_{lin}-r_H-r_z\|$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Reconstruction residual", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "reconstruction_residual.png", dpi=140)
    plt.close(fig)

    rel = relative_residual(rolls["recon_norm"], rolls["rho_norm"], FD_OUTPUT_FLOOR)
    fig, ax = plt.subplots(figsize=(6, 4))
    _bundle(ax, rel, r"$\|\rho-r_H-r_z\| / \max(\|\rho\|, floor)$")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(_title("Relative reconstruction residual", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "relative_reconstruction.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    _bundle(axes[0, 0], rolls["e_norm"], r"$\|e\|$")
    _bundle(axes[0, 0], rolls["e_lin_norm"], r"$\|e_{lin}\|$")
    axes[0, 0].set_title(r"fine $\|e\|$")
    axes[0, 0].legend(fontsize=8)
    _bundle(axes[0, 1], rolls["eta_norm"], r"$\|\eta\|$")
    _bundle(axes[0, 1], rolls["eta_lin_norm"], r"$\|\eta_{lin}\|$")
    axes[0, 1].set_title(r"latent $\|\eta\|$")
    axes[0, 1].legend(fontsize=8)
    _bundle(axes[1, 0], rolls["e_lin_err"], r"$\|e-e_{lin}\|$")
    _bundle(axes[1, 0], rolls["eta_lin_err"], r"$\|\eta-\eta_{lin}\|$")
    axes[1, 0].set_title("absolute discrepancy")
    axes[1, 0].legend(fontsize=8)
    _bundle(axes[1, 1], rolls["e_lin_rel"], r"$\|e-e_{lin}\|/\|e\|$")
    _bundle(axes[1, 1], rolls["eta_lin_rel"], r"$\|\eta-\eta_{lin}\|/\|\eta\|$")
    axes[1, 1].set_title("relative discrepancy")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xlabel("step")
    fig.suptitle(_title("Actual and linearized errors", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "linearized_error.png", dpi=140)
    plt.close(fig)

    cosine_panels = (
        (
            "cosine_retention.png",
            "Retention and channel alignment",
            (
                ("cos_rH_rZ", r"$\cos(r_H, r_z)$"),
                ("cos_e_rho", r"$\cos(e_n, \rho_n)$"),
                ("cos_e_rH", r"$\cos(e_n, r_H)$"),
                ("cos_e_rZ", r"$\cos(e_n, r_z)$"),
                ("cos_rho_rH", r"$\cos(\rho_n, r_H)$"),
                ("cos_rho_rZ", r"$\cos(\rho_n, r_z)$"),
            ),
        ),
        (
            "cosine_fine_injection.png",
            "Fine injection",
            (
                ("cos_e_inj", r"$\cos(e_n, \sum_{j<n}\epsilon^{u}_j)$"),
                ("cos_rho_inj", r"$\cos(\rho_n, \sum_{j<n}\epsilon^{u}_j)$"),
                ("cos_rH_inj", r"$\cos(r_H, \sum_{j<n}\epsilon^{u}_j)$"),
                ("cos_rZ_inj", r"$\cos(r_z, \sum_{j<n}\epsilon^{u}_j)$"),
            ),
        ),
        (
            "cosine_latent_injection.png",
            "Off-graph injection",
            (
                ("cos_eta_inj", r"$\cos(\eta_n, \sum_{j<n}\xi_j)$"),
                ("cos_nu_xi", r"$\cos(\nu_n, \sum_{j<n}\xi_j)$"),
            ),
        ),
        (
            "cosine_linearized.png",
            "Actual versus linearized",
            (
                ("cos_e_e_lin", r"$\cos(e_n, e^{lin}_n)$"),
                ("cos_eta_eta_lin", r"$\cos(\eta_n, \eta^{lin}_n)$"),
            ),
        ),
    )
    n_valid = {}
    fig_all, axes_all = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (filename, title, series) in zip(axes_all.ravel(), cosine_panels):
        fig_one, ax_one = plt.subplots(figsize=(7, 4.2))
        for key, label in series:
            if key not in rolls:
                continue
            _bundle(ax, rolls[key], label)
            _bundle(ax_one, rolls[key], label)
            _mean, _lo, _hi, count = percentile_band(rolls[key])
            n_valid[key] = count
        for target in (ax, ax_one):
            target.set_ylim(-1.0, 1.0)
            target.set_xlabel("step")
            target.legend(fontsize=8)
            target.set_title(_title(title, validated))
        fig_one.tight_layout()
        fig_one.savefig(plot_dir / filename, dpi=140)
        plt.close(fig_one)
    fig_all.suptitle(
        _title(
            "Cosines. Shade: empirical 16th–84th percentile across windows, not a CI",
            validated,
        )
    )
    note = (
        r"$e_n=\hat{u}_n-u_n$, $\eta_n=\hat{z}_{n+1}-Du_{n+1}$, "
        r"$\nu_n=\eta_n-D\Phi'_n e_n$, $\rho_n=e_n-\sum_{j<n}\epsilon^{u}_j$."
        "\n"
        r"$r_H,r_z$: truncated first-factor channels. "
        r"$\epsilon^{u},\epsilon^{z}$: one-step heads. $\xi=\epsilon^{z}-D\Phi'_{n+1}\epsilon^{u}$."
        "\n"
        r"Legacy cos_eta_inj is $\cos(\eta_n,\sum_{j<n}\xi_j)$, not $\sum\epsilon^{z}$."
    )
    fig_all.text(0.01, 0.01, note, fontsize=8)
    fig_all.tight_layout(rect=(0, 0.08, 1, 1))
    fig_all.savefig(plot_dir / "cosine_all.png", dpi=140)
    plt.close(fig_all)
    if n_valid:
        np.savez(plot_dir / "cosine_n_valid.npz", **{key: np.asarray(val) for key, val in n_valid.items()})
    _write_latent_channel_plots(plot_dir, diag, rolls, validated)


def _write_latent_channel_plots(plot_dir: Path, diag: dict, rolls: dict, validated: bool) -> None:
    """Latent figures in the high-resolution panel layout.

    Telescoping channels use r_H^eta, r_z^eta, and rho^eta.
    First-factor channels stay on their own figure.
    Off-graph xi and nu are not drawn here.
    """
    if "p_norm" not in rolls or "tr_C_Dplus_quot" not in diag:
        return
    n_valid = {}

    telescoping_note = "\n".join((
        r"$r_H^{\eta}=\sum_{k<n} C_k e_k=\Delta t\sum_{k<n} E_{u,k} e_k$",
        r"$r_z^{\eta}=\sum_{k<n}(K_k-I)\eta_k=\Delta t\sum_{k<n} E_{z,k}\eta_k$",
        r"$\rho^{\eta}=\eta_n-\sum_{k<n}\varepsilon_k^z$",
        r"Linearized recursion, zero initial error: $\rho^{\eta}=r_H^{\eta}+r_z^{\eta}$.",
        r"Nonlinear rollout: the difference is the Taylor remainder.",
        r"These sums are not the first-factor channels.",
    ))
    first_factor_note = "\n".join((
        r"$(r_H^{\eta})^{[1]}=\sum_{k<n} C_k(\sum_{i<k}\varepsilon_i^u)$",
        r"$(r_z^{\eta})^{[1]}=\sum_{k<n}(K_k-I)(\sum_{i<k}\varepsilon_i^z)$",
        r"Truncated first-factor expansion. Not $r_H^{\eta}$ or $r_z^{\eta}$.",
    ))

    def _series(key: str) -> np.ndarray | None:
        if key not in rolls:
            return None
        return np.asarray(rolls[key], dtype=np.float64)

    def _paint(ax, series, label, color, band: bool):
        values = np.asarray(series, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        x = np.arange(values.shape[1])
        if band:
            mean, lo, hi, _count = percentile_band(values)
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        else:
            ax.plot(x, values.mean(0), color=color, label=label)

    def _finish(fig, ax, title, note: str | None, cosine: bool, filename: str):
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.set_title(_title(title, validated))
        if cosine:
            ax.set_ylim(-1.0, 1.0)
        if note:
            lines = [line for line in note.split("\n") if line]
            fig.text(0.01, 0.005, "\n".join(lines), fontsize=8, va="bottom")
            fig.tight_layout(rect=(0, min(0.34, 0.032 * len(lines) + 0.02), 1, 1))
        else:
            fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=140)
        plt.close(fig)

    def _panel(filename, title, rows, note=None, cosine=False, band=False, figsize=(6, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        drawn = False
        for index, (key, label) in enumerate(rows):
            series = _series(key) if isinstance(key, str) else key
            if series is None:
                continue
            _paint(ax, series, label, f"C{index}", band)
            if isinstance(key, str):
                _mean, _lo, _hi, count = percentile_band(series)
                n_valid[key] = count
            drawn = True
        if not drawn:
            plt.close(fig)
            return
        _finish(fig, ax, title, note, cosine, filename)

    if "eta_norm" in rolls and "I_z_norm" in rolls:
        rolls["R_eta"] = (
            np.asarray(rolls["eta_norm"], dtype=np.float64)
            - np.asarray(rolls["I_z_norm"], dtype=np.float64)
        )

    _panel(
        "latent_injection_rho.png",
        r"Latent error, accumulated $\varepsilon^z$, and $\rho^{\eta}$",
        (
            ("eta_norm", r"$\|\eta\|$"),
            ("I_z_norm", r"$\|\sum_{j<n}\varepsilon_j^z\|$"),
            ("rho_eta_norm", r"$\|\rho^{\eta}\|$"),
        ),
        note=telescoping_note,
        figsize=(6.6, 4.8),
    )
    _panel(
        "latent_injection_retention.png",
        r"Latent injection and scalar retention",
        (
            ("I_z_norm", r"$\|\sum_{j<n}\varepsilon_j^z\|$"),
            ("R_eta", r"$R_n^{\eta}=\|\eta_n\|-\|\sum_{k<n}\varepsilon_k^z\|$"),
        ),
        note=r"$R_n^{\eta}$ is the difference of the two norms, not the norm of $\rho^{\eta}$.",
        figsize=(6.6, 4.6),
    )
    channel_rows = (
        ("r_u_to_z_norm", r"$\|r_H^{\eta}\|$"),
        ("r_z_to_z_norm", r"$\|r_z^{\eta}\|$"),
        ("rho_eta_norm", r"$\|\rho^{\eta}\|$"),
    )
    _panel(
        "latent_channel_norms.png",
        r"Telescoping latent channel norms",
        channel_rows,
        note=telescoping_note,
        figsize=(6.6, 4.8),
    )
    _panel(
        "latent_retention_channels.png",
        r"Telescoping latent channels",
        channel_rows,
        note=telescoping_note,
        band=True,
        figsize=(6.6, 4.8),
    )
    _panel(
        "latent_first_factor_channels.png",
        r"First-factor latent channels",
        (
            ("r_u_to_z_first_norm", r"$\|(r_H^{\eta})^{[1]}\|$"),
            ("r_z_to_z_first_norm", r"$\|(r_z^{\eta})^{[1]}\|$"),
        ),
        note=first_factor_note,
        band=True,
        figsize=(6.6, 4.6),
    )
    remainder_note = (
        "\n".join((
            r"$\|\rho^{\eta}-r_H^{\eta}-r_z^{\eta}\|$ is the Taylor remainder.",
            r"That norm is not in these saved results.",
            r"The curve is the closure $\|\rho^{\eta}-r_H^{\eta}-r_z^{\eta}-\sum\mathrm{rem}\|$.",
        ))
        if "channel_remainder_norm" not in rolls
        else "\n".join((
            r"Linearized recursion, zero initial error: $\rho^{\eta}=r_H^{\eta}+r_z^{\eta}$.",
            r"The closure also subtracts $\sum\mathrm{rem}$.",
        ))
    )
    recon_rows = []
    if "channel_remainder_norm" in rolls:
        recon_rows.append(
            ("channel_remainder_norm", r"$\|\rho^{\eta}-r_H^{\eta}-r_z^{\eta}\|$")
        )
    recon_rows.append(
        (
            "eta_recon_norm",
            r"$\|\rho^{\eta}-r_H^{\eta}-r_z^{\eta}-\sum\mathrm{rem}\|$",
        )
    )
    _panel(
        "latent_retention_reconstruction.png",
        "Latent reconstruction residual",
        tuple(recon_rows),
        note=remainder_note,
        band=True,
        figsize=(6.6, 4.8),
    )

    def _band(ax, key, label, zero: bool = True, color: str | None = None):
        series = _series(key)
        if series is None:
            return None
        mean, lo, hi, count = percentile_band(series)
        x = np.arange(mean.shape[0])
        ax.plot(x, mean, color=color, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        if zero:
            ax.axhline(0.0, color="0.45", lw=0.6)
        n_valid[key] = count
        return x

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    kind = str(np.asarray(diag["approximation"]).reshape(-1)[0])
    q_name = r"Q_n=D" if kind == "proxy" else r"Q_n=D\Phi'_n"
    for key, label in (
        ("cos_p_eta", r"$\cos(p_n,\eta_n)$"),
        ("cos_p_eta_next", r"$\cos(p_n,\eta_{n+1})$"),
        ("cos_p_Keta", r"$\cos(p_n,K_n\eta_n)$"),
        ("cos_p_eps_z", r"$\cos(p_n,\varepsilon_n^z)$"),
        ("cos_p_b", r"$\cos(p_n,b_n)$"),
        ("cos_Qe_eta", rf"$\cos({q_name} e_n,\eta_n)$"),
    ):
        _band(ax, key, label)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.set_title(_title("Fine-to-latent cosines. Band is 16–84 pct, not a CI", validated))
    fig.tight_layout()
    fig.savefig(plot_dir / "cosine_fine_to_latent.png", dpi=140)
    plt.close(fig)

    _panel(
        "cosine_latent_retention.png",
        "Latent retention and channel alignment",
        (
            ("cos_ru_rz", r"$\cos(r_H^{\eta}, r_z^{\eta})$"),
            ("cos_eta_rho", r"$\cos(\eta, \rho^{\eta})$"),
            ("cos_ru_eta", r"$\cos(\eta, r_H^{\eta})$"),
            ("cos_rz_eta", r"$\cos(\eta, r_z^{\eta})$"),
            ("cos_ru_rho", r"$\cos(\rho^{\eta}, r_H^{\eta})$"),
            ("cos_rho_rz", r"$\cos(\rho^{\eta}, r_z^{\eta})$"),
        ),
        note=telescoping_note,
        cosine=True,
        band=True,
        figsize=(7, 4.8),
    )
    _panel(
        "cosine_latent_eps_z.png",
        r"Latent injection, $\sum_{j<n}\varepsilon_j^z$",
        (
            ("cos_eta_eps_z", r"$\cos(\eta, \sum_{j<n}\varepsilon_j^z)$"),
            ("cos_rho_Iz", r"$\cos(\rho^{\eta}, \sum_{j<n}\varepsilon_j^z)$"),
            ("cos_ru_Iz", r"$\cos(r_H^{\eta}, \sum_{j<n}\varepsilon_j^z)$"),
            ("cos_rz_Iz", r"$\cos(r_z^{\eta}, \sum_{j<n}\varepsilon_j^z)$"),
        ),
        note=telescoping_note + "\n" + r"This sum is $\varepsilon^z$, not the off-graph injection $\xi$.",
        cosine=True,
        band=True,
        figsize=(7, 5.0),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key, label in (
        ("cross_bp", r"$2\langle b_n,p_n\rangle$"),
        ("p_sq", r"$\|p_n\|^2$"),
        ("delta_fine", r"$\Delta_{\mathrm{fine}}=2\langle b,p\rangle+\|p\|^2$"),
        ("eta_sq_gap", r"$\|\eta\|^2-\|\sum_{j<n}\varepsilon_j^z\|^2$"),
        ("cross_Iz_rho", r"$2\langle\sum_{j<n}\varepsilon_j^z,\rho^{\eta}\rangle$"),
        ("rho_eta_sq", r"$\|\rho^{\eta}\|^2$"),
        ("cross_Iz_ru", r"$2\langle\sum_{j<n}\varepsilon_j^z,r_H^{\eta}\rangle$"),
        ("cross_Iz_rz", r"$2\langle\sum_{j<n}\varepsilon_j^z,r_z^{\eta}\rangle$"),
        ("cross_Iz_rem", r"$2\langle\sum_{j<n}\varepsilon_j^z,\sum\mathrm{rem}\rangle$"),
    ):
        _band(ax, key, label)
    ax.set_xlabel("rollout step")
    ax.legend(fontsize=7)
    ax.set_title(_title(
        r"Local squared-error budget. $\Delta_{\mathrm{fine}}=\|b+p\|^2-\|b\|^2$",
        validated,
    ))
    fig.tight_layout()
    fig.savefig(plot_dir / "latent_fine_error_budget.png", dpi=140)
    plt.close(fig)
    if n_valid:
        np.savez(
            plot_dir / "latent_channel_n_valid.npz",
            **{f"n_{i}": np.asarray(val) for i, val in enumerate(n_valid.values())},
        )

    idx = np.arange(np.asarray(diag["tr_K_over_M"]).shape[0])
    dt = float(np.asarray(diag["dt"]))
    n_coarse = np.asarray(diag["n_coarse"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    split_rows = (
        (diag["tr_K_over_M"], diag["tr_K_ci95"] / n_coarse, r"$\mathrm{Tr}(K)/M$"),
        (
            diag["tr_Ez_over_M"],
            diag["tr_K_ci95"] / (n_coarse * dt),
            r"$\mathrm{Tr}(E_z)/M=(\mathrm{Tr}K-M)/(M\Delta t)$",
        ),
        (
            diag["tr_minus_Qnext_B_over_Mdt"],
            diag["tr_DPhi_next_B_ci95"] / (n_coarse * dt),
            r"$-\mathrm{Tr}(Q_{n+1}B)/(M\Delta t)$",
        ),
        (
            diag["tr_Sigma_over_M"],
            diag["tr_S_ci95"] / (n_coarse * dt),
            r"$\mathrm{Tr}(\Sigma)/M=(\mathrm{Tr}S-M)/(M\Delta t)$",
        ),
    )
    for y, err, label in split_rows:
        ax.errorbar(idx, y, yerr=err, marker="o", capsize=3, label=label)
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set_xlabel("snapshot")
    ax.legend(fontsize=7)
    ax.set_title(_title(
        "Paired-probe latent trace split. Bars are probe 95% CIs",
        validated,
    ))
    fig.tight_layout()
    fig.savefig(plot_dir / "latent_trace_split.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(
        idx,
        diag["tr_C_Dplus_quot"],
        yerr=diag["tr_C_Dplus_ci95"] / (n_coarse * dt),
        marker="o",
        capsize=3,
        label=r"$\mathrm{Tr}(C D^+)/(M\Delta t)$, block-constant lifts",
    )
    ax.errorbar(
        idx,
        diag["gain_C_over_N"],
        yerr=diag["gain_C_ci95"],
        marker="s",
        capsize=3,
        label=r"$\|C\|_F^2/N$ squared gain",
    )
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set_xlabel("snapshot")
    ax.legend(fontsize=7)
    ax.set_title(_title(
        "Fine-to-latent tendency under lifted coarse probes, and squared gain",
        validated,
    ))
    fig.tight_layout()
    fig.savefig(plot_dir / "fine_to_latent_trace_gain.png", dpi=140)
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda but CUDA is not available; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    if torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Task 1 requires autocast and TF32 to stay disabled")
    device = torch.device(args.device)
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
        fd_eps_rel=args.fd_eps,
    )
    ds = int(model.ds_factors[0])

    first_frames, first_stems, first_times = _load_window_from_entry(window_entries[0])
    _check_spacing(first_times, args.dt, "window 0")
    first = make_point(first_frames[:3], model, device, dtype, args.norm)
    step0 = lambda u, z, _t=first.t: model_step(ema_vit, u, z, _t)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    opening = jvp_direction_report(
        step0,
        [
            {
                "name": "step0/fine",
                "point": first,
                "du": torch.randn(first.u.shape, device=device, dtype=dtype, generator=gen),
                "dz": torch.zeros_like(first.z),
            },
            {
                "name": "step0/latent",
                "point": first,
                "du": torch.zeros_like(first.u),
                "dz": torch.randn(first.z.shape, device=device, dtype=dtype, generator=gen),
            },
        ],
        args.fd_tol,
    )
    sweep = opening
    print(
        f"[jvp] opening ok={sweep['ok']} failed={sweep['n_failed']}/{sweep['n_checks']}",
        flush=True,
    )
    if not sweep["ok"] and not args.allow_unvalidated_fd:
        worst = sweep.get("worst") or {}
        failed.append(
            "jvp validation failed "
            f"{sweep['n_failed']}/{sweep['n_checks']} checks, "
            f"worst {worst.get('name')}/{worst.get('block')} "
            f"rel={worst.get('rel')}"
        )
        raise DiagnosticsAborted(
            _run_extras(
                args, tangent, ds, weight_hash, weight_hash, sweep, failed,
                window_entries, global_indices,
                model_fd_validated=False,
                physical_tangent_validated="not_checked",
                spectral_validated="not_requested" if not args.spectral_radius else False,
                fd_validated=False,
                physical_flow=None,
                aborted_before_diagnostics=True,
                num_coarse=int(model.num_coarse),
                latent_res=list(model.latent_res),
            )
        )

    physical_flow = None
    if args.physical_jvp == "py2d":
        dns0 = load_dns_frame(first_stems[0])
        dns1 = load_dns_frame(first_stems[1])
        physical_flow = validate_physical_reference(
            tangent,
            first_frames[0],
            first_frames[1],
            args.fd_eps,
            args.fd_tol,
            args.seed,
            args.dt,
            first_stems[:2],
            dns_now=dns0["omega"],
            dns_next=dns1["omega"],
            hats=dns0["hats"],
        )
        print(
            f"[physical] task1_replay_rel={physical_flow['flow_rel']:.4e} "
            f"dns_flow_rel={physical_flow.get('dns_flow_rel')} "
            f"truncate_base_rel={physical_flow.get('truncate_base_rel')} "
            f"startup={physical_flow.get('startup')} "
            f"nx={physical_flow['nx']} dt={physical_flow['dt']} "
            f"ok={physical_flow['ok']}",
            flush=True,
        )
        if not physical_flow["ok"] and not args.allow_unvalidated_fd:
            failed.append("physical tangent replay failed")
            raise DiagnosticsAborted(
                _run_extras(
                    args, tangent, ds, weight_hash, weight_hash, sweep, failed,
                    window_entries, global_indices,
                    model_fd_validated=False,
                    physical_tangent_validated=False,
                    spectral_validated="not_requested" if not args.spectral_radius else False,
                    fd_validated=False,
                    physical_flow=physical_flow,
                    aborted_before_diagnostics=True,
                    num_coarse=int(model.num_coarse),
                    latent_res=list(model.latent_res),
                )
            )

    diags: list[dict] = []
    rolls: list[dict] = []
    for _local_i, (entry, gidx) in enumerate(zip(window_entries, global_indices)):
        frames, stems, times = _load_window_from_entry(entry)
        _check_spacing(times, args.dt, f"snap {gidx}")
        point = make_point(frames[:3], model, device, dtype, args.norm)
        point.stems = tuple(stems[:3])  # type: ignore[assignment]
        if args.physical_jvp == "py2d":
            _attach_dns(point, stems[0], stems[1])
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
            fd_tol=args.fd_tol,
        )
        diags.append(diag)
        print(
            f"[snap {gidx}] trA/N={diag['tr_A_over_N']:.4e} "
            f"trJ/N={diag['tr_J_over_N']:.4e} "
            f"amp={diag['amp_J']['mean']:.4e}",
            flush=True,
        )
        print(f"[roll {gidx}] length={args.rollout_length}", flush=True)
        sink: list = []
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
            stems=stems,
        )
        extra = jvp_direction_report(step0, sink, args.fd_tol)
        sweep = _merge_fd_reports(sweep, extra)
        print(
            f"[jvp] window {gidx} directions ok={extra['ok']} "
            f"failed={extra['n_failed']}/{extra['n_checks']}",
            flush=True,
        )
        rolls.append(roll)

    if not sweep["ok"]:
        worst = sweep.get("worst") or {}
        failed.append(
            "jvp validation failed "
            f"{sweep['n_failed']}/{sweep['n_checks']} checks, "
            f"worst {worst.get('name')}/{worst.get('block')} "
            f"rel={worst.get('rel')}"
        )
    comp_vals = [float(d["composition_rel_max"]) for d in diags]
    add_vals = [float(d["fd_additivity_rel"]) for d in diags]
    comp_ok = bool(comp_vals) and max(comp_vals) <= float(args.fd_tol)
    add_ok = bool(add_vals) and max(add_vals) <= float(args.fd_tol)
    if not comp_ok:
        failed.append(
            f"composition_rel_max {max(comp_vals) if comp_vals else 'nan'} exceeds fd_tol {args.fd_tol}"
        )
    if not add_ok:
        failed.append(
            f"fd_additivity_rel {max(add_vals) if add_vals else 'nan'} exceeds fd_tol {args.fd_tol}"
        )
    model_ok = gates_pass(
        sweep["ok"],
        max(comp_vals) if comp_vals else float("inf"),
        max(add_vals) if add_vals else float("inf"),
        args.fd_tol,
    )
    if args.physical_jvp == "py2d":
        physical_status = bool(physical_flow and physical_flow["ok"])
        if not physical_status:
            failed.append("physical tangent flow or finite-difference check failed")
        physical_ok = physical_status
    else:
        physical_status = "not_applicable"
        physical_ok = True
    if not args.spectral_radius:
        spectral_status = "not_requested"
    else:
        spectral_status = bool(diags) and all(d.get("spectral_validated") is True for d in diags)
        if not spectral_status:
            failed.append("S additivity did not certify the spectral calculation")

    after = _hash_module(ema_vit)
    if after != weight_hash:
        failed.append("checkpoint weights changed during the diagnostic")

    extras = _run_extras(
        args, tangent, ds, weight_hash, after, sweep, failed,
        window_entries, global_indices,
        model_fd_validated=model_ok,
        physical_tangent_validated=physical_status,
        spectral_validated=spectral_status,
        fd_validated=bool(model_ok and physical_ok),
        physical_flow=physical_flow,
        aborted_before_diagnostics=False,
        num_coarse=int(model.num_coarse),
        latent_res=list(model.latent_res),
    )
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
    fd_ok = all(bool(m.get("fd_validated", False)) for m in metas)
    spectral_meta = _status_all(
        [m.get("spectral_validated", "not_requested") for m in metas]
    )
    other_plots, spectral_plot = plot_validation_flags(
        fd_ok, spectral_meta, bool(args.spectral_radius)
    )
    succeeded = run_succeeded(fd_ok, spectral_meta, bool(args.spectral_radius))
    write_plots(
        save_dir,
        diag_np,
        roll_np,
        validated=other_plots,
        spectral_validated=spectral_plot,
    )

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
        "dt_meaning": (
            "Time divisor for the difference quotients, and the time between "
            "consecutive kept frames. Eval mats are 0.01 apart; "
            "--every-nth-field 5 keeps frames 0.05 apart."
        ),
        "downsampling": primary["downsampling"],
        "physical_jvp": primary["physical_jvp"],
        "is_exact_physical_jvp": False,
        "py2d_root": primary.get("py2d_root"),
        "physical_jvp_note": primary.get("physical_jvp_note"),
        "tolerances": primary.get("tolerances"),
        "legacy_names": {
            "cos_eta_inj": "cos(eta_n, sum_{j<n} xi_j)",
            "cos_e_inj": "cos(e_n, sum_{j<n} epsilon^u_j)",
            "cos_rho_inj": "cos(rho_n, sum_{j<n} epsilon^u_j)",
            "cos_rH_inj": "cos(r_H, sum_{j<n} epsilon^u_j)",
            "cos_rZ_inj": "cos(r_z, sum_{j<n} epsilon^u_j)",
            "I": "||sum_{j<n} epsilon^u_j||",
            "R": "||e_n|| - I(t_n)",
            "cos_eta_eps_z": "cos(eta_n, sum_{j<n} epsilon^z_j)",
            "cos_nu_xi": "cos(nu_n, sum_{j<n} xi_j)",
        },
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
        "fd_validated": fd_ok,
        "model_fd_validated": _status_all([m.get("model_fd_validated", False) for m in metas]),
        "physical_tangent_validated": _status_all(
            [m.get("physical_tangent_validated", "not_applicable") for m in metas]
        ),
        "spectral_validated": spectral_meta,
        "physical_flow": primary.get("physical_flow"),
        "aborted_before_diagnostics": any(m.get("aborted_before_diagnostics") for m in metas),
        "allow_unvalidated_fd": bool(args.allow_unvalidated_fd),
        "autocast_enabled": bool(primary.get("autocast_enabled", False)),
        "allow_tf32_matmul": bool(primary.get("allow_tf32_matmul", False)),
        "allow_tf32_cudnn": bool(primary.get("allow_tf32_cudnn", False)),
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
            "dns_data": args.dns_data or str(
                Path(
                    "/pscratch/sd/d/divyam/py2d_runs/paper/results/"
                    "Re10000_fkx4fky4_r0.1_b20/NoSGS/NX512/dt0.0001_IC1/data"
                )
            ),
        },
        "failed_tests": failed,
        "num_coarse": primary["num_coarse"],
        "latent_res": primary["latent_res"],
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[done] save_dir={save_dir} failed={len(failed)} "
        f"fd_validated={fd_ok} spectral_validated={spectral_meta} "
        f"succeeded={succeeded}",
        flush=True,
    )
    if failed:
        for item in failed:
            print(f"  FAIL {item}", flush=True)
    if not succeeded and not args.allow_unvalidated_fd:
        raise SystemExit(2)


def write_abort(save_dir: Path, args: argparse.Namespace, samples: list[dict], extras: dict) -> None:
    meta = {
        "task": "L2 Task 1 direct-step diagnostics",
        "checkpoint": args.checkpoint,
        "aborted_before_diagnostics": True,
        "fd_validated": False,
        "model_fd_validated": False,
        "physical_tangent_validated": extras.get("physical_tangent_validated", "not_checked"),
        "spectral_validated": extras.get(
            "spectral_validated",
            "not_requested" if not args.spectral_radius else False,
        ),
        "fd_sweep": extras.get("fd_sweep"),
        "fd_eps": args.fd_eps,
        "fd_tol": args.fd_tol,
        "allow_unvalidated_fd": False,
        "failed_tests": extras.get("failed_tests", []),
        "physical_jvp": extras.get("physical_jvp"),
        "physical_flow": extras.get("physical_flow"),
        "windows": [
            {"start": s["start_index"], "stems": s["stems"]} for s in samples
        ],
        "note": "Initial finite-difference epsilon check failed. Diagnostics were not run.",
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[abort] save_dir={save_dir} fd epsilon check failed", flush=True)
    for item in meta["failed_tests"]:
        print(f"  FAIL {item}", flush=True)


def run_single(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Restore visibility for in-process single-GPU path.
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    samples = _select_samples(args, kept, save_dir)
    try:
        diags, rolls, extras = run_windows(
            args, samples, list(range(len(samples)))
        )
    except DiagnosticsAborted as exc:
        write_abort(save_dir, args, samples, exc.extras)
        raise SystemExit(2)
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
    try:
        diags, rolls, extras = run_windows(args, entries, indices)
    except DiagnosticsAborted as exc:
        (save_dir / f"shard_{args.gpu_rank}_meta.json").write_text(
            json.dumps(exc.extras, indent=2) + "\n"
        )
        raise SystemExit(2)
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
        "--dns-data",
        args.dns_data,
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
    samples = _select_samples(args, kept, save_dir)
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
        if any(c == 2 for c in codes):
            print(f"[orchestrator] fd validation aborted workers: {codes}", flush=True)
            raise SystemExit(2)
        raise RuntimeError(f"worker exit codes: {codes}")

    diag_np, roll_np, metas = merge_shards(
        save_dir, args.n_gpus, args.n_snapshots, args.spectral_radius, args.dt
    )
    write_final(save_dir, args, diag_np, roll_np, samples, metas)
    print("[orchestrator] merge complete", flush=True)


def main() -> None:
    args = parse_args()
    if args.dns_data:
        os.environ["HINE_DNS_DATA"] = args.dns_data
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
