#!/usr/bin/env python3
"""L=2 coarse-lift diagnostics for T = C D† and U = D† C.

Does not retrain, does not edit Task 1 helpers, and writes only to --save-dir.
Primary operators are evaluated at teacher-forced joint states in float64.
Reported fields are physical vorticity (network output times --norm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

def _parent_hides_gpus(argv: list[str]) -> bool:
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


if _parent_hides_gpus(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.io
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coarse_lift_budget import rollout_budget, step_fn_at
from coarse_lift_ops import (
    BUDGET_LENGTH,
    SPECTRAL_SAMPLE_INDICES,
    SPECTRAL_STEPS,
    TWOSTEP_STEPS,
    D_dagger_torch,
    D_torch,
    LiftedBlocks,
    adjoint_dot_report,
    inspect_checkpoint_downsampler,
    jvp_fd_report,
    projector_identity_report,
)
from coarse_lift_spectral import (
    apply_complex,
    dense_operator_correspondence,
    eigen_residual,
    leading_eigenpairs,
    match_left_eigenpairs,
    partial_svd,
    sampled_directional_gains,
)
from l2_fd_jvp import assert_l2, load_ema_float64, make_point
from plot_l2_coarse_lift import (
    complex_panels,
    plot_argand,
    plot_budget,
    plot_cp_cq,
    plot_direction_fields,
    plot_eigen_summary,
    plot_singular_gains,
    plot_twostep,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "budget", "spectral", "trace"], required=True)
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument(
        "--sample-index",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_n20_float64_py2d_dt05/sample_index.json",
    )
    p.add_argument("--save-dir", required=True)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--fd-tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--smoke", action="store_true", help="Trace mode only: one window, steps 0 and 1, two probes.")
    p.add_argument("--n-probes", type=int, default=32)
    p.add_argument("--steps", default="", help="Comma-separated trace steps. Default is 0,20,38.")
    p.add_argument("--l1-checkpoint", default="", help="Optional L=1 checkpoint for tr(J-I)/N only.")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def code_hashes() -> dict[str, str]:
    names = [
        "coarse_lift_ops.py",
        "coarse_lift_spectral.py",
        "coarse_lift_budget.py",
        "eval_l2_coarse_lift.py",
        "plot_l2_coarse_lift.py",
        "feedback_trace.py",
        "plot_feedback_traces.py",
    ]
    return {name: sha256_file(_EVAL / name) for name in names}


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if len(payload.get("samples", [])) < 20:
        raise ValueError(f"manifest {path} has {len(payload.get('samples', []))} samples, expected 20")
    return payload


def load_window(entry: dict) -> tuple[list[np.ndarray], list[float]]:
    frames = []
    times = []
    for path in entry["paths"]:
        raw = scipy.io.loadmat(path)
        if "time" not in raw:
            raise RuntimeError(f"{path} has no time value")
        frames.append(np.asarray(raw["Omega"], dtype=np.float64))
        times.append(float(np.asarray(raw["time"]).reshape(-1)[0]))
    return frames, times


def check_spacing(times: list[float], dt: float, label: str) -> None:
    for left, right in zip(times, times[1:]):
        gap = float(right) - float(left)
        if abs(gap - float(dt)) > 1e-6:
            raise RuntimeError(f"{label} frame spacing {gap} != dt {dt}")


def selection_record(samples: list[dict], manifest_path: Path) -> dict:
    chosen = []
    for index in SPECTRAL_SAMPLE_INDICES:
        entry = samples[index]
        chosen.append(
            {
                "manifest_index": int(index),
                "start_index": int(entry["start_index"]),
                "stems": [int(s) for s in entry["stems"]],
                "spectral_steps": list(SPECTRAL_STEPS),
                "twostep_steps": list(TWOSTEP_STEPS),
            }
        )
    return {
        "manifest": str(manifest_path),
        "n_samples": len(samples),
        "budget_length": BUDGET_LENGTH,
        "budget_samples": list(range(len(samples))),
        "spectral_samples": list(SPECTRAL_SAMPLE_INDICES),
        "spectral_steps": list(SPECTRAL_STEPS),
        "twostep_steps": list(TWOSTEP_STEPS),
        "windows": chosen,
        "note": (
            "Indices are positions in the corrected Task 1 manifest, "
            "which is sorted by start_index. Step 0 of a fresh rollout has zero fine error."
        ),
    }


def write_launch(save_dir: Path, args, checkpoint_sha: str) -> None:
    payload = {
        "argv": sys.argv,
        "mode": args.mode,
        "git": git_head(_REPO),
        "code_sha256": code_hashes(),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "sample_index": args.sample_index,
        "dt": args.dt,
        "norm": args.norm,
        "k": args.k,
        "ncv": args.ncv,
        "maxiter": args.maxiter,
        "tol": args.tol,
        "fd_tol": args.fd_tol,
        "seed": args.seed,
        "device": args.device,
        "n_gpus": args.n_gpus,
    }
    (save_dir / "launch.json").write_text(json.dumps(payload, indent=2) + "\n")
    (save_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")


def _owned(n_items: int, n_gpus: int, rank: int) -> list[int]:
    return list(range(rank, n_items, n_gpus))


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return int(data["complete"]) == 1
    except Exception:
        return False


def _rel_np(a, b) -> float:
    diff = np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    denom = max(float(np.linalg.norm(np.asarray(b, dtype=np.float64))), 1e-12)
    return float(diff / denom)


def save_budget_file(path: Path, packed: dict, meta: dict) -> None:
    tw = packed["twostep"]
    payload = {key: value for key, value in packed.items() if key != "twostep"}
    payload["twostep_step"] = np.asarray([row["step"] for row in tw], dtype=np.int64)
    payload["twostep_ok"] = np.asarray([bool(row.get("ok", False)) for row in tw])
    payload["twostep_rel"] = np.asarray([float(row.get("sum_vs_composed_rel", np.nan)) for row in tw])
    payload["twostep_norm_AA"] = np.asarray([float(row.get("norm_AA", np.nan)) for row in tw])
    payload["twostep_norm_BC"] = np.asarray([float(row.get("norm_BC", np.nan)) for row in tw])
    payload["twostep_cosine"] = np.asarray([float(row.get("cosine_AA_BC", np.nan)) for row in tw])
    payload["twostep_initial_norm"] = np.asarray([float(row.get("initial_fine_norm", np.nan)) for row in tw])
    if tw and "AA" in tw[0]:
        payload["twostep_AA"] = np.stack([row["AA"] for row in tw])
        payload["twostep_BC"] = np.stack([row["BC"] for row in tw])
    payload["complete"] = np.int64(1)
    payload["meta_json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **payload)


def u_lift_residuals(op: LiftedBlocks, values: np.ndarray, vectors: np.ndarray, left_vectors: np.ndarray | None) -> dict:
    """Lift T eigenpairs to U. Left lift uses C* and is reported only for nonzero eigenvalues."""
    factor = op.factor
    coarse_shape = (int(op.point.z.shape[-2]), int(op.point.z.shape[-1]))
    action_u = op.matvec("U")
    action_c = op.matvec("C")
    right_res = []
    left_res = []
    for i, lam in enumerate(values):
        vec = vectors[:, i].reshape(coarse_shape)
        lifted = D_dagger_torch(
            torch.from_numpy(np.real(vec)).to(device=op.point.z.device, dtype=op.point.z.dtype).reshape_as(op.point.z),
            factor,
        )
        lifted_i = D_dagger_torch(
            torch.from_numpy(np.imag(vec)).to(device=op.point.z.device, dtype=op.point.z.dtype).reshape_as(op.point.z),
            factor,
        )
        lifted_np = _squeeze(lifted).astype(np.float64) + 1j * _squeeze(lifted_i).astype(np.float64)
        image = apply_complex(action_u.matvec, lifted_np.reshape(-1))
        denom = max(float(np.linalg.norm(lifted_np)), 1e-30)
        right_res.append(float(np.linalg.norm(image - lam * lifted_np.reshape(-1)) / denom))
        if left_vectors is None or abs(lam) <= 1e-8:
            left_res.append(float("nan"))
            continue
        y = left_vectors[:, i]
        z = apply_complex(action_c.rmatvec, y)
        if float(np.linalg.norm(z)) <= 1e-12:
            left_res.append(float("nan"))
            continue
        uz = apply_complex(action_u.rmatvec, z)
        left_res.append(float(np.linalg.norm(uz - np.conj(lam) * z) / max(float(np.linalg.norm(z)), 1e-30)))
    return {
        "right": np.asarray(right_res, dtype=np.float64),
        "left": np.asarray(left_res, dtype=np.float64),
    }


def _squeeze(tensor: torch.Tensor) -> np.ndarray:
    return np.squeeze(tensor.detach().cpu().numpy())


def pattern_triple(op: LiftedBlocks, coarse: np.ndarray) -> dict[str, np.ndarray]:
    """Input w, lifted input D†w, and output Tw. Complex inputs keep real and imag parts."""
    if np.iscomplexobj(coarse):
        real = pattern_triple(op, np.real(coarse))
        imag = pattern_triple(op, np.imag(coarse))
        return {
            "w": real["w"] + 1j * imag["w"],
            "lifted": real["lifted"] + 1j * imag["lifted"],
            "output": real["output"] + 1j * imag["output"],
        }
    w = torch.from_numpy(np.ascontiguousarray(coarse, dtype=np.float64)).to(
        device=op.point.z.device, dtype=op.point.z.dtype
    ).reshape_as(op.point.z)
    return {
        "w": _squeeze(w),
        "lifted": _squeeze(op.D_dagger(w)),
        "output": _squeeze(op.T(w)),
    }


def analyze_state(op: LiftedBlocks, args, seed: int) -> tuple[dict, list[dict]]:
    failures = []
    actions = {name: op.matvec(name) for name in ("T", "U", "C", "CQ")}
    right = leading_eigenpairs(
        actions["T"].matvec, actions["T"].n_in, args.k, args.ncv, args.maxiter, args.tol, seed
    )
    if right["values"].size == 0:
        failures.append({"stage": "eig_T", "error": right.get("error", "no eigenvalues")})
    left = leading_eigenpairs(
        actions["T"].rmatvec, actions["T"].n_in, args.k, args.ncv, args.maxiter, args.tol, seed + 17
    )
    left_ok = left["values"].size > 0
    if not left_ok:
        failures.append({"stage": "eig_T_adjoint", "error": left.get("error", "adjoint Arnoldi failed")})
    matched = None
    left_residuals = np.zeros(0, dtype=np.float64)
    if left_ok and right["values"].size:
        matched = match_left_eigenpairs(right["values"], left["values"], left["vectors"])
        left_residuals = np.array(
            [
                eigen_residual(actions["T"].rmatvec, np.conj(right["values"][i]), matched["vectors"][:, i])
                if np.isfinite(matched["values"][i])
                else float("nan")
                for i in range(right["values"].size)
            ]
        )
    lifts = {"right": np.zeros(0), "left": np.zeros(0)}
    if right["values"].size:
        left_vecs = matched["vectors"] if matched is not None else None
        lifts = u_lift_residuals(op, right["values"], right["vectors"], left_vecs)

    svds = {}
    probes = {}
    for name, action in actions.items():
        decomposition = partial_svd(
            action.matvec,
            action.rmatvec,
            action.n_out,
            action.n_in,
            args.k,
            args.ncv,
            args.maxiter,
            args.tol,
            seed + 30,
        )
        if decomposition["values"].size == 0:
            failures.append({"stage": f"svd_{name}", "error": decomposition.get("error", "svd failed")})
        decomposition["rms_values"] = decomposition["values"] * action.rms_factor
        decomposition["rms_factor"] = action.rms_factor
        probes[name] = sampled_directional_gains(action, seed + 50)
        svds[name] = decomposition

    coarse_shape = (int(op.point.z.shape[-2]), int(op.point.z.shape[-1]))
    patterns = {}
    rng = np.random.default_rng(0)
    random_w = rng.standard_normal(coarse_shape)
    random_w /= max(np.linalg.norm(random_w), 1e-30)
    patterns["random"] = pattern_triple(op, random_w)
    if right["values"].size:
        patterns["eigen_right"] = pattern_triple(op, right["vectors"][:, 0].reshape(coarse_shape))
    if svds["T"]["values"].size:
        patterns["singular_input"] = pattern_triple(op, svds["T"]["v"][:, 0].reshape(coarse_shape))
    return {
        "right": right,
        "left": left,
        "matched": matched,
        "left_residuals": left_residuals,
        "lifts": lifts,
        "svds": svds,
        "probes": probes,
        "patterns": patterns,
        "N": op.n_fine,
        "M": op.n_coarse,
        "factor": op.factor,
    }, failures


def pack_state(result: dict, meta: dict) -> dict:
    right = result["right"]
    payload = {
        "complete": np.int64(
            1
            if result["right"]["values"].size >= int(result["right"].get("k") or 0) > 0
            and all(result["svds"][name]["values"].size > 0 for name in ("T", "U", "C", "CQ"))
            else 0
        ),
        "eigenvalues": right["values"],
        "eigen_residuals": right["residuals"],
        "eigen_converged": np.asarray(right.get("mode_converged", []), dtype=bool),
        "eigen_vectors": right["vectors"],
        "u_right_residual": result["lifts"]["right"],
        "u_left_residual": result["lifts"]["left"],
        "left_residuals": result["left_residuals"],
        "N": np.int64(result["N"]),
        "M": np.int64(result["M"]),
        "factor": np.int64(result["factor"]),
        "meta_json": np.asarray(json.dumps(meta)),
    }
    matched = result["matched"]
    if matched is not None:
        payload["left_values"] = matched["values"]
        payload["left_vectors"] = matched["vectors"]
        payload["left_match_distance"] = matched["match_distance"]
    for name, decomposition in result["svds"].items():
        payload[f"sigma_{name}"] = decomposition["values"]
        payload[f"sigma_rms_{name}"] = decomposition["rms_values"]
        payload[f"svd_u_{name}"] = decomposition["u"]
        payload[f"svd_v_{name}"] = decomposition["v"]
        payload[f"svd_right_rel_{name}"] = decomposition["residuals_right_rel"]
        payload[f"svd_left_rel_{name}"] = decomposition["residuals_left_rel"]
        payload[f"svd_converged_{name}"] = np.asarray(decomposition.get("mode_converged", []), dtype=bool)
        payload[f"probe_rms_max_{name}"] = np.asarray(result["probes"][name]["rms_max"])
        payload[f"probe_rms_mean_{name}"] = np.asarray(result["probes"][name]["rms_mean"])
    for key, triple in result["patterns"].items():
        payload[f"pattern_{key}_w"] = triple["w"]
        payload[f"pattern_{key}_lifted"] = triple["lifted"]
        payload[f"pattern_{key}_output"] = triple["output"]
    return payload


def run_budget_window(ema, model, entry, args, device) -> dict:
    frames, times = load_window(entry)
    check_spacing(times, args.dt, f"sample {entry['sample']}")
    packed = rollout_budget(ema, model, frames, device, args.norm, BUDGET_LENGTH, TWOSTEP_STEPS)
    packed["times"] = np.asarray(times, dtype=np.float64)
    packed["stems"] = np.asarray(entry["stems"], dtype=np.int64)
    return packed


def run_spectral_state(ema, model, entry, step: int, args, device) -> tuple[dict, dict]:
    frames, times = load_window(entry)
    check_spacing(times, args.dt, f"sample {entry['sample']}")
    point = make_point(frames[step : step + 3], model, device, torch.float64, args.norm)
    factor = int(model.ds_factors[0])
    op = LiftedBlocks(step_fn_at(ema, point), point, factor, args.norm)
    seed = int(args.seed) + 1000 * int(entry["sample"]) + int(step)
    result, failures = analyze_state(op, args, seed)
    meta = {
        "sample": int(entry["sample"]),
        "step": int(step),
        "start_index": int(entry["start_index"]),
        "stems": [int(s) for s in entry["stems"][step : step + 3]],
        "times": times[step : step + 3],
        "failures": failures,
        "eigen_error": result["right"].get("error", ""),
        "left_error": result["left"].get("error", ""),
    }
    return pack_state(result, meta), meta


def smoke(args, samples, save_dir: Path, device) -> int:
    model, ema = load_ema_float64(args.checkpoint, device)
    assert_l2(model)
    entry = samples[0]
    frames, times = load_window(entry)
    check_spacing(times, args.dt, "smoke window")
    frame = torch.from_numpy(frames[0]).to(device=device, dtype=torch.float64).view(1, 1, *frames[0].shape)
    report = inspect_checkpoint_downsampler(model, frame)
    algebra = projector_identity_report(int(model.img_size), int(model.ds_factors[0]), seed=args.seed)
    dense = dense_operator_correspondence()
    w = torch.randn(1, 1, report.latent_res, report.latent_res, device=device, dtype=torch.float64)
    back = D_torch(D_dagger_torch(w, report.factor), report.factor)
    torch_right_inverse = _rel_np(back.cpu().numpy(), w.cpu().numpy())
    point = make_point(frames[0:3], model, device, torch.float64, args.norm)
    step = step_fn_at(ema, point)
    fd = jvp_fd_report(step, point, report.factor, args.norm, seed=args.seed)
    blocks = LiftedBlocks(step, point, report.factor, args.norm)
    adjoint_t = adjoint_dot_report(blocks.matvec("T"), seed=args.seed, n_pairs=2)
    adjoint_c = adjoint_dot_report(blocks.matvec("C"), seed=args.seed + 1, n_pairs=2)
    eig = leading_eigenpairs(blocks.matvec("T").matvec, blocks.n_coarse, args.k, max(args.ncv, args.k + 2), args.maxiter, args.tol, args.seed)
    action_t = blocks.matvec("T")
    svd = partial_svd(action_t.matvec, action_t.rmatvec, action_t.n_out, action_t.n_in, args.k, max(args.ncv, args.k + 2), args.maxiter, args.tol, args.seed)
    gates = {
        "downsampler": bool(report.ok),
        "projector_algebra": bool(algebra["ok"]),
        "dense_correspondence": bool(dense["ok"]),
        "torch_right_inverse": torch_right_inverse <= 1e-10,
        "fd": float(fd["best_rel"]) <= float(args.fd_tol) and bool(fd["ok_physical_scale"]),
        "adjoint_T": float(adjoint_t["max_rel"]) <= 1e-6,
        "adjoint_C": float(adjoint_c["max_rel"]) <= 1e-6,
        "eigen_T": bool(eig["values"].size) and bool(eig["converged"]),
        "svd_T": bool(svd["values"].size) and bool(svd["converged"]),
    }
    payload = {
        "gates": gates,
        "ok": all(gates.values()),
        "downsampler": report.as_dict(),
        "projector": algebra,
        "dense": dense,
        "torch_right_inverse": torch_right_inverse,
        "fd": fd,
        "adjoint_T": adjoint_t,
        "adjoint_C": adjoint_c,
        "eigen_T": {
            "values": [complex(v).real for v in eig["values"]],
            "values_imag": [complex(v).imag for v in eig["values"]],
            "residuals": [float(v) for v in eig["residuals"]],
            "converged": bool(eig["converged"]),
            "error": eig.get("error", ""),
        },
        "svd_T": {
            "values": [float(v) for v in svd["values"]],
            "rms_values": [float(v) for v in svd["values"]],
            "right_rel": [float(v) for v in svd["residuals_right_rel"]],
            "left_rel": [float(v) for v in svd["residuals_left_rel"]],
            "converged": bool(svd["converged"]),
            "error": svd.get("error", ""),
            "metric": "T is square, so Euclidean and physical RMS singular values match",
        },
        "sample": int(entry["sample"]),
        "step": 0,
        "stems": [int(s) for s in entry["stems"][:3]],
    }
    (save_dir / "smoke_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        save_dir / "smoke_state.npz",
        eigenvalues=eig["values"],
        eigen_residuals=eig["residuals"],
        sigma_T=svd["values"],
        complete=np.int64(1 if payload["ok"] else 0),
    )
    if eig["values"].size:
        plot_argand(
            eig["values"][None, :],
            ["sample 0 step 0"],
            np.ones((1, eig["values"].size), dtype=bool),
            save_dir / "plots" / "smoke_argand.png",
        )
    print(json.dumps({"smoke_ok": payload["ok"], "gates": gates}, indent=2), flush=True)
    return 0 if payload["ok"] else 2


def worker_tasks(args, samples: list[dict]) -> list[tuple]:
    if args.mode == "budget":
        owned = _owned(len(samples), args.n_gpus, args.gpu_rank)
        return [("budget", samples[i]) for i in owned]
    if args.mode == "spectral":
        states = [(int(s), int(step)) for s in SPECTRAL_SAMPLE_INDICES for step in SPECTRAL_STEPS]
        owned = _owned(len(states), args.n_gpus, args.gpu_rank)
        return [("spectral", samples[states[i][0]], states[i][1]) for i in owned]
    return []


def run_assigned(args, samples, save_dir: Path, device) -> list[dict]:
    model, ema = load_ema_float64(args.checkpoint, device)
    assert_l2(model)
    failures = []
    for task in worker_tasks(args, samples):
        if task[0] == "budget":
            entry = task[1]
            path = save_dir / f"budget_window_{int(entry['sample']):02d}.npz"
            if args.resume and _complete(path):
                print(f"[budget] skip complete sample {entry['sample']}", flush=True)
                continue
            print(f"[budget] sample {entry['sample']}", flush=True)
            try:
                packed = run_budget_window(ema, model, entry, args, device)
                save_budget_file(
                    path,
                    packed,
                    {"sample": int(entry["sample"]), "stems": [int(s) for s in entry["stems"]]},
                )
            except Exception as exc:
                failures.append(
                    {
                        "sample": int(entry["sample"]),
                        "stage": "budget",
                        "exception": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[budget] FAIL sample {entry['sample']}: {exc}", flush=True)
        else:
            entry, step = task[1], task[2]
            path = save_dir / f"spectral_s{int(entry['sample']):02d}_t{int(step):02d}.npz"
            if args.resume and _complete(path):
                print(f"[spectral] skip complete sample {entry['sample']} step {step}", flush=True)
                continue
            print(f"[spectral] sample {entry['sample']} step {step}", flush=True)
            try:
                payload, meta = run_spectral_state(ema, model, entry, step, args, device)
                np.savez_compressed(path, **payload)
                if meta["failures"]:
                    for item in meta["failures"]:
                        recorded = dict(item)
                        recorded["sample"] = int(entry["sample"])
                        recorded["step"] = int(step)
                        failures.append(recorded)
            except Exception as exc:
                failures.append(
                    {
                        "sample": int(entry["sample"]),
                        "step": int(step),
                        "stage": "spectral",
                        "exception": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[spectral] FAIL sample {entry['sample']} step {step}: {exc}", flush=True)
    return failures


def finalize_budget(save_dir: Path, samples: list[dict]) -> None:
    rows = []
    for entry in samples:
        path = save_dir / f"budget_window_{int(entry['sample']):02d}.npz"
        if not _complete(path):
            continue
        with np.load(path) as data:
            rows.append({key: data[key] for key in data.files})
    if not rows:
        print("[budget] no complete windows to plot", flush=True)
        return
    order = np.argsort([int(np.asarray(row["stems"])[0]) for row in rows])
    rows = [rows[i] for i in order]
    steps = rows[0]["step"]
    def stack(name):
        return np.stack([row[name] for row in rows], axis=0)
    plot_dir = save_dir / "plots"
    plot_cp_cq(steps, stack("rms_CP_e"), stack("rms_CQ_e"), plot_dir / "cp_vs_cq.png")
    plot_budget(
        steps,
        stack("rms_e_Z_next"),
        stack("rms_C_e"),
        stack("rms_K_e"),
        stack("rms_eps_Z"),
        stack("rms_remainder"),
        plot_dir / "coarse_error_budget.png",
    )
    labels = []
    norm_aa = []
    norm_bc = []
    cosine = []
    rel = []
    for row in rows:
        if "twostep_step" not in row:
            continue
        stems = row["stems"]
        for i, step in enumerate(row["twostep_step"]):
            labels.append(f"s{int(stems[0])}/n{int(step)}")
            norm_aa.append(float(row["twostep_norm_AA"][i]))
            norm_bc.append(float(row["twostep_norm_BC"][i]))
            cosine.append(float(row["twostep_cosine"][i]))
            rel.append(float(row["twostep_rel"][i]))
    if labels:
        plot_twostep(
            labels,
            np.asarray(norm_aa),
            np.asarray(norm_bc),
            np.asarray(cosine),
            np.asarray(rel),
            plot_dir / "twostep_feedback.png",
        )
    summary = {
        "n_windows": len(rows),
        "add_rel_max": float(np.nanmax(stack("add_rel"))),
        "mean_rms_remainder": float(np.nanmean(stack("rms_remainder"))),
        "mean_rms_C": float(np.nanmean(stack("rms_C_e"))),
        "mean_rms_CP": float(np.nanmean(stack("rms_CP_e"))),
        "mean_rms_CQ": float(np.nanmean(stack("rms_CQ_e"))),
    }
    (save_dir / "budget_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[budget] plotted {len(rows)} windows", flush=True)


def _state_label(meta: dict) -> str:
    stems = meta.get("stems", [0])
    return f"s{int(meta['sample'])}/n{int(meta['step'])}"


def finalize_spectral(save_dir: Path) -> None:
    files = sorted(save_dir.glob("spectral_s*_t*.npz"))
    records = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            records.append((meta, {key: data[key] for key in data.files}))
    if not records:
        print("[spectral] no state files to plot", flush=True)
        return
    labels = [_state_label(meta) for meta, _ in records]
    eigenvalues = []
    magnitudes = []
    residuals = []
    converged = []
    width = max(int(np.asarray(rec["eigenvalues"]).size) for _, rec in records)
    for _, rec in records:
        vals = np.asarray(rec["eigenvalues"], dtype=np.complex128)
        res = np.asarray(rec["eigen_residuals"], dtype=np.float64)
        flags = np.asarray(rec["eigen_converged"], dtype=bool)
        pad_v = np.full(width, np.nan + 0j, dtype=np.complex128)
        pad_r = np.full(width, np.nan, dtype=np.float64)
        pad_f = np.zeros(width, dtype=bool)
        pad_v[: vals.size] = vals
        pad_r[: res.size] = res
        pad_f[: flags.size] = flags
        eigenvalues.append(pad_v)
        magnitudes.append(np.abs(pad_v))
        residuals.append(pad_r)
        converged.append(pad_f)
    eigenvalues = np.stack(eigenvalues)
    plot_dir = save_dir / "plots"
    plot_argand(eigenvalues, labels, np.stack(converged), plot_dir / "T_argand.png")
    plot_eigen_summary(labels, np.stack(magnitudes), np.stack(residuals), np.stack(converged), plot_dir / "T_magnitudes.png")
    series = {}
    for name in ("T", "U", "C", "CQ"):
        key = f"sigma_{name}"
        rms_key = f"sigma_rms_{name}"
        if key not in records[0][1]:
            continue
        eucl = []
        rms = []
        directional = []
        for _, rec in records:
            values = np.asarray(rec[key], dtype=np.float64)
            rms_values = np.asarray(rec[rms_key], dtype=np.float64)
            eucl.append(values[0] if values.size else np.nan)
            rms.append(rms_values[0] if rms_values.size else np.nan)
            directional.append(float(rec[f"probe_rms_max_{name}"]))
        series[name] = {
            "euclidean": np.asarray(eucl, dtype=np.float64)[:, None],
            "rms": np.asarray(rms, dtype=np.float64)[:, None],
            "directional_rms": np.asarray(directional, dtype=np.float64),
        }
    if series:
        plot_singular_gains(labels, series, plot_dir / "singular_gains.png")
    for meta, rec in records:
        if int(meta["sample"]) not in SPECTRAL_SAMPLE_INDICES:
            continue
        if int(meta["step"]) not in (0, 20, 39):
            continue
        panels = []
        for key, title in (
            ("pattern_eigen_right", "T eigen right"),
            ("pattern_singular_input", "T singular input"),
            ("pattern_random", "seeded coarse w"),
        ):
            w_key = f"{key}_w"
            if w_key not in rec:
                continue
            panels.extend(complex_panels(f"{title} w", rec[w_key]))
            panels.extend(complex_panels(f"{title} D†w", rec[f"{key}_lifted"]))
            panels.extend(complex_panels(f"{title} Tw", rec[f"{key}_output"]))
        if "svd_u_T" in rec and np.asarray(rec["svd_u_T"]).size:
            side = int(round(np.sqrt(rec["svd_u_T"].shape[0])))
            panels.extend(complex_panels("T left singular", rec["svd_u_T"][:, 0].reshape(side, side)))
        if panels:
            plot_direction_fields(
                panels,
                plot_dir / f"directions_s{int(meta['sample']):02d}_t{int(meta['step']):02d}.png",
                f"Lift-and-response patterns, sample {meta['sample']} step {meta['step']}",
            )
    (save_dir / "spectral_summary.json").write_text(
        json.dumps({"n_states": len(records), "labels": labels}, indent=2) + "\n"
    )
    print(f"[spectral] plotted {len(records)} states", flush=True)


def prepare(args, save_dir: Path, samples: list[dict], checkpoint_sha: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)
    shutil.copyfile(args.sample_index, save_dir / "sample_index.json")
    selection = selection_record(samples, Path(args.sample_index))
    (save_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    write_launch(save_dir, args, checkpoint_sha)


def device_from_args(args) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was requested but is not available")
    return torch.device(args.device)


def orchestrate(args, samples, save_dir: Path) -> int:
    script = str(Path(__file__).resolve())
    procs = []
    pumps = []
    for rank in range(args.n_gpus):
        cmd = [
            sys.executable,
            script,
            "--worker",
            "--gpu-rank",
            str(rank),
            "--n-gpus",
            str(args.n_gpus),
            "--mode",
            args.mode,
            "--checkpoint",
            args.checkpoint,
            "--sample-index",
            args.sample_index,
            "--save-dir",
            str(save_dir),
            "--dt",
            str(args.dt),
            "--norm",
            str(args.norm),
            "--k",
            str(args.k),
            "--ncv",
            str(args.ncv),
            "--maxiter",
            str(args.maxiter),
            "--tol",
            str(args.tol),
            "--fd-tol",
            str(args.fd_tol),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.resume:
            cmd.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(f"[orchestrator] rank {rank}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append(proc)
        thread = threading.Thread(target=_pump, args=(proc.stdout, f"[rank{rank}] "), daemon=True)
        thread.start()
        pumps.append(thread)
    codes = [proc.wait() for proc in procs]
    for thread in pumps:
        thread.join(timeout=5)
    (save_dir / "worker_codes.json").write_text(json.dumps({"codes": codes}) + "\n")
    if any(code != 0 for code in codes):
        print(f"[orchestrator] worker codes {codes}", flush=True)
        return 1
    return 0


def _pump(stream, prefix: str) -> None:
    if stream is None:
        return
    for line in stream:
        print(prefix + line, end="", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    samples = load_manifest(Path(args.sample_index))["samples"]
    for entry, sample in zip(samples, range(len(samples))):
        entry["sample"] = int(sample)
    if args.mode == "trace":
        if args.n_gpus != 1:
            raise SystemExit("trace mode uses one GPU; start a separate job for each checkpoint")
        checkpoint_sha = sha256_file(Path(args.checkpoint)) if Path(args.checkpoint).is_file() else "missing"
        prepare(args, save_dir, samples, checkpoint_sha)
        from feedback_trace import run_trace

        return run_trace(args, samples, save_dir, device_from_args(args))
    if args.worker or args.n_gpus == 1:
        if not (save_dir / "launch.json").is_file():
            prepare(args, save_dir, samples, sha256_file(Path(args.checkpoint)) if Path(args.checkpoint).is_file() else "missing")
        if args.mode == "smoke":
            return smoke(args, samples, save_dir, device_from_args(args))
        failures = run_assigned(args, samples, save_dir, device_from_args(args))
        fail_path = save_dir / f"failed_rank{args.gpu_rank}.json"
        fail_path.write_text(json.dumps(failures, indent=2) + "\n")
        if args.n_gpus == 1:
            if args.mode == "budget":
                finalize_budget(save_dir, samples)
            elif args.mode == "spectral":
                finalize_spectral(save_dir)
        return 1 if any(item.get("exception") for item in failures) else 0
    checkpoint_sha = sha256_file(Path(args.checkpoint)) if Path(args.checkpoint).is_file() else "missing"
    prepare(args, save_dir, samples, checkpoint_sha)
    code = orchestrate(args, samples, save_dir)
    if args.mode == "budget":
        finalize_budget(save_dir, samples)
    elif args.mode == "spectral":
        finalize_spectral(save_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
