#!/usr/bin/env python3
"""Flow replay and missing injection series for the completed Task 1 run.

Reuses the saved Hutchinson traces, Arnoldi pairs, and existing cosines.
Does not overwrite the source results directory.
"""

from __future__ import annotations

import argparse
import json
import os
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
    try:
        return int(argv[idx + 1]) > 1
    except (IndexError, ValueError):
        return False


if _parent_hides_gpus(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = Path(__file__).resolve().parent
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_l2_stability_task1 import (
    FLOW_REPLAY_TOL,
    _check_spacing,
    _load_window_from_entry,
    plot_validation_flags,
    write_plots,
)
from l2_fd_jvp import load_ema_float64
from l2_physical_jvp import load_dns_frame, make_tangent, validate_physical_reference
from l2_retention import LEGACY_COSINE_NAMES, rollout_diagnostics

SOURCE_DIR = Path(
    "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/"
    "task1_l2_n20_float64_py2d_jvp_20261116_204958"
)
PRIOR_FD = Path(
    "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_n20_float64_py2d_dt05"
)
CHECKPOINT = "/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth"

NEW_ROLL = (
    "eps_z_sum_norm",
    "xi_sum_norm",
    "xi_norm",
    "nu_norm",
    "cos_eta_eps_z",
    "cos_nu_xi",
    "cos_eta_inj_recomputed",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", default=str(SOURCE_DIR))
    p.add_argument("--prior-dir", default=str(PRIOR_FD))
    p.add_argument("--save-dir", required=True)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Use the first N windows. 0 means all.")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--rollout-length", type=int, default=40)
    return p.parse_args(argv)


def _versions() -> dict:
    versions = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    try:
        import jax

        versions["jax"] = jax.__version__
    except ImportError:
        versions["jax"] = "unavailable"
    return versions


def _storage_dtype(checkpoint: str) -> list[str]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    found = set()
    for value in blob["model"].values():
        if torch.is_tensor(value):
            found.add(str(value.dtype).replace("torch.", ""))
    return sorted(found)


def _replay_one(tangent, frames, stems, index: int, role: str, dt: float) -> dict:
    dns_now = load_dns_frame(stems[index])
    dns_next = load_dns_frame(stems[index + 1])
    report = validate_physical_reference(
        tangent,
        frames[index],
        frames[index + 1],
        eps=0.0,
        tol=FLOW_REPLAY_TOL,
        seed=0,
        dt=dt,
        stems=[stems[index], stems[index + 1]],
        dns_now=dns_now["omega"],
        dns_next=dns_next["omega"],
        hats=dns_now["hats"],
    )
    report["role"] = role
    report["index"] = int(index)
    report["eval_dtype"] = str(np.asarray(frames[index]).dtype)
    report["dns_dtype"] = str(np.asarray(dns_now["omega"]).dtype)
    report["history_held_fixed"] = (
        "none; differentiated map is the single-field startup. "
        "Omega0 and convec0 are built from the same field as Omega1."
        if report.get("startup") == "Omega0=Omega1"
        else "saved Omega0_hat and Omega1_hat; Omega0 and its convection are held fixed while Omega1 follows the physical field."
    )
    return report


def _window(args, entry, tangent, model, ema, device) -> tuple[dict, list[dict]]:
    frames, stems, times = _load_window_from_entry(entry)
    _check_spacing(times, args.dt, f"sample {entry['sample']}")
    n_steps = int(args.rollout_length)
    locations = [0, n_steps // 2, n_steps - 1]
    replays = []
    for index in locations:
        replays.append(_replay_one(tangent, frames, stems, index, "state", args.dt))
        replays.append(
            _replay_one(tangent, frames, stems, index + 1, "next_inside_S", args.dt)
        )
    series = rollout_diagnostics(
        ema,
        frames,
        model,
        device,
        torch.float64,
        args.norm,
        eps_rel=0.0,
        tangent=tangent,
        n_steps=n_steps,
        stems=stems,
        retention=False,
    )
    arrays = {key: np.asarray(series[key]) for key in (
        "eps_z_sum_norm",
        "xi_sum_norm",
        "xi_norm",
        "nu_norm",
        "cos_eta_eps_z",
        "cos_nu_xi",
        "nu_available",
        "e_norm",
        "eta_norm",
        "eps_u_norm",
        "eps_z_norm",
    )}
    arrays["cos_eta_inj_recomputed"] = np.asarray(series["cos_eta_inj"])
    print(
        f"[sample {entry['sample']}] replay_ok={all(item['ok'] for item in replays)} "
        f"nu_available={bool(series['nu_available'])} "
        f"worst_task1={max(item['task1_replay_rel'] for item in replays):.3e}",
        flush=True,
    )
    return arrays, replays


def _assigned(samples: list[dict], n_gpus: int, rank: int) -> list[dict]:
    return [sample for sample in samples if int(sample["sample"]) % n_gpus == rank]


def run_worker(args) -> None:
    save_dir = Path(args.save_dir)
    samples = json.loads((save_dir / "sample_index.json").read_text())["samples"]
    mine = _assigned(samples, args.n_gpus, int(args.gpu_rank))
    if not mine:
        raise RuntimeError(f"rank {args.gpu_rank} has no windows")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the report worker")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    storage = _storage_dtype(args.checkpoint)
    model, ema = load_ema_float64(args.checkpoint, device)
    tangent = make_tangent(
        "py2d",
        Re=1e4,
        fkx=4,
        fky=4,
        alpha=0.1,
        beta=20.0,
        solver_dt=1e-4,
        emulator_dt=args.dt,
    )
    packed = {}
    replays = []
    for entry in mine:
        arrays, reports = _window(args, entry, tangent, model, ema, device)
        packed[int(entry["sample"])] = arrays
        for item in reports:
            item["sample"] = int(entry["sample"])
        replays.extend(reports)
    path = save_dir / f"report_shard_{args.gpu_rank}.npz"
    payload = {}
    for sample_id, arrays in packed.items():
        for key, value in arrays.items():
            payload[f"{key}__{sample_id}"] = value
    np.savez_compressed(path, **payload)
    meta = {
        "rank": int(args.gpu_rank),
        "samples": [int(entry["sample"]) for entry in mine],
        "replays": replays,
        "checkpoint_storage_dtype": storage,
        "evaluated_dtype": "float64",
        "versions": _versions(),
        "flow_replay_tol": FLOW_REPLAY_TOL,
        "startup_map": (
            "jax.jvp differentiates the same single-field startup used by the replay. "
            "No previous snapshot is frozen."
        ),
    }
    (save_dir / f"report_shard_{args.gpu_rank}.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(f"[rank {args.gpu_rank}] wrote {path}", flush=True)


def _load_source(source: Path) -> tuple[dict, dict]:
    blob = np.load(source / "results.npz")
    diag = {key: blob[key] for key in blob.files if not key.startswith("roll_")}
    rolls = {key[len("roll_"):]: blob[key] for key in blob.files if key.startswith("roll_")}
    return diag, rolls


def _merge_arrays(save_dir: Path, n_samples: int) -> tuple[dict, list[dict], dict]:
    merged = {key: [None] * n_samples for key in NEW_ROLL}
    extra = {key: [None] * n_samples for key in ("e_norm", "eta_norm", "eps_u_norm", "eps_z_norm", "nu_available")}
    replays = []
    versions = {}
    storage = []
    for path in sorted(save_dir.glob("report_shard_*.npz")):
        rank = int(path.stem.split("_")[-1])
        blob = np.load(path)
        meta = json.loads((save_dir / f"report_shard_{rank}.json").read_text())
        replays.extend(meta["replays"])
        versions = meta.get("versions", versions)
        storage = meta.get("checkpoint_storage_dtype", storage)
        for name in blob.files:
            key, sample = name.rsplit("__", 1)
            sample_id = int(sample)
            if key in merged:
                merged[key][sample_id] = blob[name]
            elif key in extra:
                extra[key][sample_id] = blob[name]
    out = {}
    for key, columns in {**merged, **extra}.items():
        if any(item is None for item in columns):
            raise RuntimeError(f"missing report values for {key}")
        out[key] = np.stack(columns)
    return out, replays, {"versions": versions, "checkpoint_storage_dtype": storage}


def _compare(source_rolls: dict, source_diag: dict, prior_dir: Path) -> dict:
    prior_path = prior_dir / "results.npz"
    if not prior_path.is_file():
        return {"prior": str(prior_path), "available": False}
    prior = np.load(prior_path)
    rows = []
    for key in ("rho_S", "tr_S_over_M", "tr_Heff_over_N", "tr_Fu_over_N", "tr_B_DPhi_n_quot", "tr_DFz_over_N"):
        if key in source_diag and key in prior.files:
            new = np.asarray(source_diag[key], dtype=np.float64)
            old = np.asarray(prior[key], dtype=np.float64)
            rows.append(
                {
                    "key": key,
                    "max_abs": float(np.max(np.abs(new - old))),
                    "new_mean": float(np.mean(new)),
                    "prior_fd_mean": float(np.mean(old)),
                    "kind": "numerical JVP versus finite-difference tangent",
                }
            )
    return {
        "prior": str(prior_path),
        "available": True,
        "note": (
            "Differences are the change from the dt=0.05 finite-difference physical "
            "tangent to jax.jvp. Caption and cosine-panel changes are not in this table."
        ),
        "rows": rows,
        "reused_cosine_example": "roll_cos_e_inj",
        "reused_cosine_mean": float(np.nanmean(source_rolls["cos_e_inj"])),
    }


def finalize(args) -> None:
    save_dir = Path(args.save_dir)
    source = Path(args.source_dir)
    samples = json.loads((save_dir / "sample_index.json").read_text())["samples"]
    fresh, replays, provenance = _merge_arrays(save_dir, len(samples))
    diag, rolls = _load_source(source)
    if args.limit:
        for key in list(diag):
            if np.asarray(diag[key]).shape[:1] == (20,):
                diag[key] = np.asarray(diag[key])[: args.limit]
        for key in list(rolls):
            rolls[key] = np.asarray(rolls[key])[: args.limit]
    agree = float(np.nanmax(np.abs(fresh["cos_eta_inj_recomputed"] - rolls["cos_eta_inj"])))
    rolls["cos_eta_eps_z"] = fresh["cos_eta_eps_z"]
    rolls["cos_nu_xi"] = fresh["cos_nu_xi"]
    rolls["eps_z_sum_norm"] = fresh["eps_z_sum_norm"]
    rolls["xi_sum_norm"] = fresh["xi_sum_norm"]
    rolls["xi_norm"] = fresh["xi_norm"]
    rolls["nu_norm"] = fresh["nu_norm"]
    flow_ok = all(item["ok"] for item in replays)
    nu_ok = bool(np.all(fresh["nu_available"]))
    source_meta = json.loads((source / "metadata.json").read_text())
    operator_ok = bool(source_meta.get("fd_validated"))
    spectral_ok = source_meta.get("spectral_validated") is True
    other, spectral_plot = plot_validation_flags(operator_ok, spectral_ok, True)
    write_plots(save_dir, diag, rolls, validated=other, spectral_validated=spectral_plot)
    comparison = _compare(rolls, diag, Path(args.prior_dir))
    report = {
        "source_dir": str(source),
        "source_job": "61449243",
        "reused": [
            "Hutchinson traces, probe means and stored intervals",
            "rho(S), eigenpair components, convergence, residuals",
            "existing rollout norms, R, I, rho, r_H, r_z, linearized errors, stored cosines",
        ],
        "recomputed": [
            "flow replay at early, middle, and late steps and the next state inside S",
            "epsilon^z and xi vector-sum norms",
            "nu_n = eta_n - D Phi'_n e_n and cos(nu, sum xi)",
            "cos(eta, sum epsilon^z)",
        ],
        "not_recomputed": [
            "32-probe traces and Arnoldi. Probe arrays for K, DB, B D Phi'_n, and D Phi'_{n+1} B were not stored; the mean identities in the source run are the available check.",
        ],
        "cos_eta_inj_max_abs_recompute_minus_saved": float(agree),
        "cos_eta_inj_meaning": LEGACY_COSINE_NAMES["cos_eta_inj"],
        "flow_replay_validated": flow_ok,
        "flow_replay_tol": FLOW_REPLAY_TOL,
        "nu_available": nu_ok,
        "operator_validated_from_source": operator_ok,
        "spectral_validated_from_source": spectral_ok,
        "spectral_plot_validated": spectral_plot,
        "failed_replays": [
            {
                "sample": item["sample"],
                "index": item["index"],
                "role": item["role"],
                "task1_replay_rel": item["task1_replay_rel"],
                "dns_flow_rel": item.get("dns_flow_rel"),
                "truncate_base_rel": item.get("truncate_base_rel"),
            }
            for item in replays
            if not item["ok"]
        ],
        "worst_task1_replay_rel": max(item["task1_replay_rel"] for item in replays),
        "worst_dns_flow_rel": max(item.get("dns_flow_rel", 0.0) for item in replays),
        "provenance": provenance,
        "comparison_with_dt05_fd": comparison,
        "notation": {
            "e_n": "uhat_n - u_n",
            "eta_n": "zhat_{n+1} - D u_{n+1}",
            "nu_n": "eta_n - D Phi'_n e_n",
            "epsilon_u": "g(u_n, D u_{n+1}) - u_{n+1}",
            "epsilon_z": "h(u_n, D u_{n+1}) - D u_{n+2}",
            "xi_n": "epsilon^z_n - D Phi'_{n+1} epsilon^u_n",
            "I": "||sum_{j<n} epsilon^u_j||",
            "R": "||e_n|| - I",
            "rho_n": "e_n - sum_{j<n} epsilon^u_j",
            "cos_eta_inj": "legacy name of cos(eta_n, sum xi_j)",
            "cos_eta_eps_z": "cos(eta_n, sum epsilon^z_j)",
            "cos_nu_xi": "cos(nu_n, sum xi_j)",
        },
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (save_dir / "replays.json").write_text(json.dumps(replays, indent=2) + "\n")
    np.savez_compressed(save_dir / "injections.npz", **{key: fresh[key] for key in NEW_ROLL}, nu_available=fresh["nu_available"])
    print(
        f"[report] flow_ok={flow_ok} nu_ok={nu_ok} "
        f"cos_eta_inj_delta={agree:.3e} save_dir={save_dir}",
        flush=True,
    )


def _pump(proc, rank: int) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[rank{rank}] {line}")
        sys.stdout.flush()


def orchestrate(args) -> None:
    source = Path(args.source_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((source / "sample_index.json").read_text())
    samples = list(payload["samples"])
    if args.limit:
        samples = samples[: args.limit]
    (save_dir / "sample_index.json").write_text(
        json.dumps({"n_samples": len(samples), "samples": samples}, indent=2) + "\n"
    )
    (save_dir / "launch.json").write_text(
        json.dumps({"argv": sys.argv, "source_dir": str(source)}, indent=2) + "\n"
    )
    if args.n_gpus == 1:
        args.gpu_rank = 0
        args.worker = True
        args.n_gpus = 1
        run_worker(args)
        finalize(args)
        return
    procs = []
    pumps = []
    for rank in range(args.n_gpus):
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--gpu-rank",
            str(rank),
            "--n-gpus",
            str(args.n_gpus),
            "--source-dir",
            str(source),
            "--save-dir",
            str(save_dir),
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.device,
            "--dt",
            str(args.dt),
            "--norm",
            str(args.norm),
            "--rollout-length",
            str(args.rollout_length),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(proc)
        pump = threading.Thread(target=_pump, args=(proc, rank))
        pump.start()
        pumps.append(pump)
    codes = [proc.wait() for proc in procs]
    for pump in pumps:
        pump.join()
    if any(code != 0 for code in codes):
        raise SystemExit(f"report workers failed: {codes}")
    finalize(args)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
        return
    orchestrate(args)


if __name__ == "__main__":
    main()
