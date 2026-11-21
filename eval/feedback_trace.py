"""Trace-only split of two-step latent feedback. No Arnoldi and no SVD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from coarse_lift_budget import step_fn_at, teacher_points
from coarse_lift_ops import (
    ZERO_FLOOR,
    LiftedBlocks,
    cosine,
    euclidean_norm,
    feedback_action,
    field_rms,
    inspect_checkpoint_downsampler,
    latent_shift,
    mc_mean_se,
)
from eval_l2_coarse_lift import check_spacing, load_window, sha256_file
from l2_fd_jvp import assert_l2, load_ema_float64, model_step

TRACE_ABS = 1e-8
COMPOSED_REL = 1e-6


def trace_steps(args) -> list[int]:
    if args.smoke:
        return [0, 1]
    text = str(getattr(args, "steps", "") or "").strip()
    if not text:
        from coarse_lift_ops import TWOSTEP_STEPS

        return [int(step) for step in TWOSTEP_STEPS]
    return [int(part) for part in text.split(",") if part.strip()]


def trace_probe_count(args) -> int:
    if args.smoke:
        return 2
    count = int(args.n_probes)
    if count < 1:
        raise ValueError(f"n-probes must be positive, got {count}")
    return count


def _windows(args, samples: list[dict]) -> list[dict]:
    if args.smoke:
        return samples[:1]
    return list(samples)


def _probe_rng(seed: int, stem: int, step: int, offset: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) + int(stem) + 10007 * int(step) + int(offset))


def _rademacher(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.asarray([-1.0, 1.0]), size=shape).astype(np.float64)


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _load_l1(path: str, device: torch.device):
    from eval_ablation_eigs import load_ablation

    model, ema = load_ablation(path, device)
    model = model.double()
    ema = model.ema_vit
    _freeze(model)
    ablation = str(getattr(model, "ablation", ""))
    if ablation != "l1" or int(getattr(ema, "num_aux", -1)) != 0:
        raise RuntimeError(f"L=1 checkpoint is {ablation}, num_aux={getattr(ema, 'num_aux', None)}")
    return model, ema


def _l1_shift(ema, frame: np.ndarray, probe: np.ndarray, norm: float, device: torch.device) -> float:
    state = torch.from_numpy(np.ascontiguousarray(frame)).to(device=device, dtype=torch.float64)
    state = state.reshape(1, 1, *frame.shape) / float(norm)
    direction = torch.from_numpy(np.ascontiguousarray(probe)).to(device=device, dtype=torch.float64)
    direction = direction.reshape_as(state) / float(norm)

    def forward(field):
        return ema(field)

    with torch.enable_grad():
        _value, tangent = torch.autograd.functional.jvp(forward, (state,), (direction,))
    image = (tangent * float(norm)).detach()
    probe_t = direction * float(norm)
    width = int(probe_t.numel())
    dot = torch.dot(probe_t.reshape(-1).double(), image.reshape(-1).double())
    return float(dot) / width - 1.0


def _rollout_errors(ema, points, norm: float, last_step: int) -> list[np.ndarray]:
    high = points[0].u.clone()
    latent = points[0].z.clone()
    errors = [np.zeros(points[0].u.shape[-2:], dtype=np.float64)]
    with torch.no_grad():
        for point in points[:last_step]:
            high, latent = model_step(ema, high, latent, point.t)
            high = high.detach()
            latent = latent.detach()
            gap = (high - point.u_next) * float(norm)
            errors.append(np.squeeze(gap.detach().cpu().numpy()).astype(np.float64))
    return errors


def _snapshot_path(save_dir: Path, sample: int, step: int) -> Path:
    return save_dir / f"trace_s{int(sample):02d}_t{int(step):02d}.npz"


def _done(path: Path) -> bool:
    if not path.is_file():
        return False
    stored = np.load(path)
    return "complete" in stored.files and int(np.asarray(stored["complete"]).reshape(-1)[0]) == 1


def _evaluate_step(ops, errors, step, probes, latent_probes, l1_ema, frame, norm, device) -> dict:
    op_n = ops[step]
    op_next = ops[step + 1]
    rows = {key: [] for key in ("tau_a", "tau_b", "tau_loop", "tau_A", "delta_loop", "delta_check", "residual", "composed")}
    for probe in probes:
        action = feedback_action(op_n, op_next, probe)
        if abs(action["residual"]) > TRACE_ABS or abs(action["delta_loop"] - action["delta_check"]) > TRACE_ABS:
            raise RuntimeError(
                f"probe identity failed at step {step}: residual {action['residual']:.3e}, "
                f"energy gap {action['delta_loop'] - action['delta_check']:.3e}"
            )
        if action["sum_vs_composed_rel"] > COMPOSED_REL:
            raise RuntimeError(f"two-step JVP check failed at step {step}: {action['sum_vs_composed_rel']:.3e}")
        rows["tau_a"].append(action["tau_a"])
        rows["tau_b"].append(action["tau_b"])
        rows["tau_loop"].append(action["tau_loop"])
        rows["tau_A"].append(action["tau_A"])
        rows["delta_loop"].append(action["delta_loop"])
        rows["delta_check"].append(action["delta_check"])
        rows["residual"].append(action["residual"])
        rows["composed"].append(action["sum_vs_composed_rel"])
    k_samples = [latent_shift(op_n, probe) for probe in latent_probes]
    error = errors[step]
    rollout = {
        "rms_lift": float("nan"),
        "rms_revise": float("nan"),
        "rms_loop": float("nan"),
        "cos_lift_revise": float("nan"),
        "cos_lift_loop": float("nan"),
        "rollout_rel": float("nan"),
    }
    if euclidean_norm(error) > ZERO_FLOOR:
        acted = feedback_action(op_n, op_next, error)
        if acted["sum_vs_composed_rel"] > COMPOSED_REL:
            raise RuntimeError(f"rollout two-step check failed at step {step}: {acted['sum_vs_composed_rel']:.3e}")
        rollout = {
            "rms_lift": field_rms(acted["lift"]),
            "rms_revise": field_rms(acted["revise"]),
            "rms_loop": field_rms(acted["loop"]),
            "cos_lift_revise": cosine(acted["lift"], acted["revise"]),
            "cos_lift_loop": cosine(acted["lift"], acted["loop"]),
            "rollout_rel": float(acted["sum_vs_composed_rel"]),
        }
    tau_j = float("nan")
    if l1_ema is not None:
        tau_j_samples = [_l1_shift(l1_ema, frame, probe, norm, device) for probe in probes]
        tau_j, tau_j_se = mc_mean_se(tau_j_samples)
    else:
        tau_j_samples = []
        tau_j_se = float("nan")
    tau_a, tau_a_se = mc_mean_se(rows["tau_a"])
    tau_b, tau_b_se = mc_mean_se(rows["tau_b"])
    tau_loop, tau_loop_se = mc_mean_se(rows["tau_loop"])
    tau_A, tau_A_se = mc_mean_se(rows["tau_A"])
    tau_K, tau_K_se = mc_mean_se(k_samples)
    delta, delta_se = mc_mean_se(rows["delta_loop"])
    _delta_check, delta_check_se = mc_mean_se(rows["delta_check"])
    width = op_n.n_fine
    coarse = op_n.n_coarse
    return {
        "tau_a": tau_a,
        "tau_a_se": tau_a_se,
        "tau_b": tau_b,
        "tau_b_se": tau_b_se,
        "tau_loop": tau_loop,
        "tau_loop_se": tau_loop_se,
        "tau_A": tau_A,
        "tau_A_se": tau_A_se,
        "tau_K": tau_K,
        "tau_K_se": tau_K_se,
        "tau_J": tau_j,
        "tau_J_se": tau_j_se,
        "delta_loop": delta,
        "delta_loop_se": delta_se,
        "delta_check": float(np.mean(rows["delta_check"])),
        "delta_check_se": delta_check_se,
        "residual_max": float(np.max(np.abs(rows["residual"]))),
        "composed_rel_max": float(np.max(rows["composed"])),
        "trace_joint": width * (tau_A + 1.0) + coarse * (tau_K + 1.0),
        "tau_a_probes": np.asarray(rows["tau_a"], dtype=np.float64),
        "tau_b_probes": np.asarray(rows["tau_b"], dtype=np.float64),
        "tau_loop_probes": np.asarray(rows["tau_loop"], dtype=np.float64),
        "delta_probes": np.asarray(rows["delta_loop"], dtype=np.float64),
        "tau_J_probes": np.asarray(tau_j_samples, dtype=np.float64),
        **rollout,
    }


def _write_snapshot(path: Path, row: dict, meta: dict) -> None:
    payload = {key: np.asarray(value) for key, value in row.items()}
    payload["complete"] = np.int64(1)
    payload["meta_json"] = np.asarray(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _stack(save_dir: Path, rows: list[dict]) -> None:
    keys = [
        "sample",
        "step",
        "stem",
        "tau_a",
        "tau_a_se",
        "tau_b",
        "tau_b_se",
        "tau_loop",
        "tau_loop_se",
        "tau_A",
        "tau_A_se",
        "tau_K",
        "tau_K_se",
        "tau_J",
        "tau_J_se",
        "delta_loop",
        "delta_loop_se",
        "delta_check",
        "residual_max",
        "composed_rel_max",
        "trace_joint",
        "rms_lift",
        "rms_revise",
        "rms_loop",
        "cos_lift_revise",
        "cos_lift_loop",
    ]
    packed = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in keys}
    np.savez_compressed(save_dir / "metrics.npz", **packed)


def run_trace(args, samples: list[dict], save_dir: Path, device: torch.device) -> int:
    steps = trace_steps(args)
    n_probes = trace_probe_count(args)
    windows = _windows(args, samples)
    if any(step < 0 for step in steps):
        raise ValueError(f"trace steps must be nonnegative, got {steps}")
    model, ema = load_ema_float64(args.checkpoint, device)
    assert_l2(model)
    _freeze(model)
    l1_ema = None
    l1_sha = ""
    if str(getattr(args, "l1_checkpoint", "") or "").strip():
        _l1_model, l1_ema = _load_l1(args.l1_checkpoint, device)
        l1_sha = sha256_file(Path(args.l1_checkpoint))
    checkpoint_sha = sha256_file(Path(args.checkpoint)) if Path(args.checkpoint).is_file() else "missing"
    factor = int(model.ds_factors[0])
    meta = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "l1_checkpoint": str(getattr(args, "l1_checkpoint", "") or ""),
        "l1_sha256": l1_sha,
        "sample_index": args.sample_index,
        "norm": float(args.norm),
        "factor": factor,
        "seed": int(args.seed),
        "n_probes": n_probes,
        "steps": steps,
        "smoke": bool(args.smoke),
        "probe": "unnormalized fine-grid Rademacher ±1",
        "lift": "D+",
        "trace_joint": "tr(A_n)+tr(K_n); the extra -N from the a bookkeeping row is excluded",
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    collected = []
    last = max(steps)
    for entry in windows:
        frames, times = load_window(entry)
        check_spacing(times, args.dt, f"sample {entry['sample']}")
        if len(frames) < last + 4:
            raise RuntimeError(f"sample {entry['sample']} has {len(frames)} frames, need {last + 4}")
        points = teacher_points(frames, model, device, torch.float64, args.norm, last + 2)
        report = inspect_checkpoint_downsampler(model, points[0].u * float(args.norm))
        if int(report.factor) != factor or not report.ok:
            raise RuntimeError(f"downsampler check failed: {report.as_dict()}")
        ops = [LiftedBlocks(step_fn_at(ema, point), point, factor, args.norm) for point in points]
        errors = _rollout_errors(ema, points, args.norm, last)
        for step in steps:
            path = _snapshot_path(save_dir, int(entry["sample"]), step)
            stem = int(entry["stems"][step])
            if args.resume and _done(path):
                stored = np.load(path)
                collected.append({key: float(np.asarray(stored[key]).reshape(-1)[0]) for key in (
                    "sample", "step", "stem", "tau_a", "tau_a_se", "tau_b", "tau_b_se", "tau_loop",
                    "tau_loop_se", "tau_A", "tau_A_se", "tau_K", "tau_K_se", "tau_J", "tau_J_se",
                    "delta_loop", "delta_loop_se", "delta_check", "residual_max", "composed_rel_max",
                    "trace_joint", "rms_lift", "rms_revise", "rms_loop", "cos_lift_revise", "cos_lift_loop",
                )})
                print(f"[trace] skip sample {entry['sample']} step {step}", flush=True)
                continue
            fine_rng = _probe_rng(args.seed, stem, step, 0)
            latent_rng = _probe_rng(args.seed, stem, step, 20011)
            fine_shape = points[step].u.shape[-2:]
            coarse_shape = points[step].z.shape[-2:]
            probes = [_rademacher(fine_shape, fine_rng) for _ in range(n_probes)]
            latent_probes = [_rademacher(coarse_shape, latent_rng) for _ in range(n_probes)]
            measured = _evaluate_step(
                ops, errors, step, probes, latent_probes, l1_ema, frames[step], args.norm, device
            )
            row = {
                "sample": float(entry["sample"]),
                "step": float(step),
                "stem": float(stem),
                **{key: value for key, value in measured.items() if not key.endswith("_probes")},
            }
            _write_snapshot(path, {**row, **{key: measured[key] for key in measured if key.endswith("_probes")}}, {
                **meta,
                "sample": int(entry["sample"]),
                "step": int(step),
                "stem": stem,
                "stems": [int(item) for item in entry["stems"][step : step + 3]],
                "times": times[step : step + 3],
            })
            collected.append(row)
            print(
                f"[trace] sample {entry['sample']} step {step} stem {stem} "
                f"tau_a {row['tau_a']:.4g} tau_b {row['tau_b']:.4g} "
                f"tau_loop {row['tau_loop']:.4g} delta {row['delta_loop']:.4g}",
                flush=True,
            )
    _stack(save_dir, collected)
    print(f"[trace] wrote {save_dir / 'metrics.npz'} snapshots={len(collected)}", flush=True)
    return 0
