"""Shared sampling and provenance for Tasks 3 and 5."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io

from eval_jacobian_manual_mse import load_kept_paths
from py2d_inmem import DNS_DATA, PY2D_ROOT, rms

REPO = Path(__file__).resolve().parents[1]


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_starts(n_kept: int, lookahead: int, n_samples: int, seed: int) -> np.ndarray:
    max_start = n_kept - 1 - lookahead
    if max_start < 0:
        raise ValueError(f"need at least {lookahead + 1} kept frames, have {n_kept}")
    n_valid = max_start + 1
    if n_samples > n_valid:
        raise ValueError(f"asked for {n_samples} samples, only {n_valid} valid starts")
    rng = np.random.default_rng(seed)
    starts = rng.choice(n_valid, size=n_samples, replace=False)
    starts.sort()
    return starts.astype(np.int64)


def load_omega(path: str | Path) -> tuple[np.ndarray, float]:
    raw = scipy.io.loadmat(path)
    omega = np.asarray(raw["Omega"], dtype=np.float64)
    time = float(np.asarray(raw["time"]).reshape(-1)[0]) if "time" in raw else float("nan")
    return omega, time


def sample_windows(eval_dir: str, every: int, n_samples: int, lookahead: int, seed: int) -> list[dict]:
    kept = load_kept_paths(eval_dir, every)
    starts = choose_starts(len(kept), lookahead, n_samples, seed)
    samples = []
    for i, start in enumerate(starts):
        start = int(start)
        frames = []
        for j in range(lookahead + 1):
            stem, path = kept[start + j]
            omega, time = load_omega(path)
            dns = DNS_DATA / f"{int(stem)}.mat"
            frames.append(
                {
                    "stem": int(stem),
                    "eval_path": path,
                    "dns_path": str(dns),
                    "time": time,
                    "omega": omega,
                }
            )
        dt = frames[1]["time"] - frames[0]["time"] if lookahead >= 1 else float("nan")
        samples.append({"sample": i, "start_index": start, "frames": frames, "dt_kept": dt})
    return samples


def base_metadata(args, extra: dict | None = None) -> dict:
    meta = {
        "argv": sys.argv,
        "hine_git": git_head(REPO),
        "py2d_git": git_head(PY2D_ROOT),
        "py2d_root": str(PY2D_ROOT),
        "dns_data": str(DNS_DATA),
        "python": sys.executable,
        "pid": os.getpid(),
    }
    if getattr(args, "checkpoint", None):
        ckpt = Path(args.checkpoint)
        meta["checkpoint"] = str(ckpt)
        if ckpt.is_file():
            meta["checkpoint_sha256"] = sha256_file(ckpt)
    if extra:
        meta.update(extra)
    return meta


def namespace_to_argv(ns) -> list[str]:
    """Flatten an argparse namespace into a CLI, including store_true flags."""
    argv = []
    for key, value in vars(ns).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        elif value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    denom = rms(b)
    if denom == 0.0:
        return float("nan")
    return rms(a - b) / denom
