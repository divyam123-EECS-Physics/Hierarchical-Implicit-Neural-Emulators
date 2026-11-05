#!/usr/bin/env python3
"""Block Jacobian spectra for L=3 FutureINN (U, Z1, Z2).

At each of --n-samples randomly chosen ground-truth frames (shared across
blocks), compute the requested input/output Jacobian blocks:

  tag form: inputs-outputs   e.g. Z1-Z1,Z2 or U,Z1,Z2-U,Z1,Z2

Square blocks (same level set on both sides): leading eigenvalue via
scipy.sparse.linalg.eigs on a JVP matvec.
Rectangular blocks: leading singular value via svds with JVP/VJP.

Orchestrator pulls blocks across GPUs:
  python eval_jacobian_blocks.py --n-gpus 4
  python eval_jacobian_blocks.py --blocks U-U --blocks Z1-Z1,Z2

Worker (set CUDA_VISIBLE_DEVICES by the parent):
  python eval_jacobian_blocks.py --worker --gpu-rank 0 ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

# Parent orchestrator must not bind GPUs; workers set CUDA_VISIBLE_DEVICES.
if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.io
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_high_eigs import gt_l3_state
from eval_jacobian_manual_mse import load_kept_paths, load_model
from jacobian_blocks_lib import (
    LEVELS_L3,
    BlockSpec,
    resolve_blocks,
    tag_to_dirname,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-rank", type=int, default=None)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument(
        "--blocks",
        action="append",
        default=None,
        help="Repeatable block tag. Omit to run all 49 L=3 operators.",
    )
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/checkpoint-last.pth",
    )
    p.add_argument(
        "--norm",
        type=float,
        default=10.0,
        help="Divide Omega by this before the model",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_blocks/n20",
    )
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample_XX.npz files that already exist",
    )
    return p.parse_args(argv)


def peak_gpu_mib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))


def reset_peak_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def sample_index_path(save_dir: Path) -> Path:
    return save_dir / "sample_index.json"


def queue_dirs(save_dir: Path) -> tuple[Path, Path, Path]:
    root = save_dir / "_queue"
    return root / "pending", root / "claimed", root / "done"


def choose_sample_starts(
    n_kept: int, lookahead: int, n_samples: int, seed: int
) -> np.ndarray:
    max_start = n_kept - 1 - lookahead
    if max_start < 0:
        raise ValueError(
            f"Need at least {lookahead + 1} kept frames, have {n_kept}"
        )
    n_valid = max_start + 1
    if n_samples > n_valid:
        raise ValueError(
            f"Asked for {n_samples} samples, only {n_valid} valid starts "
            f"(lookahead={lookahead})"
        )
    rng = np.random.default_rng(seed)
    starts = rng.choice(n_valid, size=n_samples, replace=False)
    starts.sort()
    return starts.astype(np.int64)


def write_sample_index(
    save_dir: Path,
    kept: list[tuple[int, str]],
    starts: np.ndarray,
    lookahead: int,
    levels: Sequence[str] | None = None,
) -> list[dict]:
    samples = []
    for i, start in enumerate(starts):
        start = int(start)
        stems = [int(kept[start + j][0]) for j in range(lookahead + 1)]
        paths = [kept[start + j][1] for j in range(lookahead + 1)]
        samples.append(
            {
                "sample": i,
                "start_index": start,
                "stems": stems,
                "paths": paths,
            }
        )
    level_names = list(levels) if levels is not None else list(LEVELS_L3)
    payload = {
        "n_samples": len(samples),
        "lookahead": lookahead,
        "levels": level_names,
        "samples": samples,
    }
    path = sample_index_path(save_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return samples


def load_sample_frames(
    sample: dict, lookahead: int
) -> list[np.ndarray]:
    paths = sample["paths"]
    if len(paths) < lookahead + 1:
        raise ValueError(
            f"sample {sample['sample']} needs {lookahead + 1} paths, "
            f"has {len(paths)}"
        )
    return [
        scipy.io.loadmat(paths[j])["Omega"] for j in range(lookahead + 1)
    ]


def _pack_named(
    names: Sequence[str], tensors: dict[str, torch.Tensor]
) -> torch.Tensor:
    return torch.cat([tensors[n].reshape(-1) for n in names], dim=0)


def _unpack_named(
    flat: torch.Tensor,
    names: Sequence[str],
    templates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        n = templates[name].numel()
        out[name] = flat[offset : offset + n].reshape_as(templates[name])
        offset += n
    return out


def compute_block_spectrum(
    ema_vit: torch.nn.Module,
    high: torch.Tensor,
    mid: torch.Tensor,
    low: torch.Tensor,
    block: BlockSpec,
    ncv: int,
    maxiter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    dtype = next(ema_vit.parameters()).dtype
    levels = {
        "U": high.detach().to(dtype),
        "Z1": mid.detach().to(dtype),
        "Z2": low.detach().to(dtype),
    }
    t = 2 * torch.ones(
        levels["U"].shape[0], device=levels["U"].device, dtype=dtype
    )
    in_names = block.inputs
    out_names = block.outputs
    frozen = {n: levels[n] for n in LEVELS_L3 if n not in in_names}
    templates = {n: levels[n] for n in in_names}
    n_in = int(sum(templates[n].numel() for n in in_names))
    n_out = int(sum(levels[n].numel() for n in out_names))

    def forward_selected(*diff_tensors: torch.Tensor) -> torch.Tensor:
        merged = dict(frozen)
        for name, tensor in zip(in_names, diff_tensors):
            merged[name] = tensor
        oh, om, ol = ema_vit(merged["U"], merged["Z1"], merged["Z2"], t)
        outs = {"U": oh, "Z1": om, "Z2": ol}
        return _pack_named(out_names, outs)

    primals = tuple(levels[n] for n in in_names)

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(v_np)).to(
            device=levels["U"].device, dtype=dtype
        )
        tangents = tuple(
            _unpack_named(v, in_names, templates)[n] for n in in_names
        )
        _, jv = torch.autograd.functional.jvp(
            forward_selected, primals, tangents
        )
        return jv.detach().reshape(-1).float().cpu().numpy()

    def rmatvec(w_np: np.ndarray) -> np.ndarray:
        w = torch.from_numpy(np.ascontiguousarray(w_np)).to(
            device=levels["U"].device, dtype=dtype
        )
        _, vjp = torch.autograd.functional.vjp(
            forward_selected, primals, w
        )
        if not isinstance(vjp, tuple):
            vjp = (vjp,)
        packed = torch.cat([part.reshape(-1) for part in vjp], dim=0)
        return packed.detach().reshape(-1).float().cpu().numpy()

    wall0 = time.perf_counter()
    converged = True
    leading: complex | float
    spectrum_kind = block.spectrum_kind
    try:
        if block.is_square:
            if n_in != n_out:
                raise RuntimeError(
                    f"Square tag {block.tag} but dims {n_out}x{n_in}"
                )
            ncv_eff = min(max(ncv, 2), n_in)
            op = spla.LinearOperator(
                (n_out, n_in), matvec=matvec, dtype=np.float32
            )
            rng = np.random.default_rng(0)
            v0 = rng.standard_normal(n_in).astype(np.float32)
            v0 /= np.linalg.norm(v0)
            vals, _ = spla.eigs(
                op,
                k=1,
                which="LM",
                ncv=ncv_eff,
                maxiter=maxiter,
                tol=tol,
                v0=v0,
                return_eigenvectors=True,
            )
            leading = complex(vals[0])
        else:
            op = spla.LinearOperator(
                (n_out, n_in),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=np.float32,
            )
            # k=1 needs n_out > 1 and n_in > 1 for svds; fall back if tiny.
            k = 1
            m = min(n_out, n_in)
            # svds needs k < ncv < min(n_out, n_in).
            if m <= 2:
                e_in = np.eye(n_in, dtype=np.float32)
                J = np.stack([matvec(e_in[:, j]) for j in range(n_in)], axis=1)
                s = np.linalg.svd(J, compute_uv=False)
                leading = float(s[0])
            else:
                ncv_eff = min(max(ncv, k + 2), m - 1)
                singular_vals = spla.svds(
                    op,
                    k=k,
                    which="LM",
                    ncv=ncv_eff,
                    maxiter=maxiter,
                    tol=tol,
                    return_singular_vectors=False,
                )
                leading = float(np.max(np.asarray(singular_vals).real))
    except Exception:
        converged = False
        raise
    finally:
        wall_s = time.perf_counter() - wall0

    if block.is_square:
        value = np.asarray(leading, dtype=np.complex128)
        radius = np.asarray(abs(complex(leading)), dtype=np.float64)
    else:
        value = np.asarray(float(leading), dtype=np.float64)
        radius = value.copy()

    return {
        "tag": np.asarray(block.tag),
        "input_names": np.asarray(list(in_names)),
        "output_names": np.asarray(list(out_names)),
        "spectrum_kind": np.asarray(spectrum_kind),
        "leading": value,
        "radius": radius,
        "n_in": np.asarray(n_in, dtype=np.int64),
        "n_out": np.asarray(n_out, dtype=np.int64),
        "converged": np.asarray(converged),
        "wall_s": np.asarray(wall_s, dtype=np.float64),
        "peak_gpu_mib": np.asarray(peak_gpu_mib(), dtype=np.float64),
    }


def sample_npz_path(block_dir: Path, sample_idx: int) -> Path:
    return block_dir / f"sample_{sample_idx:02d}.npz"


def run_block_on_samples(
    ema_vit: torch.nn.Module,
    model,
    samples: list[dict],
    block: BlockSpec,
    save_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    norm: float,
    ncv: int,
    maxiter: int,
    tol: float,
    skip_existing: bool,
) -> None:
    block_dir = save_dir / tag_to_dirname(block.tag)
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / "tag.txt").write_text(block.tag + "\n")
    for sample in samples:
        idx = int(sample["sample"])
        out_path = sample_npz_path(block_dir, idx)
        if skip_existing and out_path.is_file():
            print(
                f"[{block.tag}] sample {idx:02d} skip existing",
                flush=True,
            )
            continue
        frames = load_sample_frames(sample, lookahead=2)
        # gt_l3_state expects a list indexed by step; use step=0 on a
        # 3-frame window [k, k+1, k+2].
        high, mid, low = gt_l3_state(
            frames, 0, model, device, dtype, norm
        )
        reset_peak_gpu()
        t0 = time.perf_counter()
        try:
            result = compute_block_spectrum(
                ema_vit, high, mid, low, block, ncv, maxiter, tol
            )
        except Exception as exc:
            print(
                f"[{block.tag}] sample {idx:02d} FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        result["sample"] = np.asarray(idx, dtype=np.int64)
        result["start_index"] = np.asarray(
            sample["start_index"], dtype=np.int64
        )
        result["stems"] = np.asarray(sample["stems"], dtype=np.int64)
        result["norm"] = np.asarray(norm, dtype=np.float64)
        np.savez_compressed(out_path, **result)
        print(
            f"[{block.tag}] sample {idx:02d} "
            f"kind={block.spectrum_kind} "
            f"radius={float(result['radius']):.6g} "
            f"wall_s={time.perf_counter() - t0:.2f} "
            f"-> {out_path.name}",
            flush=True,
        )


def claim_next_tag(pending: Path, claimed: Path) -> str | None:
    pending.mkdir(parents=True, exist_ok=True)
    claimed.mkdir(parents=True, exist_ok=True)
    for path in sorted(pending.iterdir()):
        if not path.is_file():
            continue
        dest = claimed / path.name
        try:
            os.rename(path, dest)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        return dest.read_text().strip()
    return None


def mark_done(done: Path, tag: str) -> None:
    done.mkdir(parents=True, exist_ok=True)
    (done / tag_to_dirname(tag)).write_text(tag + "\n")


def seed_queue(save_dir: Path, blocks: list[BlockSpec]) -> None:
    pending, claimed, done = queue_dirs(save_dir)
    for d in (pending, claimed, done):
        d.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        name = tag_to_dirname(block.tag)
        # Skip if already fully claimed/done from a prior run.
        if (done / name).is_file():
            continue
        if (claimed / name).is_file():
            continue
        path = pending / name
        if not path.is_file():
            path.write_text(block.tag + "\n")


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu_rank is None:
        raise ValueError("--gpu-rank is required with --worker")
    save_dir = Path(args.save_dir)
    index_path = sample_index_path(save_dir)
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sample index: {index_path}")
    payload = json.loads(index_path.read_text())
    samples = payload["samples"]

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(0)
    model, ema_vit = load_model(args.checkpoint, device)
    dtype = next(ema_vit.parameters()).dtype
    allowed = LEVELS_L3
    pending, claimed, done = queue_dirs(save_dir)

    print(
        f"[worker rank={args.gpu_rank}] device={device} "
        f"n_samples={len(samples)} checkpoint={args.checkpoint}",
        flush=True,
    )

    while True:
        tag = claim_next_tag(pending, claimed)
        if tag is None:
            break
        block = resolve_blocks([tag], allowed)[0]
        print(
            f"[worker rank={args.gpu_rank}] claimed {block.tag} "
            f"({block.spectrum_kind})",
            flush=True,
        )
        run_block_on_samples(
            ema_vit,
            model,
            samples,
            block,
            save_dir,
            device,
            dtype,
            args.norm,
            args.ncv,
            args.maxiter,
            args.tol,
            args.skip_existing,
        )
        # Move claimed -> done
        claimed_path = claimed / tag_to_dirname(tag)
        if claimed_path.is_file():
            claimed_path.unlink()
        mark_done(done, block.tag)
        print(
            f"[worker rank={args.gpu_rank}] done {block.tag}",
            flush=True,
        )
    print(f"[worker rank={args.gpu_rank}] QUEUE_EMPTY", flush=True)


def _pump(stream, prefix: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    stream.close()


def run_orchestrator(args: argparse.Namespace) -> None:
    if args.n_gpus < 1:
        raise ValueError("--n-gpus must be >= 1")
    allowed = LEVELS_L3
    blocks = resolve_blocks(args.blocks, allowed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    starts = choose_sample_starts(
        len(kept), lookahead=2, n_samples=args.n_samples, seed=args.seed
    )
    samples = write_sample_index(save_dir, kept, starts, lookahead=2)
    seed_queue(save_dir, blocks)

    summary = {
        "n_blocks": len(blocks),
        "blocks": [b.tag for b in blocks],
        "n_samples": len(samples),
        "n_gpus": args.n_gpus,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "ncv": args.ncv,
        "maxiter": args.maxiter,
        "tol": args.tol,
    }
    (save_dir / "run_config.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"[orchestrator] n_blocks={len(blocks)} n_samples={len(samples)} "
        f"n_gpus={args.n_gpus} save_dir={save_dir}",
        flush=True,
    )
    for b in blocks:
        print(
            f"  block {b.tag} kind={b.spectrum_kind}",
            flush=True,
        )

    script = str(Path(__file__).resolve())
    procs: list[subprocess.Popen] = []
    pumps: list[threading.Thread] = []
    for rank in range(args.n_gpus):
        cmd = [
            sys.executable,
            "-u",
            script,
            "--worker",
            "--gpu-rank",
            str(rank),
            "--n-gpus",
            str(args.n_gpus),
            "--n-samples",
            str(args.n_samples),
            "--every-nth-field",
            str(args.every_nth_field),
            "--eval-dir",
            args.eval_dir,
            "--checkpoint",
            args.checkpoint,
            "--norm",
            str(args.norm),
            "--device",
            args.device,
            "--save-dir",
            str(save_dir),
            "--ncv",
            str(args.ncv),
            "--maxiter",
            str(args.maxiter),
            "--tol",
            str(args.tol),
            "--seed",
            str(args.seed),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        # Workers read blocks from the shared queue; --blocks not needed.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print(
            f"[orchestrator] launching rank {rank} "
            f"CUDA_VISIBLE_DEVICES={rank}",
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
    print("[orchestrator] all workers finished OK", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
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
