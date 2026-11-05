#!/usr/bin/env python3
"""Block Jacobian spectra for FutureINNPlus (U, Z1, ..., ZK).

Same tag protocol as eval_jacobian_blocks.py. For the L=2 checkpoint
(num_coarse=1) the level set is {U, Z1} and the full operator set has 9
blocks. Passing Z2 is an error.

Default checkpoint: /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth
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

if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.io
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_blocks import (
    choose_sample_starts,
    claim_next_tag,
    mark_done,
    peak_gpu_mib,
    queue_dirs,
    reset_peak_gpu,
    sample_index_path,
    sample_npz_path,
    seed_queue,
    write_sample_index,
)
from eval_jacobian_high_eigs_plus import _t_full, gt_plus_state
from eval_jacobian_manual_mse import load_kept_paths
from eval_jacobian_manual_mse_plus import load_model
from jacobian_blocks_lib import (
    BlockSpec,
    levels_plus,
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
        help="Repeatable block tag. Omit to run all subset-pair operators.",
    )
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument(
        "--checkpoint",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth",
    )
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--save-dir",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/jacobian_blocks/n20",
    )
    p.add_argument("--ncv", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def peek_num_coarse(checkpoint_path: str) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        raise KeyError("Checkpoint must contain 'args'")
    args = ckpt["args"]
    if hasattr(args, "num_coarse"):
        return int(args.num_coarse)
    if hasattr(args, "latent_res") and args.latent_res is not None:
        return len(list(args.latent_res))
    raise KeyError("Cannot infer num_coarse from checkpoint args")


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


def compute_block_spectrum_plus(
    ema_vit: torch.nn.Module,
    model,
    high: torch.Tensor,
    coarses: list[torch.Tensor],
    allowed: Sequence[str],
    block: BlockSpec,
    ncv: int,
    maxiter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    dtype = next(ema_vit.parameters()).dtype
    levels: dict[str, torch.Tensor] = {"U": high.detach().to(dtype)}
    for i, c in enumerate(coarses):
        levels[f"Z{i + 1}"] = c.detach().to(dtype)
    t = _t_full(model, levels["U"])
    in_names = block.inputs
    out_names = block.outputs
    frozen = {n: levels[n] for n in allowed if n not in in_names}
    templates = {n: levels[n] for n in in_names}
    n_in = int(sum(templates[n].numel() for n in in_names))
    n_out = int(sum(levels[n].numel() for n in out_names))
    coarse_names = [n for n in allowed if n != "U"]

    def forward_selected(*diff_tensors: torch.Tensor) -> torch.Tensor:
        merged = dict(frozen)
        for name, tensor in zip(in_names, diff_tensors):
            merged[name] = tensor
        coarse_list = [merged[n] for n in coarse_names]
        oh, ocs = ema_vit(merged["U"], coarse_list, t)
        outs = {"U": oh}
        for i, oc in enumerate(ocs):
            outs[f"Z{i + 1}"] = oc
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
    try:
        if block.is_square:
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
            leading: complex | float = complex(vals[0])
        else:
            op = spla.LinearOperator(
                (n_out, n_in),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=np.float32,
            )
            m = min(n_out, n_in)
            if m <= 2:
                e_in = np.eye(n_in, dtype=np.float32)
                J = np.stack(
                    [matvec(e_in[:, j]) for j in range(n_in)], axis=1
                )
                s = np.linalg.svd(J, compute_uv=False)
                leading = float(s[0])
            else:
                ncv_eff = min(max(ncv, 3), m - 1)
                singular_vals = spla.svds(
                    op,
                    k=1,
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
        "spectrum_kind": np.asarray(block.spectrum_kind),
        "leading": value,
        "radius": radius,
        "n_in": np.asarray(n_in, dtype=np.int64),
        "n_out": np.asarray(n_out, dtype=np.int64),
        "converged": np.asarray(converged),
        "wall_s": np.asarray(wall_s, dtype=np.float64),
        "peak_gpu_mib": np.asarray(peak_gpu_mib(), dtype=np.float64),
    }


def load_sample_frames(sample: dict, lookahead: int) -> list[np.ndarray]:
    paths = sample["paths"]
    return [
        scipy.io.loadmat(paths[j])["Omega"] for j in range(lookahead + 1)
    ]


def run_block_on_samples(
    ema_vit: torch.nn.Module,
    model,
    samples: list[dict],
    allowed: Sequence[str],
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
    lookahead = int(model.num_coarse)
    for sample in samples:
        idx = int(sample["sample"])
        out_path = sample_npz_path(block_dir, idx)
        if skip_existing and out_path.is_file():
            print(
                f"[{block.tag}] sample {idx:02d} skip existing",
                flush=True,
            )
            continue
        frames = load_sample_frames(sample, lookahead=lookahead)
        high, coarses = gt_plus_state(
            frames, 0, model, device, dtype, norm
        )
        reset_peak_gpu()
        t0 = time.perf_counter()
        result = compute_block_spectrum_plus(
            ema_vit,
            model,
            high,
            coarses,
            allowed,
            block,
            ncv,
            maxiter,
            tol,
        )
        result["sample"] = np.asarray(idx, dtype=np.int64)
        result["start_index"] = np.asarray(
            sample["start_index"], dtype=np.int64
        )
        result["stems"] = np.asarray(sample["stems"], dtype=np.int64)
        result["norm"] = np.asarray(norm, dtype=np.float64)
        result["num_coarse"] = np.asarray(
            model.num_coarse, dtype=np.int64
        )
        np.savez_compressed(out_path, **result)
        print(
            f"[{block.tag}] sample {idx:02d} "
            f"kind={block.spectrum_kind} "
            f"radius={float(result['radius']):.6g} "
            f"wall_s={time.perf_counter() - t0:.2f} "
            f"-> {out_path.name}",
            flush=True,
        )


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu_rank is None:
        raise ValueError("--gpu-rank is required with --worker")
    save_dir = Path(args.save_dir)
    index_path = sample_index_path(save_dir)
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sample index: {index_path}")
    payload = json.loads(index_path.read_text())
    samples = payload["samples"]
    allowed = tuple(payload["levels"])

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(0)
    model, ema_vit = load_model(args.checkpoint, device)
    dtype = next(ema_vit.parameters()).dtype
    pending, claimed, done = queue_dirs(save_dir)

    print(
        f"[worker rank={args.gpu_rank}] device={device} "
        f"levels={list(allowed)} n_samples={len(samples)}",
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
            allowed,
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
    num_coarse = peek_num_coarse(args.checkpoint)
    allowed = levels_plus(num_coarse)
    blocks = resolve_blocks(args.blocks, allowed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    lookahead = num_coarse
    starts = choose_sample_starts(
        len(kept),
        lookahead=lookahead,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    # write_sample_index hardcodes LEVELS_L3 in payload; rewrite levels.
    samples = write_sample_index(
        save_dir, kept, starts, lookahead=lookahead, levels=allowed
    )
    payload = json.loads(sample_index_path(save_dir).read_text())
    payload["num_coarse"] = num_coarse
    sample_index_path(save_dir).write_text(json.dumps(payload, indent=2) + "\n")

    seed_queue(save_dir, blocks)
    summary = {
        "n_blocks": len(blocks),
        "blocks": [b.tag for b in blocks],
        "levels": list(allowed),
        "num_coarse": num_coarse,
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
        f"[orchestrator] L={num_coarse + 1} levels={list(allowed)} "
        f"n_blocks={len(blocks)} n_samples={len(samples)} "
        f"n_gpus={args.n_gpus} save_dir={save_dir}",
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
