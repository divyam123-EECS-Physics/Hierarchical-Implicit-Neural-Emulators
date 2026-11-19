#!/usr/bin/env python3
"""Largest-magnitude eigenvalues of square single-level blocks on one frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import eval_jacobian_frame_spectrum as eigs
import eval_jacobian_frame_sv as svs
import numpy as np
import torch
from jacobian_blocks_lib import parse_block_tag, tag_to_dirname
from rank_spectrum_plots import write_rank_plots

_PRETTY = {"U": "u", "Z1": "z_1", "Z2": "z_2"}


def _partial(tag: str) -> str:
    left, right = tag.split("-")

    def side(text: str) -> str:
        parts = [_PRETTY[name] for name in text.split(",")]
        if len(parts) == 1:
            return parts[0]
        return "(" + ",".join(parts) + ")"

    return rf"$\partial {side(right)}/\partial {side(left)}$"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--stem", type=int, default=91836)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--ncv", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device(args.device)
    family = svs.peek_family(args.checkpoint)
    if family == "plus":
        model, ema_vit = svs.load_plus(args.checkpoint, device)
        allowed = ("U",) + tuple(f"Z{i + 1}" for i in range(int(model.num_coarse)))
        level = 1 + int(model.num_coarse)
        n_future = int(model.num_coarse)
    else:
        model, ema_vit = svs.load_l3(args.checkpoint, device)
        allowed = ("U", "Z1", "Z2")
        level = 3
        n_future = 2
    kept = svs.load_kept_paths(args.eval_dir, args.every_nth_field)
    frames, _stems = svs.frames_at_stem(kept, args.stem, n_future)
    dtype = next(ema_vit.parameters()).dtype
    levels, forward = svs.prepare_frame(
        family, model, ema_vit, frames, device, dtype, args.norm, allowed
    )
    save_dir = Path(args.save_dir)
    for tag in args.tags:
        block = parse_block_tag(tag, allowed)
        if block.inputs != block.outputs:
            raise ValueError(f"{tag} is not square")
        matvec, _rmatvec, n_in, n_out = svs.operators(
            ema_vit, levels, allowed, block.inputs, block.outputs, forward
        )
        if n_in != n_out:
            raise ValueError(f"{tag} dimensions {n_out} x {n_in}")
        spec = eigs.spectrum(
            matvec, n_in, args.k, args.ncv, args.maxiter, args.tol, args.seed
        )
        out = save_dir / "eigs_blocks" / tag_to_dirname(block.tag)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "eigenvalues.npz",
            eigenvalues=spec["eigenvalues"],
            residual=spec["residual"],
            converged=spec["converged"],
            tag=np.asarray(block.tag),
            which=np.asarray("LM"),
        )
        title = f"L={level}, {_partial(block.tag)} eigenvalues, stem {args.stem}"
        write_rank_plots(out / "plots", spec["eigenvalues"], title, kind="eigenvalue")
        print(
            f"{block.tag} leading={float(np.max(np.abs(spec['eigenvalues']))):.6g} "
            f"residual_max={float(np.nanmax(spec['residual'])):.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
