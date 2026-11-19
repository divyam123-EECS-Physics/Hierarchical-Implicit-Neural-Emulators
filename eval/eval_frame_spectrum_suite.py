#!/usr/bin/env python3
"""Top-50 spectra of one eval frame: full map, then every single-level block.

L=2 FutureINNPlus gets eigenvalues and singular values of (U,Z1) -> (U,Z1),
then singular values of U-U, U-Z1, Z1-U, and Z1-Z1.
L=3 FutureINN gets the same for (U,Z1,Z2) -> (U,Z1,Z2) and all nine
single-level pairs. Eigenvalues and singular values are the largest-modulus
ones. Plots color rank with the plasma scale.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--wavenumber", type=int, required=True)
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


def _title(wavenumber: int, level: int, mapping: str, kind: str, stem: int) -> str:
    return f"wavenumber {wavenumber}, L={level}, {mapping} {kind}, stem {stem}"


def _sv_k(requested: int, n_in: int, n_out: int) -> int:
    cap = min(n_in, n_out) - 2
    k = min(requested, cap)
    if k < 1:
        raise ValueError(
            f"block shape {(n_out, n_in)} cannot take {requested} singular values"
        )
    return k


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
    frames, stems = svs.frames_at_stem(kept, args.stem, n_future)
    dtype = next(ema_vit.parameters()).dtype
    levels, forward = svs.prepare_frame(
        family, model, ema_vit, frames, device, dtype, args.norm, allowed
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_names = ",".join(allowed)
    full_map = f"({full_names}) -> ({full_names})"
    print(
        f"family={family} L={level} wavenumber={args.wavenumber} stems={stems} "
        f"levels={list(allowed)}",
        flush=True,
    )

    if family == "plus":
        matvec, state_dim, _, _ = eigs.plus_matvec(
            ema_vit, model, levels["U"], [levels[name] for name in allowed if name != "U"]
        )
    else:
        matvec, state_dim, _, _, _ = eigs.l3_matvec(
            ema_vit, levels["U"], levels["Z1"], levels["Z2"]
        )
    spec = eigs.spectrum(
        matvec, state_dim, args.k, args.ncv, args.maxiter, args.tol, args.seed
    )
    vals = spec["eigenvalues"]
    eigs_dir = save_dir / "eigs"
    eigs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eigs_dir / "eigenvalues.npz",
        eigenvalues=vals,
        residual=spec["residual"],
        converged=spec["converged"],
        k_requested=spec["k_requested"],
        k_returned=spec["k_returned"],
        state_dim=np.asarray(state_dim, dtype=np.int64),
        stems=np.asarray(stems, dtype=np.int64),
        which=np.asarray("LM"),
    )
    write_rank_plots(
        eigs_dir / "plots",
        vals,
        _title(args.wavenumber, level, full_map, "eigenvalues", args.stem),
        kind="eigenvalue",
    )
    print(
        f"eigs returned={int(spec['k_returned'])}/{args.k} "
        f"leading={float(np.max(np.abs(vals))):.6g} "
        f"residual_max={float(np.nanmax(spec['residual'])):.3e}",
        flush=True,
    )

    tags = [f"{full_names}-{full_names}"]
    tags.extend(f"{src}-{dst}" for src in allowed for dst in allowed)
    summaries = []
    for index, tag in enumerate(tags):
        block = parse_block_tag(tag, allowed)
        matvec_b, rmatvec, n_in, n_out = svs.operators(
            ema_vit, levels, allowed, block.inputs, block.outputs, forward
        )
        k = _sv_k(args.k, n_in, n_out)
        got = svs.largest_singular_values(
            matvec_b,
            rmatvec,
            n_out,
            n_in,
            k,
            args.ncv,
            args.maxiter,
            args.tol,
            args.seed + 1 + index,
        )
        out_dir = save_dir / "sv" / tag_to_dirname(block.tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "singular_values.npz",
            singular_values=got["singular_values"],
            residual=got["residual"],
            converged=np.asarray(got["converged"]),
            k_requested=np.asarray(args.k, dtype=np.int64),
            k_used=np.asarray(got["singular_values"].size, dtype=np.int64),
            n_in=np.asarray(n_in, dtype=np.int64),
            n_out=np.asarray(n_out, dtype=np.int64),
            tag=np.asarray(block.tag),
            which=np.asarray("LM"),
        )
        mapping = f"({','.join(block.inputs)}) -> ({','.join(block.outputs)})"
        write_rank_plots(
            out_dir / "plots",
            got["singular_values"],
            _title(args.wavenumber, level, mapping, "singular values", args.stem),
            kind="singular",
        )
        leading = float(got["singular_values"][0])
        summaries.append(
            {
                "tag": block.tag,
                "n_in": n_in,
                "n_out": n_out,
                "k": int(got["singular_values"].size),
                "largest": leading,
                "converged": bool(got["converged"]),
                "residual_max": float(np.nanmax(got["residual"])),
            }
        )
        print(
            f"sv {block.tag} shape={n_out}x{n_in} k={got['singular_values'].size} "
            f"largest={leading:.6g} converged={bool(got['converged'])}",
            flush=True,
        )

    manifest = {
        "checkpoint": args.checkpoint,
        "eval_dir": args.eval_dir,
        "wavenumber": args.wavenumber,
        "level": level,
        "family": family,
        "stem": args.stem,
        "stems": stems,
        "k": args.k,
        "which": "LM",
        "full_map": full_map,
        "eigenvalue_leading_modulus": float(np.max(np.abs(vals))),
        "singular_blocks": summaries,
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {save_dir}", flush=True)


if __name__ == "__main__":
    main()
