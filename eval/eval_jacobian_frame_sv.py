#!/usr/bin/env python3
"""Largest singular values of one rectangular step-Jacobian block on one frame.

Default tags, at the training stagger used by the eigenvalue scripts:
  L=2 FutureINNPlus: U,Z1-Z1, the map (high, low) -> low
  L=3 FutureINN:     U,Z1,Z2-Z2, the map (high, mid, low) -> low

The count is --k, capped by the output dimension. ARPACK which=LM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse.linalg as spla
import torch

from eval_jacobian_high_eigs import gt_l3_state, load_kept_paths
from eval_jacobian_high_eigs_plus import _t_full, gt_plus_state
from eval_jacobian_manual_mse import load_model as load_l3
from eval_jacobian_manual_mse_plus import load_model as load_plus
from jacobian_blocks_lib import parse_block_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stem", type=int, default=91836)
    p.add_argument(
        "--n-frames",
        type=int,
        default=1,
        help="1 uses --stem. Above 1 draws that many kept-frame starts with --seed.",
    )
    p.add_argument("--tag", default="")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--ncv", type=int, default=0, help="0 means min(2k+1, dim-1).")
    p.add_argument("--maxiter", type=int, default=0, help="0 means 10*ncv.")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm", type=float, default=10.0)
    p.add_argument("--every-nth-field", type=int, default=5)
    p.add_argument("--eval-dir", default="/global/homes/d/divyam/TACSLAB/eval")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--save-dir", default="")
    return p.parse_args()


def peek_family(checkpoint: str) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    getter = args.get if isinstance(args, dict) else lambda name: getattr(args, name, None)
    if getter("num_coarse") is not None or getter("latent_res") is not None:
        return "plus"
    return "l3"


def choose_starts(n_kept: int, lookahead: int, n_samples: int, seed: int) -> np.ndarray:
    n_valid = n_kept - lookahead
    if n_samples > n_valid:
        raise ValueError(f"Asked for {n_samples} frames, only {n_valid} valid starts")
    rng = np.random.default_rng(seed)
    starts = rng.choice(n_valid, size=n_samples, replace=False)
    starts.sort()
    return starts.astype(np.int64)


def frames_at_stem(kept, stem: int, n_future: int):
    stems = [item[0] for item in kept]
    if stem not in stems:
        raise ValueError(f"stem {stem} is not in the kept eval list")
    pos = stems.index(stem)
    if pos + n_future >= len(kept):
        raise ValueError(f"stem {stem} needs {n_future} following kept frames")
    chosen = kept[pos : pos + n_future + 1]
    frames = [np.asarray(scipy.io.loadmat(path)["Omega"]) for _, path in chosen]
    return frames, [item[0] for item in chosen]


def _pack(names, tensors) -> torch.Tensor:
    return torch.cat([tensors[name].reshape(-1) for name in names], dim=0)


def _unpack(flat, names, templates):
    out = {}
    offset = 0
    for name in names:
        n = templates[name].numel()
        out[name] = flat[offset : offset + n].reshape_as(templates[name])
        offset += n
    return out


def operators(ema_vit, levels, allowed, in_names, out_names, forward):
    dtype = next(ema_vit.parameters()).dtype
    frozen = {name: levels[name] for name in allowed if name not in in_names}
    templates = {name: levels[name] for name in in_names}
    n_in = int(sum(templates[name].numel() for name in in_names))
    n_out = int(sum(levels[name].numel() for name in out_names))
    primals = tuple(levels[name] for name in in_names)

    def selected(*diff_tensors):
        merged = dict(frozen)
        for name, tensor in zip(in_names, diff_tensors):
            merged[name] = tensor
        return _pack(out_names, forward(merged))

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(v_np)).to(
            device=levels["U"].device, dtype=dtype
        )
        tangents = tuple(_unpack(v, in_names, templates)[name] for name in in_names)
        _, jv = torch.autograd.functional.jvp(selected, primals, tangents)
        return jv.detach().reshape(-1).float().cpu().numpy()

    def rmatvec(w_np: np.ndarray) -> np.ndarray:
        w = torch.from_numpy(np.ascontiguousarray(w_np)).to(
            device=levels["U"].device, dtype=dtype
        ).reshape(-1)
        _, vjp = torch.autograd.functional.vjp(selected, primals, w)
        if not isinstance(vjp, tuple):
            vjp = (vjp,)
        packed = torch.cat([part.reshape(-1) for part in vjp], dim=0)
        return packed.detach().reshape(-1).float().cpu().numpy()

    return matvec, rmatvec, n_in, n_out


def residuals(matvec, rmatvec, values, u, vt) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    if u is None or vt is None:
        return out
    for i, sigma in enumerate(values):
        left = matvec(np.asarray(vt[i], dtype=np.float32)) - sigma * u[:, i]
        right = rmatvec(np.asarray(u[:, i], dtype=np.float32)) - sigma * vt[i]
        scale = max(float(np.linalg.norm(u[:, i])), 1e-30)
        out[i] = float(np.hypot(np.linalg.norm(left), np.linalg.norm(right)) / scale)
    return out


def largest_singular_values(matvec, rmatvec, n_out, n_in, k, ncv, maxiter, tol, seed):
    m = min(n_out, n_in)
    if k >= m:
        raise ValueError(f"k={k} must be smaller than min(n_out, n_in)={m}")
    ncv_eff = ncv if ncv > 0 else min(max(2 * k + 1, 20), m - 1)
    ncv_eff = min(max(ncv_eff, k + 2), m - 1)
    if not (k < ncv_eff < m):
        raise ValueError(f"need k < ncv < min(dim), got k={k} ncv={ncv_eff} dim={m}")
    maxiter_eff = maxiter if maxiter > 0 else max(10 * ncv_eff, 300)
    op = spla.LinearOperator((n_out, n_in), matvec=matvec, rmatvec=rmatvec, dtype=np.float32)
    converged = True
    u = vt = None

    def _call(ncv_now, maxiter_now):
        return spla.svds(
            op,
            k=k,
            which="LM",
            ncv=ncv_now,
            maxiter=maxiter_now,
            tol=tol,
            return_singular_vectors=True,
            rng=seed,
        )

    try:
        u, values, vt = _call(ncv_eff, maxiter_eff)
    except spla.ArpackNoConvergence:
        ncv_retry = min(m - 1, max(ncv_eff * 2, k + 10))
        print(
            f"retrying singular values with ncv={ncv_retry} maxiter={maxiter_eff * 4}",
            flush=True,
        )
        try:
            u, values, vt = _call(ncv_retry, maxiter_eff * 4)
        except spla.ArpackNoConvergence as exc:
            values = np.asarray(exc.eigenvalues, dtype=np.float64)
            values = values[np.isfinite(values)]
            converged = False
            if values.size == 0:
                raise
            print(f"partial convergence: kept {values.size}/{k} singular values", flush=True)
    values = np.asarray(values, dtype=np.float64).real
    res = residuals(matvec, rmatvec, values, u, vt)
    order = np.argsort(-values)
    return {
        "singular_values": values[order],
        "residual": res[order],
        "converged": converged,
        "k_requested": k,
        "ncv": ncv_eff,
        "maxiter": maxiter_eff,
    }


def prepare_frame(family, model, ema_vit, frames, device, dtype, norm, allowed):
    if family == "plus":
        high, coarses = gt_plus_state(frames, 0, model, device, dtype, norm)
        levels = {"U": high.detach().to(dtype)}
        for i, coarse in enumerate(coarses):
            levels[f"Z{i + 1}"] = coarse.detach().to(dtype)
        t = _t_full(model, levels["U"])
        coarse_names = [name for name in allowed if name != "U"]

        def forward(merged):
            oh, ocs = ema_vit(merged["U"], [merged[name] for name in coarse_names], t)
            outs = {"U": oh}
            for i, oc in enumerate(ocs):
                outs[f"Z{i + 1}"] = oc
            return outs
    else:
        high, mid, low = gt_l3_state(frames, 0, model, device, dtype, norm)
        levels = {
            "U": high.detach().to(dtype),
            "Z1": mid.detach().to(dtype),
            "Z2": low.detach().to(dtype),
        }
        t = 2 * torch.ones(levels["U"].shape[0], device=levels["U"].device, dtype=dtype)

        def forward(merged):
            oh, om, ol = ema_vit(merged["U"], merged["Z1"], merged["Z2"], t)
            return {"U": oh, "Z1": om, "Z2": ol}
    return levels, forward


def write_curve_plot(path: Path, values: np.ndarray, title: str) -> None:
    values = np.asarray(values)
    if values.ndim == 1:
        from rank_spectrum_plots import write_rank_plots

        write_rank_plots(path.parent, values, title, kind="singular")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rank = np.arange(values.shape[-1])
    for row in values:
        ax.plot(rank, row, color="0.75", lw=0.8)
    ax.plot(rank, np.nanmean(values, axis=0), color="C0", lw=2.0, label="mean")
    ax.legend()
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_frames < 1:
        raise ValueError(f"--n-frames must be >= 1, got {args.n_frames}")
    family = peek_family(args.checkpoint)
    tag = args.tag or ("U,Z1-Z1" if family == "plus" else "U,Z1,Z2-Z2")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    kept = load_kept_paths(args.eval_dir, args.every_nth_field)
    if family == "plus":
        model, ema_vit = load_plus(args.checkpoint, device)
        allowed = ("U",) + tuple(f"Z{i + 1}" for i in range(int(model.num_coarse)))
        n_future = int(model.num_coarse)
    else:
        model, ema_vit = load_l3(args.checkpoint, device)
        allowed = ("U", "Z1", "Z2")
        n_future = 2
    block = parse_block_tag(tag, allowed)
    dtype = next(ema_vit.parameters()).dtype
    if args.n_frames == 1:
        windows = [frames_at_stem(kept, args.stem, n_future)]
    else:
        starts = choose_starts(len(kept), n_future, args.n_frames, args.seed)
        windows = []
        for start in starts:
            chosen = kept[int(start) : int(start) + n_future + 1]
            frames = [np.asarray(scipy.io.loadmat(path)["Omega"]) for _, path in chosen]
            windows.append((frames, [item[0] for item in chosen]))

    rows = []
    residual_rows = []
    high_stems = []
    converged = []
    shape = None
    k_used = None
    for index, (frames, stems) in enumerate(windows):
        levels, forward = prepare_frame(family, model, ema_vit, frames, device, dtype, args.norm, allowed)
        matvec, rmatvec, n_in, n_out = operators(
            ema_vit, levels, allowed, block.inputs, block.outputs, forward
        )
        k = min(args.k, min(n_out, n_in) - 2)
        if k < 1:
            raise ValueError(f"output dimension {n_out} is too small for a partial SVD")
        if k_used is None:
            k_used = k
            shape = (n_out, n_in)
            if k < args.k:
                print(
                    f"capping k from {args.k} to {k} because min(n_out, n_in)={min(n_out, n_in)}",
                    flush=True,
                )
        spec = largest_singular_values(
            matvec, rmatvec, n_out, n_in, k, args.ncv, args.maxiter, args.tol, args.seed + index
        )
        values = spec["singular_values"]
        pad = k - values.size
        if pad:
            values = np.concatenate([values, np.full(pad, np.nan)])
            residual = np.concatenate([spec["residual"], np.full(pad, np.nan)])
        else:
            residual = spec["residual"]
        rows.append(values)
        residual_rows.append(residual)
        high_stems.append(int(stems[0]))
        converged.append(bool(spec["converged"]))
        print(
            f"[{index + 1}/{len(windows)}] stem={stems[0]} returned={int(np.sum(np.isfinite(values)))}/{k} "
            f"converged={bool(spec['converged'])} largest={float(values[0]):.6g} "
            f"residual_max={float(np.nanmax(residual)):.3e}",
            flush=True,
        )

    table = np.vstack(rows)
    residual_table = np.vstack(residual_rows)
    tag_slug = block.tag.replace(",", "_")
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif args.n_frames == 1:
        save_dir = (
            Path(args.checkpoint).resolve().parent
            / "jacobian_frame_spectrum"
            / f"stem{args.stem}_sv_{tag_slug}"
        )
    else:
        save_dir = (
            Path(args.checkpoint).resolve().parent
            / "jacobian_frame_spectrum"
            / f"n{args.n_frames}_sv_{tag_slug}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / "singular_values.npz",
        singular_values=table,
        residual=residual_table,
        converged=np.asarray(converged),
        high_stems=np.asarray(high_stems, dtype=np.int64),
        k_requested=np.asarray(args.k, dtype=np.int64),
        k_used=np.asarray(k_used, dtype=np.int64),
        n_in=np.asarray(shape[1], dtype=np.int64),
        n_out=np.asarray(shape[0], dtype=np.int64),
        tag=np.asarray(block.tag),
        which=np.asarray("LM"),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    finite = table[np.isfinite(table)]
    summary = {
        "family": family,
        "tag": block.tag,
        "inputs": list(block.inputs),
        "outputs": list(block.outputs),
        "checkpoint": args.checkpoint,
        "n_frames": len(windows),
        "high_stems": high_stems,
        "n_in": shape[1],
        "n_out": shape[0],
        "k_requested": args.k,
        "k_used": k_used,
        "n_converged": int(sum(converged)),
        "largest_mean": float(np.nanmean(table[:, 0])),
        "largest_min": float(np.nanmin(table[:, 0])),
        "largest_max": float(np.nanmax(table[:, 0])),
        "residual_max": float(np.nanmax(residual_table)),
        "which": "LM",
        "seed": args.seed,
        "note": "Largest singular values of this rectangular block on each frame. which=LM.",
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    title = f"{family} ${block.tag}$, {len(windows)} frame" + ("s" if len(windows) != 1 else "")
    write_curve_plot(plot_dir / "singular_values.png", table if len(windows) > 1 else table[0], title)
    print(
        f"wrote {save_dir} largest mean/min/max="
        f"{summary['largest_mean']:.6g}/{summary['largest_min']:.6g}/{summary['largest_max']:.6g} "
        f"finite={finite.size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
