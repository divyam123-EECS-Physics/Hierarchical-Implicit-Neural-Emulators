#!/usr/bin/env python3
"""Redraw saved stem-91836 spectra as permutation strip plots.

The x-axis is the map. Each column is the 50 largest-modulus values,
colored so rank 1 is hottest.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import numpy as np

from rank_spectrum_plots import write_permutation_strip, write_rank_plots

LEVELS = ("U", "Z1", "Z2")
ROOTS = {
    "k4_l2": (
        4,
        2,
        Path("/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/jacobian_frame_spectrum/stem91836_k50_rankcolor"),
    ),
    "k5_l2": (
        5,
        2,
        Path("/pscratch/sd/d/divyam/hine_runs/k5/l2/jacobian_frame_spectrum/stem91836_k50_rankcolor"),
    ),
    "k4_l3": (
        4,
        3,
        Path("/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_frame_spectrum/stem91836_k50_rankcolor"),
    ),
    "k5_l3": (
        5,
        3,
        Path("/pscratch/sd/d/divyam/hine_runs/k5/l3/jacobian_frame_spectrum/stem91836_k50_rankcolor"),
    ),
}


_PRETTY = {"U": "u", "Z1": "z_1", "Z2": "z_2"}


def _side(text: str) -> str:
    parts = [_PRETTY[name] for name in text.split(",")]
    if len(parts) == 1:
        return parts[0]
    return "(" + ",".join(parts) + ")"


def _label(tag: str) -> str:
    left, right = tag.split("-")
    return rf"$\frac{{\partial {_side(right)}}}{{\partial {_side(left)}}}$"


def _sort_key(tag: str) -> tuple:
    left, right = tag.split("-")
    if "," in left or "," in right:
        return (1, 0, 0)
    return (0, LEVELS.index(left), LEVELS.index(right))


def redraw(name: str, wavenumber: int, level: int, root: Path) -> None:
    eigs = np.load(root / "eigs" / "eigenvalues.npz")["eigenvalues"]
    full = "U,Z1" if level == 2 else "U,Z1,Z2"
    eigs_title = f"L={level}, {_label(f'{full}-{full}')} eigenvalues, stem 91836"
    write_rank_plots(root / "eigs" / "plots", eigs, eigs_title, kind="eigenvalue")
    write_permutation_strip(
        root / "eigenvalues_by_map.png",
        [(_label(f"{full}-{full}"), eigs)],
        eigs_title,
        r"$|\lambda|$",
    )

    series = []
    for path in sorted((root / "sv").iterdir()):
        tag = str(np.load(path / "singular_values.npz")["tag"])
        values = np.load(path / "singular_values.npz")["singular_values"]
        series.append((tag, values))
        block_title = f"L={level}, {_label(tag)} singular values, stem 91836"
        write_rank_plots(path / "plots", values, block_title, kind="singular")
    series.sort(key=lambda item: _sort_key(item[0]))
    labeled = [(_label(tag), values) for tag, values in series]
    write_permutation_strip(
        root / "singular_values_by_block.png",
        labeled,
        f"L={level}, singular values by block, stem 91836",
        "singular value",
        log_y=True,
    )
    by_tag = {tag: values for tag, values in series}
    named = [
        ("U-U", "argand_du_du.png", "singular"),
        ("Z1-U", "argand_du_dz1.png", "singular"),
        (f"{full}-{full}", "argand_full.png", "eigenvalue"),
    ]
    if level == 3:
        named.insert(2, ("Z2-U", "argand_du_dz2.png", "singular"))
    spectra = dict(by_tag)
    spectra[f"{full}-{full}"] = eigs
    for tag, filename, kind in named:
        noun = "eigenvalues" if kind == "eigenvalue" else "singular values"
        title = f"L={level}, {_label(tag)} {noun}, stem 91836"
        write_rank_plots(root / "_argand_tmp", spectra[tag], title, kind=kind)
        (root / "_argand_tmp" / "argand.png").replace(root / filename)
    print(f"{name} blocks={len(labeled)} argands={len(named)}", flush=True)


def main() -> None:
    for name, (wavenumber, level, root) in ROOTS.items():
        redraw(name, wavenumber, level, root)


if __name__ == "__main__":
    main()
