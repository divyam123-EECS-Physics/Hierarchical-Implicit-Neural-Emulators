#!/usr/bin/env python3
"""Eigenvalue Argands for square blocks from saved npz. No Arnoldi."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import numpy as np

from plot_saved_permutation_strips import ROOTS, _label
from rank_spectrum_plots import write_rank_plots

HOME = Path("/global/homes/d/divyam/eig_argands")


def _draw(root: Path, dest_name: str, values: np.ndarray, title: str, home_dir: Path) -> None:
    tmp = root / "_eig_argand_tmp"
    write_rank_plots(tmp, values, title, kind="eigenvalue")
    src = tmp / "argand.png"
    src.replace(root / dest_name)
    home_dir.mkdir(parents=True, exist_ok=True)
    (home_dir / dest_name).write_bytes((root / dest_name).read_bytes())
    modulus = tmp / "modulus.png"
    if modulus.exists():
        modulus.unlink()
    if tmp.exists():
        tmp.rmdir()


def redraw(name: str, level: int, root: Path) -> None:
    home_dir = HOME / name
    full = "U,Z1" if level == 2 else "U,Z1,Z2"
    jobs = [
        (
            root / "eigs" / "eigenvalues.npz",
            "argand_eig_full.png",
            f"{full}-{full}",
        ),
        (
            root / "eigs_blocks" / "Z1-Z1" / "eigenvalues.npz",
            "argand_eig_dz1_dz1.png",
            "Z1-Z1",
        ),
    ]
    if level == 3:
        jobs.append(
            (
                root / "eigs_blocks" / "Z2-Z2" / "eigenvalues.npz",
                "argand_eig_dz2_dz2.png",
                "Z2-Z2",
            )
        )
    uu = root / "eigs_blocks" / "U-U" / "eigenvalues.npz"
    if uu.is_file():
        jobs.append((uu, "argand_eig_du_du.png", "U-U"))
    written = []
    for path, filename, tag in jobs:
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path)["eigenvalues"]
        title = f"L={level}, {_label(tag)} eigenvalues, stem 91836"
        _draw(root, filename, values, title, home_dir)
        if filename == "argand_eig_full.png":
            (root / "argand_full.png").write_bytes((root / filename).read_bytes())
        written.append(filename)
    print(f"{name} {written}", flush=True)


def main() -> None:
    for name, (_wavenumber, level, root) in ROOTS.items():
        redraw(name, level, root)


if __name__ == "__main__":
    main()
