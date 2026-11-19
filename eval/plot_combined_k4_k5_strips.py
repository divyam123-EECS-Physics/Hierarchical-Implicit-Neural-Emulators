#!/usr/bin/env python3
"""Overlay saved k=4 and k=5 singular-value strips. No spectra are recomputed."""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

import numpy as np

from plot_saved_permutation_strips import ROOTS, _label, _sort_key
from rank_spectrum_plots import write_paired_permutation_strip

OUT = Path("/global/homes/d/divyam")


def _series(root: Path) -> list[tuple[str, np.ndarray]]:
    rows = []
    for path in sorted((root / "sv").iterdir()):
        blob = np.load(path / "singular_values.npz")
        rows.append((str(blob["tag"]), blob["singular_values"]))
    rows.sort(key=lambda item: _sort_key(item[0]))
    return [(_label(tag), values) for tag, values in rows]


def main() -> None:
    for level in (2, 3):
        k4 = _series(ROOTS[f"k4_l{level}"][2])
        k5 = _series(ROOTS[f"k5_l{level}"][2])
        path = OUT / f"combined_sv_L{level}_k4_k5.png"
        write_paired_permutation_strip(
            path,
            k4,
            k5,
            f"L={level}, singular values by block, stem 91836",
            "singular value",
            "k = 4",
            "k = 5",
            log_y=True,
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()
