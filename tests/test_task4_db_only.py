#!/usr/bin/env python3
"""Task 4 flag and Argand bookkeeping. No checkpoint and no GPU."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "eval"))

from task3_parasitic_arnoldi import parse_args, plot_db_argand


def test_db_only_accepts_the_production_width():
    args = parse_args(
        [
            "--db-only",
            "--n-gpus",
            "4",
            "--n-states",
            "20",
            "--k",
            "12",
            "--ncv",
            "32",
        ]
    )
    assert args.db_only
    assert args.n_states == 20
    assert args.k == 12
    assert not args.db


def test_db_only_rejects_a_narrow_spectrum():
    try:
        parse_args(["--db-only", "--k", "4"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("k=4 should be rejected")


def test_argand_marks_zero_and_one_and_stores_distances():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        values = np.array([0.2 + 0.1j, 0.9 + 0.0j], dtype=np.complex128)
        residuals = np.array([1e-6, 1e-4])
        np.savez_compressed(
            root / "state_00.npz",
            values_real=np.real(values),
            values_imag=np.imag(values),
            residuals=residuals,
            distance_from_1=np.abs(values - 1.0),
            value_converged=residuals <= 1e-3,
            sample=np.asarray(0),
        )
        plot_db_argand(root)
        assert (root / "db_argand.png").is_file()
        saved = np.load(root / "db_eigenvalues.npz")
        assert np.allclose(saved["distance_from_1"], np.abs(values - 1.0))
        assert saved["value_converged"].dtype == bool


if __name__ == "__main__":
    test_db_only_accepts_the_production_width()
    test_db_only_rejects_a_narrow_spectrum()
    test_argand_marks_zero_and_one_and_stores_distances()
    print("ok")
