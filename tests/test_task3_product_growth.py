#!/usr/bin/env python3
"""product_growth returns rates for a valid segment and rejects a short one."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "eval"))

import eval_jacobian_manual_mse as frames
import task3_parasitic_arnoldi as task3


def test_valid_segment_and_short_segment():
    originals = (
        frames.load_kept_paths,
        task3.load_omega,
        task3.pool_avg,
        task3.hine_S_action,
        task3.DNS_DATA,
    )
    try:
        frames.load_kept_paths = lambda _directory, _every: [(i, f"{i}.mat") for i in range(6)]
        task3.load_omega = lambda _path: (np.ones((8, 8), dtype=np.float64), 0.0)
        task3.pool_avg = lambda _field, _factor: np.ones((4, 4), dtype=np.float64)

        def half(_model, _solver, _u, _z, _u512, v, *_rest):
            return 0.5 * np.asarray(v, dtype=np.float64)

        task3.hine_S_action = half
        task3.DNS_DATA = Path(".")
        args = argparse.Namespace(
            every_nth_field=1,
            burn_in=0,
            product_steps=2,
            product_start=0,
            n_starts=1,
            seed=0,
            delta_t=0.05,
            fd_eps=1e-3,
            fd_ref_scale=1.0,
        )
        rates = task3.product_growth(args, None, None, 2, None, "eval")
        assert len(rates) == 1
        expected = 2.0 * np.log(0.5) / 0.1
        assert abs(rates[0]["rate"] - expected) < 1e-10
        assert rates[0]["underflow"] is False
        args.product_steps = 20
        try:
            task3.product_growth(args, None, None, 2, None, "eval")
        except ValueError as exc:
            assert "does not fit" in str(exc)
        else:
            raise AssertionError("a segment past the kept list must raise")
    finally:
        (
            frames.load_kept_paths,
            task3.load_omega,
            task3.pool_avg,
            task3.hine_S_action,
            task3.DNS_DATA,
        ) = originals


if __name__ == "__main__":
    test_valid_segment_and_short_segment()
    print("ok")
