#!/usr/bin/env python3
"""Exact scalar parasitic-root checks. No network and no solver.

The on-graph fit is alpha = a (1 - beta_s), gamma = a (a - kappa_s).
The parasitic root is s = kappa_s - a beta_s. |s| = |a| is not absolute
damping: the mode decays only when |a| < 1, and it is not damped relative
to the physical root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_matrix(beta_s: float, kappa_s: float, a: float) -> tuple[np.ndarray, float]:
    alpha = a * (1.0 - beta_s)
    gamma = a * (a - kappa_s)
    matrix = np.array([[alpha, beta_s], [gamma, kappa_s]], dtype=np.float64)
    s = kappa_s - a * beta_s
    return matrix, float(s)


def check_case(beta_s: float, kappa_s: float, a: float, expected_s: float, name: str) -> dict:
    matrix, s = fit_matrix(beta_s, kappa_s, a)
    trace = float(np.trace(matrix))
    det = float(np.linalg.det(matrix))
    # lambda^2 - (a+s) lambda + a s
    poly_ok = abs(trace - (a + s)) < 1e-12 and abs(det - (a * s)) < 1e-12
    roots = np.linalg.eigvals(matrix)
    # Match roots to {a, s} regardless of order.
    target = np.array([a, s], dtype=np.complex128)
    order = [0, 1]
    err = min(
        abs(roots[0] - target[0]) + abs(roots[1] - target[1]),
        abs(roots[0] - target[1]) + abs(roots[1] - target[0]),
    )
    return {
        "name": name,
        "beta_s": beta_s,
        "kappa_s": kappa_s,
        "a": a,
        "s": s,
        "expected_s": expected_s,
        "s_ok": abs(s - expected_s) < 1e-12,
        "poly_ok": bool(poly_ok),
        "eig_err": float(err),
        "abs_s": abs(s),
        "abs_a": abs(a),
        "grows": abs(s) > 1.0,
        "relative_to_physical": (
            "same magnitude as the physical root, not extra damping"
            if abs(abs(s) - abs(a)) < 1e-12
            else "different magnitude from the physical root"
        ),
    }


def run_checks(a: float = 0.8) -> list[dict]:
    cases = [
        check_case(1.0, 0.0, a, -a, "beta_s=1, kappa_s=0 -> s=-a"),
        check_case(1.0, a, a, 0.0, "beta_s=1, kappa_s=a -> s=0"),
        check_case(0.0, 0.3, a, 0.3, "beta_s=0 -> s=kappa_s"),
        check_case(0.0, 1.5, 0.5, 1.5, "exact fit with |s|>1"),
    ]
    bad = [c["name"] for c in cases if not (c["s_ok"] and c["poly_ok"] and c["eig_err"] < 1e-10)]
    if bad:
        raise AssertionError(f"toy root checks failed: {bad}")
    growing = cases[-1]
    if not growing["grows"]:
        raise AssertionError("expected one exact-fit parasitic root outside the unit disk")
    return cases


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default="/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task3_parasitic_debug_test/toy_roots.json",
    )
    p.add_argument("--a", type=float, default=0.8)
    args = p.parse_args()
    cases = run_checks(a=args.a)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2) + "\n")
    for case in cases:
        print(
            f"{case['name']}: s={case['s']:.6g} |s|={case['abs_s']:.6g} "
            f"({case['relative_to_physical']})",
            flush=True,
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
