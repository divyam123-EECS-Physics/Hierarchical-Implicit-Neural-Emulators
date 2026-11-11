#!/usr/bin/env python3
"""Dense checks for the homogeneous tangent and the (p, w) form."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "eval"))

from l2_coupled_growth import augmented_step_dense, isolated_s_dense, pw_step_dense
from l2_fd_jvp import L2Point
from l2_retention import propagate_augmented, propagate_homogeneous


def test_injections_still_start_from_zero():
    u = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    z = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    point = L2Point(u, z, torch.ones(1), u, z, (0, 0, 0))

    def boom(_n):
        raise AssertionError("the zero initial state must not request a derivative")

    eps_u = [torch.ones_like(u)]
    eps_z = [2.0 * torch.ones_like(z)]
    e, eta = propagate_augmented(boom, [point], eps_u, eps_z, 1e-3)
    assert torch.allclose(e[0], torch.zeros_like(u))
    assert torch.allclose(e[1], eps_u[0])
    assert torch.allclose(eta[1], eps_z[0])


def test_feedback_can_grow_while_s_is_damped():
    A = np.array([[0.2]])
    B = np.array([[3.0]])
    C = np.array([[0.5]])
    K = np.array([[0.2]])
    zero = np.zeros((1, 1))
    block = np.block([[A, B], [C, K]])
    assert np.max(np.abs(np.linalg.eigvals(block))) > 1.0
    assert abs(float(isolated_s_dense(B, K, zero, np.array([1.0]))[0])) < 1.0
    p = np.array([0.0])
    q = np.array([1.0])
    w = np.array([1.0])
    for _ in range(6):
        p, q = augmented_step_dense(A, B, C, K, p, q)
        w = isolated_s_dense(B, K, zero, w)
    assert math.sqrt(float(p[0] ** 2 + q[0] ** 2)) > 2.0
    assert abs(float(w[0])) < 0.2

    damped = np.block(
        [[np.array([[0.2]]), zero], [zero, np.array([[0.2]])]]
    )
    assert np.max(np.abs(np.linalg.eigvals(damped))) < 1.0


def test_pw_form_matches_the_augmented_map():
    rng = np.random.default_rng(0)
    A, B, C, K, M, M_next = (rng.normal(size=(2, 2)) for _ in range(6))
    p = rng.normal(size=2)
    q = rng.normal(size=2)
    w = q - M @ p
    p_next, q_next = augmented_step_dense(A, B, C, K, p, q)
    p_pw, w_pw = pw_step_dense(A, B, C, K, M, M_next, p, w)
    assert np.allclose(p_pw, p_next)
    assert np.allclose(w_pw, q_next - M_next @ p_next)


def test_shared_rescale_reconstructs_the_linear_product():
    base = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    point = L2Point(base, base, torch.ones(1), base, base, (0, 0, 0))

    def grow(_u, z):
        return 100.0 * _u, 100.0 * z

    args = dict(eps_rel=1e-3, lo=1e-6, hi=10.0)
    plain, _q_plain, _logs = propagate_homogeneous(
        lambda _n: grow, [point, point], base.clone(), base.clone(), rescale=False, **args
    )
    scaled, q_scaled, logs = propagate_homogeneous(
        lambda _n: grow, [point, point], base.clone(), base.clone(), rescale=True, **args
    )
    assert logs[-1] > 0.0
    for got, ref, log_s in zip(scaled, plain, logs):
        assert torch.allclose(got * math.exp(log_s), ref, rtol=1e-8, atol=1e-8)
    assert torch.allclose(q_scaled[-1] * math.exp(logs[-1]), plain[-1], rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    test_injections_still_start_from_zero()
    test_feedback_can_grow_while_s_is_damped()
    test_pw_form_matches_the_augmented_map()
    test_shared_rescale_reconstructs_the_linear_product()
    print("ok")
