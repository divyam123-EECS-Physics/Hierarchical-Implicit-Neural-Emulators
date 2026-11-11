#!/usr/bin/env python3
"""CPU checks for Tasks 3 and 5. No checkpoint and no 512^2 DNS."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import torch
from torch import nn

_EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(_EVAL))

from coarse_projectors import PoolingProjector, SharpProjector, q_model
from l2_model_jvp import ModelStepPhysical
from py2d_inmem import (
    Py2DConfig,
    Py2DInMemory,
    embed_field,
    fd_tangent,
    rms,
    steps_for,
    truncate_field,
)
from task3_operators import compose_S, hine_S_action
from task3_toy_roots import run_checks


def test_toy_roots():
    cases = run_checks()
    growing = [c for c in cases if c["grows"]]
    assert len(growing) == 1
    assert growing[0]["s_ok"] and growing[0]["poly_ok"]


def test_steps_for_and_fd():
    assert steps_for(1e-4, 0.05) == 500
    try:
        steps_for(1e-4, 0.05001)
    except ValueError:
        pass
    else:
        raise AssertionError("non-integer interval was accepted")

    def flow(x):
        return 3.0 * x

    x = np.array([1.0, -2.0])
    v = np.array([2.0, 0.5])
    got = fd_tangent(flow, x, v, 1e-3, 1.0)
    assert np.allclose(got, 3.0 * v)
    assert np.allclose(fd_tangent(flow, x, np.zeros_like(v), 1e-3, 1.0), 0.0)


def test_embed_truncate_and_pooling():
    rng = np.random.default_rng(0)
    n = 16
    raw = rng.standard_normal((n, n))
    hat = np.fft.rfft2(raw)
    hat[n // 2, :] = 0.0
    hat[:, n // 2] = 0.0
    field = np.fft.irfft2(hat)
    back = truncate_field(embed_field(field, 32), n)
    assert np.allclose(back, field, atol=1e-10)
    proj = PoolingProjector(16, 8, 2)
    assert proj.right_inverse_error() < 1e-8


def test_sharp_projector_and_conditional():
    solver = Py2DInMemory(Py2DConfig(nx=16, Re=100.0, dt_dns=1e-2, fkx=1, fky=1, alpha=0.1, beta=1.0))
    proj = SharpProjector(solver, kc=3, shape="square")
    rng = np.random.default_rng(1)
    omega = rng.standard_normal((16, 16))
    hat = solver.jnp.asarray(np.fft.rfft2(omega))
    coarse = proj.coarse_hat(hat)
    assert np.allclose(np.asarray(proj.coarse_hat(coarse)), np.asarray(coarse), atol=1e-10)
    fine = proj.Q_hat(hat)
    assert rms(np.asarray(proj.coarse_hat(fine))) < 1e-10

    interval = 0.03
    path = solver.record_coarse_path(omega, proj, interval)
    pert = np.array(solver.to_hat_phys(proj.Q_hat(solver.jnp.asarray(np.fft.rfft2(rng.standard_normal((16, 16)))))))
    pert *= 1e-3 / rms(pert)
    out = solver.advance_conditional(omega + pert, path, proj, interval)
    got = np.asarray(proj.coarse_hat(solver.jnp.asarray(np.fft.rfft2(out))))
    assert np.allclose(got, path[-1], atol=1e-8)

    free = solver.advance_physical(omega, interval)
    ref = solver.startup(omega)
    fol = solver.startup(omega)
    ref, _fol, _series = solver.clamped_block(ref, fol, 3, None)
    assert np.allclose(solver.to_phys(ref), free, atol=1e-8)


def test_s_uses_phi_at_n_plus_1():
    class Model:
        def jvp_latent(self, u, z, w):
            return np.ones_like(u) * float(w.reshape(-1)[0]), 0.2 * w

    class Solver:
        def __init__(self):
            self.cfg = SimpleNamespace(nx=2)
            self.seen = []

        def advance_physical(self, u, interval):
            self.seen.append(np.asarray(u, dtype=np.float64))
            return 0.5 * np.asarray(u, dtype=np.float64)

    u_n = np.ones((2, 2))
    u_np1 = np.full((2, 2), 3.0)
    solver = Solver()
    out = hine_S_action(
        Model(), solver, u_n, np.ones((1, 1)), u_np1, np.ones((1, 1)),
        0.05, 1e-4, 1.0, 2, 2,
    )
    assert abs(float(np.asarray(out).reshape(-1)[0]) + 0.3) < 1e-6
    mid = 0.5 * (solver.seen[0] + solver.seen[1])
    assert np.allclose(mid, u_np1)
    assert not np.allclose(mid, u_n)

    def g_z(v):
        return np.array([2.0 * v[0]])

    def h_z(v):
        return np.array([0.25 * v[0]])

    phi_n = lambda w: np.array([9.0 * w[0]])
    phi_np1 = lambda w: np.array([0.5 * w[0]])
    got = compose_S(np.array([1.0]), g_z, h_z, phi_np1, lambda w: w)
    wrong = compose_S(np.array([1.0]), g_z, h_z, phi_n, lambda w: w)
    # K v - Phi_np1 B v = 0.25 - 0.5 * 2 = -0.75
    assert abs(float(np.asarray(got).reshape(-1)[0]) + 0.75) < 1e-12
    assert not np.allclose(got, wrong)


def test_continuous_clamp_differs_from_endpoint():
    def step(c, f):
        return c + 0.2 * f, 0.5 * f + 0.3 * c

    def rollout(c, f, n, mode, reference):
        for i in range(n):
            c, f = step(c, f)
            if mode == "every":
                c = reference[i]
            elif mode == "end" and i == n - 1:
                c = reference[i]
        return c, f

    n = 4
    reference = [1.0 + 0.1 * i for i in range(n)]
    c0, f0 = 1.0, 0.2
    every = rollout(c0, f0, n, "every", reference)
    end = rollout(c0, f0, n, "end", reference)
    free = rollout(c0, f0, n, "none", reference)
    assert every[1] != end[1]
    assert free[1] != every[1]


def test_model_jvp_norm_cancels():
    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.3))
            self.b = nn.Parameter(torch.tensor(-0.4))
            self.k = nn.Parameter(torch.tensor(0.2))

        def forward(self, u, zs, t):
            z = zs[0]
            z_up = z.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            return self.a * u + self.b * z_up, [self.k * z]

    lin = Lin().eval()
    rng = np.random.default_rng(2)
    u = rng.standard_normal((4, 4))
    z = rng.standard_normal((2, 2))
    w = rng.standard_normal((2, 2))
    v = q_model(rng.standard_normal((4, 4)), 2)
    b1, k1 = ModelStepPhysical(lin, 10.0).jvp_latent(u, z, w)
    b2, k2 = ModelStepPhysical(lin, 2.0).jvp_latent(u, z, w)
    assert np.allclose(b1, b2, atol=1e-5)
    assert np.allclose(k1, k2, atol=1e-5)
    assert np.allclose(k1, 0.2 * w, atol=1e-5)
    a1 = ModelStepPhysical(lin, 10.0).jvp_fine_fixed_latent(u, z, v)
    a2 = ModelStepPhysical(lin, 4.0).jvp_fine_fixed_latent(u, z, v)
    assert np.allclose(a1, a2, atol=1e-5)
    assert np.allclose(a1, 0.3 * v, atol=1e-5)


def test_task5_partitions_cover_every_case():
    from task5_hine_vs_conditional import hine_case_keys
    from task5_sync_screen import _row_from_npz, sync_task_keys

    keys = sync_task_keys(3, [0, 4, 8, 12, 16, 20, 24, 32, 64], 3)
    assert len(keys) == 81
    assert len(set(keys)) == 81
    counts = [sum(index % 4 == rank for index in range(len(keys))) for rank in range(4)]
    assert counts == [21, 20, 20, 20]
    cases = hine_case_keys(8, 4, 1e-3, 2, [1e-4, 1e-3, 1e-2])
    assert len(cases) == 32 + 16
    per_sample = {}
    for sample, _direction, eps in cases:
        per_sample.setdefault(sample, set()).add(eps)
    assert len(per_sample[0]) == 3
    assert len(per_sample[2]) == 1
    path = Path("/tmp/task5_old_segment.npz")
    np.savez_compressed(
        path,
        kc=np.asarray(4),
        stem=np.asarray(98456),
        lambda_c=np.asarray(-0.3),
        log_gains=np.asarray([-0.1, -0.2]),
        first_block_E=np.asarray([1e-3, 9e-4]),
        dt_dns=np.asarray(1e-4),
        coarse_rms=np.asarray(1e-15),
    )
    row = _row_from_npz(path, 0, 1, 984.56, 0.05)
    assert row["kc"] == 4 and row["direction"] == 1
    assert row["underflow"] is False
    assert abs(row["step_factor"] - np.exp(-0.3 * 0.05)) < 1e-12


def test_bands_partition_and_pooling_fine_part():
    from coarse_projectors import PoolingProjector, apply_band, band_real_dof, radial_band_masks
    from task5_sync_screen import screen_levels, segment_filename

    masks = radial_band_masks(32)
    covered = np.zeros(masks[0][1].shape, dtype=np.int32)
    for _label, mask in masks:
        covered += mask.astype(np.int32)
    assert np.all(covered == 1)
    assert sum(band_real_dof(mask) for _label, mask in masks) == 32 * 32
    field = np.random.default_rng(0).standard_normal((32, 32))
    limited = apply_band(field, masks[2][1])
    hat = np.fft.rfft2(limited)
    assert np.max(np.abs(hat[~masks[2][1]])) < 1e-8

    class _Args:
        cutoff_shape = "pooling"
        pool_factors = [2, 4, 8, 16, 32]
        kc = [0, 4]

    levels, kind = screen_levels(_Args())
    assert levels == [2, 4, 8, 16, 32] and kind == "pool"
    assert segment_filename(8, 0, 1, "pool") == "pool8_seg0_dir1.npz"
    assert segment_filename(8, 0, 1) == "kc8_seg0_dir1.npz"

    solver_side = 32
    projector = PoolingProjector(solver_side, solver_side, 8)
    hat_in = projector.jnp.fft.rfft2(
        projector.jnp.asarray(np.random.default_rng(1).standard_normal((solver_side, solver_side)))
    )
    fine = np.array(projector.Q_phys_hat(hat_in))
    coarse = np.array(projector.D_phys(projector.jnp.asarray(fine)))
    assert rms(coarse) < 1e-8
    assert rms(fine) > 1e-3


if __name__ == "__main__":
    test_toy_roots()
    test_steps_for_and_fd()
    test_embed_truncate_and_pooling()
    test_sharp_projector_and_conditional()
    test_s_uses_phi_at_n_plus_1()
    test_continuous_clamp_differs_from_endpoint()
    test_model_jvp_norm_cancels()
    test_task5_partitions_cover_every_case()
    test_bands_partition_and_pooling_fine_part()
    print("ok")
