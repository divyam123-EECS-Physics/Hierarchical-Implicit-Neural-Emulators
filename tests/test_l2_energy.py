#!/usr/bin/env python3
"""Synthetic checks for the L=2 energy budget. No checkpoint and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, svds

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for entry in (_REPO, _EVAL):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from energy_l2_model import EnergyL2Step, WeightedBlocks, prepare_image
from energy_metric import (
    EnergyMetric,
    first_exit,
    principal_angles,
    ratio_or_nan,
    sort_singular_triplets,
    subspace_fraction,
)
from eval_kinetic_energy_survival import kinetic_energy_torch
from eval_l2_energy import analyze_window, window_closures_ok


class _Affine(torch.nn.Module):
    def __init__(self, quadratic: bool = False):
        super().__init__()
        self.quadratic = quadratic
        self.fine = torch.nn.Conv2d(1, 1, 1, bias=False).double()
        self.lat = torch.nn.Conv2d(1, 1, 1, bias=False).double()
        with torch.no_grad():
            self.fine.weight.fill_(0.3)
            self.lat.weight.fill_(0.2)

    def forward(self, u, coarses, t):
        z = coarses[0]
        factor_y = u.shape[-2] // z.shape[-2]
        factor_x = u.shape[-1] // z.shape[-1]
        z_up = z.repeat_interleave(factor_y, dim=-2).repeat_interleave(factor_x, dim=-1)
        high = self.fine(u) + 0.1 * z_up
        if self.quadratic:
            high = high + 0.05 * u * u
        coarse = self.lat(z)
        return high, [coarse]


class _Recording(EnergyL2Step):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_means = []
        self.z_means = []

    def step(self, u_phys, z_phys):
        self.u_means.append(float(u_phys.detach().mean()))
        self.z_means.append(float(z_phys.detach().mean()))
        return super().step(u_phys, z_phys)


def _metric_and_torch(nx: int = 16):
    metric = EnergyMetric(nx, nx)
    kx = torch.from_numpy(np.ascontiguousarray(metric.Kx))
    ky = torch.from_numpy(np.ascontiguousarray(metric.Ky))
    inv = torch.from_numpy(np.ascontiguousarray(metric.invKsq))
    return metric, kx, ky, inv


def test_energy_matches_kinetic_energy_and_exact_split():
    metric, kx, ky, inv = _metric_and_torch(16)
    rng = np.random.default_rng(1)
    field = rng.normal(size=(16, 16))
    other = rng.normal(size=(16, 16))
    torch_e = float(kinetic_energy_torch(torch.from_numpy(field).view(1, 16, 16), kx, ky, inv))
    assert abs(metric.energy(field) - torch_e) <= 1e-10 * max(abs(torch_e), 1.0)
    split = metric.energy_split(field, other)
    assert abs(split["closure"]) <= 1e-10 * max(abs(split["signed_energy_error"]), 1.0)
    assert abs(metric.quad_shells(field).sum() - metric.energy(field)) <= 1e-10
    err = other - field
    assert abs(metric.cross_shells(field, err).sum() - split["L"]) <= 1e-10
    assert abs(metric.quad_shells(err).sum() - split["Q"]) <= 1e-9
    bands = metric.band_sums(metric.quad_shells(field))
    assert abs(sum(bands.values()) + metric.quad_shells(field)[0] - metric.energy(field)) <= 1e-10


def test_pack_norm_rms_and_adjoint():
    metric = EnergyMetric(16, 16)
    rng = np.random.default_rng(2)
    field = rng.normal(size=(16, 16))
    packed = metric.pack(field)
    assert abs(np.dot(packed, packed) - metric.inner_G(field, field)) <= 1e-10
    centered = field - field.mean()
    u_vel, v_vel = metric.velocities(centered)
    u_back, v_back = metric.velocities_from_coord(metric.pack(centered))
    assert np.max(np.abs(u_vel - u_back)) <= 1e-10
    assert np.max(np.abs(v_vel - v_back)) <= 1e-10
    coord = rng.normal(size=metric.n_coord)
    adjoint = metric.pack_adjoint(coord)
    assert abs(np.dot(metric.pack(field), coord) - np.sum(field * adjoint)) <= 1e-8 * max(np.linalg.norm(coord), 1.0)
    unit = rng.normal(size=(16, 16))
    unit = unit / np.linalg.norm(unit)
    physical = metric.physical_from_rms_coord(unit, (16, 16))
    assert abs(np.sqrt(np.mean(physical * physical)) - 1.0) <= 1e-12
    assert np.allclose(metric.rms_coord_from_physical(physical), unit.ravel())


def test_mean_mode_is_removed_and_nyquist_is_kept():
    metric = EnergyMetric(16, 16)
    assert metric.invKsq[0, 0] == 0.0
    assert np.all(np.isfinite(metric.invKsq[:, -1]))
    assert metric.weight[0, 0] == 1.0
    assert metric.weight[0, -1] == 1.0
    assert metric.weight[0, 1] == 2.0
    constant = np.ones((16, 16))
    assert abs(metric.energy(constant)) <= 1e-12


def test_staggered_indices_and_linear_budget():
    module = _Affine(quadratic=False).double()
    step = _Recording(module, norm=10.0, ds=2, device=torch.device("cpu"))
    metric = EnergyMetric(8, 8)
    frames = [(0.2 * n + 0.1) * np.ones((8, 8)) for n in range(6)]
    result = analyze_window(step, metric, frames, n_steps=3, mu_E=0.0, sigma_E=1.0)
    assert np.allclose(step.u_means[0::2], [0.1, 0.3, 0.5])
    assert np.allclose(step.z_means[0::2], [0.3, 0.5, 0.7])
    assert result["init_e_H_norm"] == 0.0 and result["init_e_Z_norm"] == 0.0
    assert window_closures_ok(result)
    assert np.max(result["abs_euclid_delta"]) < 1e-10
    assert np.all(result["e_Z_norm"][0] == 0.0)


def test_nonlinear_budget_keeps_exact_identities():
    module = _Affine(quadratic=True).double()
    step = EnergyL2Step(module, norm=10.0, ds=2, device=torch.device("cpu"))
    metric = EnergyMetric(8, 8)
    rng = np.random.default_rng(3)
    frames = [rng.normal(size=(8, 8)) for _ in range(6)]
    result = analyze_window(step, metric, frames, n_steps=3, mu_E=0.0, sigma_E=1.0)
    assert window_closures_ok(result)
    assert np.max(result["abs_euclid_delta"]) > 1e-6
    assert np.isnan(ratio_or_nan(1.0, 0.0))
    assert first_exit(np.array([1.0, 4.0, 6.0])) == 2
    assert first_exit(np.array([1.0, 5.0])) == -1


def test_jvp_matches_fd_and_vjp():
    for quadratic in (False, True):
        module = _Affine(quadratic=quadratic).double()
        step = EnergyL2Step(module, norm=10.0, ds=2, device=torch.device("cpu"))
        rng = np.random.default_rng(4 + int(quadratic))
        u = prepare_image(rng.normal(size=(8, 8)), torch.device("cpu"))
        z = prepare_image(rng.normal(size=(4, 4)), torch.device("cpu"))
        v = prepare_image(rng.normal(size=(8, 8)), torch.device("cpu"))
        w = prepare_image(rng.normal(size=(4, 4)), torch.device("cpu"))
        cot = prepare_image(rng.normal(size=(8, 8)), torch.device("cpu"))
        jvp_a = step.jvp_A(u, z, v)
        jvp_b = step.jvp_B(u, z, w)
        fd_a = step.finite_difference_A(u, z, v, 1e-6)
        fd_b = step.finite_difference_B(u, z, w, 1e-6)
        assert jvp_a.dtype == torch.float64 and jvp_b.dtype == torch.float64
        rel_a = float((jvp_a - fd_a).norm() / jvp_a.norm())
        rel_b = float((jvp_b - fd_b).norm() / jvp_b.norm())
        assert rel_a < 1e-5 and rel_b < 1e-5
        back_a = step.vjp_A(u, z, cot)
        back_b = step.vjp_B(u, z, cot)
        left_a = float(torch.sum(jvp_a * cot))
        right_a = float(torch.sum(v * back_a))
        left_b = float(torch.sum(jvp_b * cot))
        right_b = float(torch.sum(w * back_b))
        assert abs(left_a - right_a) <= 1e-8 * max(abs(left_a), 1.0)
        assert abs(left_b - right_b) <= 1e-8 * max(abs(left_b), 1.0)


def test_dense_partial_svd_and_q_e():
    rng = np.random.default_rng(5)
    dense_a = rng.normal(size=(8, 6))
    dense_b = rng.normal(size=(8, 3))
    op_a = LinearOperator(
        dense_a.shape,
        matvec=lambda x: dense_a @ x,
        rmatvec=lambda y: dense_a.T @ y,
        dtype=np.float64,
    )
    _u, values, vh = svds(op_a, k=3, ncv=5, which="LM", v0=rng.standard_normal(6))
    left, values, right = sort_singular_triplets(_u, values, vh.T)
    exact = np.linalg.svd(dense_a, compute_uv=False)
    assert np.allclose(values, exact[:3], rtol=1e-6, atol=1e-8)
    assert np.all(np.diff(values) <= 1e-12)
    assert orthonormality(left) < 1e-8 and orthonormality(right) < 1e-8
    product = np.concatenate([dense_a, dense_b], axis=1)
    q_matrix = product.T @ product
    op_q = LinearOperator(
        q_matrix.shape,
        matvec=lambda x: q_matrix @ x,
        rmatvec=lambda x: q_matrix @ x,
        dtype=np.float64,
    )
    evals, _evecs = eigsh(op_q, k=4, which="LA", ncv=8, v0=rng.standard_normal(9))
    exact_evals = np.sort(np.linalg.eigvalsh(q_matrix))[::-1][:4]
    assert np.allclose(np.sort(evals)[::-1], exact_evals, rtol=1e-6, atol=1e-8)
    angles = principal_angles(np.eye(4, 2), np.eye(4, 2))
    assert np.allclose(angles, 0.0, atol=1e-12)
    assert abs(subspace_fraction(np.eye(4, 2), np.array([1.0, 0.0, 3.0, 0.0])) - 0.1) < 1e-12
    assert np.isnan(subspace_fraction(np.eye(3, 1), np.zeros(3)))


def orthonormality(basis: np.ndarray) -> float:
    gram = basis.T @ basis
    return float(np.linalg.norm(gram - np.eye(gram.shape[0])))


def test_weighted_blocks_match_dense_svd():
    module = _Affine(quadratic=False).double()
    step = EnergyL2Step(module, norm=10.0, ds=2, device=torch.device("cpu"))
    metric = EnergyMetric(8, 8)
    rng = np.random.default_rng(6)
    u = prepare_image(rng.normal(size=(8, 8)), torch.device("cpu"))
    z = prepare_image(rng.normal(size=(4, 4)), torch.device("cpu"))
    blocks = WeightedBlocks(step, metric, u, z)
    dense_a = np.stack([blocks.apply_A(np.eye(blocks.n_fine)[i]) for i in range(blocks.n_fine)], axis=1)
    dense_b = np.stack([blocks.apply_B(np.eye(blocks.n_latent)[i]) for i in range(blocks.n_latent)], axis=1)
    op_a = LinearOperator(
        (blocks.n_out, blocks.n_fine),
        matvec=blocks.apply_A,
        rmatvec=blocks.apply_A_adjoint,
        dtype=np.float64,
    )
    _u, values, vh = svds(op_a, k=3, which="LM", ncv=8, v0=rng.standard_normal(blocks.n_fine))
    _left, values, _right = sort_singular_triplets(_u, values, vh.T)
    exact = np.linalg.svd(dense_a, compute_uv=False)
    assert np.allclose(values, exact[:3], rtol=1e-5, atol=1e-8)
    v = rng.normal(size=blocks.n_fine)
    y = rng.normal(size=blocks.n_out)
    assert abs(np.dot(blocks.apply_A(v), y) - np.dot(v, blocks.apply_A_adjoint(y))) <= 1e-7 * max(np.linalg.norm(v), 1.0)
    w = rng.normal(size=blocks.n_latent)
    assert abs(np.dot(blocks.apply_B(w), y) - np.dot(w, blocks.apply_B_adjoint(y))) <= 1e-7 * max(np.linalg.norm(w), 1.0)
    cross = dense_a.T @ dense_b
    m_dense = np.stack([blocks.apply_M_HZ(np.eye(blocks.n_latent)[i]) for i in range(blocks.n_latent)], axis=1)
    assert np.allclose(m_dense, cross, rtol=1e-6, atol=1e-8)
    product = np.concatenate([dense_a, dense_b], axis=1)
    q_dense = product.T @ product
    prod = rng.normal(size=blocks.n_fine + blocks.n_latent)
    assert np.allclose(blocks.apply_Q_E(prod), q_dense @ prod, rtol=1e-6, atol=1e-8)
    # A and B right vectors live in different dimensions and are not dotted.
    assert dense_a.shape[1] != dense_b.shape[1]


def test_reference_cache_interval_if_present():
    path = Path("/pscratch/sd/d/divyam/hine_runs/paper_1gpu/jacobian_manual_mse/reference_kinetic_energy.npz")
    if not path.is_file():
        return
    cache = np.load(path)
    stems = np.asarray(cache["stems"])
    energies = np.asarray(cache["energies"], dtype=np.float64)
    assert int(stems[0]) == 10000 and int(stems[-1]) == 105000 and stems.size == 95001
    assert abs(float(cache["mu_E"]) - float(energies.mean())) <= 1e-12
    assert abs(float(cache["sigma_E"]) - float(energies.std(ddof=0))) <= 1e-12


if __name__ == "__main__":
    test_energy_matches_kinetic_energy_and_exact_split()
    test_pack_norm_rms_and_adjoint()
    test_mean_mode_is_removed_and_nyquist_is_kept()
    test_staggered_indices_and_linear_budget()
    test_nonlinear_budget_keeps_exact_identities()
    test_jvp_matches_fd_and_vjp()
    test_dense_partial_svd_and_q_e()
    test_weighted_blocks_match_dense_svd()
    test_reference_cache_interval_if_present()
    print("L2_ENERGY_TESTS_OK")
