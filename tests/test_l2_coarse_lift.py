#!/usr/bin/env python3
"""CPU checks for L=2 coarse-lift operators. No checkpoint and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coarse_lift_ops import (
    SPECTRAL_SAMPLE_INDICES,
    SPECTRAL_STEPS,
    TWOSTEP_STEPS,
    D_dagger_numpy,
    D_numpy,
    D_star_numpy,
    LiftedBlocks,
    Q_numpy,
    cosine,
    directional_gains,
    jvp_fd_report,
    feedback_action,
    latent_shift,
    probe_energy,
    probe_quadratics,
    projector_identity_report,
    rms_scale,
    route_matrices,
    sign_probes,
    torch_matches_numpy_pool,
    two_step_parts,
)
from coarse_lift_spectral import dense_operator_correspondence, leading_eigenpairs, partial_svd
from l2_fd_jvp import L2Point
from plot_l2_coarse_lift import plot_argand, plot_twostep


def _point(u, z) -> L2Point:
    t = torch.ones(u.shape[0], dtype=u.dtype)
    return L2Point(u=u, z=z, t=t, u_next=u.clone(), z_next=z.clone(), stems=(0, 0, 0))


def test_projector_algebra_and_torch_match():
    report = projector_identity_report(8, 2, seed=0)
    assert report["ok"], report
    report_r = projector_identity_report(12, 3, seed=1)
    assert report_r["ok"], report_r
    match = torch_matches_numpy_pool(8, 2, seed=2)
    assert match["ok"], match
    w = np.arange(4, dtype=np.float64).reshape(2, 2)
    lifted = D_dagger_numpy(w, 2)
    star = D_star_numpy(w, 2)
    assert np.allclose(lifted, 4.0 * star)
    assert not np.allclose(lifted, star)
    assert np.allclose(D_numpy(lifted, 2), w)
    assert not np.allclose(D_numpy(star, 2), w)


def test_q_kills_block_means_and_split_is_additive():
    rng = np.random.default_rng(3)
    field = rng.standard_normal((8, 8))
    assert np.allclose(D_numpy(Q_numpy(field, 2), 2), 0.0)
    assert np.allclose(field, (field - Q_numpy(field, 2)) + Q_numpy(field, 2))

    def step(u, z):
        return 0.2 * u, 0.3 * D_torch_local(u) + 0.4 * z

    def D_torch_local(u):
        return torch.nn.functional.avg_pool2d(u, 2)

    u = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    z = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    op = LiftedBlocks(step, _point(u, z), 2, norm=4.0)
    e = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    total = op.C(e)
    parts = op.C(op.P(e)) + op.CQ(e)
    assert torch.allclose(total, parts, atol=1e-10, rtol=1e-10)
    # Physical scaling cancels: C_phys(v) equals the model JVP applied to v.
    v = torch.randn_like(u)
    from l2_fd_jvp import jvp_both_u

    _g, h = jvp_both_u(step, op.point, v)
    assert torch.allclose(op.C(v * 4.0), h * 4.0, atol=1e-10)


def test_jvp_matches_centered_difference_and_adjoint():
    def step(u, z):
        # Subsampling keeps a Q component, so CQ is not the zero map.
        sampled = u[:, :, ::2, ::2]
        fine = 0.2 * u + 0.1 * z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        coarse = 0.5 * sampled + 0.25 * torch.nn.functional.avg_pool2d(u, 2) - 0.3 * z
        return fine, coarse

    u = torch.randn(1, 1, 6, 6, dtype=torch.float64)
    z = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    point = _point(u, z)
    report = jvp_fd_report(step, point, 2, norm=2.5, eps_list=(1e-6,), seed=4)
    assert report["best_rel"] < 1e-8, report
    assert report["ok_physical_scale"]
    op = LiftedBlocks(step, point, 2, norm=2.5)
    for name in ("T", "U", "C", "CQ"):
        action = op.matvec(name)
        gaps = []
        rng = np.random.default_rng(5)
        for _ in range(3):
            v = rng.standard_normal(action.n_in)
            w = rng.standard_normal(action.n_out)
            left = float(np.dot(action.matvec(v), w))
            right = float(np.dot(v, action.rmatvec(w)))
            gaps.append(abs(left - right) / max(abs(left), 1e-12))
        assert max(gaps) < 1e-8, (name, gaps)
    assert op.matvec("T").rms_factor == 1.0
    assert op.matvec("U").rms_factor == 1.0
    assert abs(op.matvec("C").rms_factor - rms_scale(op.n_coarse, op.n_fine)) < 1e-12
    assert abs(op.matvec("C").rms_factor - np.sqrt(op.n_fine / op.n_coarse)) < 1e-12


def test_dense_correspondence_and_arnoldi_svd():
    report = dense_operator_correspondence(fine_n=4, factor=2, seed=0)
    assert report["ok"], report
    rng = np.random.default_rng(6)
    matrix = rng.standard_normal((12, 12))

    def matvec(v):
        return matrix @ np.asarray(v, dtype=np.float64)

    def rmatvec(v):
        return matrix.T @ np.asarray(v, dtype=np.float64)

    short = leading_eigenpairs(matvec, 12, k=4, ncv=6, maxiter=1, tol=1e-14, seed=4)
    assert short["values"].shape[0] == short["vectors"].shape[1]
    assert short["vectors"].shape[0] == 12
    assert short["converged"] or short["error"]
    eig = leading_eigenpairs(matvec, 12, k=3, ncv=8, maxiter=200, tol=1e-10, seed=1)
    assert eig["converged"], eig
    truth = np.linalg.eigvals(matrix)
    truth = truth[np.argsort(-np.abs(truth))][:3]
    for got in eig["values"]:
        assert np.min(np.abs(truth - got)) < 1e-6
    assert np.max(eig["residuals"]) < 1e-6
    decomposition = partial_svd(matvec, rmatvec, 12, 12, k=3, ncv=8, maxiter=200, tol=1e-10, seed=2)
    assert decomposition["converged"], decomposition
    exact = np.linalg.svd(matrix, compute_uv=False)[:3]
    assert np.allclose(decomposition["values"], exact, rtol=1e-6, atol=1e-6)
    assert np.max(decomposition["residuals_right_rel"]) < 1e-6
    wide = rng.standard_normal((6, 15))

    def wide_matvec(v):
        return wide @ np.asarray(v, dtype=np.float64)

    def wide_rmatvec(v):
        return wide.T @ np.asarray(v, dtype=np.float64)

    wide_svd = partial_svd(wide_matvec, wide_rmatvec, 6, 15, k=2, ncv=5, maxiter=100, tol=1e-10, seed=3)
    assert wide_svd["converged"], wide_svd
    assert np.allclose(wide_svd["values"], np.linalg.svd(wide, compute_uv=False)[:2], rtol=1e-5, atol=1e-5)
    from coarse_lift_spectral import match_left_eigenpairs

    lam = np.array([0.2 + 0.5j, 0.2 - 0.5j])
    y_same = np.array([[1 + 1j, 1 - 1j], [0.3 - 0.2j, 0.3 + 0.2j]])
    # Adjoint returned the same member as λ, not conj(λ), for the first mode.
    adjoint_vals = np.array([0.2 + 0.5j, 0.2 - 0.5j])
    matched = match_left_eigenpairs(lam[:1], adjoint_vals[:1], y_same[:, :1])
    assert matched["conjugated"][0]
    assert abs(matched["values"][0] - np.conj(adjoint_vals[0])) < 1e-12
    assert np.allclose(matched["vectors"][:, 0], np.conj(y_same[:, 0]))
    matched_pair = match_left_eigenpairs(lam, np.conj(lam), np.conj(y_same))
    assert not np.any(matched_pair["conjugated"])
    assert np.allclose(matched_pair["vectors"], np.conj(y_same))


def test_two_step_identity_and_masked_cosine():
    def op_from(scale_u, scale_z):
        def step(u, z):
            fine = scale_u * u + scale_z * z.repeat_interleave(2, -2).repeat_interleave(2, -1)
            coarse = 0.25 * torch.nn.functional.avg_pool2d(u, 2) + 0.1 * z
            return fine, coarse

        u = torch.randn(1, 1, 4, 4, dtype=torch.float64)
        z = torch.randn(1, 1, 2, 2, dtype=torch.float64)
        return LiftedBlocks(step, _point(u, z), 2, norm=3.0)

    first = op_from(0.4, 0.2)
    second = op_from(-0.3, 0.5)
    e = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    parts = two_step_parts(first, second, e)
    assert parts["sum_vs_composed_rel"] < 1e-8, parts
    # B C alone is a different vector from the sum unless A A e vanishes.
    assert parts["norm_sum"] > 0
    zero = two_step_parts(first, second, torch.zeros_like(e))
    assert zero["norm_AA"] == 0.0 and zero["norm_BC"] == 0.0
    assert np.isnan(cosine(np.zeros(4), np.ones(4)))
    assert np.isnan(cosine(np.ones(4), np.zeros(4)))
    assert abs(cosine(np.ones(4), np.ones(4)) - 1.0) < 1e-12
    mixed = directional_gains(torch.ones(4, dtype=torch.float64), np.ones(4))
    assert abs(mixed["euclidean"] - 1.0) < 1e-12
    assert abs(mixed["rms"] - 1.0) < 1e-12


def test_feedback_route_dense_and_torch():
    """N=4, M=1 example with different operators at the two frames."""
    eye = np.eye(4)
    lift = np.ones((4, 1))
    A0 = np.diag([0.8, 0.9, 1.0, 1.1])
    A1 = np.diag([1.0, 0.9, 0.8, 0.7])
    B0 = np.array([[0.6, 0.7, 0.8, 0.9]]).T
    B1 = np.array([[1.6, 1.2, 0.8, 1.2]]).T
    C0 = np.array([[-0.10, -0.05, -0.03, -0.02]])
    C1 = 0.5 * C0
    K0 = np.array([[0.75]])
    Q = eye - lift @ np.full((1, 4), 0.25)
    routes = route_matrices(A0, A1, B1, C0, lift)
    assert abs(routes["tau_a"] + 0.05) < 1e-12
    assert abs(routes["tau_b"] + 0.017) < 1e-12
    assert abs(routes["tau_loop"] + 0.067) < 1e-12
    assert abs(routes["delta_loop"] + 0.086164) < 5e-7
    assert abs(routes["tau_a"] + routes["tau_b"] - routes["tau_loop"]) < 1e-12
    probes = sign_probes(4)
    for name, matrix in (("tau_a", routes["La"]), ("tau_b", routes["Lb"]), ("tau_loop", routes["L"])):
        samples = probe_quadratics(matrix, probes)
        assert abs(float(samples.mean()) - routes[name]) < 1e-12
    energy, check = probe_energy(routes["T"], routes["L"], probes)
    assert abs(float(energy.mean()) - routes["delta_loop"]) < 1e-12
    assert np.allclose(check, energy, atol=1e-12)
    star = route_matrices(A0, A1, B1, C0, lift / 4.0)
    assert abs(star["tau_a"] + 0.05) > 1e-3
    wrong_frame = route_matrices(A0, A1, B0, C0, lift)
    assert abs(wrong_frame["tau_loop"] + 0.067) > 1e-3

    a = np.array([0.2, -0.1, 0.3, 0.5])
    b = np.array([0.6, -0.7, 0.4, -0.2])
    eta = np.array([0.15])
    u1 = np.array([1.0, 2.0, -1.0, 0.0])
    eu = np.array([0.01, -0.02, 0.03, 0.04])
    ez = np.array([-0.04])
    block = np.block([[np.zeros((4, 4)), np.zeros((4, 4)), lift], [A0, A0, B0 - lift], [C0, C0, K0]])
    forcing = np.concatenate([-Q @ u1, Q @ u1 + eu, ez])
    e1 = A0 @ (a + b) + (B0 @ eta).reshape(-1) + eu
    eta1 = (C0 @ (a + b)).reshape(-1) + (K0 @ eta).reshape(-1) + ez
    z_hat = 0.25 * float(u1.sum()) + float(eta.reshape(-1)[0])
    expected = np.concatenate([lift.reshape(-1) * z_hat - u1, u1 + e1 - lift.reshape(-1) * z_hat, eta1])
    got = block @ np.concatenate([a, b, eta]) + forcing
    assert np.allclose(got, expected, atol=1e-12)

    def step_of(A, B, C, K):
        mats = [torch.tensor(item, dtype=torch.float64) for item in (A, B, C, K)]

        def step(u, z, _mats=mats):
            fine = _mats[0] @ u.reshape(-1) + _mats[1] @ z.reshape(-1)
            coarse = _mats[2] @ u.reshape(-1) + _mats[3] @ z.reshape(-1)
            return fine.reshape(u.shape), coarse.reshape(z.shape)

        return step

    blank = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    coarse = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    first = LiftedBlocks(step_of(A0, B0, C0, K0), _point(blank, coarse), 2, norm=10.0)
    second = LiftedBlocks(step_of(A1, B1, C1, np.array([[0.2]])), _point(blank.clone(), coarse.clone()), 2, norm=10.0)
    collected = {key: [] for key in ("tau_a", "tau_b", "tau_loop", "delta_loop", "delta_check", "residual")}
    rels = []
    for vector in probes:
        action = feedback_action(first, second, vector.reshape(1, 1, 2, 2))
        for key in collected:
            collected[key].append(action[key])
        rels.append(action["sum_vs_composed_rel"])
    assert max(rels) < 1e-8
    assert max(abs(np.asarray(collected["residual"]))) < 1e-12
    assert abs(float(np.mean(collected["tau_a"])) + 0.05) < 1e-12
    assert abs(float(np.mean(collected["tau_b"])) + 0.017) < 1e-12
    assert abs(float(np.mean(collected["tau_loop"])) + 0.067) < 1e-12
    assert abs(float(np.mean(collected["delta_loop"])) - routes["delta_loop"]) < 1e-12
    assert np.allclose(collected["delta_check"], collected["delta_loop"], atol=1e-12)
    assert abs(latent_shift(first, np.array([1.0])) + 0.25) < 1e-12
    assert abs(latent_shift(first, np.array([-1.0])) + 0.25) < 1e-12


def test_selection_and_tiny_plot(tmp_path=None):
    assert SPECTRAL_SAMPLE_INDICES == (0, 10, 19)
    assert SPECTRAL_STEPS == (0, 20, 39)
    assert TWOSTEP_STEPS == (0, 20, 38)
    out = Path(__file__).resolve().parent / "_coarse_lift_plot_tmp"
    out.mkdir(exist_ok=True)
    vals = np.array([[0.2 + 0.1j, -0.4]], dtype=np.complex128)
    plot_argand(vals, ["state"], np.array([[True, False]]), out / "argand.png")
    plot_twostep(["a"], np.array([1.0]), np.array([0.5]), np.array([np.nan]), np.array([0.0]), out / "two.png")
    assert (out / "argand.png").is_file()
    assert (out / "two.png").is_file()
    for path in out.glob("*.png"):
        path.unlink()
    out.rmdir()


def main():
    test_projector_algebra_and_torch_match()
    test_q_kills_block_means_and_split_is_additive()
    test_jvp_matches_centered_difference_and_adjoint()
    test_dense_correspondence_and_arnoldi_svd()
    test_two_step_identity_and_masked_cosine()
    test_feedback_route_dense_and_torch()
    test_selection_and_tiny_plot()
    print("ok")


if __name__ == "__main__":
    main()
