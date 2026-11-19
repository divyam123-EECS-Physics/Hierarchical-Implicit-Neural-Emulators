#!/usr/bin/env python3
"""Synthetic checks for L=2 Task 1 diagnostics. No checkpoint and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for p in (_REPO, _EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from l2_fd_jvp import (
    L2Point,
    apply_A,
    apply_B,
    apply_C,
    apply_D,
    apply_Dplus,
    apply_K,
    assert_l2,
    block_step,
    fd_additivity,
    fd_both_uz,
    fd_both_z,
    fd_pair,
    jvp_both_uz,
    jvp_direction_report,
    make_point,
    validate_fd_eps,
)
from eval_l2_stability_task1 import (
    DIAG_KEYS,
    ROLL_KEYS,
    parse_args,
    percentile_band,
    plot_validation_flags,
    run_succeeded,
    trace_rows,
)
from l2_operators import _phi_image, diagnose_point, flat_diag, gates_pass, quotient_halfwidth, rho_S
from l2_physical_jvp import ProxyTangent, Py2DFDTangent, validate_physical_reference
from l2_retention import (
    LATENT_CHANNEL_KEYS,
    LATENT_LIN_KEYS,
    _cos,
    account_injections,
    latent_error_channels,
    propagate_augmented,
    relative_residual,
    running_channels,
    validation_directions,
)
from main_plus import finalize_args
from model_plus import FutureINNPlus
import argparse


def _ns(**kwargs):
    defaults = dict(
        latent_res=None,
        hierarchy_L=None,
        ds_factors=None,
        inject_res=None,
        ratio=0.85,
        img_size=256,
        num_of_frame=-1,
        ds_factor=-1,
        ds_factor_low=-1,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class LinearL2:
    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(16, 16, generator=g, dtype=torch.float64) * 0.05
        self.B = torch.randn(16, 4, generator=g, dtype=torch.float64) * 0.05
        self.C = torch.randn(4, 16, generator=g, dtype=torch.float64) * 0.05
        self.K = torch.randn(4, 4, generator=g, dtype=torch.float64) * 0.05

    def step(self, u, z):
        uf = u.reshape(-1).to(torch.float64)
        zf = z.reshape(-1).to(torch.float64)
        g = (self.A @ uf + self.B @ zf).reshape_as(u).to(dtype=u.dtype)
        h = (self.C @ uf + self.K @ zf).reshape_as(z).to(dtype=z.dtype)
        return g, h


def _point(gen: torch.Generator) -> L2Point:
    u = torch.randn(1, 1, 4, 4, generator=gen, dtype=torch.float64)
    z = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
    t = torch.ones(1, dtype=torch.float64)
    return L2Point(u, z, t, u.clone(), z.clone(), (0, 1, 2))


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).to(torch.float64)


def test_fd_matches_dense_and_is_linear():
    lin = LinearL2()
    gen = torch.Generator().manual_seed(1)
    pt = _point(gen)
    v = torch.randn(1, 1, 4, 4, generator=gen, dtype=torch.float64)
    w = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
    Av = apply_A(lin.step, pt, v, 1e-2)
    Bv = apply_B(lin.step, pt, w, 1e-2)
    Cv = apply_C(lin.step, pt, v, 1e-2)
    Kv = apply_K(lin.step, pt, w, 1e-2)
    assert Av.shape == v.shape and Bv.shape == v.shape
    assert Cv.shape == w.shape and Kv.shape == w.shape
    assert torch.allclose(Av, (lin.A @ _flat(v)).reshape_as(v), atol=1e-8, rtol=1e-6)
    assert torch.allclose(Bv, (lin.B @ _flat(w)).reshape_as(v), atol=1e-8, rtol=1e-6)
    assert torch.allclose(Cv, (lin.C @ _flat(v)).reshape_as(w), atol=1e-8, rtol=1e-6)
    assert torch.allclose(Kv, (lin.K @ _flat(w)).reshape_as(w), atol=1e-8, rtol=1e-6)
    a, b = 0.3, -0.7
    v2 = torch.randn_like(v)
    left = apply_A(lin.step, pt, a * v + b * v2, 1e-2)
    right = a * apply_A(lin.step, pt, v, 1e-2) + b * apply_A(lin.step, pt, v2, 1e-2)
    assert torch.allclose(left, right, atol=1e-7, rtol=1e-6)


def test_eps_sweep_exact_on_linear():
    lin = LinearL2()
    pt = _point(torch.Generator().manual_seed(2))
    sweep = validate_fd_eps(lin.step, [("step0", pt)], 1e-2, tol=1e-6, seed=2, ds=2)
    assert sweep["ok"], sweep
    assert sweep["n_checks"] == 12
    assert max(c["rel"] for c in sweep["checks"]) < 1e-6
    fine = next(c for c in sweep["checks"] if c["name"].endswith("/fine") and c["block"] == "g")
    latent = next(c for c in sweep["checks"] if c["name"].endswith("/latent") and c["block"] == "h")
    assert fine["rel_z"] == 0.0 and abs(fine["rel_u"] - 1e-2) < 1e-8
    assert latent["rel_u"] == 0.0 and abs(latent["rel_z"] - 1e-2) < 1e-8


def test_compositions_and_trace_identity():
    lin = LinearL2()
    pt = _point(torch.Generator().manual_seed(3))
    ds = 2
    v = torch.randn_like(pt.u)
    Dv = apply_D(v, ds)
    Jv, CKv = fd_both_uz(lin.step, pt, v, Dv, 1e-3)
    Av = apply_A(lin.step, pt, v, 1e-3)
    BDv = apply_B(lin.step, pt, Dv, 1e-3)
    assert torch.allclose(Jv, Av + BDv, atol=1e-7, rtol=1e-6)
    w = torch.randn_like(pt.z)
    Bw, Kw = fd_both_z(lin.step, pt, w, 1e-3)
    Sw = Kw - apply_D(Bw, ds)
    # Dense S on this one vector.
    dense_S = (lin.K @ _flat(w) - _flat(apply_D((lin.B @ _flat(w)).reshape_as(pt.u), ds))).reshape_as(w)
    assert torch.allclose(Sw, dense_S, atol=1e-7, rtol=1e-6)
    L = CKv - apply_D(Jv, ds)
    dense_L = (
        lin.C @ _flat(v)
        + lin.K @ _flat(Dv)
        - _flat(apply_D((lin.A @ _flat(v) + lin.B @ _flat(Dv)).reshape_as(pt.u), ds))
    ).reshape_as(w)
    assert torch.allclose(L, dense_L, atol=1e-6, rtol=1e-5)

    diag = diagnose_point(
        lin.step, pt, ds, 1e-3, ProxyTangent(), dt=0.25, norm=10.0,
        n_probes=8, seed=0, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6,
    )
    assert diag["approximation"] == "proxy"
    assert "identity_gap" not in diag
    assert diag["composition_rel_mean"] < 1e-6
    assert diag["trace_estimator_same_operator"] is True
    flat = flat_diag(diag)
    missing = [key for key in DIAG_KEYS if key not in flat]
    assert not missing, missing
    # Hutchinson of A should sit near the true trace.
    diag_h = diagnose_point(
        lin.step, pt, ds, 1e-3, ProxyTangent(), dt=0.25, norm=10.0,
        n_probes=256, seed=1, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6,
    )
    true_tr = float(torch.trace(lin.A))
    assert abs(diag_h["tr_A"]["mean"] - true_tr) < 3.0 * max(diag_h["tr_A"]["std"], 1e-6)
    # Difference-quotient split: Tr(H)/N = Tr(Fu)/N + Tr(DFz)/N, using the paired J.
    # tr_Heff uses tr_J; tr_Fu uses tr_A; the coarse estimator tr_DB is a second sample.
    # The paired identity is the one that must hold exactly.
    assert abs(
        (diag["tr_J"]["mean"] - diag["n_fine"]) / (diag["n_fine"] * 0.25)
        - diag["tr_Heff_over_N"]
    ) < 1e-9


def test_rho_S_matches_dense():
    lin = LinearL2(seed=4)
    pt = _point(torch.Generator().manual_seed(5))
    # Build dense S, 4x4, via the FD operator on the basis.
    cols = []
    eye = torch.eye(4, dtype=torch.float64)
    for i in range(4):
        w = eye[i].reshape(1, 1, 2, 2)
        Bw, Kw = fd_both_z(lin.step, pt, w, 1e-3)
        cols.append((Kw - apply_D(Bw, 2)).reshape(-1))
    S = torch.stack(cols, dim=1)
    true = torch.linalg.eigvals(S).abs().max().item()
    est = rho_S(lin.step, pt, 2, 1e-3, ProxyTangent(), 10.0, ncv=3, maxiter=50, tol=1e-8, seed=0)
    assert est["converged"], est
    assert est["spectral_validated"] is True
    assert est["residual"] < 1e-4
    assert abs(est["radius"] - true) / max(true, 1e-8) < 1e-3


def test_retention_bookkeeping():
    g = torch.Generator().manual_seed(6)
    eu = [torch.randn(1, 1, 4, 4, generator=g) for _ in range(3)]
    xi = [torch.randn(1, 1, 2, 2, generator=g) for _ in range(3)]
    e = [torch.zeros(1, 1, 4, 4)]
    acc = torch.zeros(1, 1, 4, 4)
    for k in range(3):
        acc = acc + eu[k]
        # Extra retained piece so e is not the plain injection sum.
        e.append(acc + (k + 1) * torch.ones_like(acc))

    def Jminus(k, vec):
        return (0.1 + 0.01 * k) * vec

    def Bop(k, w):
        return F.interpolate(w, size=(4, 4), mode="nearest") * (0.2 + 0.05 * k)

    out = running_channels(e, eu, xi, Jminus, Bop)
    # Explicit double sums at n=3.
    rH = torch.zeros_like(e[0])
    rZ = torch.zeros_like(e[0])
    for k in range(3):
        S = sum(eu[:k], torch.zeros_like(eu[0]))
        rH = rH + Jminus(k, S)
        if k >= 1:
            Z = sum(xi[:k], torch.zeros_like(xi[0]))
            rZ = rZ + Bop(k, Z)
    rho = e[3] - sum(eu, torch.zeros_like(eu[0]))
    assert abs(out["I"][3] - float(sum(eu, torch.zeros_like(eu[0])).norm())) < 1e-5
    assert abs(out["R"][3] - (out["e_norm"][3] - out["I"][3])) < 1e-8
    assert abs(out["rH_norm"][3] - float(rH.norm())) < 1e-5
    assert abs(out["rZ_norm"][3] - float(rZ.norm())) < 1e-5
    assert abs(out["recon_norm"][3] - float((rho - rH - rZ).norm())) < 1e-5
    # R is not ||rho||.
    assert abs(out["R"][3] - out["rho_norm"][3]) > 1e-4
    injected = sum(eu, torch.zeros_like(eu[0]))
    assert abs(out["cos_e_rho"][3] - _cos(e[3], rho)) < 1e-6
    assert abs(out["cos_e_inj"][3] - _cos(e[3], injected)) < 1e-6
    assert abs(out["cos_rho_inj"][3] - _cos(rho, injected)) < 1e-6
    assert np.isnan(out["cos_e_rho"][0])
    eta = [torch.zeros(1, 1, 2, 2)]
    acc_z = torch.zeros(1, 1, 2, 2)
    for k in range(3):
        acc_z = acc_z + xi[k]
        eta.append(acc_z + torch.ones_like(acc_z))
    eta_lin = [torch.full_like(item, 0.5) for item in eta]
    with_eta = running_channels(e, eu, xi, Jminus, Bop, eta=eta, eta_lin=eta_lin)
    assert abs(with_eta["cos_eta_inj"][3] - _cos(eta[3], acc_z)) < 1e-6
    assert abs(with_eta["cos_eta_eta_lin"][3] - _cos(eta[3], eta_lin[3])) < 1e-6
    assert np.isnan(with_eta["cos_eta_inj"][0])


def test_large_eps_on_cubic():
    def step(u, z):
        return u ** 3, z ** 3

    gen = torch.Generator().manual_seed(11)
    pt = _point(gen)
    v = torch.randn_like(pt.u)
    true = (3.0 * pt.u ** 2 * v).reshape_as(v)
    small = apply_A(step, pt, v, 1e-5)
    large = apply_A(step, pt, v, 0.5)
    assert torch.allclose(small, true, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(large, true, atol=1e-3, rtol=1e-3)
    bad = validate_fd_eps(step, [("step0", pt)], 0.5, tol=1e-3, seed=11, ds=2)
    good = validate_fd_eps(step, [("step0", pt)], 1e-4, tol=1e-3, seed=11, ds=2)
    assert not bad["ok"], bad["worst"]
    assert good["ok"], good["worst"]


def test_latent_step_uses_latent_scale():
    u = torch.ones(1, 1, 4, 4, dtype=torch.float64) * 10.0
    z = torch.ones(1, 1, 2, 2, dtype=torch.float64) * 1e-3
    du = torch.zeros_like(u)
    dz = torch.ones_like(z)
    info = block_step(u, z, du, dz, 1e-2)
    assert info["rel_u"] == 0.0
    assert abs(info["rel_z"] - 1e-2) < 1e-8
    joint = torch.sqrt(u.reshape(-1).pow(2).sum() + z.reshape(-1).pow(2).sum())
    old_h = 1e-2 * float(joint) / float(dz.reshape(-1).norm())
    assert info["h"] < 0.1 * old_h
    dg, dh, got = fd_pair(lambda a, b: (a, b), u, z, torch.zeros_like(u), torch.zeros_like(z), 1e-2)
    assert got["zero"]
    assert float(dg.abs().sum()) == 0.0 and float(dh.abs().sum()) == 0.0


def test_independent_composition_not_tautology():
    def step(u, z):
        return u + 0.2 * u ** 3, z

    gen = torch.Generator().manual_seed(12)
    pt = _point(gen)
    ds = 2
    v = torch.randn_like(pt.u)
    # A large latent direction changes the shared step h, so the mixed
    # finite difference is not the fine-only difference plus B(w).
    w = torch.ones_like(pt.z) * 20.0
    Jv, _ = fd_both_uz(step, pt, v, w, 0.4)
    Av = apply_A(step, pt, v, 0.4)
    Bw = apply_B(step, pt, w, 0.4)
    independent = float((Jv - Av - Bw).norm() / Jv.norm().clamp(min=1e-12))
    tautology = float((Jv - Av - (Jv - Av)).norm())
    assert tautology == 0.0
    assert independent > 1e-4
    diag = diagnose_point(
        step, pt, ds, 0.4, ProxyTangent(), dt=0.25, norm=10.0,
        n_probes=4, seed=12, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6,
    )
    assert diag["composition_rel_mean"] < 1e-6


class _ScaleTangent:
    name = "py2d_fd"
    is_exact = False

    def apply(self, u_phys, v_phys):
        factor = 2.0 if float(np.mean(u_phys)) >= 0.0 else 0.5
        return np.asarray(v_phys, dtype=np.float64) * factor


def test_distinct_tangents_at_n_and_n_plus_1():
    lin = LinearL2(seed=8)
    gen = torch.Generator().manual_seed(9)
    u = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    u_next = -torch.ones_like(u)
    z = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
    pt = L2Point(u, z, torch.ones(1, dtype=torch.float64), u_next, z.clone(), (0, 1, 2))
    common = dict(dt=0.25, norm=10.0, n_probes=6, seed=3, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6)
    proxy = diagnose_point(lin.step, pt, 2, 1e-4, ProxyTangent(), **common)
    scaled = diagnose_point(lin.step, pt, 2, 1e-4, _ScaleTangent(), **common)
    assert np.allclose(proxy["tr_DB"]["probes"], scaled["tr_DB"]["probes"], atol=1e-5, rtol=1e-5)
    assert np.allclose(scaled["tr_DPhi_next_B"]["probes"], 0.5 * scaled["tr_DB"]["probes"], atol=1e-4, rtol=1e-4)
    assert np.allclose(scaled["tr_B_DPhi_n"]["probes"], 2.0 * proxy["tr_B_DPhi_n"]["probes"], atol=1e-4, rtol=1e-4)
    assert abs(scaled["tr_DB"]["mean"] - scaled["tr_DPhi_next_B"]["mean"]) > 1e-8


def test_augmented_linearization_matches_dense_affine():
    lin = LinearL2(seed=7)
    gen = torch.Generator().manual_seed(13)
    bu = torch.randn(16, generator=gen, dtype=torch.float64) * 0.01
    bz = torch.randn(4, generator=gen, dtype=torch.float64) * 0.01

    def step(u, z):
        g, h = lin.step(u, z)
        return g + bu.reshape_as(u).to(dtype=u.dtype), h + bz.reshape_as(z).to(dtype=z.dtype)

    us = [torch.randn(1, 1, 4, 4, generator=gen, dtype=torch.float64) for _ in range(5)]
    zs = [torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64) for _ in range(5)]
    points = [
        L2Point(us[n], zs[n], torch.ones(1, dtype=torch.float64), us[n + 1], zs[n + 1], (n, n + 1, n + 2))
        for n in range(4)
    ]
    eps_u, eps_z = [], []
    for pt in points:
        g, h = step(pt.u, pt.z)
        eps_u.append(g - pt.u_next)
        eps_z.append(h - pt.z_next)
    e_lin, eta_lin = propagate_augmented(lambda _n: step, points, eps_u, eps_z, 1e-5)

    e = torch.zeros_like(points[0].u)
    eta = torch.zeros_like(points[0].z)
    dense_e = [e.clone()]
    dense_eta = [eta.clone()]
    for n in range(4):
        e_next = (lin.A @ _flat(e) + lin.B @ _flat(eta)).reshape_as(e).to(dtype=e.dtype) + eps_u[n]
        eta_next = (lin.C @ _flat(e) + lin.K @ _flat(eta)).reshape_as(eta).to(dtype=eta.dtype) + eps_z[n]
        e, eta = e_next, eta_next
        dense_e.append(e)
        dense_eta.append(eta)
    for got, ref in zip(e_lin, dense_e):
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)
    for got, ref in zip(eta_lin, dense_eta):
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)

    high = points[0].u.clone()
    z = points[0].z.clone()
    actual_e = [torch.zeros_like(high)]
    actual_eta = [torch.zeros_like(z)]
    for n, pt in enumerate(points):
        high, z = step(high, z)
        actual_e.append(high - pt.u_next)
        actual_eta.append(z - pt.z_next)
    for got, ref in zip(actual_e, e_lin):
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)
    for got, ref in zip(actual_eta, eta_lin):
        assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)
    for key in ("e_lin_norm", "eta_norm", "eta_lin_norm", "e_lin_err", "eta_lin_err", "recon_lin_norm"):
        assert key in ROLL_KEYS


def test_confidence_interval_and_relative_residual():
    scaled = quotient_halfwidth(1.96, 16, 0.25)
    assert abs(float(scaled) - 1.96 / (16 * 0.25)) < 1e-12
    assert abs(float(scaled) - 1.96 / 16) > 1e-3
    recon = np.array([[2.0, 0.0], [4.0, 1.0]])
    rho = np.array([[4.0, 1e-16], [1.0, 2.0]])
    rel = relative_residual(recon, rho, floor=1e-12)
    assert np.isnan(rel[0, 1])
    ratios = np.array([0.5, 4.0, 0.5])
    assert abs(np.nanmean(rel) - ratios.mean()) < 1e-12
    ratio_of_means = recon.mean() / rho.mean()
    assert abs(ratio_of_means - np.nanmean(rel)) > 0.1


def test_labels_and_l2_model():
    assert ProxyTangent().is_exact is False
    assert Py2DFDTangent().is_exact is False
    assert Py2DFDTangent().name == "py2d_fd"
    model = FutureINNPlus(finalize_args(_ns(hierarchy_L=2, ds_factors=[8])))
    assert_l2(model)
    x = torch.randn(2, 1, 256, 256)
    pooled = model._pool_seq(x)[0]
    assert torch.equal(apply_D(x, 8), pooled)

    class _K2:
        num_coarse = 2

    try:
        assert_l2(_K2())
        raise AssertionError("K=2 should be rejected")
    except ValueError:
        pass


def test_composition_gate_is_not_the_epsilon_sweep():
    assert gates_pass(True, 0.0, 0.0, 1e-2)
    assert not gates_pass(True, 0.2, 0.0, 1e-2)
    lin = LinearL2()
    pt = _point(torch.Generator().manual_seed(2))
    sweep = validate_fd_eps(lin.step, [("step0", pt)], 1e-2, tol=1e-6, seed=2, ds=2)
    assert sweep["ok"]

    def cubic(u, z):
        z_up = z.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        return u + 0.8 * (u ** 2) * z_up, z

    gen = torch.Generator().manual_seed(21)
    cpt = _point(gen)
    diag = diagnose_point(
        cubic, cpt, 2, 0.5, ProxyTangent(), dt=0.25, norm=1.0,
        n_probes=4, seed=21, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6,
    )
    assert diag["composition_rel_max"] < 1e-6
    assert not gates_pass(True, 0.2, 0.0, 1e-2)


class _CubicTangent:
    name = "py2d_fd"
    is_exact = False

    def apply(self, u_phys, v_phys, eps_rel=None):
        v = np.asarray(v_phys, dtype=np.float64)
        return v + 0.5 * v ** 3


def test_s_additivity_gates_arnoldi():
    lin = LinearL2(seed=4)
    pt = _point(torch.Generator().manual_seed(5))
    full = fd_additivity(lin.step, pt, 1e-3, seed=4)
    assert full["rel"] < 1e-6
    assert gates_pass(True, 0.0, full["rel"], 1e-4)
    est = rho_S(
        lin.step, pt, 2, 1e-3, _CubicTangent(), 1.0,
        ncv=3, maxiter=40, tol=1e-8, seed=0, fd_tol=1e-6,
    )
    assert est["spectral_validated"] is False
    assert "Arnoldi skipped" in est["error"]
    assert est["converged"] is False


def test_retention_direction_is_mixed():
    eu = [torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), torch.full((1, 1, 4, 4), 2.0)]
    xi = [torch.ones(1, 1, 2, 2) for _ in range(3)]
    e_lin = [torch.zeros(1, 1, 4, 4)] + [torch.full((1, 1, 4, 4), float(k)) for k in range(1, 4)]
    eta_lin = [torch.zeros(1, 1, 2, 2)] + [torch.ones(1, 1, 2, 2) for _ in range(3)]
    points = []
    for n in range(3):
        points.append(L2Point(
            torch.ones(1, 1, 4, 4),
            torch.ones(1, 1, 2, 2),
            torch.ones(1),
            torch.ones(1, 1, 4, 4),
            torch.ones(1, 1, 2, 2),
            (n, n + 1, n + 2),
        ))
    found = validation_directions(points, eu, xi, e_lin, eta_lin, ProxyTangent(), 2, 1.0)
    rh = next(item for item in found if item["name"] == "rH_step1")
    assert torch.allclose(rh["du"], eu[0])
    assert torch.allclose(rh["dz"], apply_D(eu[0], 2))
    assert float(rh["dz"].abs().sum()) > 0.0
    assert any(item["name"].startswith("lin_step") for item in found)


def test_phi_image_keeps_float64():
    seen = {}

    class _Rec:
        name = "py2d_fd"
        is_exact = False

        def apply(self, u_phys, v_phys, eps_rel=None):
            seen["u"] = np.asarray(u_phys).dtype
            seen["v"] = np.asarray(v_phys).dtype
            return np.asarray(v_phys, dtype=np.float64)

    u = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    v = torch.full((1, 1, 2, 2), 0.25, dtype=torch.float64)
    out = _phi_image(_Rec(), u, v, 10.0)
    assert seen["u"] == np.float64 and seen["v"] == np.float64
    assert out.dtype == torch.float64
    assert torch.allclose(out, v)


def test_physical_reference_uses_the_supplied_frames():
    class _Id:
        def flow(self, u):
            return np.array(u, dtype=np.float64, copy=True)

        def apply(self, u, v, eps_rel=None):
            return np.array(v, dtype=np.float64, copy=True)

    field = np.ones((8, 8))
    good = validate_physical_reference(_Id(), field, field, 1e-3, 1e-6, 0, 0.25, [11, 12])
    assert good["ok"]
    assert good["nx"] == 8 and good["dt"] == 0.25
    assert good["stems"] == [11, 12] and good["source"] == "task1_window"
    bad = validate_physical_reference(_Id(), field, 2.0 * field, 1e-3, 1e-6, 0, 0.25, [13, 14])
    assert not bad["ok"]
    assert bad["stems"] == [13, 14]


def test_spectral_failure_fails_the_job_and_its_plot():
    assert run_succeeded(True, False, True) is False
    assert run_succeeded(True, True, True) is True
    assert run_succeeded(True, "not_requested", False) is True
    assert run_succeeded(False, True, True) is False
    other, spectral = plot_validation_flags(True, False, True)
    assert other is True and spectral is False
    other, spectral = plot_validation_flags(True, True, True)
    assert other is True and spectral is True
    other, spectral = plot_validation_flags(False, True, True)
    assert other is False and spectral is False


def test_physical_operator_and_check_share_epsilon():
    tang = Py2DFDTangent(nx=4, fd_eps_rel=3e-4, emulator_dt=0.25)
    assert tang.fd_eps_rel == 3e-4
    seen = {}

    def _fd(flow, u, v, eps, floor):
        seen["eps"] = float(eps)
        return np.array(v, copy=True)

    tang._solver = object()
    tang._fd = _fd
    field = np.ones((4, 4))
    direction = np.full((4, 4), 0.2)
    out = tang.apply(field, direction)
    assert seen["eps"] == 3e-4
    assert np.allclose(out, direction)

    class _Rec:
        def __init__(self, eps):
            self.fd_eps_rel = float(eps)
            self.seen = []

        def flow(self, u):
            return np.array(u, dtype=np.float64, copy=True)

        def apply(self, u, v, eps_rel=None):
            used = self.fd_eps_rel if eps_rel is None else float(eps_rel)
            self.seen.append(used)
            return np.array(v, dtype=np.float64, copy=True)

    rec = _Rec(tang.fd_eps_rel)
    report = validate_physical_reference(
        rec, field, field, rec.fd_eps_rel, 1e-6, 0, 0.05, [1, 2]
    )
    assert report["ok"]
    assert report["flow_rel"] == 0.0
    assert report["source"] == "task1_window"
    proxy = ProxyTangent()
    assert not hasattr(proxy, "fd_eps_rel")


def test_jvp_direction_report_is_exact_on_a_linear_map():
    def step_fn(u, z):
        return 0.3 * u + 0.2 * z, 0.4 * z - 0.1 * u

    point = make_point(
        [np.ones((4, 4)), np.full((4, 4), 0.5), np.ones((4, 4))],
        type("M", (), {"ds_factors": [1], "img_size": 4})(),
        torch.device("cpu"),
        torch.float64,
        1.0,
    )
    # make_point needs a real model. Build the point directly if that fails.
    du = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    dz = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    report = jvp_direction_report(
        step_fn, [{"name": "linear", "point": point, "du": du, "dz": dz}], 1e-8
    )
    assert report["ok"], report
    assert report["n_failed"] == 0


def test_dense_jacobian_matches_jvp_on_both_heads():
    def step(u, z):
        return 0.3 * u + 0.2 * z, 0.4 * z - 0.1 * u

    u = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    z = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    du = torch.randn_like(u)
    dz = torch.randn_like(z)
    _primals, tangents = torch.autograd.functional.jvp(step, (u, z), (du, dz))
    jac = torch.autograd.functional.jacobian(step, (u, z))

    def contract(block, direction):
        return (block * direction).sum(dim=tuple(range(-direction.ndim, 0)))

    g_ref = contract(jac[0][0], du) + contract(jac[0][1], dz)
    h_ref = contract(jac[1][0], du) + contract(jac[1][1], dz)
    assert torch.allclose(tangents[0], g_ref, atol=1e-8, rtol=1e-8)
    assert torch.allclose(tangents[1], h_ref, atol=1e-8, rtol=1e-8)


def test_validation_fd_does_not_replace_the_reported_jvp():
    def step(u, z):
        return 0.2 * u + 0.1 * z, -0.3 * z + 0.05 * u

    point = make_point(
        [np.ones((4, 4)), np.full((4, 4), 0.5), np.ones((4, 4))],
        type("M", (), {"ds_factors": [1]})(),
        torch.device("cpu"),
        torch.float64,
        1.0,
    )
    du = torch.randn_like(point.u)
    dz = torch.randn_like(point.z)
    jg, jh = jvp_both_uz(step, point, du, dz)
    fg, fh = fd_both_uz(step, point, du, dz, 1e-5)
    assert torch.allclose(jg, fg, atol=1e-5, rtol=1e-5)
    assert torch.allclose(jh, fh, atol=1e-5, rtol=1e-5)
    report = jvp_direction_report(
        step, [{"name": "linear", "point": point, "du": du, "dz": dz}], 1e-6
    )
    assert report["derivative"] == "torch.autograd.functional.jvp"
    assert report["ok"]


def test_injection_names_and_vector_sums():
    eps_z = [torch.tensor([[[[1.0]]]]), torch.tensor([[[[3.0]]]])]
    xi = [torch.tensor([[[[0.0]]]]), torch.tensor([[[[4.0]]]])]
    eta = [
        torch.zeros(1, 1, 1, 1),
        eps_z[0].clone(),
        eps_z[0] + eps_z[1],
    ]
    nu = [item + 2.0 for item in eta]
    out = account_injections(eta, eps_z, xi, nu)
    assert np.isnan(out["cos_eta_inj"][0])
    assert abs(out["eps_z_sum_norm"][2] - float((eps_z[0] + eps_z[1]).norm())) < 1e-8
    assert abs(out["xi_sum_norm"][2] - float((xi[0] + xi[1]).norm())) < 1e-8
    assert abs(out["cos_eta_eps_z"][2] - 1.0) < 1e-8
    assert abs(out["cos_nu_xi"][2] - _cos(nu[2], xi[0] + xi[1])) < 1e-8
    assert "cos_eta_eps_z" in ROLL_KEYS and "cos_nu_xi" in ROLL_KEYS


def test_percentile_band_is_not_a_zero_fill():
    values = np.array([[0.0, np.nan], [1.0, np.nan]])
    mean, lo, hi, n_valid = percentile_band(values)
    assert abs(mean[0] - 0.5) < 1e-12
    assert n_valid[0] == 2 and n_valid[1] == 0
    assert np.isnan(mean[1])
    assert lo[0] <= mean[0] <= hi[0]


def test_j_quotient_keeps_its_probe_bar():
    diag = {
        "dt": np.array(0.05),
        "n_fine": np.array([4.0, 4.0]),
        "n_coarse": np.array([1.0, 1.0]),
        "tr_A_mean": np.zeros(2),
        "tr_A_ci95": np.full(2, 1.96),
        "tr_K_over_M": np.ones(2),
        "tr_J_mean": np.zeros(2),
        "tr_J_ci95": np.full(2, 1.96),
        "tr_S_mean": np.zeros(2),
        "tr_S_ci95": np.ones(2),
        "tr_DB_mean": np.zeros(2),
        "tr_DB_ci95": np.ones(2),
        "tr_B_DPhi_n_mean": np.zeros(2),
        "tr_B_DPhi_n_ci95": np.ones(2),
        "tr_DPhi_next_B_mean": np.zeros(2),
        "tr_DPhi_next_B_ci95": np.ones(2),
        "tr_Fu_over_N": np.zeros(2),
        "tr_Heff_over_N": np.zeros(2),
        "tr_Sigma_over_M": np.zeros(2),
        "tr_DFz_over_N": np.zeros(2),
        "tr_B_DPhi_n_quot": np.zeros(2),
    }
    _raw, quotients = trace_rows(diag)
    heff = next(row for row in quotients if r"H_{\mathrm{eff}}" in row[3])
    assert r"A_n+B_n D\Phi'_n" in heff[0]
    assert heff[2] is not None
    assert abs(float(np.asarray(heff[2])[0]) - 1.96 / (4.0 * 0.05)) < 1e-12


def test_production_preset_keeps_explicit_overrides():
    args = parse_args(["--production", "--n-snapshots", "1", "--device", "cpu"])
    assert args.model_dtype == "float64"
    assert args.physical_jvp == "py2d"
    assert args.rollout_length == 40
    assert args.n_probes == 32
    assert args.dt == 0.05
    assert args.spectral_radius is True
    assert args.n_snapshots == 1
    bare = parse_args(["--device", "cpu"])
    assert bare.model_dtype == "float32"
    assert bare.physical_jvp == "proxy"
    assert bare.n_snapshots == 4


def test_latent_fine_error_channels_on_an_affine_model():
    C = torch.zeros(4, 16, dtype=torch.float64)
    C[:, :4] = torch.eye(4)
    K = torch.eye(4, dtype=torch.float64)
    direction = torch.ones(4, dtype=torch.float64)

    def _fine_from_latent(lat: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(16, dtype=torch.float64)
        flat[:4] = lat.reshape(-1)
        return flat.reshape(1, 1, 4, 4)

    def _lat(vec: torch.Tensor) -> torch.Tensor:
        return vec.reshape(1, 1, 2, 2)

    def apply_c(_k, vec):
        return _lat(C @ vec.reshape(-1))

    def apply_k(_k, vec):
        return _lat(K @ vec.reshape(-1))

    # Step 0 is zero. Step 1 reinforces. Step 2 opposes. Remainder is zero.
    e0 = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    eta0 = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    e1 = _fine_from_latent(direction)
    eta1 = _lat(2.0 * direction)
    ez1 = _lat(direction)
    eta2 = apply_c(1, e1) + apply_k(1, eta1) + ez1
    e2 = _fine_from_latent(direction)
    ez2 = _lat(-direction - (K @ eta2.reshape(-1)))
    eta3 = apply_c(2, e2) + apply_k(2, eta2) + ez2
    e = [e0, e1, e2, torch.zeros_like(e0)]
    eta = [eta0, eta1, eta2, eta3]
    eps_u = [torch.randn(1, 1, 4, 4, dtype=torch.float64) for _ in range(3)]
    eps_u[0] = torch.zeros_like(e0)
    eps_u[1] = _fine_from_latent(direction)
    eps_z = [torch.zeros_like(eta0), ez1, ez2]

    Qe = [apply_D(item, 2) for item in e]
    out = latent_error_channels(
        e, eta, eps_u, eps_z, apply_c, apply_k, Qe=Qe, nu=list(eta),
    )
    assert float(np.max(out["eta_recon_norm"])) < 1e-8
    assert float(np.max(np.abs(out["budget_resid"]))) < 1e-8
    r_u = torch.zeros_like(eta[0])
    r_z = torch.zeros_like(eta[0])
    i_z = torch.zeros_like(eta[0])
    for n in range(4):
        rho = eta[n] - i_z
        assert abs(out["channel_remainder_norm"][n] - float((rho - r_u - r_z).norm())) < 1e-8
        for key, left, right in (
            ("cos_eta_rho", eta[n], rho),
            ("cos_rz_eta", r_z, eta[n]),
            ("cos_rho_rz", rho, r_z),
            ("cos_rho_Iz", rho, i_z),
        ):
            expected = _cos(left, right)
            got = out[key][n]
            if np.isnan(expected):
                assert np.isnan(got)
            else:
                assert abs(got - expected) < 1e-8
        if n == 3:
            break
        r_u = r_u + apply_c(n, e[n])
        r_z = r_z + (apply_k(n, eta[n]) - eta[n])
        i_z = i_z + eps_z[n]
    assert out["cross_bp"][1] > 0.0 and out["delta_fine"][1] > 0.0
    assert out["cross_bp"][2] < 0.0 and out["delta_fine"][2] < 0.0
    for n in range(3):
        p = apply_c(n, e[n])
        b = apply_k(n, eta[n]) + eps_z[n]
        bp = (b + p).reshape(-1)
        bb = b.reshape(-1)
        delta = float(bp.dot(bp) - bb.dot(bb))
        assert abs(out["delta_fine"][n] - delta) < 1e-8

    sum_u = torch.zeros_like(e[0])
    sum_z = torch.zeros_like(eta[0])
    acc_u = torch.zeros_like(eta[0])
    acc_z = torch.zeros_like(eta[0])
    for n in range(3):
        assert abs(out["r_u_to_z_first_norm"][n] - float(acc_u.norm())) < 1e-8
        assert abs(out["r_z_to_z_first_norm"][n] - float(acc_z.norm())) < 1e-8
        acc_u = acc_u + apply_c(n, sum_u)
        acc_z = acc_z + (apply_k(n, sum_z) - sum_z)
        sum_u = sum_u + eps_u[n]
        sum_z = sum_z + eps_z[n]
    assert abs(out["r_u_to_z_first_norm"][3] - float(acc_u.norm())) < 1e-8

    z = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    assert torch.allclose(apply_D(apply_Dplus(z, 2), 2), z, atol=1e-12)
    uneven = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    assert not torch.allclose(apply_Dplus(apply_D(uneven, 2), 2), uneven, atol=1e-3)

    lin = LinearL2(seed=4)
    dense_lift = 0.0
    for i in range(4):
        basis = torch.zeros(4, dtype=torch.float64)
        basis[i] = 1.0
        lifted = apply_Dplus(basis.reshape(1, 1, 2, 2), 2).reshape(-1)
        dense_lift += float(basis @ (lin.C @ lifted))
    dense_gain = float((lin.C * lin.C).sum()) / 16.0
    pt = _point(torch.Generator().manual_seed(4))
    diag = diagnose_point(
        lin.step, pt, 2, 1e-4, ProxyTangent(), dt=0.25, norm=1.0,
        n_probes=64, seed=4, spectral_radius=False, ncv=3, maxiter=20, tol=1e-6,
    )
    assert diag["tr_S_split_probe_max"] < 1e-6
    assert abs(
        diag["tr_Ez_over_M"] - (diag["tr_K"]["mean"] - 4.0) / (4.0 * 0.25)
    ) < 1e-10
    flat = flat_diag(diag)
    assert abs(flat["tr_C_Dplus_quot"] - diag["tr_C_Dplus"]["mean"] / (4.0 * 0.25)) < 1e-12
    assert "tr_C_mean" not in flat and "tr_Eu_over_N" not in flat
    lift_se = diag["tr_C_Dplus"]["std"] / np.sqrt(64)
    gain_se = diag["gain_C"]["std"] / np.sqrt(64)
    assert abs(diag["tr_C_Dplus"]["mean"] - dense_lift) < max(6.0 * lift_se, 1e-8)
    assert abs(diag["gain_C_over_N"] - dense_gain) < max(6.0 * gain_se, 1e-8)
    assert abs(flat["gain_Eu_sq_over_N"] - dense_gain / (0.25 ** 2)) < max(
        6.0 * gain_se / (0.25 ** 2), 1e-8
    )

    zeros_e = [torch.zeros(1, 1, 4, 4, dtype=torch.float64) for _ in range(2)]
    zeros_z = [torch.zeros(1, 1, 2, 2, dtype=torch.float64) for _ in range(2)]
    quiet = latent_error_channels(
        zeros_e, zeros_z, zeros_e[:1], zeros_z[:1], apply_c, apply_k,
    )
    assert np.isnan(quiet["cos_p_eta"][0])
    assert np.isnan(quiet["cos_ru_Iz"][0])

    assert set(LATENT_CHANNEL_KEYS).issubset(ROLL_KEYS)
    assert set(LATENT_LIN_KEYS).issubset(ROLL_KEYS)
    assert "cos_eta_inj" in ROLL_KEYS
    left = {key: np.arange(4, dtype=np.float64) for key in ROLL_KEYS}
    right = {key: np.arange(4, dtype=np.float64) + 1.0 for key in ROLL_KEYS}
    merged = {key: np.stack([left[key], right[key]]) for key in ROLL_KEYS}
    assert merged["p_norm"].shape == (2, 4)
    assert merged["lin_eta_recon_norm"].shape == (2, 4)
    missing = [key for key in DIAG_KEYS if key not in flat]
    assert not missing, missing


def test_float64_path_does_not_downcast():
    class _Model:
        ds_factors = [2]

    value = np.float64(1.0 + 2.0 ** -40)
    field = np.full((4, 4), value, dtype=np.float64)
    point = make_point(
        [field, field, field], _Model(), torch.device("cpu"), torch.float64, 1.0
    )
    assert point.u.dtype == torch.float64
    assert abs(float(point.u.reshape(-1)[0]) - float(value)) < 1e-18

    from ufno import AttentionOp
    from util.fourier import SpectralConv2dSteerable

    q = torch.randn(2, 4, 6, dtype=torch.float64)
    k = torch.randn(2, 4, 6, dtype=torch.float64)
    w = AttentionOp.apply(q, k)
    assert w.dtype == torch.float64
    w32 = AttentionOp.apply(q.float(), k.float())
    assert w32.dtype == torch.float32

    layer = SpectralConv2dSteerable(1, 1, 2, 2).double()
    x = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    y = layer(x)
    assert y.dtype == torch.float64
    y32 = SpectralConv2dSteerable(1, 1, 2, 2).float()(x.float())
    assert y32.dtype == torch.float32


if __name__ == "__main__":
    test_fd_matches_dense_and_is_linear()
    test_eps_sweep_exact_on_linear()
    test_large_eps_on_cubic()
    test_latent_step_uses_latent_scale()
    test_compositions_and_trace_identity()
    test_independent_composition_not_tautology()
    test_distinct_tangents_at_n_and_n_plus_1()
    test_augmented_linearization_matches_dense_affine()
    test_confidence_interval_and_relative_residual()
    test_rho_S_matches_dense()
    test_retention_bookkeeping()
    test_labels_and_l2_model()
    test_composition_gate_is_not_the_epsilon_sweep()
    test_s_additivity_gates_arnoldi()
    test_retention_direction_is_mixed()
    test_phi_image_keeps_float64()
    test_physical_reference_uses_the_supplied_frames()
    test_spectral_failure_fails_the_job_and_its_plot()
    test_physical_operator_and_check_share_epsilon()
    test_jvp_direction_report_is_exact_on_a_linear_map()
    test_dense_jacobian_matches_jvp_on_both_heads()
    test_validation_fd_does_not_replace_the_reported_jvp()
    test_injection_names_and_vector_sums()
    test_percentile_band_is_not_a_zero_fill()
    test_j_quotient_keeps_its_probe_bar()
    test_latent_fine_error_channels_on_an_affine_model()
    test_production_preset_keeps_explicit_overrides()
    test_float64_path_does_not_downcast()
    print("ok")
