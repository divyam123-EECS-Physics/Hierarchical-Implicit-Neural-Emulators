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
    apply_K,
    assert_l2,
    block_step,
    fd_both_uz,
    fd_both_z,
    fd_pair,
    validate_fd_eps,
)
from eval_l2_stability_task1 import DIAG_KEYS, ROLL_KEYS
from l2_operators import diagnose_point, flat_diag, quotient_halfwidth, rho_S
from l2_physical_jvp import ProxyTangent, Py2DFDTangent
from l2_retention import propagate_augmented, relative_residual, running_channels
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
    w = apply_D(v, ds)
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
    assert diag["composition_rel_mean"] > 1e-4


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

    points = []
    for _ in range(4):
        u = torch.randn(1, 1, 4, 4, generator=gen, dtype=torch.float64)
        z = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
        u_next = torch.randn(1, 1, 4, 4, generator=gen, dtype=torch.float64)
        z_next = torch.randn(1, 1, 2, 2, generator=gen, dtype=torch.float64)
        points.append(L2Point(u, z, torch.ones(1, dtype=torch.float64), u_next, z_next, (0, 1, 2)))
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
    print("ok")
