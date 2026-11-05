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
    eps_sweep,
    fd_both_uz,
    fd_both_z,
)
from l2_operators import diagnose_point, rho_S
from l2_physical_jvp import ProxyTangent, Py2DFDTangent
from l2_retention import running_channels
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
    sweep = eps_sweep(lin.step, pt, [1e-1, 1e-2, 1e-3], tol=1e-6)
    assert sweep["ok"], sweep
    assert max(sweep["consecutive_rel"]) < 1e-6


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
    assert abs(diag["identity_gap"]) < 1e-6
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


def test_labels_and_l2_model():
    assert ProxyTangent().is_exact is False
    assert Py2DFDTangent().is_exact is True
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
    test_compositions_and_trace_identity()
    test_rho_S_matches_dense()
    test_retention_bookkeeping()
    test_labels_and_l2_model()
    print("ok")
