#!/usr/bin/env python3
"""Unit tests for FutureINNPlus L=2 (K=1) and L=3 CLI regression."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main_plus import finalize_args, get_args_parser
from model_plus import FutureINNPlus


def _ns(**kwargs) -> argparse.Namespace:
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


def test_cli_l2_hierarchy_and_latent_res_agree():
    a = finalize_args(_ns(hierarchy_L=2, ds_factors=[8]))
    assert a.latent_res == [32]
    assert a.num_coarse == 1
    assert a.hierarchy_L == 2
    assert a.num_of_frame == 3
    assert a.ds_factors == [8]
    assert a.inject_res == [64]

    b = finalize_args(_ns(latent_res=[32]))
    assert b.latent_res == [32]
    assert b.hierarchy_L == 2
    assert b.num_of_frame == 3

    c = finalize_args(_ns(latent_res=[32], hierarchy_L=2, ds_factors=[8]))
    assert c.latent_res == [32]


def test_cli_l3_regression():
    a = finalize_args(_ns(latent_res=[32, 8]))
    assert a.num_coarse == 2
    assert a.hierarchy_L == 3
    assert a.num_of_frame == 4
    assert a.ds_factors == [8, 32]
    assert a.inject_res == [64, 32]

    b = finalize_args(_ns(hierarchy_L=3, ds_factors=[8, 32]))
    assert b.latent_res == [32, 8]
    assert b.num_of_frame == 4


def test_cli_rejects_bad_specs():
    try:
        finalize_args(_ns())
        assert False, "expected missing-spec error"
    except ValueError:
        pass
    try:
        finalize_args(_ns(hierarchy_L=2))
        assert False, "expected incomplete hierarchy error"
    except ValueError:
        pass
    try:
        finalize_args(_ns(hierarchy_L=2, ds_factors=[8, 32]))
        assert False, "expected length mismatch"
    except ValueError:
        pass
    try:
        finalize_args(_ns(latent_res=[32], hierarchy_L=3, ds_factors=[8, 32]))
        assert False, "expected disagreement"
    except ValueError:
        pass


def _build_l2(ds: int = 8) -> FutureINNPlus:
    args = finalize_args(_ns(hierarchy_L=2, ds_factors=[ds]))
    return FutureINNPlus(args)


def test_l2_forward_shapes_and_backward():
    for ds in (8, 4, 16):
        model = _build_l2(ds)
        model.train()
        b = 2
        x = torch.randn(b, 3, 256, 256)
        loss, high_loss, coarse_losses = model(x)
        assert len(coarse_losses) == 1
        assert torch.isfinite(loss)
        loss.backward()
        grads = [
            p.grad
            for n, p in model.vit.named_parameters()
            if "low_res_out" in n and p.grad is not None
        ]
        assert grads, f"no coarse-head grads for ds={ds}"
        assert all(torch.isfinite(g).all() for g in grads)

        lat = 256 // ds
        high = torch.randn(b, 1, 256, 256)
        zeros = [torch.zeros(b, 1, lat, lat)]
        t0 = torch.zeros(b)
        out_h, outs = model.vit(high, zeros, t0)
        assert out_h.shape == (b, 1, 256, 256)
        assert len(outs) == 1
        assert outs[0].shape == (b, 1, lat, lat)

        coarse = torch.randn(b, 1, lat, lat)
        t1 = torch.ones(b)
        out_h2, outs2 = model.vit(high, [coarse], t1)
        assert out_h2.shape == (b, 1, 256, 256)
        assert outs2[0].shape == (b, 1, lat, lat)


def test_l2_sample_rollout_and_checkpoint():
    model = _build_l2(8)
    model.eval()
    b = 1
    high = torch.randn(b, 1, 256, 256)
    high1, coarses1 = model.sample(high, coarses=None, ema=False)
    assert high1.shape == (b, 1, 256, 256)
    assert len(coarses1) == 1
    assert coarses1[0].shape == (b, 1, 32, 32)

    high2, coarses2 = model.sample(high1, coarses=coarses1, ema=False)
    assert high2.shape == (b, 1, 256, 256)
    assert coarses2[0].shape == (b, 1, 32, 32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"model": model.state_dict(), "args": model.vit.latent_res}, path)
        loaded = _build_l2(8)
        state = torch.load(path, map_location="cpu", weights_only=False)
        loaded.load_state_dict(state["model"])
        out_a, c_a = model.vit(
            high, [torch.zeros(b, 1, 32, 32)], torch.zeros(b)
        )
        out_b, c_b = loaded.vit(
            high, [torch.zeros(b, 1, 32, 32)], torch.zeros(b)
        )
        assert torch.allclose(out_a, out_b)
        assert torch.allclose(c_a[0], c_b[0])


def test_parser_accepts_hierarchy_flags():
    p = get_args_parser()
    args = p.parse_args(["--hierarchy_L", "2", "--ds_factors", "8"])
    args = finalize_args(args)
    assert args.latent_res == [32]
    assert args.hierarchy_L == 2


def main() -> None:
    test_cli_l2_hierarchy_and_latent_res_agree()
    test_cli_l3_regression()
    test_cli_rejects_bad_specs()
    test_parser_accepts_hierarchy_flags()
    test_l2_forward_shapes_and_backward()
    test_l2_sample_rollout_and_checkpoint()
    print("ALL_L2_PLUS_TESTS_PASSED")


if __name__ == "__main__":
    main()
