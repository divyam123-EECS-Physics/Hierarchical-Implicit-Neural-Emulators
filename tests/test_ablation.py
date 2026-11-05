#!/usr/bin/env python3
"""Synthetic tests for FineUNet ablations and unchanged FutureINNPlus L=3."""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from main_ablation import finalize_args, get_args_parser
from main_plus import finalize_args as finalize_plus
from model_ablation import AblationEmulator, field_loss
from model_plus import FutureINNPlus

IMG = 64


def _ablation_ns(**kwargs) -> argparse.Namespace:
    defaults = dict(
        ablation="l1",
        train_unroll=1,
        aux_levels=0,
        aux_loss_weight=1.0,
        ds_factors=None,
        img_size=IMG,
        num_of_frame=-1,
        output_dir="",
        log_dir="",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _plus_ns(**kwargs) -> argparse.Namespace:
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


def test_l1_shapes_and_forward_signature():
    args = finalize_args(_ablation_ns(ablation="l1"))
    model = AblationEmulator(args)
    model.train()
    x = torch.randn(2, 2, IMG, IMG)
    loss, high_loss, aux = model(x)
    assert aux == []
    assert torch.isfinite(loss)
    assert high_loss.ndim == 0

    sig = inspect.signature(model.vit.forward)
    assert list(sig.parameters) == ["x"]
    out = model.vit(x[:, [0]])
    assert out.shape == (2, 1, IMG, IMG)


def test_aux_forward_has_no_coarse_argument():
    args = finalize_args(
        _ablation_ns(ablation="l1_aux", aux_levels=2, ds_factors=[8, 32])
    )
    model = AblationEmulator(args)
    sig = inspect.signature(model.vit.forward)
    assert "coarses" not in sig.parameters
    assert list(sig.parameters) == ["x"]

    x = torch.randn(2, 4, IMG, IMG)
    high, aux = model.vit(x[:, [0]])
    assert high.shape == (2, 1, IMG, IMG)
    assert len(aux) == 2
    assert aux[0].shape == (2, 1, IMG // 8, IMG // 8)
    assert aux[1].shape == (2, 1, IMG // 32, IMG // 32)

    loss, high_loss, aux_losses = model(x)
    assert len(aux_losses) == 2
    assert torch.isfinite(loss)


def test_aux_loss_changes_shared_backbone_weight():
    args = finalize_args(
        _ablation_ns(ablation="l1_aux", aux_levels=1, ds_factors=[8])
    )
    model = AblationEmulator(args)
    model.train()
    with torch.no_grad():
        model.vit.out_conv.weight.zero_()
        model.vit.out_conv.bias.zero_()
        model.vit.out_conv.weight.requires_grad_(False)
        model.vit.out_conv.bias.requires_grad_(False)

    enc_param = next(model.vit.enc.parameters())
    model.zero_grad(set_to_none=True)
    x = torch.randn(2, 3, IMG, IMG)
    loss, high_loss, aux_losses = model(x)
    assert len(aux_losses) == 1
    (model.aux_loss_weight * aux_losses[0]).backward()
    assert enc_param.grad is not None
    assert enc_param.grad.abs().sum().item() > 0


def test_unroll_p2_sends_grad_through_first_predicted_state():
    args = finalize_args(
        _ablation_ns(ablation="l1_unrolled", train_unroll=2)
    )
    model = AblationEmulator(args)
    model.train()
    x = torch.randn(2, 3, IMG, IMG)

    pred1 = model.vit(x[:, [0]])
    pred2 = model.vit(pred1)
    l2 = field_loss(pred2, x[:, [2]])
    (g,) = torch.autograd.grad(l2, pred1, retain_graph=False, allow_unused=False)
    assert g is not None
    assert g.abs().sum().item() > 0

    model.zero_grad(set_to_none=True)
    first_w = model.vit.enc[f"{IMG}x{IMG}_conv"].weight
    loss, high_l, _ = model(x)
    (loss - high_l).backward()
    assert first_w.grad is not None
    assert first_w.grad.abs().sum().item() > 0


def test_p1_loss_matches_one_step_loss():
    args_u = finalize_args(
        _ablation_ns(ablation="l1_unrolled", train_unroll=1)
    )
    args_l1 = finalize_args(_ablation_ns(ablation="l1"))
    m_u = AblationEmulator(args_u)
    m_l1 = AblationEmulator(args_l1)
    m_l1.load_state_dict(m_u.state_dict())
    m_u.eval()
    m_l1.eval()
    x = torch.randn(2, 2, IMG, IMG)
    with torch.no_grad():
        loss_u, high_u, _ = m_u(x)
        loss_l1, high_l1, _ = m_l1(x)
    assert torch.allclose(loss_u, loss_l1)
    assert torch.allclose(high_u, high_l1)
    assert torch.allclose(loss_u, high_u)


def test_ar_rollout_shape():
    args = finalize_args(_ablation_ns(ablation="l1"))
    model = AblationEmulator(args)
    model.eval()
    high = torch.randn(2, 1, IMG, IMG)
    out = model.sample(high, ema=False)
    assert out.shape == (2, 1, IMG, IMG)
    cur = high
    for _ in range(3):
        cur = model.sample(cur, ema=False)
    assert cur.shape == (2, 1, IMG, IMG)


def test_save_and_load():
    args = finalize_args(
        _ablation_ns(ablation="l1_aux", aux_levels=1, ds_factors=[8])
    )
    model = AblationEmulator(args)
    model.eval()
    x = torch.randn(1, 1, IMG, IMG)
    with torch.no_grad():
        out_a, aux_a = model.vit(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"model": model.state_dict(), "args": args}, path)
        loaded = AblationEmulator(args)
        state = torch.load(path, map_location="cpu", weights_only=False)
        loaded.load_state_dict(state["model"])
        loaded.eval()
        with torch.no_grad():
            out_b, aux_b = loaded.vit(x)
        assert torch.allclose(out_a, out_b)
        assert torch.allclose(aux_a[0], aux_b[0])


def test_futureinnplus_l3_still_returns_high_plus_two_coarses():
    args = finalize_plus(_plus_ns(hierarchy_L=3, ds_factors=[8, 32]))
    model = FutureINNPlus(args)
    b = 1
    high = torch.randn(b, 1, 256, 256)
    coarses = [
        torch.zeros(b, 1, 32, 32),
        torch.zeros(b, 1, 8, 8),
    ]
    t = torch.zeros(b)
    out_h, outs = model.vit(high, coarses, t)
    assert out_h.shape == (b, 1, 256, 256)
    assert len(outs) == 2
    assert outs[0].shape == (b, 1, 32, 32)
    assert outs[1].shape == (b, 1, 8, 8)


def test_cli_rejects_hine_and_accepts_flags():
    p = get_args_parser()
    args = p.parse_args(["--ablation", "l1"])
    args = finalize_args(args)
    assert args.num_of_frame == 2

    args = finalize_args(
        _ablation_ns(ablation="l1_aux", aux_levels=2, ds_factors=None, img_size=256)
    )
    assert args.ds_factors == [8, 32]
    assert args.num_of_frame == 4

    args = finalize_args(
        _ablation_ns(ablation="l1_unrolled", train_unroll=15)
    )
    assert args.num_of_frame == 16

    try:
        finalize_args(_ablation_ns(ablation="hine"))
        assert False, "expected SystemExit for hine"
    except SystemExit as e:
        assert e.code == 2


def main() -> None:
    test_l1_shapes_and_forward_signature()
    test_aux_forward_has_no_coarse_argument()
    test_aux_loss_changes_shared_backbone_weight()
    test_unroll_p2_sends_grad_through_first_predicted_state()
    test_p1_loss_matches_one_step_loss()
    test_ar_rollout_shape()
    test_save_and_load()
    test_futureinnplus_l3_still_returns_high_plus_two_coarses()
    test_cli_rejects_hine_and_accepts_flags()
    print("ALL_ABLATION_TESTS_PASSED")


if __name__ == "__main__":
    main()
