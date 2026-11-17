#!/usr/bin/env python3
"""Unit tests for block Jacobian tag parsing (no GPU, no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for path in (_REPO, _EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jacobian_blocks_lib import (
    LEVELS_L3,
    all_blocks,
    canonicalize_names,
    levels_plus,
    parse_block_tag,
    resolve_blocks,
    tag_to_dirname,
)


def _raises(fn, *args, match: str | None = None):
    try:
        fn(*args)
    except ValueError as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(
                f"expected match {match!r} in {exc!r}"
            ) from exc
        return
    raise AssertionError("expected ValueError")


def test_levels_plus_l2():
    assert levels_plus(1) == ("U", "Z1")


def test_levels_plus_l3():
    assert levels_plus(2) == ("U", "Z1", "Z2")


def test_all_blocks_counts():
    assert len(all_blocks(LEVELS_L3)) == 49
    assert len(all_blocks(levels_plus(1))) == 9


def test_parse_z1_dash_z1_z2():
    spec = parse_block_tag("Z1-Z1,Z2", LEVELS_L3)
    assert spec.inputs == ("Z1",)
    assert spec.outputs == ("Z1", "Z2")
    assert spec.tag == "Z1-Z1,Z2"
    assert not spec.is_square
    assert spec.spectrum_kind == "singular_value"


def test_parse_full_and_high():
    full = parse_block_tag("U,Z1,Z2-U,Z1,Z2", LEVELS_L3)
    assert full.is_square
    assert full.spectrum_kind == "eigenvalue"
    high = parse_block_tag("U-U", LEVELS_L3)
    assert high.is_square
    assert high.tag == "U-U"


def test_canonicalize_reorders():
    names = canonicalize_names(["Z2", "U"], LEVELS_L3)
    assert names == ("U", "Z2")
    # Same set after reorder => square.
    spec = parse_block_tag("Z2,U-U,Z2", LEVELS_L3)
    assert spec.inputs == ("U", "Z2")
    assert spec.outputs == ("U", "Z2")
    assert spec.is_square


def test_resolve_default_and_subset():
    all49 = resolve_blocks(None, LEVELS_L3)
    assert len(all49) == 49
    subset = resolve_blocks(["U-U", "Z1-Z1,Z2", "U-U"], LEVELS_L3)
    assert [s.tag for s in subset] == ["U-U", "Z1-Z1,Z2"]


def test_l2_rejects_z2():
    _raises(parse_block_tag, "Z2-Z2", levels_plus(1), match="Unknown level")


def test_bad_tag_forms():
    _raises(parse_block_tag, "U", LEVELS_L3)
    _raises(parse_block_tag, "U-Z1-Z2", LEVELS_L3)
    _raises(parse_block_tag, "-U", LEVELS_L3)


def test_tag_to_dirname():
    assert tag_to_dirname("U,Z1,Z2-U,Z1,Z2") == "U_Z1_Z2-U_Z1_Z2"
    assert tag_to_dirname("Z1-Z1,Z2") == "Z1-Z1_Z2"


def test_rectangular_svds_accepts_column_cotangent():
    """Tall blocks (fine from coarse) and wide blocks (coarse from fine)."""
    import numpy as np
    import torch
    from torch import nn

    from eval_jacobian_blocks import compute_block_spectrum
    from eval_jacobian_blocks_plus import compute_block_spectrum_plus
    from jacobian_blocks_lib import LEVELS_L3, levels_plus, parse_block_tag

    class TinyL3(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, u, z1, z2, t):
            oh = self.scale * u + z2.repeat_interleave(2, dim=-1).repeat_interleave(
                2, dim=-2
            )
            om = self.scale * z1
            ol = self.scale * z2 + u[..., ::2, ::2]
            return oh, om, ol

    class TinyL2(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, u, coarses, t):
            z1 = coarses[0]
            oh = self.scale * u + z1.repeat_interleave(2, dim=-1).repeat_interleave(
                2, dim=-2
            )
            oz = self.scale * z1 + u[..., ::2, ::2]
            return oh, [oz]

    class Model:
        num_coarse = 1

    def _check(result, tag: str) -> None:
        radius = float(np.asarray(result["radius"]))
        assert result["spectrum_kind"].item() == "singular_value", tag
        assert np.isfinite(radius) and radius > 0, (tag, radius)

    torch.manual_seed(0)
    high = torch.randn(1, 1, 4, 4)
    mid = torch.randn(1, 1, 3, 3)
    low = torch.randn(1, 1, 2, 2)
    l3 = TinyL3()
    for tag in ("Z2-U", "U-Z2"):
        block = parse_block_tag(tag, LEVELS_L3)
        result = compute_block_spectrum(
            l3, high, mid, low, block, ncv=20, maxiter=200, tol=1e-3
        )
        _check(result, tag)

    z1 = torch.randn(1, 1, 2, 2)
    allowed = levels_plus(1)
    l2 = TinyL2()
    for tag in ("Z1-U", "U-Z1"):
        block = parse_block_tag(tag, allowed)
        result = compute_block_spectrum_plus(
            l2,
            Model(),
            high,
            [z1],
            allowed,
            block,
            ncv=20,
            maxiter=200,
            tol=1e-3,
        )
        _check(result, tag)


def test_choose_sample_starts():
    from eval_jacobian_blocks import choose_sample_starts

    starts = choose_sample_starts(100, lookahead=2, n_samples=20, seed=0)
    assert len(starts) == 20
    assert len(set(starts.tolist())) == 20
    assert starts.min() >= 0
    assert starts.max() <= 100 - 1 - 2
    # Deterministic.
    starts2 = choose_sample_starts(100, lookahead=2, n_samples=20, seed=0)
    assert (starts == starts2).all()


if __name__ == "__main__":
    test_levels_plus_l2()
    test_levels_plus_l3()
    test_all_blocks_counts()
    test_parse_z1_dash_z1_z2()
    test_parse_full_and_high()
    test_canonicalize_reorders()
    test_resolve_default_and_subset()
    test_l2_rejects_z2()
    test_bad_tag_forms()
    test_tag_to_dirname()
    test_choose_sample_starts()
    test_rectangular_svds_accepts_column_cotangent()
    print("ok")
