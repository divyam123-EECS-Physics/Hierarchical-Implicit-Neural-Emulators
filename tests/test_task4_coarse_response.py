#!/usr/bin/env python3
"""Synthetic maps for the coarse-response metrics. No checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "eval"))

from l2_fd_jvp import L2Point
from task4_coarse_response import (
    aggregate_rows,
    band_field,
    eta_direction,
    measure_step,
    passes_fd,
    random_field,
    response_metrics,
    rms,
)


def _point(z: torch.Tensor) -> L2Point:
    fine = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    t = torch.ones(1, dtype=torch.float64)
    return L2Point(fine, z, t, fine, z, (0, 0, 0))


def _upsample(z: torch.Tensor, ds: int = 2) -> torch.Tensor:
    return z.repeat_interleave(ds, dim=-1).repeat_interleave(ds, dim=-2)


def _one(step, field: np.ndarray, z: torch.Tensor | None = None):
    if z is None:
        z = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    point = _point(z)
    probes = [{"family": "random", "index": 0, "seed": 1, "field": field}]
    value, rows = measure_step(
        step, point, probes, point.z, ds=2, norm=1.0, eps=1e-3, rtol=1e-5, atol=1e-12
    )
    return value, rows[0]


def test_exact_copy_and_constant_offset():
    def copy_step(_u, z):
        return _upsample(z), z.clone()

    def offset_step(_u, z):
        return _upsample(z) + 0.4, z.clone()

    field = np.array([[2.0, 0.0], [0.0, 0.0]])
    value, row = _one(copy_step, field)
    assert value["rel_undefined"] is False
    assert value["rel_c"] < 1e-8
    assert row["validated"]
    assert abs(row["copy_defect"]) < 1e-6
    assert abs(row["gain"] - 1.0) < 1e-6
    assert abs(row["alignment"] - 1.0) < 1e-6

    offset, offset_row = _one(offset_step, field)
    assert abs(offset["rms_c"] - 0.4) < 1e-8
    assert abs(offset["rel_c"] - 0.4) < 1e-8
    assert abs(offset_row["copy_defect"] - row["copy_defect"]) < 1e-6
    assert abs(offset_row["gain"] - row["gain"]) < 1e-6
    assert abs(offset_row["alignment"] - row["alignment"]) < 1e-6


def test_scaled_map_and_rotation():
    def scaled(_u, z):
        return 0.5 * _upsample(z), 0.5 * z

    field = np.array([[2.0, 0.0], [0.0, 0.0]])
    _value, row = _one(scaled, field)
    assert abs(row["gain"] - 0.5) < 1e-6
    assert abs(row["copy_defect"] - 0.5) < 1e-6
    assert abs(row["alignment"] - 1.0) < 1e-6

    def rotate(_u, z):
        turned = torch.rot90(z, 1, dims=(-2, -1))
        return _upsample(turned), turned

    _value, turned = _one(rotate, field)
    assert abs(turned["gain"] - 1.0) < 1e-6
    assert turned["copy_defect"] > 0.5
    assert turned["alignment"] < 0.5


def test_bands_exclude_the_constant_mode_and_stay_real():
    field, mask, definition = band_field(32, 1, 4, seed=3)
    hat = np.fft.rfft2(field)
    assert definition["mean_removed"] is False
    assert abs(hat[0, 0]) < 1e-8
    assert np.max(np.abs(hat[~mask])) < 1e-6
    assert abs(rms(field) - 1.0) < 1e-8
    assert np.isrealobj(field)


def test_random_probe_keeps_its_mean():
    field = random_field(16, seed=11)
    rng = np.random.default_rng(11)
    raw = rng.standard_normal((16, 16))
    assert abs(float(field.mean()) - float((raw / rms(raw)).mean())) < 1e-12


def test_zero_eta_is_skipped_and_failed_probes_stay_unvalidated():
    assert eta_direction(torch.zeros(1, 1, 2, 2, dtype=torch.float64)) is None
    work = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    ref = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    assert passes_fd(work, ref, rtol=1e-2, atol=1e-12, norm=1.0)["ok"] is False
    rows = [
        {
            "mode": "teacher_forced",
            "family": "random",
            "window": 0,
            "gain": 2.0,
            "copy_defect": 1.0,
            "alignment": 0.0,
            "validated": False,
        },
        {
            "mode": "teacher_forced",
            "family": "random",
            "window": 0,
            "gain": 0.5,
            "copy_defect": 0.1,
            "alignment": 1.0,
            "validated": True,
        },
        {
            "mode": "teacher_forced",
            "family": "random",
            "window": 1,
            "gain": 1.5,
            "copy_defect": 0.2,
            "alignment": 0.5,
            "validated": True,
        },
    ]
    validated = aggregate_rows(rows, validated_only=True)
    block = validated["teacher_forced|random"]
    assert block["per_window"][0]["n"] == 1
    assert block["per_window"][0]["median_gain"] == 0.5
    assert "not an operator norm" in block["across_windows"]["sampled_maximum_is"]
    assert block["across_windows"]["sampled_maximum_gain"] == 1.5
    assert response_metrics(ref, ref)["gain"] == 1.0


if __name__ == "__main__":
    test_exact_copy_and_constant_offset()
    test_scaled_map_and_rotation()
    test_bands_exclude_the_constant_mode_and_stay_real()
    test_random_probe_keeps_its_mean()
    test_zero_eta_is_skipped_and_failed_probes_stay_unvalidated()
    print("ok")
