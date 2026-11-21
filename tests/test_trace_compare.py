#!/usr/bin/env python3
"""Synthetic checks for trace identities. No checkpoint and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "eval"
for _path in (_REPO, _EVAL):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from l2_fd_jvp import apply_D, apply_Dplus
from trace_compare_ops import (
    dense_square_reference,
    directional_quadratic,
    enumerate_rademacher,
    expanding_negative_trace,
    finite_difference_check,
    loop_increases_despite_negative_trace,
    project_blocks,
    square_probe_metrics,
    two_step_energy,
)


def _apply_all(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probes = enumerate_rademacher(matrix.shape[0])
    images = probes @ matrix.T
    dots = np.sum(probes * images, axis=1)
    fro = np.sum(images * images, axis=1)
    return dots, fro


def test_square_trace_matches_dense_reference():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(3, 3))
    dots, images = _apply_all(matrix)
    got = square_probe_metrics(dots, images, 3)
    ref = dense_square_reference(matrix)
    assert abs(got["trace"] - ref["trace"]) < 1e-10
    assert abs(got["tau"] - ref["tau"]) < 1e-10
    assert abs(got["energy"] - ref["energy"]) < 1e-10
    assert abs(got["identity_residual"]) < 1e-10
    assert abs(ref["identity_residual"]) < 1e-12


def test_negative_trace_need_not_contract():
    matrix = expanding_negative_trace()
    ref = dense_square_reference(matrix)
    assert ref["tau"] < 0.0
    assert ref["spectral_radius"] > 1.0
    assert ref["energy"] > 0.0


def test_negative_loop_trace_can_increase_energy():
    example = loop_increases_despite_negative_trace()
    assert example["trace_loop"] < 0.0
    assert example["metrics"]["interference"] < 0.0
    assert example["metrics"]["loop_net"] > 0.0
    assert abs(example["metrics"]["identity_residual"]) < 1e-12


def test_cycle_keeps_denominator_n():
    rng = np.random.default_rng(1)
    fine = 3
    latent = 2
    b_map = rng.normal(size=(fine, latent))
    c_map = rng.normal(size=(latent, fine))
    product = b_map @ c_map
    fine_probes = enumerate_rademacher(fine)
    fine_images = fine_probes @ product.T
    fine_dots = np.sum(fine_probes * fine_images, axis=1)
    latent_probes = enumerate_rademacher(latent)
    latent_product = c_map @ b_map
    latent_images = latent_probes @ latent_product.T
    latent_dots = np.sum(latent_probes * latent_images, axis=1)
    fine_trace = float(fine_dots.mean())
    latent_trace = float(latent_dots.mean())
    assert abs(fine_trace - latent_trace) < 1e-10
    assert abs(fine_trace - np.trace(product)) < 1e-10
    assert abs(fine_trace / fine - np.trace(product) / latent) > 1e-6


def test_frozen_and_consecutive_products_differ():
    b_now = np.array([[1.0, 0.0], [0.0, 0.0]])
    c_now = np.array([[1.0, 0.0], [0.0, 1.0]])
    b_next = np.array([[0.0, 4.0], [0.0, 0.0]])
    frozen = float(np.trace(b_now @ c_now))
    consecutive = float(np.trace(b_next @ c_now))
    assert frozen != consecutive


def test_projector_matches_pool_and_is_orthogonal():
    rng = np.random.default_rng(2)
    side, factor = 4, 2
    flat = rng.normal(size=side * side)
    projected, complement = project_blocks(flat, side, factor)
    image = torch.from_numpy(flat.reshape(1, 1, side, side))
    pooled = apply_D(image, factor)
    lifted = apply_Dplus(pooled, factor).reshape(-1).numpy()
    assert np.allclose(projected, lifted)
    again, _rest = project_blocks(projected, side, factor)
    assert np.allclose(again, projected)
    other = rng.normal(size=flat.size)
    other_p, _other_q = project_blocks(other, side, factor)
    assert abs(float(np.dot(complement, other_p))) < 1e-10
    coarse = apply_D(torch.from_numpy(complement.reshape(1, 1, side, side)), factor)
    assert float(torch.linalg.vector_norm(coarse)) < 1e-10


def test_zero_error_is_masked():
    assert np.isnan(directional_quadratic(np.ones(4), np.zeros(4), subtract_identity=True))


def test_two_step_identity_on_shared_probes():
    probes = enumerate_rademacher(2)
    direct = np.sum(probes * probes, axis=1)
    loop = np.sum((0.5 * probes) * (0.5 * probes), axis=1)
    cross = np.sum(probes * (0.5 * probes), axis=1)
    metrics = two_step_energy(direct, loop, cross, 2)
    assert abs(metrics["identity_residual"]) < 1e-12
    assert abs(metrics["total"] - (metrics["direct"] + metrics["interference"] + metrics["loop"])) < 1e-12


def test_finite_difference_matches_linear_map():
    weight = torch.tensor([[1.2, -0.3], [0.4, 0.7]], dtype=torch.float64)

    def forward(state):
        return weight @ state

    state = torch.tensor([0.3, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.2, 0.5], dtype=torch.float64)
    _value, image = torch.autograd.functional.jvp(forward, (state,), (direction,))
    assert torch.allclose(image, weight @ direction)
    report = finite_difference_check(forward, state, direction, image)
    assert report["best_relative"] < 1e-8


def test_plots_omit_missing_latent_values(tmp_path=None):
    import matplotlib

    matplotlib.use("Agg")
    from plot_trace_compare import render

    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/trace_compare_plot_test")
    if tmp_path is None:
        root.mkdir(parents=True, exist_ok=True)
    for name, level, tau_k in (("l1_k4", 1, np.nan), ("l2_k4", 2, -0.2)):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        count = 4
        payload = {
            "stem": np.arange(count),
            "trajectory_id": np.asarray(["loader-0"] * count),
            "tau": np.linspace(-0.2, 0.1, count),
            "tau_se": np.full(count, 0.01),
            "energy": np.linspace(0.05, -0.02, count),
            "energy_se": np.full(count, 0.01),
            "tau_K": np.full(count, tau_k),
            "tau_K_se": np.full(count, 0.01 if level == 2 else np.nan),
            "energy_K": np.full(count, 0.1 if level == 2 else np.nan),
            "tau_BC": np.full(count, -0.05 if level == 2 else np.nan),
            "tau_BC_se": np.full(count, 0.01 if level == 2 else np.nan),
            "tau_loop_next": np.full(count, 0.02 if level == 2 else np.nan),
            "b_gain": np.full(count, 0.3 if level == 2 else np.nan),
            "direct": np.full(count, 1.1),
            "interference": np.full(count, -0.2 if level == 2 else np.nan),
            "loop": np.full(count, 0.4 if level == 2 else np.nan),
            "total": np.full(count, 1.3 if level == 2 else 1.1),
            "loop_net": np.full(count, 0.2 if level == 2 else np.nan),
            "dir_tau": np.linspace(-0.1, 0.05, count),
            "dir_energy": np.linspace(0.02, -0.01, count),
            "dir_bc": np.full(count, -0.01 if level == 2 else np.nan),
            "dir_k": np.full(count, -0.04 if level == 2 else np.nan),
            "e_norm": np.linspace(0.0, 1.0, count),
            "eps_u_norm": np.full(count, 0.1),
            "R_value": np.linspace(0.0, -0.2, count),
            "rho_norm": np.linspace(0.0, 0.4, count),
            "memory_frozen": np.ones((count, 2)),
            "memory_varying": np.full((count, 2), 1.2),
            "horizons": np.asarray([1, 2]),
        }
        if level == 1:
            payload["memory_frozen"] = np.full((count, 2), np.nan)
            payload["memory_varying"] = np.full((count, 2), np.nan)
        np.savez_compressed(directory / "metrics.npz", **payload)
        (directory / "metadata.json").write_text(
            '{"label": "%s", "name": "%s", "level": %d}' % (name, name, level)
        )
    render(root, root / "report.md", smoke=True)
    text = (root / "report.md").read_text()
    assert "N/A" in text
    assert "not prove contraction" in text
    table = (root / "snapshots.csv").read_text().splitlines()[1]
    assert table.split(",")[0] == "l1_k4"
    assert (root / "plots" / "direct_trace.png").is_file()


def main() -> None:
    test_square_trace_matches_dense_reference()
    test_negative_trace_need_not_contract()
    test_negative_loop_trace_can_increase_energy()
    test_cycle_keeps_denominator_n()
    test_frozen_and_consecutive_products_differ()
    test_projector_matches_pool_and_is_orthogonal()
    test_zero_error_is_masked()
    test_two_step_identity_on_shared_probes()
    test_finite_difference_matches_linear_map()
    test_plots_omit_missing_latent_values()
    print("trace compare tests passed")


if __name__ == "__main__":
    main()
