"""Matrix-free parasitic block S_n = K_n - D Phi'_{n+1} B_n.

Phi' is whatever callable the caller passes. The Task 3 runner passes the
solver tangent at the true field u_{n+1}, not at u_n.
"""

from __future__ import annotations

import numpy as np

from py2d_inmem import embed_field, fd_tangent, pool_avg, truncate_field


def compose_S(v: np.ndarray, g_z, h_z, phi_at_np1, downsample) -> np.ndarray:
    """S v. g_z and h_z are the fine and latent heads' latent derivatives."""
    bv = g_z(v)
    kv = h_z(v)
    return kv - downsample(phi_at_np1(bv))


def hine_S_action(
    model,
    solver,
    u_n: np.ndarray,
    z_n: np.ndarray,
    u_np1_512: np.ndarray,
    v: np.ndarray,
    delta_t: float,
    eps_rel: float,
    ref_scale: float,
    pool_factor: int,
    nx_model: int,
) -> np.ndarray:
    """One latent-space action. The finite-difference base point is u_{n+1}."""
    bv, kv = model.jvp_latent(u_n, z_n, v)
    direction = embed_field(bv, solver.cfg.nx)

    def flow(u512):
        return solver.advance_physical(u512, delta_t)

    phi = fd_tangent(flow, u_np1_512, direction, eps_rel, ref_scale)
    if phi.shape[0] != nx_model:
        phi = truncate_field(phi, nx_model)
    return kv - pool_avg(phi, pool_factor)
