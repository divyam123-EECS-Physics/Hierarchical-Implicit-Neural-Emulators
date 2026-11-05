"""Physical-coordinate L=2 model step and Jacobian actions.

The network is evaluated on Omega / norm and its outputs are multiplied
back by norm, so a tangent returned here maps a physical perturbation to
a physical perturbation. The latent input is held fixed when differentiating
the fine head, and a latent tangent produces both heads from one JVP.
"""

from __future__ import annotations

import numpy as np
import torch


def assert_l2(model) -> None:
    if int(getattr(model, "num_coarse", -1)) != 1:
        raise ValueError(
            f"Tasks 3 and 5 are L=2 only (num_coarse=1); got "
            f"{getattr(model, 'num_coarse', None)}"
        )


class ModelStepPhysical:
    def __init__(self, ema_vit: torch.nn.Module, norm: float):
        self.ema = ema_vit
        self.norm = float(norm)
        self.dtype = next(ema_vit.parameters()).dtype
        self.device = next(ema_vit.parameters()).device

    def _image(self, field: np.ndarray) -> torch.Tensor:
        arr = np.asarray(field, dtype=np.float64)
        return (
            torch.from_numpy(np.ascontiguousarray(arr))
            .to(device=self.device, dtype=self.dtype)
            .view(1, 1, arr.shape[0], arr.shape[1])
            / self.norm
        )

    def _out(self, tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.detach().reshape(tensor.shape[-2], tensor.shape[-1]).float().cpu().numpy()
            * self.norm
        ).astype(np.float64)

    def step(self, u_phys: np.ndarray, z_phys: np.ndarray):
        u = self._image(u_phys)
        z = self._image(z_phys)
        t = torch.ones(u.shape[0], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            high, coarses = self.ema(u, [z], t)
        return self._out(high), self._out(coarses[0])

    def jvp_latent(self, u_phys: np.ndarray, z_phys: np.ndarray, w_phys: np.ndarray):
        """(B w, K w) in physical units, from one autograd JVP."""
        u = self._image(u_phys)
        z = self._image(z_phys)
        w = self._image(w_phys)
        t = torch.ones(u.shape[0], device=self.device, dtype=self.dtype)

        def f(z_in):
            high, coarses = self.ema(u, [z_in], t)
            return high, coarses[0]

        _primals, tangents = torch.autograd.functional.jvp(f, (z,), (w,))
        return self._out(tangents[0]), self._out(tangents[1])

    def jvp_fine_fixed_latent(self, u_phys: np.ndarray, z_phys: np.ndarray, v_phys: np.ndarray):
        """A v in physical units. The latent tensor is not an input of the JVP."""
        u = self._image(u_phys)
        z = self._image(z_phys)
        v = self._image(v_phys)
        t = torch.ones(u.shape[0], device=self.device, dtype=self.dtype)

        def f(u_in):
            high, _coarses = self.ema(u_in, [z], t)
            return high

        _primal, tangent = torch.autograd.functional.jvp(f, (u,), (v,))
        return self._out(tangent)
