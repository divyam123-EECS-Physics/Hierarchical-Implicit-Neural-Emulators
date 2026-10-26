"""2D FNO baseline matching the paper's L=1 FNO comparison.

Paper (arXiv:2506.04528): 3 FNO layers, modes grid-searched in {64,96,128},
~130M params (4x the L=3 UNet), trained with ell_1 + ell_2 (not Sobolev).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.fourier import SpectralConv2d


class FNO2dBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.w(x))


class FNO2d(nn.Module):
    """Autoregressive one-step map: vorticity(t) -> vorticity(t+1)."""

    def __init__(
        self,
        modes: int = 96,
        width: int = 32,
        n_layers: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNO2dBlock(width, modes, modes) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W). SpectralConv2d's complex GEMM has no ComplexHalf
        # CUDA kernel, so this map always runs in float32.
        x = x.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.lift(x)
            for block in self.blocks:
                x = block(x)
            return self.project(x)


def fno_step_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper FNO baseline uses ell_1 + ell_2 (same as other Table 4 runs)."""
    return ((pred - target) ** 2).mean() + (pred - target).abs().mean()
