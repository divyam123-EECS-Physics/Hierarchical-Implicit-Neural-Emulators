"""UNetPlus: variable-K coarse inject/output heads. Does not modify ufno.UNet."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu

from ufno import (
    Conv2d,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
)

F = torch.nn.functional

PYRAMID_RES = (128, 64, 32, 16, 8)


class UNetPlus(torch.nn.Module):
    """Fourier U-Net with K coarse inject points and K 1-channel output heads.

    Args:
        latent_res: length-K list of spatial sizes for coarse futures (also output heads).
        inject_res: length-K list of encoder resolutions where each coarse is added.
    """

    def __init__(
        self,
        img_resolution: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_res: list[int] | None = None,
        inject_res: list[int] | None = None,
        model_channels: int = 16,
        channel_mult: list[int] | None = None,
        channel_mult_emb: int = 4,
        num_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        dropout: float = 0.2,
        label_dropout: float = 0,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8, 8, 8]
        if attn_resolutions is None:
            attn_resolutions = []
        if latent_res is None:
            latent_res = [32, 8]
        if inject_res is None:
            inject_res = list(latent_res)

        latent_res = [int(r) for r in latent_res]
        inject_res = [int(r) for r in inject_res]
        if len(latent_res) != len(inject_res):
            raise ValueError(
                f"latent_res and inject_res must have the same length, "
                f"got {len(latent_res)} and {len(inject_res)}"
            )
        if len(set(inject_res)) != len(inject_res):
            raise ValueError(f"inject_res must be unique, got {inject_res}")
        if len(set(latent_res)) != len(latent_res):
            raise ValueError(f"latent_res must be unique, got {latent_res}")

        self.label_dropout = label_dropout
        self.latent_res = list(latent_res)
        self.inject_res = list(inject_res)
        self.num_coarse = len(latent_res)

        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        self.enc = torch.nn.ModuleDict()
        self.low_res_inject = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if res in inject_res:
                lat = latent_res[inject_res.index(res)]
                fno_mode = min(lat // 2 + 1, 96)
                self.low_res_inject[f"{res}"] = nn.Sequential(
                    nn.Conv2d(1, cout, kernel_size=3, padding=1),
                    nn.SiLU(),
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        fno_mode=fno_mode,
                        disable_emb=True,
                        **block_kwargs,
                    ),
                    GroupNorm(num_channels=cout),
                )
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                fno_mode = min(res // 2 + 1, 96)
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    fno_mode=fno_mode,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        self.dec = torch.nn.ModuleDict()
        self.low_res_out = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=True,
                    **block_kwargs,
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                fno_mode = min(res // 2 + 1, 96)
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    fno_mode=fno_mode,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
            if res in latent_res:
                fno_mode = min(res // 2 + 1, 96)
                self.low_res_out[f"{res}"] = nn.Sequential(
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        fno_mode=fno_mode,
                        disable_emb=True,
                        **block_kwargs,
                    ),
                    GroupNorm(num_channels=cout),
                    nn.SiLU(),
                    nn.Conv2d(cout, 1, kernel_size=3, padding=1),
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3
        )

    def forward(
        self, x: torch.Tensor, coarses: list[torch.Tensor], t: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(coarses) != self.num_coarse:
            raise ValueError(
                f"expected {self.num_coarse} coarse maps, got {len(coarses)}"
            )

        emb = self.map_noise(t)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        emb = silu(emb)

        inject_list = [
            self.low_res_inject[f"{self.inject_res[i]}"](coarses[i])
            for i in range(self.num_coarse)
        ]

        skips = []
        for key, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
            if x.shape[-1] in self.inject_res and "down" in key:
                x = x + F.interpolate(
                    inject_list.pop(0),
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                )

        coarse_outs: dict[int, torch.Tensor] = {}
        for key, block in self.dec.items():
            if "up" in key:
                for i, c in enumerate(coarses):
                    res = x.shape[-1]
                    if c.shape[-1] == res and self.latent_res[i] == res:
                        coarse_outs[i] = self.low_res_out[f"{res}"](x)
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))

        missing = [i for i in range(self.num_coarse) if i not in coarse_outs]
        if missing:
            raise RuntimeError(
                f"decoder never produced heads for coarse indices {missing}; "
                f"latent_res={self.latent_res}"
            )
        return x, [coarse_outs[i] for i in range(self.num_coarse)]
