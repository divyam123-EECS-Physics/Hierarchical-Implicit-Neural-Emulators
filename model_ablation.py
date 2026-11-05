"""L=1 FineUNet ablations: plain one-step, aux heads, unrolled training.

No coarse inject modules. Forward takes only the fine field.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu

from ufno import Conv2d, GroupNorm, Linear, PositionalEmbedding, UNetBlock

F = torch.nn.functional


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def field_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = pred - target
    return (err**2 + err.abs()).mean()


class FineUNet(torch.nn.Module):
    """Fourier U-Net backbone without coarse inject.

    Optional aux heads read decoder features at latent resolutions derived from
    ``ds_factors`` (or ``latent_res``). Forward accepts only the fine field.
    """

    def __init__(
        self,
        img_resolution: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_res: list[int] | None = None,
        model_channels: int = 16,
        channel_mult: list[int] | None = None,
        channel_mult_emb: int = 4,
        num_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channel_mult is None:
            channel_mult = [1, 2, 4, 8, 8, 8]
        if attn_resolutions is None:
            attn_resolutions = []
        if latent_res is None:
            latent_res = []

        latent_res = [int(r) for r in latent_res]
        if len(set(latent_res)) != len(latent_res):
            raise ValueError(f"latent_res must be unique, got {latent_res}")

        self.img_resolution = img_resolution
        self.latent_res = list(latent_res)
        self.num_aux = len(latent_res)

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
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
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
        self.aux_out = torch.nn.ModuleDict()
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
                self.aux_out[f"{res}"] = nn.Sequential(
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

    def forward(self, x: torch.Tensor):
        """Predict next fine field (and optional aux coarse maps) from fine ``x`` only."""
        b = x.shape[0]
        t = torch.zeros(b, device=x.device, dtype=x.dtype)
        emb = self.map_noise(t)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        emb = silu(emb)

        skips = []
        for key, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        aux_outs: dict[int, torch.Tensor] = {}
        for key, block in self.dec.items():
            if "up" in key and self.num_aux > 0:
                res = x.shape[-1]
                if res in self.latent_res:
                    idx = self.latent_res.index(res)
                    aux_outs[idx] = self.aux_out[f"{res}"](x)
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))

        if self.num_aux == 0:
            return x

        missing = [i for i in range(self.num_aux) if i not in aux_outs]
        if missing:
            raise RuntimeError(
                f"decoder never produced aux heads for indices {missing}; "
                f"latent_res={self.latent_res}"
            )
        return x, [aux_outs[i] for i in range(self.num_aux)]


class AblationEmulator(nn.Module):
    """Wrapper for l1 / l1_aux / l1_unrolled training and fine-only sampling."""

    def __init__(self, args=None):
        super().__init__()
        self.ablation = str(args.ablation)
        self.img_size = int(args.img_size)
        self.train_unroll = int(getattr(args, "train_unroll", 1))
        self.aux_levels = int(getattr(args, "aux_levels", 0))
        self.aux_loss_weight = float(getattr(args, "aux_loss_weight", 1.0))
        self.num_frame = int(args.num_of_frame)

        ds_factors = [int(d) for d in (getattr(args, "ds_factors", None) or [])]
        if self.ablation == "l1_aux":
            if self.aux_levels not in (1, 2):
                raise ValueError(
                    f"l1_aux requires --aux_levels 1 or 2, got {self.aux_levels}"
                )
            if len(ds_factors) != self.aux_levels:
                raise ValueError(
                    f"l1_aux needs len(--ds_factors)==aux_levels={self.aux_levels}, "
                    f"got {ds_factors}"
                )
            for d in ds_factors:
                if d < 1 or self.img_size % d != 0:
                    raise ValueError(
                        f"img_size {self.img_size} must be divisible by ds_factor {d}"
                    )
            latent_res = [self.img_size // d for d in ds_factors]
            self.num_coarse = self.aux_levels
        else:
            latent_res = []
            ds_factors = []
            self.aux_levels = 0
            self.num_coarse = 0

        self.ds_factors = ds_factors
        self.latent_res = latent_res

        self.vit = FineUNet(
            in_channels=1,
            out_channels=1,
            img_resolution=self.img_size,
            latent_res=latent_res,
        )
        self.ema_vit = FineUNet(
            in_channels=1,
            out_channels=1,
            img_resolution=self.img_size,
            latent_res=latent_res,
        )

    def _predict(self, net: FineUNet, high: torch.Tensor):
        out = net(high)
        if self.aux_levels > 0:
            return out  # (high, aux_list)
        return out, []

    def compute_loss(self, img_seq: torch.Tensor, use_ema: bool = False):
        net = self.ema_vit if use_ema else self.vit
        if self.ablation == "l1_aux":
            high_in = img_seq[:, [0]]
            pred_high, aux_preds = self._predict(net, high_in)
            high_tgt = img_seq[:, [1]]
            high_l = field_loss(pred_high, high_tgt)
            aux_losses = []
            for i, (ds, pred) in enumerate(zip(self.ds_factors, aux_preds)):
                tgt = F.avg_pool2d(
                    img_seq[:, [i + 2]], kernel_size=ds, stride=ds
                )
                aux_losses.append(field_loss(pred, tgt))
            loss = high_l + self.aux_loss_weight * sum(aux_losses)
            return loss, high_l, aux_losses

        # l1 and l1_unrolled share the same module; P comes from train_unroll.
        p = 1 if self.ablation == "l1" else self.train_unroll
        if p < 1:
            raise ValueError(f"train_unroll must be >= 1, got {p}")
        if img_seq.shape[1] < p + 1:
            raise ValueError(
                f"need {p + 1} frames for unroll P={p}, got {img_seq.shape[1]}"
            )
        pred = img_seq[:, [0]]
        step_losses = []
        for step in range(p):
            pred, _ = self._predict(net, pred)
            step_losses.append(field_loss(pred, img_seq[:, [step + 1]]))
        high_l = step_losses[0]
        loss = sum(step_losses)
        return loss, high_l, []

    def forward(self, img_seq: torch.Tensor):
        return self.compute_loss(img_seq, use_ema=False)

    def ema_step(self, ema_ratio: float = 0.9995):
        update_ema(self.ema_vit, self.vit, ema_ratio)

    @torch.no_grad()
    def sample(self, high_x: torch.Tensor, ema: bool = True) -> torch.Tensor:
        """Fine-only AR step; returns only the fine prediction."""
        model = self.ema_vit if ema else self.vit
        self.eval()
        device_type = "cuda" if high_x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=high_x.is_cuda):
            out = model(high_x)
            if self.aux_levels > 0:
                high_x = out[0]
            else:
                high_x = out
        self.train()
        return high_x
