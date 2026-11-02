"""FutureINNPlus: L-level hierarchical emulator (K = L-1 coarse grids).

Does not modify model.FutureINN. Coarse sizes come from args.latent_res.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from ufno_plus import PYRAMID_RES, UNetPlus

F = torch.nn.functional


def default_inject_res(latent_res: list[int]) -> list[int]:
    """Encoder inject maps for each latent. Paper pair (32, 8) -> (64, 32)."""
    latent_res = [int(r) for r in latent_res]
    if latent_res == [32, 8]:
        return [64, 32]
    out: list[int] = []
    used: set[int] = set()
    for lr in latent_res:
        r = min(2 * lr, 128)
        if r in used or r not in PYRAMID_RES:
            r = lr
        if r in used:
            for cand in PYRAMID_RES:
                if cand not in used:
                    r = cand
                    break
        if r in used:
            raise ValueError(
                f"cannot assign unique inject_res for latent_res={latent_res}"
            )
        used.add(r)
        out.append(r)
    return out


def validate_latent_args(img_size: int, latent_res: list[int], inject_res: list[int]):
    allowed = set(PYRAMID_RES)
    if img_size != 256:
        # still require dyadic factors under img_size
        allowed = {img_size >> level for level in range(1, 6)}
    for r in latent_res:
        if r not in allowed:
            raise ValueError(
                f"latent_res value {r} not in allowed pyramid {sorted(allowed)}"
            )
        if img_size % r != 0:
            raise ValueError(f"img_size {img_size} must be divisible by latent {r}")
    for r in inject_res:
        if r not in allowed:
            raise ValueError(
                f"inject_res value {r} not in allowed pyramid {sorted(allowed)}"
            )
    if len(inject_res) != len(latent_res):
        raise ValueError("inject_res length must match latent_res")
    if len(set(inject_res)) != len(inject_res):
        raise ValueError(f"inject_res must be unique, got {inject_res}")
    if inject_res != sorted(inject_res, reverse=True):
        raise ValueError(
            f"inject_res must be strictly descending (encoder order), got {inject_res}"
        )
    if latent_res != sorted(latent_res, reverse=True):
        raise ValueError(
            f"latent_res should be descending (finer to coarser), got {latent_res}"
        )


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class FutureINNPlus(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_frame = int(args.num_of_frame)
        self.img_size = int(args.img_size)
        self.ratio = float(args.ratio)

        latent_res = [int(r) for r in args.latent_res]
        if getattr(args, "inject_res", None):
            inject_res = [int(r) for r in args.inject_res]
        else:
            inject_res = default_inject_res(latent_res)
        validate_latent_args(self.img_size, latent_res, inject_res)

        self.latent_res = latent_res
        self.inject_res = inject_res
        self.num_coarse = len(latent_res)
        self.ds_factors = [self.img_size // r for r in latent_res]
        # Keep paper-style aliases when K>=2 for logging / eval helpers.
        self.ds_factor = self.ds_factors[0]
        self.ds_res = latent_res[0]
        self.ds_factor_low = self.ds_factors[-1]
        self.low_res = latent_res[-1]

        expected_frames = self.num_coarse + 2
        if self.num_frame != expected_frames:
            raise ValueError(
                f"num_of_frame must be num_coarse+2={expected_frames}, "
                f"got {self.num_frame}"
            )

        self.vit = UNetPlus(
            in_channels=1,
            out_channels=1,
            img_resolution=self.img_size,
            latent_res=latent_res,
            inject_res=inject_res,
        )
        self.ema_vit = UNetPlus(
            in_channels=1,
            out_channels=1,
            img_resolution=self.img_size,
            latent_res=latent_res,
            inject_res=inject_res,
        )

    def _pool_seq(self, img_seq: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.avg_pool2d(img_seq, kernel_size=ds, stride=ds) for ds in self.ds_factors
        ]

    def forward(self, img_seq: torch.Tensor):
        b = img_seq.shape[0]
        k = self.num_coarse
        device = img_seq.device
        pooled = self._pool_seq(img_seq)

        zeros_high = torch.zeros_like(img_seq[:, [1]])
        zeros_c = [
            torch.zeros(
                b, 1, self.latent_res[i], self.latent_res[i], device=device, dtype=img_seq.dtype
            )
            for i in range(k)
        ]

        # Case c < K: warmup. Case K: full autoregressive.
        # inputs_cases[c] = (high, list of K coarses, indicator)
        # targets_cases[c] = (high_tgt, list of K coarse tgts)
        inputs_cases = []
        targets_cases = []
        for c in range(k):
            coarses = []
            for i in range(k):
                if i < c:
                    coarses.append(pooled[i][:, [i + 1]])
                else:
                    coarses.append(zeros_c[i])
            inputs_cases.append(
                (
                    img_seq[:, [0]],
                    coarses,
                    float(c) * torch.ones(b, device=device, dtype=img_seq.dtype),
                )
            )
            tgts = []
            for i in range(k):
                if i == c:
                    tgts.append(pooled[i][:, [c + 1]])
                else:
                    tgts.append(zeros_c[i])
            targets_cases.append((zeros_high, tgts))

        # Full case
        coarses_full = [pooled[i][:, [i + 1]] for i in range(k)]
        inputs_cases.append(
            (
                img_seq[:, [0]],
                coarses_full,
                float(k) * torch.ones(b, device=device, dtype=img_seq.dtype),
            )
        )
        targets_cases.append(
            (
                img_seq[:, [1]],
                [pooled[i][:, [i + 2]] for i in range(k)],
            )
        )

        n_cases = k + 1
        probs = torch.full((k,), (1.0 - self.ratio) / k)
        probs = torch.cat([probs, torch.tensor([self.ratio])])
        case_idx = torch.multinomial(probs, b, replacement=True)
        mask = F.one_hot(case_idx, num_classes=n_cases).to(
            device=device, dtype=img_seq.dtype
        )[
            :, :, None, None, None
        ]  # (B, n_cases, 1, 1, 1)

        high_in = sum(inputs_cases[c][0] * mask[:, c] for c in range(n_cases))
        coarses_in = []
        for i in range(k):
            coarses_in.append(
                sum(inputs_cases[c][1][i] * mask[:, c] for c in range(n_cases))
            )
        t_in = sum(
            inputs_cases[c][2] * mask[:, c].reshape(-1) for c in range(n_cases)
        )

        high_tgt = sum(targets_cases[c][0] * mask[:, c] for c in range(n_cases))
        coarses_tgt = []
        for i in range(k):
            coarses_tgt.append(
                sum(targets_cases[c][1][i] * mask[:, c] for c in range(n_cases))
            )

        out_high, out_coarses = self.vit(high_in, coarses_in, t_in)

        high_loss = ((high_tgt - out_high) ** 2) + (high_tgt - out_high).abs()
        coarse_losses = [
            ((coarses_tgt[i] - out_coarses[i]) ** 2)
            + (coarses_tgt[i] - out_coarses[i]).abs()
            for i in range(k)
        ]

        # High: only full case. Coarse i: warmup case i or full case.
        high_loss = high_loss * (mask[:, k] == 1).float()
        for i in range(k):
            keep = (mask[:, i] == 1).float() + (mask[:, k] == 1).float()
            coarse_losses[i] = coarse_losses[i] * keep

        high_loss_m = (2 * high_loss).mean()
        coarse_loss_m = [cl.mean() for cl in coarse_losses]
        loss = high_loss_m + sum(coarse_loss_m)
        return loss, high_loss_m, coarse_loss_m

    def ema_step(self, ema_ratio: float = 0.9995):
        update_ema(self.ema_vit, self.vit, ema_ratio)

    @torch.no_grad()
    def sample(self, high_x, coarses=None, ema: bool = True):
        model = self.ema_vit if ema else self.vit
        b = high_x.shape[0]
        k = self.num_coarse
        device = high_x.device
        dtype = high_x.dtype
        self.eval()
        with torch.no_grad():
            if coarses is None:
                coarses = [
                    torch.zeros(
                        b, 1, self.latent_res[i], self.latent_res[i], device=device, dtype=dtype
                    )
                    for i in range(k)
                ]
                with torch.cuda.amp.autocast():
                    for c in range(k):
                        t = float(c) * torch.ones(b, device=device, dtype=dtype)
                        _, outs = model(high_x, coarses, t)
                        coarses[c] = outs[c]
                # high unchanged during warmup
            else:
                with torch.cuda.amp.autocast():
                    t = float(k) * torch.ones(b, device=device, dtype=dtype)
                    high_x, coarses = model(high_x, coarses, t)
        self.train()
        return high_x, coarses
