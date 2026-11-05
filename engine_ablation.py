"""Training / validation for FineUNet ablations. No Jacobian or inject diagnostics."""

from __future__ import annotations

import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.lr_sched as lr_sched
import util.misc as misc


@torch.no_grad()
def validation_loss(model, dataset_validation, device, args):
    model.eval()
    net = model.module if hasattr(model, "module") else model
    n_aux = int(getattr(net, "aux_levels", 0) or 0)
    loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    sum_loss = torch.zeros(2 + n_aux, device=device)
    n_samples = torch.zeros(1, device=device)
    for samples in loader:
        x = samples.to(device).float()
        b = x.shape[0]
        with torch.cuda.amp.autocast():
            loss, high_loss, aux_losses = net(x)
        sum_loss[0] += float(loss) * b
        sum_loss[1] += float(high_loss) * b
        for i, al in enumerate(aux_losses):
            sum_loss[2 + i] += float(al) * b
        n_samples[0] += b
    if misc.get_world_size() > 1:
        dist.all_reduce(sum_loss)
        dist.all_reduce(n_samples)
    model.train()
    n = max(float(n_samples.item()), 1.0)
    out = {
        "val_loss": (sum_loss[0] / n).item(),
        "val_high_loss": (sum_loss[1] / n).item(),
    }
    for i in range(n_aux):
        out[f"val_aux{i}_loss"] = (sum_loss[2 + i] / n).item()
    return out


def train_one_epoch(
    model,
    model_params,
    ema_params,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    dataset_validation=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 12
    optimizer.zero_grad()

    net = model.module if hasattr(model, "module") else model

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        x = samples.to(args.device).float()

        with torch.cuda.amp.autocast():
            loss, high_loss, aux_losses = model(x)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(high_loss=high_loss.item())
        for i, al in enumerate(aux_losses):
            metric_logger.update(**{f"aux{i}_loss": al.item()})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=None,
            parameters=model.parameters(),
            update_grad=True,
        )
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if hasattr(model, "module"):
            model.module.ema_step(args.ema_rate)
        else:
            net.ema_step(args.ema_rate)
        metric_logger.update(grad_norm=grad_norm.mean())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k_: meter.global_avg for k_, meter in metric_logger.meters.items()}, x[:, [0]]
