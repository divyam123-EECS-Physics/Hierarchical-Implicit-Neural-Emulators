#!/usr/bin/env python3
"""Train the paper's L=1 FNO baseline on 2D NS vorticity (arXiv:2506.04528).

Uses the same data loader / AdamW / lr schedule conventions as main.py.
One-step map: frame[t] -> frame[t+1] with ell_1 + ell_2 loss.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import util.lr_sched as lr_sched
import util.misc as misc
from dataloader import DistributedPreloadDataset
from fno_baseline import FNO2d, fno_step_loss
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("FNO L=1 baseline", add_help=False)
    p.add_argument("--img_size", default=256, type=int)
    p.add_argument("--num_of_frame", default=2, type=int, help="L=1: current + next")
    p.add_argument("--modes", default=96, type=int, help="Fourier modes; paper grid {64,96,128}")
    p.add_argument("--width", default=32, type=int, help="Channel width (~130M with modes=96, 3 layers)")
    p.add_argument("--n_layers", default=3, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--eval_freq", type=int, default=5)
    p.add_argument("--save_last_freq", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--lr_schedule", type=str, default="constant")
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--output_dir", default="./output_fno")
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--num_workers", default=10, type=int)
    p.add_argument("--pin_mem", action="store_true")
    p.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    p.set_defaults(pin_mem=True)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url", default="env://")
    p.add_argument("--start_epoch", default=0, type=int)
    return p


@torch.no_grad()
def validation_loss(model, dataset_validation, device, args) -> dict[str, float]:
    model.eval()
    net = model.module if hasattr(model, "module") else model
    loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    total = 0.0
    n = 0
    for samples in loader:
        x = samples.to(device).float()
        pred = net(x[:, [0]])
        loss = fno_step_loss(pred, x[:, [1]])
        b = x.shape[0]
        total += float(loss) * b
        n += b
    model.train()
    return {"val_loss": total / max(n, 1)}


@torch.no_grad()
def short_rollout_mse(model, dataset, device, sidx: int, tms: int = 20) -> float:
    """Autoregressive MSE vs ground truth over tms steps from start index sidx."""
    net = model.module if hasattr(model, "module") else model
    n_frames = min(sidx + tms + 1, len(dataset.images))
    if n_frames - sidx < 2:
        return float("nan")
    truth = torch.stack(
        [dataset.images[i] for i in range(sidx, n_frames)], dim=0
    ).float().to(device)  # (T, H, W)
    pred = truth[0:1].unsqueeze(0)  # (1, 1, H, W)
    errs = []
    for t in range(1, truth.shape[0]):
        pred = net(pred)
        errs.append(float(((pred[0, 0] - truth[t]) ** 2).mean().item()))
    return float(np.mean(errs)) if errs else float("nan")


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 12
    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        x = samples.to(device).float()
        pred = model(x[:, [0]])
        loss = fno_step_loss(pred, x[:, [1]])

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise SystemExit(1)

        loss.backward()
        grad_norm = misc.get_grad_norm_(model.parameters())
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(grad_norm=grad_norm.mean() if grad_norm is not None else 0.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)
    print("job dir:", os.path.dirname(os.path.realpath(__file__)))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = DistributedPreloadDataset(args.num_of_frame, args.img_size)
    dataset_validation = DistributedPreloadDataset(
        args.num_of_frame, args.img_size, validation=True
    )
    print("train dataset", len(dataset_train))
    print("validation dataset", len(dataset_validation))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )

    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO2d modes={args.modes} width={args.width} layers={args.n_layers}")
    print(f"Number of trainable parameters: {n_params / 1e6:.2f}M")

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.99, 0.999),
    )
    # lr_sched expects param_groups with optional lr_scale
    for g in optimizer.param_groups:
        g["lr_scale"] = 1.0
    loss_scaler = NativeScaler()

    ckpt_last = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else ""
    if args.resume and os.path.isfile(ckpt_last):
        checkpoint = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resumed from {ckpt_last} at epoch {args.start_epoch}")
        del checkpoint
    else:
        print("Training from scratch")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, args
        )
        val_stats = validation_loss(model, dataset_validation, device, args)

        if epoch % args.eval_freq == 0 and epoch > 0 and misc.is_main_process():
            model.eval()
            for sidx in (0, 100):
                mse = short_rollout_mse(model, dataset_validation, device, sidx, tms=20)
                print(f"short rollout val sidx={sidx}: mean MSE={mse:.6g}")
            model.train()

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=None,
                epoch_name="last",
            )
        if epoch > 100 and epoch % args.save_last_freq == 0:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=None,
                epoch_name=f"{epoch}",
            )

        if misc.is_main_process():
            metrics_dir = os.path.join(args.output_dir, "csv")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "train_metrics.csv")
            fieldnames = ["epoch", "lr", "loss", "grad_norm", "val_loss"]
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "epoch": epoch,
                        "lr": train_stats.get("lr", ""),
                        "loss": train_stats.get("loss", ""),
                        "grad_norm": train_stats.get("grad_norm", ""),
                        "val_loss": val_stats.get("val_loss", ""),
                    }
                )

    total_time = time.time() - start_time
    print("Training time", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "FNO L=1 baseline", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
