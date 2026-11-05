#!/usr/bin/env python3
"""Train FineUNet L=1 ablations (l1, l1_aux, l1_unrolled).

Does not modify main.py / main_plus.py. HINE stays on those entry points.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from dataloader import DistributedPreloadDataset
from engine_ablation import train_one_epoch, validation_loss
from model_ablation import AblationEmulator
from util.misc import NativeScalerWithGradNormCount as NativeScaler

HINE_HELP = """\
'hine' is not implemented in main_ablation.py. Use the existing trainers:

  # L=2 (high 256, one latent at 32)
  torchrun --nproc_per_node=1 main_plus.py \\
    --hierarchy_L 2 --ds_factors 8 --img_size 256 \\
    --ratio 0.85 --lr 3e-4 --batch_size 32 --epochs 200 \\
    --output_dir /pscratch/sd/d/divyam/hine_runs/plus_L2_m32

  # L=3 paper-style
  torchrun --nproc_per_node=1 main.py \\
    --ds_factor 8 --ds_factor_low 32 --img_size 256 ...

  # or Plus L=3
  torchrun --nproc_per_node=1 main_plus.py \\
    --hierarchy_L 3 --ds_factors 8 32 --img_size 256 ...
"""


def get_args_parser():
    parser = argparse.ArgumentParser("FineUNet ablations", add_help=False)
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["l1", "l1_aux", "l1_unrolled", "hine"],
        help="Ablation mode. 'hine' prints the main/main_plus commands and exits.",
    )
    parser.add_argument(
        "--train_unroll",
        type=int,
        default=1,
        help="Unroll length P for l1_unrolled (P=1 matches l1). Ignored for l1/l1_aux.",
    )
    parser.add_argument(
        "--aux_levels",
        type=int,
        default=0,
        help="Number of coarse aux heads for l1_aux (1 or 2).",
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=1.0,
        help="Scale for summed aux losses (l1_aux).",
    )
    parser.add_argument(
        "--ds_factors",
        nargs="+",
        type=int,
        default=None,
        help="Downsample factors for aux heads, e.g. 8 or 8 32 (l1_aux).",
    )
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--num_of_frame", type=int, default=-1)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument("--save_last_freq", type=int, default=50)
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--acc_grad", type=int, default=1)
    parser.add_argument("--ema_rate", default=0.9995, type=float)

    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_epoch", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def default_output_dir(ablation: str, aux_levels: int, train_unroll: int) -> str:
    base = "/pscratch/sd/d/divyam/hine_runs"
    if ablation == "l1":
        name = "ablation_l1"
    elif ablation == "l1_aux":
        name = f"ablation_l1_aux_{aux_levels}"
    elif ablation == "l1_unrolled":
        name = f"ablation_l1_unrolled_p{train_unroll}"
    else:
        name = f"ablation_{ablation}"
    return f"{base}/{name}"


def finalize_args(args):
    if args.ablation == "hine":
        print(HINE_HELP)
        sys.exit(2)

    img_size = int(args.img_size)
    if args.ablation == "l1":
        args.aux_levels = 0
        args.ds_factors = []
        args.train_unroll = 1
        args.num_of_frame = 2
    elif args.ablation == "l1_aux":
        if args.aux_levels not in (1, 2):
            raise ValueError(
                f"--aux_levels must be 1 or 2 for l1_aux, got {args.aux_levels}"
            )
        if args.ds_factors is None:
            args.ds_factors = [8] if args.aux_levels == 1 else [8, 32]
        else:
            args.ds_factors = [int(d) for d in args.ds_factors]
        if len(args.ds_factors) != args.aux_levels:
            raise ValueError(
                f"--ds_factors length must equal --aux_levels={args.aux_levels}, "
                f"got {args.ds_factors}"
            )
        for d in args.ds_factors:
            if d < 1 or img_size % d != 0:
                raise ValueError(
                    f"img_size {img_size} must be divisible by ds_factor {d}"
                )
        args.num_of_frame = 2 + args.aux_levels
        args.train_unroll = 1
    elif args.ablation == "l1_unrolled":
        if args.train_unroll < 1:
            raise ValueError(f"--train_unroll must be >= 1, got {args.train_unroll}")
        args.aux_levels = 0
        args.ds_factors = []
        args.num_of_frame = args.train_unroll + 1
    else:
        raise ValueError(f"unknown ablation {args.ablation}")

    if not args.output_dir:
        args.output_dir = default_output_dir(
            args.ablation, args.aux_levels, args.train_unroll
        )
    if not args.log_dir:
        args.log_dir = args.output_dir
    return args


def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
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
    print(
        f"ablation={args.ablation} frames={args.num_of_frame} "
        f"unroll={args.train_unroll} aux_levels={args.aux_levels} "
        f"ds_factors={args.ds_factors}"
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    model = AblationEmulator(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    model_without_ddp.ema_step(0)

    params = list(model_without_ddp.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [], "lr": args.lr},
            {"params": params, "lr": args.lr, "lr_scale": 1.0},
        ],
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )
    loss_scaler = NativeScaler()

    ckpt_pth = (
        "checkpoint-last.pth"
        if args.resume_epoch < 1
        else f"checkpoint-{args.resume_epoch}.pth"
    )
    if args.resume and os.path.exists(os.path.join(args.resume, ckpt_pth)):
        checkpoint = torch.load(
            os.path.join(args.resume, ckpt_pth),
            map_location="cpu",
            weights_only=False,
        )
        model_without_ddp.load_state_dict(checkpoint["model"])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint["model_ema"]
        ema_params = [
            ema_state_dict[name].cuda()
            for name, _ in model_without_ddp.named_parameters()
        ]
        print("Resume checkpoint %s" % args.resume)
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    if args.evaluate:
        print("--evaluate is not wired for ablations; use eval/eval_ablation_mse.py")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataset_train.set_epoch(epoch)

        train_stats, _ = train_one_epoch(
            model,
            model_params,
            ema_params,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=None,
            args=args,
            dataset_validation=dataset_validation,
        )
        val_stats = validation_loss(model, dataset_validation, device, args)

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                ema_params=ema_params,
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
                ema_params=ema_params,
                epoch_name=f"{epoch}",
            )

        if misc.is_main_process():
            metrics_dir = os.path.join(args.output_dir, "csv")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, "train_metrics.csv")
            fieldnames = [
                "epoch",
                "lr",
                "loss",
                "high_loss",
                "grad_norm",
                "val_loss",
                "val_high_loss",
            ]
            for i in range(args.aux_levels):
                fieldnames.append(f"aux{i}_loss")
                fieldnames.append(f"val_aux{i}_loss")
            write_header = not os.path.exists(metrics_path)
            row = {
                "epoch": epoch,
                "lr": train_stats.get("lr", ""),
                "loss": train_stats.get("loss", ""),
                "high_loss": train_stats.get("high_loss", ""),
                "grad_norm": train_stats.get("grad_norm", ""),
                "val_loss": val_stats.get("val_loss", ""),
                "val_high_loss": val_stats.get("val_high_loss", ""),
            }
            for i in range(args.aux_levels):
                row[f"aux{i}_loss"] = train_stats.get(f"aux{i}_loss", "")
                row[f"val_aux{i}_loss"] = val_stats.get(f"val_aux{i}_loss", "")
            with open(metrics_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.ablation == "hine":
        print(HINE_HELP)
        sys.exit(2)
    args = finalize_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
