#!/usr/bin/env python3
"""Train / eval FutureINNPlus (L>=2 via latent_res or hierarchy_L/ds_factors).

Does not modify main.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from dataloader import DistributedPreloadDataset
from engine_plus import eval_model_rec, train_one_epoch, validation_loss
from model_plus import FutureINNPlus, default_inject_res, validate_latent_args
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("FutureINNPlus", add_help=False)

    parser.add_argument(
        "--latent_res",
        nargs="+",
        type=int,
        default=None,
        help="K coarse grid sizes, finer to coarser, e.g. 32 for L=2 or 32 8 for L=3",
    )
    parser.add_argument(
        "--hierarchy_L",
        type=int,
        default=None,
        help="Hierarchy depth L = K+1. Use with --ds_factors (length K = L-1).",
    )
    parser.add_argument(
        "--ds_factors",
        nargs="+",
        type=int,
        default=None,
        help="Downsample factors for each coarse, e.g. 8 for L=2 or 8 32 for L=3",
    )
    parser.add_argument(
        "--inject_res",
        nargs="+",
        type=int,
        default=None,
        help="Optional encoder inject maps (same length as latent_res, descending)",
    )
    parser.add_argument("--ratio", default=0.85, type=float)
    parser.add_argument("--img_size", default=256, type=int)
    # Set automatically from latent_res; kept for checkpoint/args compatibility.
    parser.add_argument("--num_of_frame", type=int, default=-1)
    # Legacy aliases stored for eval scripts that peek at ckpt args.
    parser.add_argument("--ds_factor", default=-1, type=int)
    parser.add_argument("--ds_factor_low", default=-1, type=int)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument("--save_last_freq", type=int, default=50)
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--acc_grad", type=int, default=1)
    parser.add_argument("--ema_rate", default=0.9995, type=float)
    parser.add_argument("--eval_idx", type=int, default=0)

    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--output_dir", default="./output/output_dir_plus")
    parser.add_argument("--log_dir", default="./output/output_dir_plus")
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


def finalize_args(args):
    img_size = int(args.img_size)
    has_latent = args.latent_res is not None and len(args.latent_res) > 0
    has_hier = args.hierarchy_L is not None or args.ds_factors is not None

    if not has_latent and not has_hier:
        raise ValueError(
            "Provide --latent_res ... or both --hierarchy_L and --ds_factors"
        )

    latent_from_ds = None
    if has_hier:
        if args.hierarchy_L is None or args.ds_factors is None:
            raise ValueError(
                "--hierarchy_L and --ds_factors must be set together "
                "(or use --latent_res alone)"
            )
        hierarchy_L = int(args.hierarchy_L)
        ds_factors = [int(d) for d in args.ds_factors]
        if hierarchy_L < 2:
            raise ValueError(f"--hierarchy_L must be >= 2, got {hierarchy_L}")
        if len(ds_factors) != hierarchy_L - 1:
            raise ValueError(
                f"--ds_factors length must be hierarchy_L-1={hierarchy_L - 1}, "
                f"got {len(ds_factors)}"
            )
        for d in ds_factors:
            if d < 1 or img_size % d != 0:
                raise ValueError(
                    f"img_size {img_size} must be divisible by ds_factor {d}"
                )
        latent_from_ds = [img_size // d for d in ds_factors]

    if has_latent:
        latent_res = [int(r) for r in args.latent_res]
        if latent_from_ds is not None and latent_res != latent_from_ds:
            raise ValueError(
                f"--latent_res {latent_res} disagrees with "
                f"--hierarchy_L/--ds_factors -> {latent_from_ds}"
            )
        if args.hierarchy_L is not None and int(args.hierarchy_L) != len(latent_res) + 1:
            raise ValueError(
                f"--hierarchy_L {args.hierarchy_L} disagrees with "
                f"len(latent_res)+1={len(latent_res) + 1}"
            )
    else:
        latent_res = latent_from_ds

    if args.inject_res is None:
        inject_res = default_inject_res(latent_res)
    else:
        inject_res = [int(r) for r in args.inject_res]
    validate_latent_args(img_size, latent_res, inject_res)
    args.latent_res = latent_res
    args.inject_res = inject_res
    k = len(latent_res)
    args.num_of_frame = k + 2
    args.ds_factor = img_size // latent_res[0]
    args.ds_factor_low = img_size // latent_res[-1]
    args.num_coarse = k
    args.hierarchy_L = k + 1
    args.ds_factors = [img_size // r for r in latent_res]
    return args


def main(args):
    args = finalize_args(args)
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
    dataset_test = DistributedPreloadDataset(
        args.num_of_frame, args.img_size, eval=True
    )
    dataset_train[0]
    sampler_train = dataset_train
    print("train dataset", len(dataset_train))
    print("validation dataset", len(dataset_validation))
    print("test dataset", len(dataset_test))
    print(
        f"L={args.hierarchy_L} K={args.num_coarse} "
        f"latent_res={args.latent_res} inject_res={args.inject_res} "
        f"num_of_frame={args.num_of_frame}"
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

    model = FutureINNPlus(args)
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

    if args.gpu == 0 and args.evaluate:
        import pandas as pd
        from tqdm import tqdm

        def save_folder(prefix, args_):
            pth = "/".join(os.getcwd().split("/")[:-1]) + f"/{prefix[1]}"
            os.makedirs(pth, exist_ok=True)
            tag = "_".join(str(r) for r in args_.latent_res)
            pth = pth + f"/{prefix[0]}_{tag}_{args_.start_epoch}"
            os.makedirs(pth, exist_ok=True)
            return pth

        save_pth = save_folder([f"ours_level{args.hierarchy_L}", "mse"], args)
        save_data_pth = save_folder([f"ours_level{args.hierarchy_L}", "data"], args)

        mse_list, mse_save = [], []
        for sidx in tqdm(range(0, 2500, 25)):
            model.eval()
            data = eval_model_rec(
                model,
                dataset_test,
                args.start_epoch,
                btz=1,
                sidx=sidx,
                output_dir=args.output_dir,
                tms=200,
            )
            mse_list.append(data["mse"])
            torch.save(data, save_data_pth + f"/{sidx}.pth")

        mse_array = torch.stack(mse_list, dim=0)
        mse_mean = mse_array.mean(dim=0)
        mse_std = mse_array.std(dim=0)
        for i in (0, 5, 10, 20, 30, 40, 50, -1):
            print(i, f"mse mean: {mse_mean[i]:.5f}, mse std: {mse_std[i]:.5f}")
            mse_save.append(
                [i, mse_mean[i].cpu().data.numpy(), mse_std[i].cpu().data.numpy()]
            )
        df = pd.DataFrame(
            dict(
                time=np.stack(mse_save)[:, 0],
                mse=np.stack(mse_save)[:, 1],
                mse_std=np.stack(mse_save)[:, 2],
            )
        )
        df.to_csv(
            f"{args.output_dir}/csv/eval_mse_{args.start_epoch:04d}.csv",
            index=False,
            float_format="%.5f",
        )
        df.to_csv(save_pth + ".csv", index=False, float_format="%.5f")

    if not args.evaluate:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

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
                for i in range(args.num_coarse):
                    fieldnames.append(f"coarse{i}_loss")
                    fieldnames.append(f"val_coarse{i}_loss")
                fieldnames += ["mid_loss", "low_loss", "val_mid_loss", "val_low_loss"]
                write_header = not os.path.exists(metrics_path)
                row = {
                    "epoch": epoch,
                    "lr": train_stats.get("lr", ""),
                    "loss": train_stats.get("loss", ""),
                    "high_loss": train_stats.get("high_loss", ""),
                    "grad_norm": train_stats.get("grad_norm", ""),
                    "val_loss": val_stats.get("val_loss", ""),
                    "val_high_loss": val_stats.get("val_high_loss", ""),
                    "mid_loss": train_stats.get("mid_loss", ""),
                    "low_loss": train_stats.get("low_loss", ""),
                    "val_mid_loss": val_stats.get("val_mid_loss", ""),
                    "val_low_loss": val_stats.get("val_low_loss", ""),
                }
                for i in range(args.num_coarse):
                    row[f"coarse{i}_loss"] = train_stats.get(f"coarse{i}_loss", "")
                    row[f"val_coarse{i}_loss"] = val_stats.get(
                        f"val_coarse{i}_loss", ""
                    )
                with open(metrics_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

        total_time = time.time() - start_time
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
