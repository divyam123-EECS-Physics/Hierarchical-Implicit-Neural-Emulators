"""Training / validation / AR eval for FutureINNPlus. Does not modify engine_mar.py."""

from __future__ import annotations

import math
import os
import sys
from typing import Iterable

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm

import util.lr_sched as lr_sched
import util.misc as misc

F = torch.nn.functional


def gather_imgs(dataset, sidx, eidx):
    return torch.stack([dataset.images[i] for i in range(sidx, eidx)])


@torch.no_grad()
def validation_loss(model, dataset_validation, device, args):
    model.eval()
    net = model.module if hasattr(model, "module") else model
    k = net.num_coarse
    loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    sum_loss = torch.zeros(2 + k, device=device)
    n_samples = torch.zeros(1, device=device)
    for samples in loader:
        x = samples.to(device).float()
        b = x.shape[0]
        with torch.cuda.amp.autocast():
            loss, high_loss, coarse_losses = net(x)
        sum_loss[0] += float(loss) * b
        sum_loss[1] += float(high_loss) * b
        for i, cl in enumerate(coarse_losses):
            sum_loss[2 + i] += float(cl) * b
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
    for i in range(k):
        out[f"val_coarse{i}_loss"] = (sum_loss[2 + i] / n).item()
    # aliases for K>=2 CSV compatibility
    if k >= 1:
        out["val_mid_loss"] = out["val_coarse0_loss"]
    if k >= 2:
        out["val_low_loss"] = out[f"val_coarse{k-1}_loss"]
    return out


def eval_model_rec(
    model,
    data_loader,
    epoch=5000,
    device=0,
    sidx=5,
    tms=20,
    btz=1,
    output_dir="output/output_dir",
    prefix="",
    test=False,
    save_folder="data_long",
):
    num_of_frame = 1
    if tms < 2000:
        test_img = gather_imgs(data_loader, sidx, sidx + tms * num_of_frame + 1).float().to(
            device
        )
    else:
        test_img = gather_imgs(data_loader, sidx, sidx + 1000).float().to(device)
    x_min = test_img.min()
    x_max = test_img.max()

    def video_sampling(init_cond, times, x_min=None, x_max=None):
        x_min, x_max = (
            (init_cond.min(), init_cond.max())
            if x_min is None and x_max is None
            else (x_min, x_max)
        )
        sample_img = [(init_cond[:, None, :, :] - x_min) / (x_max - x_min)]
        pred = init_cond[:, None]
        coarses = None
        with torch.cuda.amp.autocast():
            for t in tqdm(range(times + 1)):
                pred, coarses = model.module.sample(pred, coarses)
                if tms >= 10000:
                    if t % 100 == 0:
                        torch.save(
                            {"pred": pred, "coarses": coarses},
                            f"{save_folder}/{t:07d}.pth",
                        )
                else:
                    if t > 0:
                        sample_img.append((pred[:, [0]] - x_min) / (x_max - x_min))
        if tms < 10000:
            return torch.cat(sample_img, dim=1)

    def eval_mse(predict, test_img):
        mse = ((predict - test_img[None, :, :]) ** 2).reshape(
            predict.shape[0], predict.shape[1], -1
        ).mean(dim=2)
        return mse.mean(dim=0)[1:], mse.std(dim=0)[1:]

    def plot(predict_img, extra_prefix=""):
        diff = predict_img - test_img[None, :, :]
        predict_img = (
            torch.stack([predict_img, diff], dim=1)
            .flatten(1, 2)
            .reshape(predict_img.size(0) * 2, *predict_img.size()[1:])
        )
        output = torch.cat([test_img[None, :, :], predict_img], dim=0)
        output = torch.tensor(
            mpl.colormaps["bwr"](output.flatten(0, 1).cpu().data.numpy())[:, :, :, :3]
        ).permute(0, 3, 1, 2)
        save_image(
            output,
            f"{output_dir}/plots/{epoch:04d}_{sidx}{prefix}_{extra_prefix}.png",
            nrow=tms * num_of_frame + 1,
            normalize=True,
        )

    predict_img = video_sampling(
        test_img[0].repeat(btz, 1, 1), tms, x_min=x_min, x_max=x_max
    )

    if tms < 2000:
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        test_img = (test_img - x_min) / (x_max - x_min)
        mse, mse_std = eval_mse(predict_img, test_img)
        df = pd.DataFrame(
            dict(
                time=list(np.arange(tms * num_of_frame)),
                mse=mse.cpu().data.numpy(),
                mse_std=mse_std.cpu().data.numpy(),
            )
        )
        df.to_csv(
            f"{output_dir}/csv/mse_{epoch:04d}_{sidx}_{prefix}.csv",
            index=False,
            float_format="%.5f",
        )
        plot(predict_img)
        return {
            "test_img": test_img,
            "predict_img": predict_img,
            "mse": mse,
            "min_max": [x_min, x_max],
        }


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
    k = net.num_coarse

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / len(data_loader) + epoch, args
        )
        x = samples.to(args.device).float()

        with torch.cuda.amp.autocast():
            loss, high_loss, coarse_losses = model(x)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(high_loss=high_loss.item())
        for i, cl in enumerate(coarse_losses):
            metric_logger.update(**{f"coarse{i}_loss": cl.item()})
        if k >= 1:
            metric_logger.update(mid_loss=coarse_losses[0].item())
        if k >= 2:
            metric_logger.update(low_loss=coarse_losses[-1].item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        grad_norm = loss_scaler(
            3 * loss,
            optimizer,
            clip_grad=None,
            parameters=model.parameters(),
            update_grad=True,
        )
        optimizer.zero_grad()
        torch.cuda.synchronize()
        model.module.ema_step(args.ema_rate)
        metric_logger.update(grad_norm=grad_norm.mean())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if epoch % (args.eval_freq) == 0 and args.gpu == 0 and epoch > 0:
        model.eval()
        eval_model_rec(
            model,
            data_loader.dataset,
            epoch,
            x.device,
            0,
            prefix="train",
            output_dir=args.output_dir,
        )
        eval_model_rec(
            model,
            data_loader.dataset,
            epoch,
            x.device,
            100,
            prefix="train",
            output_dir=args.output_dir,
        )
        eval_model_rec(
            model, dataset_validation, epoch, x.device, 0, output_dir=args.output_dir
        )
        eval_model_rec(
            model, dataset_validation, epoch, x.device, 100, output_dir=args.output_dir
        )
        model.train()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k_: meter.global_avg for k_, meter in metric_logger.meters.items()}, x[:, [0]]
