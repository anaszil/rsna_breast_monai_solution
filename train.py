import argparse
import gc
import importlib
import os
import sys
import shutil
import wandb


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *


from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    Activationsd,
    AsDiscreted,
    LoadImage,
)
import json

from dataset import CustomDataset
from sklearn.metrics import roc_auc_score
import timm
import nextvit


def main(cfg, logger=None, track_wandb=False):

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)
    set_seed(cfg.seed)

    # set dataset, dataloader
    if (
        hasattr(cfg, "sweep_dataset_size")
        and hasattr(cfg, "do_sweep")
        and cfg.do_sweep == True
    ):
        df_ = pd.read_csv(cfg.data_df)
        # sample only sweep_dataset_size % of the whole dataframe
        df = stratified_sample(df_, cfg.sweep_dataset_size)
        print(f"Using {cfg.sweep_dataset_size*100}% of train set for sweeps")
    else:
        df = pd.read_csv(cfg.data_df)

    if hasattr(cfg, "run_val_whole_data") and cfg.run_val_whole_data == True:
        val_df = df
        train_df = df[df["fold"] == cfg.fold]  # not used !!!
    else:
        val_df = df[df["fold"] == cfg.fold]
        train_df = df[df["fold"] != cfg.fold]

    train_dataset = CustomDataset(df=train_df, cfg=cfg, aug=cfg.train_transforms)
    val_dataset = CustomDataset(df=val_df, cfg=cfg, aug=cfg.val_transforms)
    print("train: ", len(train_dataset), " val: ", len(val_dataset))
    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    # set model
    if "nextvit" in cfg.backbone:
        model = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
        )
    else:
        model = timm.create_model(
            cfg.backbone,
            in_chans=cfg.in_channels,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
        )

    # model = EfnNet()
    model = torch.nn.DataParallel(model)
    model.to(cfg.device)

    # set optimizer, lr scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=int(len(train_dataset) / cfg.batch_size),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=cfg.lr_div,
        final_div_factor=cfg.lr_final_div,
    )
    if cfg.weights is not None:

        state_dict = torch.load(
            os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights),
        )["model"]
        state_dict = {k.replace("model", "module"): v for k, v in state_dict.items()}
        for k in state_dict.keys():
            if "nn_cancer.0" in k:
                state_dict[
                    k.replace("nn_cancer.0", "module.classifier")
                ] = state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if hasattr(cfg, "load_spec") and "optimizer" in cfg.load_spec:
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights)
                )["optimizer"]
            )
        if hasattr(cfg, "load_spec") and "scheduler" in cfg.load_spec:
            scheduler.load_state_dict(
                torch.load(
                    os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights)
                )["scheduler"]
            )
        print(f"weights from: {cfg.weights} are loaded.")

    # set loss
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor([cfg.pos_weight])
    ).to(cfg.device)

    if track_wandb == "all":
        wandb.watch(model, loss_function, log="all", log_freq=1000)  # WANDB WATCH

    # set other tools
    scaler = GradScaler()

    # train and val loop
    step = 0
    i = 0
    best_metric = 0.0
    optimizer.zero_grad()
    print("start from: ", best_metric)
    for epoch in range(cfg.epochs):
        print("EPOCH:", epoch)
        gc.collect()

        if not (hasattr(cfg, "run_train") and cfg.run_train == False):
            run_train(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                scaler=scaler,
                epoch=epoch,
                iteration=i,
                step=step,
                loss_function=loss_function,
                logger=logger,
            )

        val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            cfg=cfg,
            epoch=epoch,
            logger=logger,
        )

        checkpoint = create_checkpoint(
            model,
            optimizer,
            epoch,
            scheduler=scheduler,
            scaler=scaler,
        )

        torch.save(
            checkpoint,
            f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_{epoch}.pth",
        )

        if val_metric > best_metric:
            print(f"SAVING CHECKPOINT: val_metric {best_metric:.5} -> {val_metric:.5}")
            best_metric = val_metric
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_best_metric.pth",
            )


def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    cfg,
    scaler,
    epoch,
    iteration,
    step,
    loss_function,
    logger,
):
    model.train()
    losses = []
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)

    all_outputs, all_labels = [], []

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["image"].to(cfg.device), batch["label"].float().to(
            cfg.device
        )
        iteration += 1

        step += cfg.batch_size
        torch.set_grad_enabled(True)
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        losses.append(loss.item())

        outputs = list(torch.sigmoid(outputs).detach().cpu().numpy())
        labels = list(labels.detach().cpu().numpy())

        all_outputs.extend(outputs)
        all_labels.extend(labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        progress_bar.set_description(
            f"loss: {np.mean(losses):.2f} lr: {scheduler.get_last_lr()[0]:.6f}"
        )
    score = pfbeta(all_labels, all_outputs, 1.0)
    auc = roc_auc_score(all_labels, all_outputs)
    print("Train pF1: ", score, "AUC: ", auc)

    if logger is None:
        wandb.log(
            {"Loss": np.mean(losses), "Train pF1": score, "Train AUC": auc}, step=epoch
        )
    else:
        logger.log(
            {"Loss": np.mean(losses), "Train pF1": score, "Train AUC": auc}, step=epoch
        )


def run_eval(model, val_dataloader, cfg, epoch, logger):

    model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(range(len(val_dataloader)))
    tr_it = iter(val_dataloader)

    all_labels = []
    all_outputs = []
    all_ids = []
    all_imgs = []

    for itr in progress_bar:
        batch = next(tr_it)
        inputs, labels = batch["image"].to(cfg.device), batch["label"].float().to(
            cfg.device
        )
        ids = batch["prediction_id"]
        path_imgs = batch["image"]
        outputs = model(inputs)
        outputs = list(torch.sigmoid(outputs).detach().cpu().numpy()[:, 0])
        labels = list(labels.detach().cpu().numpy()[:, 0])

        all_outputs.extend(outputs)
        all_labels.extend(labels)
        all_ids.extend(ids)
        all_imgs.extend(path_imgs)

    df_pred = pd.DataFrame.from_dict(all_ids)
    df_pred.columns = ["prediction_id"]
    df_pred["path_imgs"] = path_imgs
    df_pred["all_labels"] = all_labels
    df_pred["all_outputs"] = all_outputs

    if hasattr(cfg, "run_val_whole_data") and cfg.run_val_whole_data == True:
        print("saving results of validation : ")
        df_pred.to_csv("results.csv")

    df_pred = df_pred.groupby(["prediction_id"]).mean().reset_index()
    all_labels = df_pred["all_labels"]
    all_outputs = df_pred["all_outputs"]

    score = pfbeta(all_labels, all_outputs, 1.0)
    auc = roc_auc_score(all_labels, all_outputs)

    thresh = optimize_preds(
        torch.tensor(all_outputs.values),
        torch.tensor(all_labels.values),
        return_thresh=True,
        print_results=True,
    )

    all_outputs = (np.array(all_outputs) > thresh).astype(np.int8).tolist()
    try:
        bin_score = pfbeta(all_labels, all_outputs, 1.0)
    except:
        bin_score = 0.0
    print(
        "Val_pF1: ",
        score,
        "thresh: ",
        thresh,
        "val pF1-thresh: ",
        bin_score,
        "AUC: ",
        auc,
    )
    nested_metrics = {
        "Val_pF1": score,
        "thresh": thresh,
        "val pF1-thresh": bin_score,
        "Val AUC": auc,
    }

    if logger is None:
        wandb.log(nested_metrics, step=epoch)
        # the metric to be optimized needs to be logged at top-level for Wandb:
        # wandb.log({"Val_pF1", nested_metrics["Val_pF1"]}, step=epoch)
    else:
        logger.log(nested_metrics, step=epoch)
        # the metric to be optimized needs to be logged at top-level for Wandb:
        # logger.log({"Val_pF1", nested_metrics["Val_pF1"]}, step=epoch)

    return score


if __name__ == "__main__":

    sys.path.append("configs")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-c", "--config", default="cfg_clf_baseline", help="config filename"
    )
    parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
    parser.add_argument(
        "-backbone", "--backbone", default="tf_efficientnetv2_s", help="backbone"
    )
    parser.add_argument("-s", "--seed", type=int, default=20220421, help="seed")
    parser.add_argument("-w", "--weights", default=None, help="the path of weights")

    parser_args, _ = parser.parse_known_args(sys.argv)

    cfg = importlib.import_module(parser_args.config).cfg
    cfg.fold = parser_args.fold
    cfg.seed = parser_args.seed
    cfg.weights = parser_args.weights
    cfg.backbone = parser_args.backbone

    main(cfg)
