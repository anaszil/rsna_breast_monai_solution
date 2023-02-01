import os

import numpy as np
import torch
from monai.utils import set_determinism
from torch.utils.data import DataLoader, WeightedRandomSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    if ctp + cfp == 0:
        return 0
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def set_seed(seed):
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_dataloader(train_dataset, cfg):
    df = train_dataset.df.copy()
    df["weight"] = 1
    df.loc[df.cancer == 1, "weight"] = (
        len(df.loc[df.cancer == 0]) / len(df.loc[df.cancer == 1]) / cfg.alpha
    )
    # wrs = WeightedRandomSampler(
    #     weights=df.weight.tolist(), num_samples=len(df), replacement=True
    # )
    ewrs = ExhaustiveWeightedRandomSampler(df.weight.tolist(), num_samples=len(df))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=ewrs,
        shuffle=cfg.shuffle,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=True,
    )

    return train_dataloader


def get_val_dataloader(val_dataset, cfg):

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
    )

    return val_dataloader


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def pfbeta_torch(preds, labels, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0.0


def pfbeta_torch_thresh(preds, labels):
    optimized_preds = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels)


def optimize_preds(
    preds, labels=None, thresh=None, return_thresh=False, print_results=False
):
    preds = preds.clone()
    if labels is not None:
        without_thresh = pfbeta_torch(preds, labels)

    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 101)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f"without optimization: {without_thresh}")
        pfbeta = pfbeta_torch(preds, labels)
        print(f"with optimization: {pfbeta}")
        print(f"best_thresh = {thresh}")
    if return_thresh:
        return thresh
    return preds
