import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# import wandb
from network.kgnet import KneeGraphNetwork, KneeLoss
from network.dataloader import KGNetDataloader as Dataloader
from utils.Config import Config
from utils.Result_cls import Result
from utils.Result_ordinal import ResultOrdinal
from utils.Result_multilabel import ResultMultiLabel
from utils.utils_net import init_train, save_model
from utils.parser import args


def train():
    st = time.time()
    running_loss = 0.0
    lr = scheduler.get_last_lr()[0]
    net.train()
    for data in tqdm(dataloader["train"], ncols=60, desc="train", unit="b", leave=None):
        data.to(cfg.device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            preds = net(data)
            loss = lossfunc.cls(data, preds)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
    scheduler.step()
    ft = time.time()
    e_loss = running_loss / len(dataloader["train"])
    logging.info(f"\n\nEPOCH: {epoch}, TRAIN_LOSS : {e_loss:.3f}, TIME: {ft - st:.1f}s, LR: {lr:.2e}")
    # if not args.t:
    #     wandb.log({"train_loss": e_loss, "learning_rate": lr}, step=epoch)
    return e_loss


@torch.no_grad()
def eval(dataset_type, _result):
    _result.init()
    net.eval()
    for data in dataloader[dataset_type]:
        data.to(cfg.device)
        preds = net(data)
        if cfg.label_mode == "multi_label":
            # ResultMultiLabel.eval takes raw logits + binary float targets
            _result.eval(preds["cls"], data.multi_label)
        else:
            # Result / ResultOrdinal both accept the full preds dict + grade
            _result.eval(preds, data.grade)
    pars = _result.stastic()
    _result.print()
    return


if __name__ == "__main__":

    scaler = GradScaler()
    cfg = Config(args)
    dataloader = Dataloader(cfg)
    init_train(cfg)

    # Build network — passes label_mode and num_labels so DiagnosisHead is configured
    net = KneeGraphNetwork(
        num_cls=cfg.num_cls,
        pretrain_from_imagenet=False,
        label_mode=cfg.label_mode,
        num_labels=cfg.num_labels,
    )
    net.load_pretrain(cfg.ckpt)
    net = net.to(cfg.device)
    print(f"dataset={args.dataset}  label_mode={cfg.label_mode}")

    # Build loss — passes label_mode and num_cls for dispatch
    lossfunc = KneeLoss(
        cfg.device, args.dataset,
        label_mode=cfg.label_mode,
        num_cls=cfg.num_cls,
    )
    optimizer = optim.Adam(net.parameters(), cfg.lr, weight_decay=cfg.wd)
    scheduler = CosineAnnealingLR(optimizer, cfg.num_epoch, 1e-8)

    # Build result trackers appropriate for the selected label_mode
    if cfg.label_mode == "multi_label":
        result_valid = ResultMultiLabel(label_names=["abnormality", "acl"])
        result_test  = ResultMultiLabel(label_names=["abnormality", "acl"])
    elif cfg.label_mode in ("ordinal", "soft"):
        result_valid = ResultOrdinal(cfg, "valid")
        result_test  = ResultOrdinal(cfg, "test")
    else:
        # label_mode == 'single'  — original behaviour unchanged
        result_valid = Result(cfg, "valid")
        result_test  = Result(cfg, "test")
    

    net.forward = net.finetune

    for epoch in range(cfg.num_epoch):
        train()
        eval('valid', result_valid)
        eval('test', result_test)
        save_model(result_valid, net, cfg)
    # wandb.finish()
