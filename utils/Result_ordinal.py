"""
Ordinal-aware evaluation metrics for KGNet fine-tuning.

Extends Result_cls.py by adding:
  - MAE  (mean absolute error between predicted ordinal class and ground truth)
  - QWK  (quadratic weighted kappa — penalises large ordinal disagreements more)

These are appropriate for ordinal (CORN) and soft-label modes where the
class ordering Normal < PTCD < FTCD is clinically meaningful.

For label_mode='single' these metrics can still be computed as a bonus
comparison to the standard ACC/AUC reported in the original paper.
"""
import logging
import time

import numpy as np
import torch
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    roc_auc_score,
)

from utils.Config import Config


def get_one_hot(label: np.ndarray, num_cls: int) -> np.ndarray:
    label = label.reshape(-1)
    return np.eye(num_cls)[label]


class ResultOrdinal:
    """
    Drop-in companion to Result_cls.Result for ordinal label modes.

    Tracks standard classification metrics (ACC, AUC …) AND ordinal-specific
    metrics (MAE, QWK).
    """

    def __init__(self, cfg: Config, dtype: str = "test") -> None:
        self.cfg        = cfg
        self.type       = dtype
        self.epoch      = 0
        self.best_epoch = 0
        self.best_result = 0.0   # tracked by QWK
        self.save_path  = cfg.log_dir

    def init(self) -> None:
        self.st    = time.time()
        self.preds = []   # raw logits (before softmax / corn_label_to_probas)
        self.trues = []

    def eval(self, pred: dict, true: torch.Tensor) -> None:
        """
        Args:
            pred: dict with key "cls" → Tensor [B, num_cls] or [B, num_cls-1]
            true: Tensor [B] int64 ground-truth class labels
        """
        self.preds.append(pred["cls"])
        self.trues.append(true)

    def stastic(self) -> list:
        from utils.ordinal import corn_label_to_probas

        logits = torch.cat(self.preds, dim=0)   # [N, K] or [N, K-1]
        true   = torch.cat(self.trues, dim=0).cpu().numpy()

        # Determine mode by output shape
        if logits.shape[1] == self.cfg.num_cls - 1:
            # CORN mode: convert to class probabilities
            probs = corn_label_to_probas(logits).cpu().detach().numpy()
        else:
            # soft / single mode
            probs = torch.softmax(logits, dim=1).cpu().detach().numpy()

        preds_cls     = np.argmax(probs, axis=1)
        true_one_hot  = get_one_hot(true, self.cfg.num_cls)

        self.acc  = accuracy_score(true, preds_cls)
        self.rec  = sensitivity_score(true, preds_cls, average="macro")
        self.spe  = specificity_score(true, preds_cls, average="macro")
        self.pre  = precision_score(true, preds_cls, average="macro", zero_division=0)
        self.f1   = f1_score(true, preds_cls, average="macro")
        self.auc  = roc_auc_score(true_one_hot, probs, average="macro")
        self.cm   = confusion_matrix(true, preds_cls)
        self.mae  = mean_absolute_error(true, preds_cls)
        self.qwk  = cohen_kappa_score(true, preds_cls, weights="quadratic")
        self.time = round(time.time() - self.st, 1)

        self.pars = [self.acc, self.rec, self.spe, self.pre, self.f1, self.auc,
                     self.mae, self.qwk]
        self.pars = [round(float(p), 4) for p in self.pars]
        return self.pars

    def print(self) -> None:
        self.epoch += 1
        titles = ["dataset", "ACC", "REC", "SPE", "PRE", "F1S", "AUC", "MAE", "QWK"]
        items  = [self.type.upper()] + self.pars
        fmt1   = "\n|{:^8}" + "|{:^6}" * (len(titles) - 1) + "|"
        fmt2   = "\n|{:^8}" + "|{:^.4f}" * (len(titles) - 1) + "|"
        logging.info(f"{self.type.upper()} QWK: {self.qwk:.4f}, TIME: {self.time}s")
        logging.info((fmt1 + fmt2).format(*titles, *items))
        logging.debug(f"\n{self.cm}")
        if self.qwk > self.best_result:
            self.best_epoch  = self.epoch
            self.best_result = self.qwk
