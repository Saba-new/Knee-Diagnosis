"""
Evaluation metrics for multi-label classification (MRNet joint ACL + Abnormality).

Kept separate from Result_cls.py to preserve the original single-label pipeline
and allow independent evolution of each metric class.

Usage:
    result = ResultMultiLabel(label_names=["abnormality", "acl"], threshold=0.5)
    result.init()
    for data in dataloader:
        preds = net(data)
        result.eval(preds["cls"], data.multi_label)
    metrics = result.stastic()
    result.print(metrics)
"""
import logging
import time

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ResultMultiLabel:
    def __init__(self, label_names: list, threshold: float = 0.5) -> None:
        """
        Args:
            label_names: list of str, e.g. ["abnormality", "acl"]
            threshold:   float, sigmoid threshold for binary prediction (default 0.5).
                         IMPORTANT: tune this on the validation split only.
        """
        self.label_names = label_names
        self.num_labels = len(label_names)
        self.threshold = threshold
        self.all_probs: list = []
        self.all_targets: list = []
        self.best_result = 0.0
        self.best_epoch = 0
        self.epoch = 0

    def init(self) -> None:
        """Reset per-epoch accumulators."""
        self.st = time.time()
        self.all_probs = []
        self.all_targets = []

    def eval(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate one batch of predictions.

        Args:
            logits:  Tensor [B, num_labels]  raw model outputs (before sigmoid)
            targets: Tensor [B, num_labels]  float binary ground-truth labels
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
        self.all_probs.append(probs)
        self.all_targets.append(targets.cpu().numpy())

    def stastic(self) -> dict:
        """
        Compute all metrics over accumulated predictions.

        Returns:
            dict of metric_name → float value
        """
        probs   = np.concatenate(self.all_probs,   axis=0)   # [N, L]
        targets = np.concatenate(self.all_targets, axis=0)   # [N, L]
        preds   = (probs >= self.threshold).astype(int)
        targets_int = targets.astype(int)

        metrics = {}
        for j, name in enumerate(self.label_names):
            p  = probs[:, j]
            t  = targets_int[:, j]
            pr = preds[:, j]
            try:
                metrics[f"{name}_auc"] = roc_auc_score(t, p)
            except ValueError:
                metrics[f"{name}_auc"] = float("nan")
            metrics[f"{name}_f1"]        = f1_score(t, pr, zero_division=0)
            metrics[f"{name}_precision"] = precision_score(t, pr, zero_division=0)
            metrics[f"{name}_recall"]    = recall_score(t, pr, zero_division=0)

        # Micro / macro aggregates
        metrics["micro_f1"]  = f1_score(
            targets_int.ravel(), preds.ravel(), average="micro", zero_division=0
        )
        metrics["macro_f1"]  = f1_score(
            targets_int, preds, average="macro", zero_division=0
        )
        metrics["hamming"]   = float(np.mean(targets_int != preds))
        metrics["time"]      = round(time.time() - self.st, 1)

        self.metrics = metrics
        return metrics

    def print(self, metrics: dict = None) -> None:
        """Log all metrics. Updates best epoch tracking based on macro_f1."""
        self.epoch += 1
        if metrics is None:
            metrics = self.metrics
        logging.info("\n[MULTI-LABEL EVAL]")
        for k, v in metrics.items():
            if k == "time":
                logging.info(f"  TIME: {v}s")
            else:
                logging.info(f"  {k}: {v:.4f}")
        macro_f1 = metrics.get("macro_f1", 0.0)
        if macro_f1 > self.best_result:
            self.best_result = macro_f1
            self.best_epoch  = self.epoch
        logging.info(f"  best_macro_f1: {self.best_result:.4f} (epoch {self.best_epoch})")
