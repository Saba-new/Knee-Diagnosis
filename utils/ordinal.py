"""
CORN ordinal regression loss.
Reference: Shi et al. (2021) "Deep Neural Networks for Rank Consistent Ordinal
           Regression based on Conditional Probabilities", arXiv:2111.08851
"""
import torch
import torch.nn.functional as F


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Conditioned Ordinal Regression with Neural networks (CORN).

    Args:
        logits:      Tensor [B, num_classes - 1]  raw outputs (before sigmoid)
        labels:      Tensor [B]                   integer class labels in {0, ..., num_classes-1}
        num_classes: int

    Returns:
        Scalar loss tensor (mean over all conditioning sets)
    """
    sets = []
    for i in range(num_classes - 1):
        # Conditioning set: samples whose true label >= i  (i.e., label > i-1)
        label_mask = labels > (i - 1)
        logit_i = logits[label_mask, i]
        y_i = (labels[label_mask] > i).float()
        sets.append((logit_i, y_i))

    num_samples = 0
    loss = torch.zeros(1, device=logits.device)
    for logit_i, y_i in sets:
        train_examples = y_i.size(0)
        if train_examples == 0:
            continue
        loss += F.binary_cross_entropy_with_logits(
            logit_i, y_i, reduction="sum"
        )
        num_samples += train_examples

    return loss / max(num_samples, 1)


def corn_label_to_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits to class probabilities.

    P(Y = k) = P(Y > k-1) - P(Y > k)

    Args:
        logits: Tensor [B, num_classes - 1]

    Returns:
        probas: Tensor [B, num_classes], each row sums to 1, all values >= 0
    """
    cumprobs = torch.sigmoid(logits)           # P(Y > k) for k = 0 .. K-2
    # P(Y > -1) = 1 (always true), P(Y > K-1) = 0 (never true)
    ones  = torch.ones(logits.size(0),  1, device=logits.device)
    zeros = torch.zeros(logits.size(0), 1, device=logits.device)
    cumprobs = torch.cat([ones, cumprobs, zeros], dim=1)   # [B, K+1]
    probas   = cumprobs[:, :-1] - cumprobs[:, 1:]          # [B, K]
    # CORN sigmoid outputs are independent and not guaranteed monotone at
    # random init, so differences can be negative. Clamp then renormalize
    # to maintain a valid probability distribution (sums to 1, all >= 0).
    probas = torch.clamp(probas, min=0.0)
    probas = probas / (probas.sum(dim=1, keepdim=True) + 1e-8)
    return probas


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Argmax over CORN class probabilities → predicted class index.

    Args:
        logits: Tensor [B, num_classes - 1]

    Returns:
        preds: Tensor [B] int64
    """
    probas = corn_label_to_probas(logits)
    return torch.argmax(probas, dim=1)
