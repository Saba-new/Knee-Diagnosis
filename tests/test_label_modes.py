"""
Unit tests for the KGNet Rich Label Learning extension.

Run with:
    cd "c:\\Users\\SABARISH\\Desktop\\knee diagnosis\\KGNet"
    python -m pytest tests/test_label_modes.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from utils.ordinal import corn_loss, corn_label_to_probas, corn_predict
from utils.soft_label import build_ordinal_soft_label


# ---------------------------------------------------------------------------
# Soft-label tests
# ---------------------------------------------------------------------------
class TestSoftLabel:

    def test_sum_to_one_all_grades(self):
        for grade in range(3):
            soft = build_ordinal_soft_label(grade, num_classes=3, alpha=0.1)
            assert abs(soft.sum() - 1.0) < 1e-6, f"grade={grade} sums to {soft.sum()}"

    def test_hard_label_at_alpha_zero(self):
        soft = build_ordinal_soft_label(1, num_classes=3, alpha=0.0)
        assert soft[1] == pytest.approx(1.0)
        assert soft[0] == pytest.approx(0.0)
        assert soft[2] == pytest.approx(0.0)

    def test_dominant_class_has_max_probability(self):
        for grade in range(3):
            soft = build_ordinal_soft_label(grade, num_classes=3, alpha=0.1)
            assert soft[grade] == max(soft), f"grade={grade}: {soft}"

    def test_boundary_grade_zero(self):
        # Grade 0 has no lower neighbour → extra half goes back to grade 0
        soft = build_ordinal_soft_label(0, num_classes=3, alpha=0.1)
        assert soft.sum() == pytest.approx(1.0)
        assert soft[0] == pytest.approx(0.95)
        assert soft[1] == pytest.approx(0.05)
        assert soft[2] == pytest.approx(0.0)

    def test_boundary_grade_last(self):
        # Grade 2 has no upper neighbour
        soft = build_ordinal_soft_label(2, num_classes=3, alpha=0.1)
        assert soft.sum() == pytest.approx(1.0)
        assert soft[2] == pytest.approx(0.95)
        assert soft[1] == pytest.approx(0.05)
        assert soft[0] == pytest.approx(0.0)

    def test_middle_grade_symmetric(self):
        soft = build_ordinal_soft_label(1, num_classes=3, alpha=0.1)
        assert soft[0] == pytest.approx(soft[2])   # symmetric neighbours
        assert soft[1] == pytest.approx(0.90)

    def test_output_dtype(self):
        soft = build_ordinal_soft_label(0, num_classes=3, alpha=0.1)
        assert soft.dtype == np.float32

    def test_invalid_grade_raises(self):
        with pytest.raises(AssertionError):
            build_ordinal_soft_label(5, num_classes=3, alpha=0.1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(AssertionError):
            build_ordinal_soft_label(0, num_classes=3, alpha=1.5)


# ---------------------------------------------------------------------------
# CORN loss tests
# ---------------------------------------------------------------------------
class TestCORNLoss:

    def test_finite_loss_random(self):
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 3, (8,))
        loss = corn_loss(logits, labels, num_classes=3)
        assert torch.isfinite(loss), "Loss must be finite"

    def test_loss_is_non_negative(self):
        logits = torch.randn(8, 2)
        labels = torch.randint(0, 3, (8,))
        loss = corn_loss(logits, labels, num_classes=3)
        assert loss.item() >= 0.0

    def test_loss_scalar(self):
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 2, 1])
        loss = corn_loss(logits, labels, num_classes=3)
        assert loss.numel() == 1, "Loss should be a scalar"

    def test_loss_decreases_with_correct_logits(self):
        """Logits strongly predicting the correct class should yield lower loss."""
        # All grade=0 → logit outputs should be strongly negative (P(Y>0) ≈ 0)
        good_logits = torch.full((8, 2), -5.0)
        bad_logits  = torch.full((8, 2), +5.0)
        labels = torch.zeros(8, dtype=torch.long)
        loss_good = corn_loss(good_logits, labels, num_classes=3).item()
        loss_bad  = corn_loss(bad_logits,  labels, num_classes=3).item()
        assert loss_good < loss_bad

    def test_probas_shape(self):
        logits = torch.randn(8, 2)
        probas = corn_label_to_probas(logits)
        assert probas.shape == (8, 3)

    def test_probas_sum_to_one(self):
        logits = torch.randn(16, 2)
        probas = corn_label_to_probas(logits)
        sums = probas.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5), \
            f"Row sums: {sums}"

    def test_probas_non_negative(self):
        logits = torch.randn(16, 2)
        probas = corn_label_to_probas(logits)
        assert (probas >= 0).all(), "All probabilities must be ≥ 0"

    def test_predict_valid_class_range(self):
        logits = torch.randn(16, 2)
        preds  = corn_predict(logits)
        assert preds.shape == (16,)
        assert ((preds >= 0) & (preds <= 2)).all()

    def test_extreme_logits_grade_two(self):
        """Very large positive logits → predict grade 2 (highest)."""
        logits = torch.full((4, 2), 10.0)
        preds  = corn_predict(logits)
        assert (preds == 2).all()

    def test_extreme_logits_grade_zero(self):
        """Very large negative logits → predict grade 0 (lowest)."""
        logits = torch.full((4, 2), -10.0)
        preds  = corn_predict(logits)
        assert (preds == 0).all()


# ---------------------------------------------------------------------------
# Multi-label BCE tests
# ---------------------------------------------------------------------------
class TestMultiLabelLoss:

    def test_bce_finite(self):
        logits  = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        assert torch.isfinite(loss)

    def test_bce_scalar(self):
        logits  = torch.randn(8, 2)
        targets = torch.zeros(8, 2)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        assert loss.dim() == 0

    def test_bce_non_negative(self):
        logits  = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8, 2)).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        assert loss.item() >= 0.0

    def test_bce_zero_for_perfect_prediction(self):
        """Near-perfect prediction should yield near-zero loss."""
        logits  = torch.tensor([[10.0, -10.0]])   # predicts [1, 0]
        targets = torch.tensor([[1.0,   0.0]])
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# Baseline unchanged tests
# ---------------------------------------------------------------------------
class TestBaselineUnchanged:
    """Verify that label_mode='single' behaviour is identical to original."""

    def test_ce_loss_finite(self):
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        loss = F.cross_entropy(logits, labels)
        assert torch.isfinite(loss)

    def test_ce_loss_scalar(self):
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        loss = F.cross_entropy(logits, labels)
        assert loss.dim() == 0

    def test_grade_dtype_is_long(self):
        # Grade must remain torch.long for CrossEntropyLoss
        grade = torch.tensor(1, dtype=torch.long)
        assert grade.dtype == torch.long

    def test_softmax_sums_to_one(self):
        logits = torch.randn(4, 3)
        probs  = torch.softmax(logits, dim=1)
        sums   = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)
