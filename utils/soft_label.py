"""
Soft label construction with ordinal prior.

Only valid for ordinal targets where a natural ordering exists:
Normal (0) < PTCD (1) < FTCD (2) for the in-house cartilage dataset.

Scientific justification:
    WORMS criteria define an explicit severity ordering. A subject graded
    PTCD (WORMS 2–4) sits closer to FTCD than to Normal, but the released
    dataset collapses raw WORMS scores to 3 classes. This function
    approximates that uncertainty by distributing a small probability mass
    `alpha` to adjacent classes only — not uniformly to all classes.
    Alpha should be selected on the validation split, never on the test set.

NOT valid for MRNet multi-label targets (ACL and Abnormality are independent,
not ordered), so do not call this function in multi_label mode.
"""
import numpy as np


def build_ordinal_soft_label(grade: int, num_classes: int, alpha: float = 0.1) -> np.ndarray:
    """
    Build a soft target distribution for an ordinal label.

    Args:
        grade:       int   true class in {0, ..., num_classes-1}
        num_classes: int   total number of ordered classes
        alpha:       float smoothing strength in [0, 1).
                          0.0 = hard one-hot label (no smoothing)
                          0.1 = 5% probability moved to each adjacent class

    Returns:
        soft: np.ndarray of shape [num_classes], dtype float32, sums to 1.0

    Example (num_classes=3, alpha=0.1):
        grade=0 → [0.95, 0.05, 0.00]  (left boundary: no lower neighbour)
        grade=1 → [0.05, 0.90, 0.05]
        grade=2 → [0.00, 0.05, 0.95]  (right boundary: no upper neighbour)
    """
    assert 0 <= grade < num_classes, f"grade {grade} out of range [0, {num_classes-1}]"
    assert 0.0 <= alpha < 1.0, f"alpha {alpha} must be in [0, 1)"

    soft = np.zeros(num_classes, dtype=np.float32)
    soft[grade] = 1.0 - alpha

    half = alpha / 2.0

    if grade > 0:
        soft[grade - 1] += half
    else:
        soft[grade] += half          # leftmost: no lower neighbour → keep on self

    if grade < num_classes - 1:
        soft[grade + 1] += half
    else:
        soft[grade] += half          # rightmost: no upper neighbour → keep on self

    assert abs(soft.sum() - 1.0) < 1e-6, f"Soft label sums to {soft.sum()}, expected 1.0"
    return soft
