"""
Unit test cho imbalanced_learning/losses.py.
Chạy: pytest imbalanced_learning/tests/test_losses.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from losses import FocalLoss, WeightedCrossEntropyLoss, compute_class_weights


# ---- FocalLoss ----

def test_focal_loss_gamma0_equals_ce_no_alpha():
    """gamma=0 và không có alpha -> Focal Loss phải trùng Cross-Entropy thường."""
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    target = torch.randint(0, 3, (16,))

    ce = nn.CrossEntropyLoss()
    focal = FocalLoss(gamma=0.0)

    assert torch.allclose(ce(logits, target), focal(logits, target), atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    """Với sample đã được phân loại rất đúng (p_t cao), Focal Loss phải nhỏ hơn CE
    nhiều hơn so với sample bị phân loại sai (p_t thấp) — đúng tinh thần focusing."""
    focal = FocalLoss(gamma=2.0, reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    # Sample "dễ": logit của đúng class rất cao -> p_t gần 1
    easy_logits = torch.tensor([[10.0, 0.0, 0.0]])
    easy_target = torch.tensor([0])

    # Sample "khó": logit gần đều nhau -> p_t gần 1/3
    hard_logits = torch.tensor([[0.1, 0.0, -0.1]])
    hard_target = torch.tensor([0])

    ce_easy, ce_hard = ce(easy_logits, easy_target).item(), ce(hard_logits, hard_target).item()
    focal_easy, focal_hard = focal(easy_logits, easy_target).item(), focal(hard_logits, hard_target).item()

    ratio_easy = focal_easy / ce_easy
    ratio_hard = focal_hard / ce_hard

    # Tỉ lệ focal/CE của sample dễ phải nhỏ hơn nhiều so với sample khó
    assert ratio_easy < ratio_hard
    assert ratio_easy < 0.05  # sample rất dễ -> focal loss giảm mạnh


def test_focal_loss_reduction_modes():
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))

    loss_none = FocalLoss(reduction="none")(logits, target)
    loss_mean = FocalLoss(reduction="mean")(logits, target)
    loss_sum = FocalLoss(reduction="sum")(logits, target)

    assert loss_none.shape == (8,)
    assert loss_mean.dim() == 0
    assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)
    assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)


def test_focal_loss_with_alpha_class_weight():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    target = torch.tensor([2, 2, 2, 2, 0, 0, 1, 1])
    alpha = torch.tensor([1.0, 1.0, 5.0])  # class 2 (minority) được nhân 5x

    focal_no_alpha = FocalLoss(gamma=2.0)(logits, target)
    focal_with_alpha = FocalLoss(gamma=2.0, alpha=alpha)(logits, target)

    # Class 2 chiếm nửa batch và được weight x5 -> loss trung bình phải tăng lên
    assert focal_with_alpha.item() > focal_no_alpha.item()


def test_focal_loss_invalid_gamma():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)


def test_focal_loss_gradient_flows():
    logits = torch.randn(8, 3, requires_grad=True)
    target = torch.randint(0, 3, (8,))
    loss = FocalLoss(gamma=2.0)(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


# ---- WeightedCrossEntropyLoss ----

def test_weighted_ce_matches_nn_crossentropy():
    weights = torch.tensor([1.0, 2.0, 3.0])
    logits = torch.randn(10, 3)
    target = torch.randint(0, 3, (10,))

    wce = WeightedCrossEntropyLoss(class_weights=weights)
    ref = nn.CrossEntropyLoss(weight=weights)

    assert torch.allclose(wce(logits, target), ref(logits, target), atol=1e-6)


def test_weighted_ce_no_weight_equals_plain_ce():
    logits = torch.randn(10, 3)
    target = torch.randint(0, 3, (10,))

    wce = WeightedCrossEntropyLoss(class_weights=None)
    ce = nn.CrossEntropyLoss()

    assert torch.allclose(wce(logits, target), ce(logits, target), atol=1e-6)


# ---- compute_class_weights ----

def test_inverse_frequency_known_values():
    # 3 class: 600 / 300 / 100 sample -> total=1000, num_classes=3
    # raw weight = total / (num_classes * count) = 1000/(3*count)
    # class0: 1000/1800=0.5556, class1: 1000/900=1.1111, class2: 1000/300=3.3333
    # sau chuẩn hóa theo mean (mean_raw = 1.6667) -> [0.3333, 0.6667, 2.0]
    labels = [0] * 600 + [1] * 300 + [2] * 100
    weights = compute_class_weights(labels, num_classes=3, strategy="inverse_frequency")

    expected = torch.tensor([0.3333, 0.6667, 2.0])
    assert torch.allclose(weights, expected, atol=1e-3)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)


def test_effective_number_minority_gets_higher_weight():
    labels = [0] * 700 + [1] * 250 + [2] * 50
    weights = compute_class_weights(labels, num_classes=3, strategy="effective_number")

    # Class càng ít sample (class 2) phải có weight càng cao
    assert weights[2] > weights[1] > weights[0]
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-4)


def test_compute_class_weights_missing_class_neutral_weight():
    # Missing class nhận weight = 1.0 (neutral) thay vì raise error,
    # vì rare aspect-sentiment classes có thể không xuất hiện trong train split
    labels = [0, 0, 1, 1]  # thiếu class 2
    weights = compute_class_weights(labels, num_classes=3)
    assert weights.shape == (3,)
    assert weights[2].item() > 0  # không crash, weight > 0


def test_compute_class_weights_balanced_gives_equal_weight():
    labels = [0] * 100 + [1] * 100 + [2] * 100
    weights = compute_class_weights(labels, num_classes=3, strategy="inverse_frequency")
    assert torch.allclose(weights, torch.ones(3), atol=1e-5)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        compute_class_weights([0, 1, 2], num_classes=3, strategy="not_a_real_strategy")
