"""
Imbalanced Learning module cho Vietnamese Targeted Aspect Sentiment Analysis (TASA).

Mục đích: cung cấp loss function thay thế Cross-Entropy thường để xử lý
class imbalance giữa các polarity (Positive/Negative/Neutral) — limitation
thứ hai được nêu trong paper ViTASA gốc.

Gồm:
    - FocalLoss: giảm trọng số đóng góp của các sample đã được phân loại
      tốt (easy example), tập trung học vào sample khó / minority class.
    - WeightedCrossEntropyLoss: Cross-Entropy có trọng số theo class.
    - compute_class_weights(): tính trọng số class từ label distribution,
      hỗ trợ 2 chiến lược "inverse_frequency" và "effective_number"
      (Class-Balanced Loss, Cui et al. 2019).

Tất cả đều là drop-in replacement cho torch.nn.CrossEntropyLoss trong
training loop fine-tune PhoBERT/ViSoBERT (nhận logits [N, C] và label [N]).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss cho multi-class classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    trong đó p_t là xác suất model dự đoán cho đúng class. Khi gamma=0,
    FocalLoss tương đương (weighted) Cross-Entropy.

    Args:
        gamma: hệ số focusing, càng lớn càng giảm mạnh đóng góp của sample
            dễ (p_t cao). Giá trị phổ biến: 1.0 - 2.0. gamma=0 -> CE thường.
        alpha: trọng số theo class, shape [num_classes] hoặc None. Dùng để
            kết hợp Focal Loss với class weighting (vd lấy từ
            compute_class_weights()).
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma phải >= 0, nhận được {gamma}")
        self.gamma = gamma
        self.reduction = reduction
        # register_buffer để alpha tự move theo .to(device) của module
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] raw scores (chưa softmax)
            target: [N] class index (long), giá trị trong [0, C-1]
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [N, C]
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-Entropy có trọng số theo class — wrapper mỏng quanh
    nn.CrossEntropyLoss(weight=...) để cùng interface với FocalLoss,
    tiện swap qua lại khi chạy ablation."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, target)


def compute_class_weights(
    labels: Iterable[int],
    num_classes: int,
    strategy: Literal["inverse_frequency", "effective_number"] = "effective_number",
    beta: float = 0.999,
) -> torch.Tensor:
    """Tính trọng số class từ label distribution thực tế (vd của ViTASA dataset).

    Args:
        labels: list/array nhãn (int), vd toàn bộ y_train.
        num_classes: số class (TASA: thường 3 — Positive/Negative/Neutral).
        strategy:
            - "inverse_frequency": weight_c = N / (num_classes * count_c)
              (chuẩn hóa để trung bình weight = 1)
            - "effective_number": Class-Balanced Loss (Cui et al., CVPR 2019)
              weight_c = (1 - beta) / (1 - beta^count_c), sau đó chuẩn hóa
              để trung bình = 1. Mềm hơn inverse_frequency khi count_c lớn,
              tránh weight quá cực đoan cho class rất ít sample.
        beta: chỉ dùng cho "effective_number", thường 0.99 - 0.9999.

    Returns:
        Tensor [num_classes], dtype float32.
    """
    counts = Counter(int(l) for l in labels)
    count_per_class = torch.tensor(
        [counts.get(c, 0) for c in range(num_classes)], dtype=torch.float32
    )
    if (count_per_class == 0).any():
        missing = [c for c in range(num_classes) if count_per_class[c] == 0]
        raise ValueError(
            f"Class {missing} không có sample nào trong `labels` — "
            "không thể tính weight. Kiểm tra lại label distribution."
        )

    if strategy == "inverse_frequency":
        total = count_per_class.sum()
        weights = total / (num_classes * count_per_class)
    elif strategy == "effective_number":
        effective_num = 1.0 - torch.pow(beta, count_per_class)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Không hỗ trợ strategy={strategy!r}")

    # Chuẩn hóa để trung bình weight = 1 (giữ scale loss ổn định, dễ so sánh giữa config)
    weights = weights / weights.mean()
    return weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # Giả lập label distribution mất cân bằng kiểu ViTASA:
    # Positive nhiều, Negative ít, Neutral rất ít
    labels = [0] * 700 + [1] * 250 + [2] * 50  # 0=Positive, 1=Negative, 2=Neutral
    weights = compute_class_weights(labels, num_classes=3, strategy="effective_number")
    print("Class weights (effective_number):", weights.tolist())

    weights_inv = compute_class_weights(labels, num_classes=3, strategy="inverse_frequency")
    print("Class weights (inverse_frequency):", weights_inv.tolist())

    # So sánh loss giữa CE thường, Weighted CE, Focal Loss trên 1 batch giả lập
    logits = torch.randn(8, 3)
    target = torch.tensor([0, 0, 0, 1, 1, 2, 0, 1])

    ce = nn.CrossEntropyLoss()
    wce = WeightedCrossEntropyLoss(class_weights=weights)
    focal = FocalLoss(gamma=2.0, alpha=weights)

    print("\nSo sánh trên 1 batch giả lập:")
    print("  CE loss          :", ce(logits, target).item())
    print("  Weighted CE loss :", wce(logits, target).item())
    print("  Focal loss       :", focal(logits, target).item())
