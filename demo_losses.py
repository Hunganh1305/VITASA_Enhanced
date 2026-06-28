"""
CLI demo cho Imbalanced Learning Module.
Chạy: python3 demo_losses.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from imbalanced_learning.losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
)

CLASS_NAMES = ["Positive", "Negative", "Neutral"]

SAMPLE_DISTRIBUTIONS = {
    "1": ("Mobile (giả lập theo paper ViTASA — Positive áp đảo)", [620, 280, 100]),
    "2": ("Restaurant (giả lập — imbalance nặng hơn)", [500, 350, 150]),
    "3": ("Hotel (giả lập)", [580, 300, 120]),
    "4": ("Cân bằng hoàn toàn (đối chứng)", [300, 300, 300]),
}


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_distribution(counts: list[int]) -> None:
    labels = []
    for class_idx, n in enumerate(counts):
        labels += [class_idx] * n

    print(
        "\nLabel distribution: "
        + ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in enumerate(counts))
    )

    w_inv = compute_class_weights(labels, num_classes=3, strategy="inverse_frequency")
    w_eff = compute_class_weights(labels, num_classes=3, strategy="effective_number")

    print("\nClass weights:")
    print(f"  inverse_frequency : {[round(x, 3) for x in w_inv.tolist()]}")
    print(f"  effective_number  : {[round(x, 3) for x in w_eff.tolist()]}")

    # Tạo 1 batch giả lập (16 sample) theo đúng tỉ lệ distribution người dùng nhập
    random.seed(0)
    torch.manual_seed(0)
    target_list = random.choices(population=range(3), weights=counts, k=16)
    target = torch.tensor(target_list)
    logits = torch.randn(16, 3)

    ce = nn.CrossEntropyLoss()
    wce = WeightedCrossEntropyLoss(class_weights=w_eff)
    focal = FocalLoss(gamma=2.0, alpha=w_eff)

    print("\nSo sánh loss trên 1 batch giả lập (16 sample, lấy theo tỉ lệ distribution trên):")
    print(f"  CE thường   : {ce(logits, target).item():.4f}")
    print(f"  Weighted CE : {wce(logits, target).item():.4f}")
    print(f"  Focal Loss  : {focal(logits, target).item():.4f}")


def run_samples() -> None:
    print_header("CHẠY VỚI SAMPLE CÓ SẴN")
    for key, (name, counts) in SAMPLE_DISTRIBUTIONS.items():
        print(f"{key}. {name} -> {counts}")
    try:
        choice = input("\nChọn sample (1-4): ").strip()
    except EOFError:
        return
    if choice in SAMPLE_DISTRIBUTIONS:
        _, counts = SAMPLE_DISTRIBUTIONS[choice]
        run_distribution(counts)
    else:
        print("Lựa chọn không hợp lệ.")


def run_user_input() -> None:
    print_header("TỰ NHẬP LABEL DISTRIBUTION")
    print(f"Nhập số lượng sample cho mỗi class: {CLASS_NAMES}")
    counts = []
    for name in CLASS_NAMES:
        while True:
            try:
                raw = input(f"  Số lượng {name}: ").strip()
            except EOFError:
                return
            if raw.isdigit() and int(raw) > 0:
                counts.append(int(raw))
                break
            print("  Vui lòng nhập số nguyên dương.")
    run_distribution(counts)


def main() -> None:
    while True:
        print_header("IMBALANCED LEARNING — DEMO CLI")
        print("1. Chạy với sample có sẵn")
        print("2. Tự nhập label distribution")
        print("0. Thoát")
        try:
            choice = input("\nChọn (0/1/2): ").strip()
        except EOFError:
            break

        if choice == "1":
            run_samples()
        elif choice == "2":
            run_user_input()
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, thử lại.")


if __name__ == "__main__":
    main()
