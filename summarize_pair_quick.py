"""summarize_pair_quick.py — bảng tổng hợp nhanh kết quả pair-classification,
1 dòng / 1 config, kèm cột "vs ViTASD".

✨ NEW (2026-07-21): thay thế bản cũ bị bug đọc nhầm key `d["test_f1"]`
(field phẳng, chỉ có ở format BIO cũ của train.py). File results.json của
train_pair.py lưu test dạng LỒNG NHAU: d["test"]["macro_f1"] — đọc nhầm key
khiến Test F1 luôn hiện 0.00% dù Dev F1 vẫn đúng (field đó không đổi tên).

Cách dùng (chạy trong VITASA_Enhanced/, kể cả trên Colab):
    python3 summarize_pair_quick.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "experiments" / "results_pair"
BASELINE_VITASD = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/results.json")):
        d = json.load(open(f, encoding="utf-8"))
        domain = d["domain"]
        rows.append({
            "config": d["config"],
            "domain": domain,
            "dev_f1": round(d["best_dev_f1"] * 100, 2),
            # 🔧 FIX: đọc đúng key lồng nhau thay vì d["test_f1"]
            "test_f1": round(d["test"]["macro_f1"] * 100, 2),
            "vs_vitasd": round(d["test"]["macro_f1"] * 100 - BASELINE_VITASD[domain], 2),
        })

    if not rows:
        print(f"⚠️  Chưa có kết quả nào trong {RESULTS_DIR}")
        return

    print("-" * 80)
    print(f'{"Config":<36} {"Domain":<10} {"Dev F1":>8} {"Test F1":>9} {"vs ViTASD":>11}')
    print("-" * 80)
    for r in sorted(rows, key=lambda x: (x["domain"], x["config"])):
        print(f'{r["config"]:<36} {r["domain"]:<10} {r["dev_f1"]:>7.2f}% '
              f'{r["test_f1"]:>8.2f}% {r["vs_vitasd"]:>10.2f}%')
    print("-" * 80)
    print(f"Total: {len(rows)} results")


if __name__ == "__main__":
    main()
