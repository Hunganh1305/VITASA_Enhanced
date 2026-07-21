"""Chỉ giữ lại best_model.pt của config có test_f1 CAO NHẤT mỗi domain (trong
experiments/results/, model=visobert — nhánh ablation đang chạy) — xóa .pt của
mọi config còn lại để tiết kiệm dung lượng, KHÔNG đụng vào results.json (mọi
số liệu Macro-F1 vẫn giữ nguyên 100%, chỉ mất khả năng load lại weight của
các config không phải "tốt nhất").

Lý do (2026-07-16): ablation có nhiều chục config/domain, mỗi checkpoint
ViSoBERT ~187MB (fp16, xem train.py). Chỉ config tốt nhất mới thực sự cần giữ
weight để dùng cho error analysis / demo / báo cáo sau này — các config còn
lại chỉ cần results.json để so sánh trong bảng ablation.

Cách dùng:
    python3 cleanup_checkpoints.py                  # tất cả domain
    python3 cleanup_checkpoints.py --domain mobile   # 1 domain
    python3 cleanup_checkpoints.py --dry-run         # chỉ in ra, không xóa

Được gọi tự động sau MỖI run trong run_cross_ablation.sh (chỉ domain vừa
train xong) — "tối ưu dung lượng sau mỗi lần train" theo yêu cầu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "experiments" / "results"


def cleanup_domain(domain: str, dry_run: bool = False) -> None:
    candidates = []
    for f in sorted(RESULTS_DIR.glob("*/results.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("domain") != domain or d.get("model") != "visobert":
            continue
        pt = f.parent / "best_model.pt"
        candidates.append((f.parent.name, d["test_f1"], pt))

    if not candidates:
        print(f"[{domain}] chưa có config nào (model=visobert), bỏ qua.")
        return

    candidates.sort(key=lambda c: c[1], reverse=True)
    best_name, best_f1, _ = candidates[0]
    print(f"[{domain}] config tốt nhất: {best_name} (test_f1={best_f1*100:.2f}%) — GIỮ checkpoint")

    freed = 0
    for name, f1, pt in candidates[1:]:
        if not pt.exists():
            continue
        size = pt.stat().st_size
        freed += size
        action = "sẽ xóa" if dry_run else "đã xóa"
        print(f"  {action}: {name} (test_f1={f1*100:.2f}%, {size/1e6:.0f}MB)")
        if not dry_run:
            pt.unlink()

    print(f"[{domain}] {'sẽ giải phóng' if dry_run else 'đã giải phóng'}: {freed/1e9:.2f} GB\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["mobile", "restaurant", "hotel"], default=None,
                         help="Chỉ dọn 1 domain (default: cả 3)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra, không xóa gì")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else ["mobile", "restaurant", "hotel"]
    for domain in domains:
        cleanup_domain(domain, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
