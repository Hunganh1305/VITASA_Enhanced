"""
summarize_pair_results.py — ✨ NEW (2026-07-20) tổng hợp kết quả ablation
(formulation pair classification đúng) thành bảng so sánh với baseline ViTASD.

Thay thế summarize_results.py (cũ dùng BIO tagging sai).

Chạy: python3 summarize_pair_results.py
Xuất: bảng ra màn hình + experiments/results_pair_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "experiments" / "results_pair"

BASELINE = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}

CONFIG_LABELS = [
    ("loss-ce", "C1 — Baseline (CE)"),
    ("loss-ce_norm", "C2 — + Normalization"),
    ("loss-focal", "C3 — + Imbalanced"),
    ("loss-focal_norm", "C4 — Full Model"),
]


def load_all() -> list[dict]:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/results.json")):
        try:
            rows.append(json.load(open(f, encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] bỏ qua {f}: {e}")
    return rows


def main():
    if not RESULTS_DIR.exists():
        print(f"Chưa có kết quả nào ở {RESULTS_DIR}. Chạy ./run_pair_ablation.sh trước.")
        return

    rows = load_all()
    if not rows:
        print(f"Không tìm thấy results.json nào trong {RESULTS_DIR}.")
        return

    # index: (domain, config_key) -> test macro F1 (%)
    table: dict[tuple[str, str], float] = {}
    for r in rows:
        cfg = r["config"]  # vd pair_mobile_loss-focal_norm_visobert
        domain = r["domain"]
        key = cfg.split(f"pair_{domain}_", 1)[-1]
        for suffix in ("_visobert", "_phobert"):
            key = key.replace(suffix, "")
        table[(domain, key)] = r["test"]["macro_f1"] * 100

    print("\n" + "=" * 78)
    print("KẾT QUẢ ABLATION — formulation: target-aspect pair classification")
    print("Metric: macro F1 trên 3 lớp sentiment (loại 'none') — cùng thang đo paper")
    print("=" * 78)

    header = f"{'Cấu hình':<26}" + "".join(f"{d.capitalize():>15}" for d in BASELINE)
    print(header)
    print("-" * 78)

    for key, label in CONFIG_LABELS:
        cells = ""
        for d in BASELINE:
            v = table.get((d, key))
            cells += f"{v:>14.2f}%" if v is not None else f"{'—':>15}"
        print(f"{label:<26}{cells}")

    print("-" * 78)
    print(f"{'ViTASD (paper gốc)':<26}" + "".join(f"{v:>14.2f}%" for v in BASELINE.values()))
    print("=" * 78)

    # chênh lệch của config tốt nhất mỗi domain so với baseline
    print("\nCấu hình tốt nhất mỗi domain so với baseline ViTASD:")
    for d, base in BASELINE.items():
        cands = {k: v for (dom, k), v in table.items() if dom == d}
        if not cands:
            print(f"  {d:<12} — chưa có kết quả")
            continue
        best_k = max(cands, key=cands.get)
        best_v = cands[best_k]
        label = dict(CONFIG_LABELS).get(best_k, best_k)
        print(f"  {d:<12} {best_v:6.2f}%  ({label})   vs baseline {base}%  →  {best_v - base:+.2f}")

    out_csv = ROOT / "experiments" / "results_pair_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "config", "test_macro_f1_pct", "baseline_vitasd_pct", "diff"])
        for (d, k), v in sorted(table.items()):
            w.writerow([d, dict(CONFIG_LABELS).get(k, k), f"{v:.2f}",
                        BASELINE[d], f"{v - BASELINE[d]:+.2f}"])
    print(f"\nCSV → {out_csv}")


if __name__ == "__main__":
    main()
