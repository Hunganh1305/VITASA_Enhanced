"""Bảng đánh giá tổng hợp — so sánh Test F1 qua các mốc cải tiến pipeline (bug fix,
word segmentation, đổi backbone), đối chiếu với baseline ViTASD.

4 mốc so sánh:
    1. "Trước fix bug"      — lần train Colab đầu tiên (2026-07-03 → 07-08), trước khi
                                phát hiện bug padding-mask/Unicode offset. Dữ liệu gốc nằm
                                rải rác ở nhiều thư mục cũ ngoài project (đã archive), nên
                                hardcode lại đây làm mốc tham chiếu — KHÔNG tính lại từ file.
    2. "Sau fix (chưa segment)" — đọc live từ experiments/results_presegment_backup/ (PhoBERT)
    3. "PhoBERT — final"       — đọc live từ experiments/results_phobert_final/ (PhoBERT,
                                   sau fix bug + segmentation, 30 epoch — archive 2026-07-12
                                   trước khi đổi backbone sang ViSoBERT)
    4. "ViSoBERT — hiện tại"   — đọc live từ experiments/results/, LỌC model="visobert"
                                   (backbone mới, thay thế PhoBERT sau khi quick-test 15 epoch
                                   cho tín hiệu tốt hơn hẳn — xem experiments/quicktest/)

Mỗi mốc từ folder được lọc theo "model" trong results.json để PhoBERT và ViSoBERT
không bị lẫn vào nhau dù nằm chung 1 thư mục experiments/results/.

Cách dùng:
    python3 evaluation_table.py
    python3 evaluation_table.py --csv out.csv     # xuất thêm file CSV
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent

BASELINE_VITASD = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}

# Mốc 1 — hardcode (xem docstring). Chỉ có số của config C4 (Focal+Norm) vì đây là
# config duy nhất còn số liệu rõ ràng từ trước khi dọn dẹp; các config khác lúc đó
# đều ra ~0% (bug khiến model chỉ đoán "O").
ROUND1_BEFORE_FIX = {
    ("mobile", "C4"): 9.05,
    ("restaurant", "C4"): 0.71,
    ("hotel", "C4"): 0.37,
}

# (tên hiển thị, thư mục, model_filter) — model_filter=None nghĩa là không lọc
# (dùng cho các mốc cũ chỉ có 1 backbone duy nhất trong thư mục).
ROUNDS_FROM_FOLDER = [
    ("Sau fix (chưa segment)", ROOT / "experiments" / "results_presegment_backup", None),
    ("PhoBERT — final", ROOT / "experiments" / "results_phobert_final", None),
    ("ViSoBERT — hiện tại", ROOT / "experiments" / "results", "visobert"),
]


def config_label(d: dict) -> str:
    """Nhãn config ngắn gọn để hiển thị bảng — PHẢI trùng logic với
    summarize_results.py (xem docstring bên đó để hiểu lý do thêm hậu tố
    keep_expressive/weight_strategy: tránh 2 combo khác nhau bị gộp cùng
    key "C4" trong load_round() bên dưới, dẫn đến đè mất kết quả)."""
    if d["normalize"] and d["loss"] == "focal":
        base = "C4"
    elif not d["normalize"] and d["loss"] == "focal":
        base = "C3"
    elif d["normalize"] and d["loss"] == "ce":
        base = "C2"
    elif not d["normalize"] and d["loss"] == "ce":
        base = "C1"
    else:
        base = d["loss"] + ("_norm" if d["normalize"] else "")

    extra = []
    if d.get("keep_expressive"):
        extra.append("keepexpr")
    weight_strategy = d.get("weight_strategy", "effective_number")
    if weight_strategy != "effective_number":
        extra.append(weight_strategy)
    if extra:
        base += "+" + "+".join(extra)
    return base


def load_round(folder: Path, model_filter: str | None = None) -> dict[tuple[str, str], float]:
    """Trả về {(domain, config): test_f1_percent} từ 1 thư mục experiments/results/.

    model_filter: nếu set, chỉ lấy các run có d["model"] == model_filter — cần thiết
    khi 1 thư mục chứa lẫn kết quả của nhiều backbone (vd cả phobert lẫn visobert
    cùng nằm trong experiments/results/ do tên config khác nhau nên không bị ghi đè).
    """
    out = {}
    if not folder.exists():
        return out
    for f in sorted(folder.glob("*/results.json")):
        d = json.load(open(f, encoding="utf-8"))
        if model_filter is not None and d.get("model", "phobert") != model_filter:
            continue
        key = (d["domain"], config_label(d))
        out[key] = round(d["test_f1"] * 100, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Xuất thêm bảng ra file CSV")
    args = parser.parse_args()

    rounds = [("Trước fix bug", None, None)] + ROUNDS_FROM_FOLDER
    round_data = [ROUND1_BEFORE_FIX] + [load_round(folder, mf) for _, folder, mf in ROUNDS_FROM_FOLDER]

    domains = ["mobile", "restaurant", "hotel"]
    # C1-C4 luôn hiển thị trước (đúng thứ tự cũ); các combo cross ablation mới
    # (vd "C4+keepexpr", "C4+pos_neg_ratio") chỉ tồn tại ở mốc cuối
    # ("ViSoBERT — hiện tại") nên được tự động thêm vào cuối danh sách nếu có,
    # thay vì hardcode — tránh bị "biến mất" khỏi bảng như trước khi fix.
    base_configs = ["C1", "C2", "C3", "C4"]
    extra_configs = sorted({
        cfg for (_, cfg) in round_data[-1].keys() if cfg not in base_configs
    })
    configs = base_configs + extra_configs

    headers = ["Domain", "Config"] + [name for name, _, _ in rounds] + ["Baseline ViTASD", "Chênh lệch (mốc cuối)"]
    # Config column rộng hơn 7 vì nhãn cross-ablation có thể dài (vd
    # "C4+keepexpr+pos_neg_ratio")
    col_w = [12, 28] + [22] * len(rounds) + [16, 22]

    def fmt_row(cells):
        return "".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print(fmt_row(headers))
    print("-" * sum(col_w))

    csv_rows = [headers]
    for domain in domains:
        for cfg in configs:
            key = (domain, cfg)
            values = []
            for data in round_data:
                v = data.get(key)
                values.append(f"{v:.2f}%" if v is not None else "—")
            baseline = BASELINE_VITASD[domain]
            last = round_data[-1].get(key)
            diff = f"{last - baseline:+.2f}" if last is not None else "—"
            row = [domain, cfg] + values + [f"{baseline:.2f}%", diff]
            print(fmt_row(row))
            csv_rows.append(row)

    print()
    print("=== Best config mỗi domain (theo mốc cuối) so với baseline ===")
    for domain in domains:
        best_key, best_val = None, -1.0
        for cfg in configs:
            v = round_data[-1].get((domain, cfg))
            if v is not None and v > best_val:
                best_key, best_val = cfg, v
        baseline = BASELINE_VITASD[domain]
        if best_key is None:
            print(f"  {domain:<12}: chưa có dữ liệu")
            continue
        diff = best_val - baseline
        mark = "✅" if diff > 0 else "❌"
        print(f"  {domain:<12}: {best_key} = {best_val:.2f}% vs baseline {baseline:.2f}% ({mark} {diff:+.2f} điểm)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nĐã xuất CSV → {args.csv}")


if __name__ == "__main__":
    main()
