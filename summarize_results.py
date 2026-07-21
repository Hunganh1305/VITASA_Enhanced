"""Tổng hợp kết quả training từ experiments/results/*/results.json (bản local,
tương đương Cell 6 của colab_run.ipynb) và so với baseline ViTASD.

Cách dùng:
    python summarize_results.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "experiments" / "results"
BASELINE_VITASD = {"mobile": 61.77, "restaurant": 41.12, "hotel": 52.64}


def config_label(d: dict) -> str:
    """Nhãn config ngắn gọn để hiển thị bảng.

    Chú ý (2026-07-16): thêm hậu tố cho 2 option ablation mới
    (keep_expressive, weight_strategy) — nếu không, các combo của cross
    ablation (keep_expressive x weight_strategy) đều bị gộp chung nhãn "C4",
    khiến bảng hiển thị nhầm lẫn (nhiều dòng "C4" khác nhau) và — nghiêm
    trọng hơn ở evaluation_table.py — bị ĐÈ MẤT DỮ LIỆU do dict key trùng.
    d.get(...) có default nên results.json cũ (chưa có 2 field này) không bị
    ảnh hưởng, vẫn ra đúng nhãn C1-C4 như trước.
    """
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


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/results.json")):
        d = json.load(open(f, encoding="utf-8"))
        rows.append({
            "domain": d["domain"],
            "config": config_label(d),
            "loss": d["loss"],
            "normalize": d["normalize"],
            "epochs": d["epochs"],
            "dev_f1": round(d["best_dev_f1"] * 100, 2),
            "test_f1": round(d["test_f1"] * 100, 2),
        })

    if not rows:
        print("⚠️  Chưa có kết quả nào trong experiments/results/ — chạy run_all_local.sh trước.")
        return

    print(f'{"Domain":<12}{"Config":<22}{"Loss":<12}{"Norm":<6}{"Epochs":>7}{"DevF1":>8}{"TestF1":>9}')
    print("-" * 78)
    for r in sorted(rows, key=lambda x: (x["domain"], x["config"])):
        print(f'{r["domain"]:<12}{r["config"]:<22}{r["loss"]:<12}{str(r["normalize"]):<6}'
              f'{r["epochs"]:>7}{r["dev_f1"]:>7.2f}%{r["test_f1"]:>8.2f}%')
    print("-" * 78)
    print(f"Tổng: {len(rows)} config đã chạy (12 config C1-C4 x 3 domain + các combo "
          f"cross ablation nếu có, xem run_cross_ablation.sh)")

    print("\n=== So sánh Test F1 với baseline ViTASD ===")
    for domain, base in BASELINE_VITASD.items():
        best = max((r for r in rows if r["domain"] == domain), key=lambda r: r["test_f1"], default=None)
        if best is None:
            print(f"  {domain:<12}: chưa có kết quả")
            continue
        diff = best["test_f1"] - base
        arrow = "✅ vượt" if diff > 0 else "❌ thấp hơn"
        print(f"  {domain:<12}: best={best['config']} test_f1={best['test_f1']:.2f}% "
              f"vs baseline={base:.2f}% ({arrow} {diff:+.2f} điểm)")


if __name__ == "__main__":
    main()
