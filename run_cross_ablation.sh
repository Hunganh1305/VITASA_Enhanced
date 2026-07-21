#!/usr/bin/env bash
# Cross ablation cho 2 finding từ paper ViGoEmotions (EACL 2026) — xem
# text_normalization/normalizer.py và imbalanced_learning/losses.py.
#
# Chạy 4 combo (keep_expressive x weight_strategy) x 3 domain = 12 run, trên
# nền config "Full Model" hiện tại (loss=focal, --normalize, model=visobert):
#
#   combo 1: keep_expressive=false  weight_strategy=effective_number  (= config
#            C4 gốc đã có sẵn từ run_all_local.sh trước đó -> tự động SKIP)
#   combo 2: keep_expressive=true   weight_strategy=effective_number
#   combo 3: keep_expressive=false  weight_strategy=pos_neg_ratio
#   combo 4: keep_expressive=true   weight_strategy=pos_neg_ratio
#
# -> thực tế chỉ chạy MỚI 9 run (3 domain x 3 combo còn lại), vì combo 1 dùng
#    lại kết quả "<domain>_loss-focal_norm_visobert" đã có.
#
# Tự động BỎ QUA run nào đã có experiments/results/<config>/results.json, nên
# có thể Ctrl-C giữa chừng rồi chạy lại script để tiếp tục từ chỗ dừng.
#
# Cách dùng:
#   chmod +x run_cross_ablation.sh
#   ./run_cross_ablation.sh                 # mặc định 30 epoch, backbone visobert
#   ./run_cross_ablation.sh 20              # tùy chỉnh số epoch
#   ./run_cross_ablation.sh 30 phobert      # đổi backbone (không khuyến nghị, xem README)
#
# Ước lượng thời gian (đo thật trên M3/MPS, 30 epoch, ViSoBERT — xem
# logs/run_all_20260713_003438.log): ~29 phút/run -> 9 run mới ~ 4.3 giờ.
# 2 option mới không đổi kiến trúc/batch size nên tốc độ mỗi run xấp xỉ y hệt
# các run C4 cũ, không chậm hơn đáng kể.
#
# Sau khi chạy xong, dùng summarize_results.py hoặc evaluation_table.py để
# xem bảng so sánh (đã cập nhật để phân biệt 4 combo, không còn gộp chung "C4").
#
# Tối ưu dung lượng (2026-07-16): sau MỖI run, script tự gọi
# cleanup_checkpoints.py --domain <domain> — chỉ giữ checkpoint (.pt) của
# config có test_f1 cao nhất trong domain đó, xóa .pt của các config còn lại.
# results.json (số liệu Macro-F1 để so sánh ablation) KHÔNG bị đụng tới.

set -eo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="python3"
command -v python3 >/dev/null 2>&1 || PYTHON_BIN="python"

EPOCHS=${1:-30}
MODEL=${2:-visobert}
MODEL_SUFFIX=""
if [ "$MODEL" != "phobert" ]; then
  MODEL_SUFFIX="_${MODEL}"
fi

DOMAINS=("mobile" "restaurant" "hotel")
# keep_expressive weight_strategy
COMBOS=(
  "false effective_number"   # = C4 gốc, thường đã có sẵn -> auto-skip
  "true  effective_number"
  "false pos_neg_ratio"
  "true  pos_neg_ratio"
)

mkdir -p logs
LOGFILE="logs/run_cross_$(date +%Y%m%d_%H%M%S).log"
echo "Bắt đầu cross ablation (keep_expressive x weight_strategy) — EPOCHS=$EPOCHS, MODEL=$MODEL" | tee -a "$LOGFILE"
echo "Log chi tiết: $LOGFILE"
overall_start=$(date +%s)

total=0
done_count=0
skip_count=0

for domain in "${DOMAINS[@]}"; do
  for combo in "${COMBOS[@]}"; do
    total=$((total+1))
  done
done

for domain in "${DOMAINS[@]}"; do
  for combo in "${COMBOS[@]}"; do
    read -r keep_expr weight_strategy <<< "$combo"

    expr_flag=""
    expr_suffix=""
    if [ "$keep_expr" = "true" ]; then
      expr_flag="--keep-expressive"
      expr_suffix="_keepexpr"
    fi

    weight_flag="--weight-strategy $weight_strategy"
    weight_suffix=""
    if [ "$weight_strategy" != "effective_number" ]; then
      weight_suffix="_w-${weight_strategy}"
    fi

    # Phải khớp đúng cách train.py build config_name (xem hàm main() trong train.py):
    # {domain}_loss-focal_norm{expr_suffix}{model_suffix}{seg_suffix}{weight_suffix}
    config_name="${domain}_loss-focal_norm${expr_suffix}${MODEL_SUFFIX}${weight_suffix}"
    result_file="experiments/results/${config_name}/results.json"

    if [ -f "$result_file" ]; then
      echo "⏭  [$config_name] đã có kết quả, bỏ qua." | tee -a "$LOGFILE"
      skip_count=$((skip_count+1))
      continue
    fi

    echo "" | tee -a "$LOGFILE"
    echo "=== [$config_name] bắt đầu $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOGFILE"
    start=$(date +%s)
    "$PYTHON_BIN" train.py --domain "$domain" --loss focal --normalize $expr_flag $weight_flag \
      --model "$MODEL" --epochs "$EPOCHS" 2>&1 | tee -a "$LOGFILE"
    end=$(date +%s)
    mins=$(( (end-start)/60 ))
    echo "=== [$config_name] xong, mất ${mins} phút ===" | tee -a "$LOGFILE"
    done_count=$((done_count+1))

    # Tối ưu dung lượng ngay sau mỗi lần train: chỉ giữ checkpoint của config
    # tốt nhất (test_f1 cao nhất) trong domain này, xóa .pt của các config còn
    # lại — results.json vẫn giữ nguyên 100% (không mất số liệu ablation).
    "$PYTHON_BIN" cleanup_checkpoints.py --domain "$domain" | tee -a "$LOGFILE"
  done
done

overall_end=$(date +%s)
hours=$(( (overall_end-overall_start)/3600 ))
echo "" | tee -a "$LOGFILE"
echo "✅ HOÀN TẤT — chạy $done_count/$total run (bỏ qua $skip_count đã có sẵn), tổng ${hours} giờ." | tee -a "$LOGFILE"
echo "Chạy '$PYTHON_BIN summarize_results.py' hoặc '$PYTHON_BIN evaluation_table.py' để xem bảng tổng hợp."
