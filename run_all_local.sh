#!/usr/bin/env bash
# Chạy toàn bộ ablation (4 config x 3 domain = 12 run) trên máy local (M3/MPS).
# Tự động BỎ QUA config nào đã có experiments/results/<config>/results.json,
# nên có thể Ctrl-C giữa chừng rồi chạy lại script để tiếp tục từ chỗ dừng.
#
# Cách dùng:
#   chmod +x run_all_local.sh
#   ./run_all_local.sh                    # mặc định 30 epoch, backbone visobert
#   ./run_all_local.sh 20                 # tùy chỉnh số epoch
#   ./run_all_local.sh 30 phobert         # đổi lại backbone phobert nếu cần
#
# Backbone mặc định ĐÃ ĐỔI sang ViSoBERT (2026-07-12): quick-test 15 epoch cho thấy
# ViSoBERT hội tụ nhanh hơn PhoBERT nhiều và dev F1 đỉnh cao hơn (37.58% vs 30.50%)
# trên domain mobile — xem experiments/quicktest/mobile_visobert_seg/results.json.
# Kết quả PhoBERT cũ (30 epoch, đã fix bug + segmentation) được giữ lại tham khảo ở
# experiments/results_phobert_final/ (KHÔNG bị ghi đè vì tên config khác nhau).
#
# Khuyến nghị: chạy thử 1 config nhỏ trước để đo tốc độ thật trên máy bạn:
#   python3 train.py --domain mobile --loss focal --model visobert --epochs 2
# rồi nhân ra ước lượng tổng thời gian trước khi chạy full qua đêm.

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
# loss normalize
CONFIGS=(
  "ce false"      # C1 - baseline
  "ce true"       # C2 - + Text Normalization
  "focal false"   # C3 - + Focal Loss
  "focal true"    # C4 - + cả hai (full model)
)

mkdir -p logs
LOGFILE="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "Bắt đầu chạy toàn bộ ablation — EPOCHS=$EPOCHS, MODEL=$MODEL" | tee -a "$LOGFILE"
echo "Log chi tiết: $LOGFILE"
overall_start=$(date +%s)

total=0
done_count=0
skip_count=0

for domain in "${DOMAINS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    total=$((total+1))
  done
done

for domain in "${DOMAINS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    read -r loss normalize <<< "$cfg"
    norm_flag=""
    suffix=""
    if [ "$normalize" = "true" ]; then
      norm_flag="--normalize"
      suffix="_norm"
    fi
    config_name="${domain}_loss-${loss}${suffix}${MODEL_SUFFIX}"
    result_file="experiments/results/${config_name}/results.json"

    if [ -f "$result_file" ]; then
      echo "⏭  [$config_name] đã có kết quả, bỏ qua." | tee -a "$LOGFILE"
      skip_count=$((skip_count+1))
      continue
    fi

    echo "" | tee -a "$LOGFILE"
    echo "=== [$config_name] bắt đầu $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOGFILE"
    start=$(date +%s)
    "$PYTHON_BIN" train.py --domain "$domain" --loss "$loss" $norm_flag --model "$MODEL" --epochs "$EPOCHS" 2>&1 | tee -a "$LOGFILE"
    end=$(date +%s)
    mins=$(( (end-start)/60 ))
    echo "=== [$config_name] xong, mất ${mins} phút ===" | tee -a "$LOGFILE"
    done_count=$((done_count+1))
  done
done

overall_end=$(date +%s)
hours=$(( (overall_end-overall_start)/3600 ))
echo "" | tee -a "$LOGFILE"
echo "✅ HOÀN TẤT — chạy $done_count/$total config (bỏ qua $skip_count đã có sẵn), tổng ${hours} giờ." | tee -a "$LOGFILE"
echo "Chạy '$PYTHON_BIN summarize_results.py' để xem bảng tổng hợp."
