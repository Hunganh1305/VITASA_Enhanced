#!/bin/bash
# run_pair_ablation.sh — ✨ NEW (2026-07-20) chạy ablation study với
# formulation ĐÚNG (pair classification).
#
# Thay thế run_all_local.sh (dùng formulation BIO cũ — KHÔNG dùng nữa).
# Xem FORMULATION_FIX_2026-07-20.md để chi tiết tại sao formulation cũ sai.
#
# Cách dùng:
#   ./run_pair_ablation.sh                    # full: 4 config × 3 domain
#   ./run_pair_ablation.sh mobile             # chỉ 1 domain
#   EPOCHS=5 ./run_pair_ablation.sh mobile    # đổi số epoch
#
# Script tự bỏ qua config đã có results.json — nên chạy lại an toàn sau khi bị
# ngắt giữa chừng.

set -u

EPOCHS="${EPOCHS:-3}"
MODEL="${MODEL:-visobert}"
DOMAINS="${1:-mobile restaurant hotel}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
LOG="logs/run_pair_${TS}.log"

echo "Ablation pair-classification — EPOCHS=$EPOCHS, MODEL=$MODEL" | tee "$LOG"
echo "Domains: $DOMAINS" | tee -a "$LOG"
echo "Log: $LOG"

# 4 config ablation: "tên|flags"
CONFIGS=(
  "C1_baseline|--loss ce"
  "C2_norm|--loss ce --normalize"
  "C3_imbalanced|--loss focal"
  "C4_full|--loss focal --normalize"
)

for domain in $DOMAINS; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"
    flags="${entry#*|}"

    # tên thư mục phải khớp cách train_pair.py sinh config_name
    suffix=""
    [[ "$flags" == *"--normalize"* ]] && suffix="_norm"
    loss=$(echo "$flags" | grep -o -- '--loss [a-z_]*' | awk '{print $2}')
    outdir="experiments/results_pair/pair_${domain}_loss-${loss}${suffix}_${MODEL}"

    if [ -f "$outdir/results.json" ]; then
      echo "⏭  [$domain/$name] đã có kết quả, bỏ qua." | tee -a "$LOG"
      continue
    fi

    echo "" | tee -a "$LOG"
    echo "=== [$domain/$name] bắt đầu $(date '+%F %T') ===" | tee -a "$LOG"
    start=$(date +%s)

    python3 train_pair.py --domain "$domain" $flags \
      --model "$MODEL" --epochs "$EPOCHS" 2>&1 | tee -a "$LOG"

    mins=$(( ($(date +%s) - start) / 60 ))
    echo "=== [$domain/$name] xong, mất ${mins} phút ===" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "Hoàn tất. Tổng hợp kết quả:" | tee -a "$LOG"
python3 summarize_pair_results.py 2>&1 | tee -a "$LOG"
