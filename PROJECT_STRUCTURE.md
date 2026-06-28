# Project Structure — VITASA_Enhanced

Đề tài: *Enhancing Vietnamese Targeted Aspect Sentiment Analysis with Social
Media Text Normalization and Imbalanced Learning* (GVHD: TS. Nguyễn Văn Kiệt).

## Cấu trúc hiện tại

Nguyên tắc: **module = thư mục** (logic + data + tests đóng gói riêng), còn
**script để chạy nằm ở root** (train.py, demo_*.py) — import vào các module.

```
VITASA_Enhanced/
├── README.md                     # project overview (English)
├── PROJECT_STRUCTURE.md          # file này
│
├── train.py                      # ★ ENTRY POINT — fine-tune PhoBERT, 4 config ablation
├── demo_normalizer.py            # demo CLI module Text Normalization
├── demo_losses.py                # demo CLI module Imbalanced Learning
│
├── text_normalization/           # Contribution #1 — DONE (v1, rule-based)
│   ├── normalizer.py              # class TextNormalizer
│   ├── data/
│   │   └── teencode_dict.json     # từ điển teencode/viết tắt (151 entries)
│   ├── tests/
│   │   └── test_normalizer.py     # unit test
│   └── README.md                  # thiết kế, cách dùng, limitation
│
├── imbalanced_learning/          # Contribution #2 — DONE (v1)
│   ├── losses.py                  # FocalLoss, WeightedCrossEntropyLoss, compute_class_weights
│   ├── tests/
│   │   └── test_losses.py         # unit test
│   └── README.md                  # công thức, cách dùng, mapping vào config ablation
│
├── baseline/                     # Reproduce ViTASD
│   └── data/                      # ViTASA dataset (mobile/restaurant/hotel)
│       ├── mobile/mobile.jsonl
│       ├── restaurant/restaurant.jsonl
│       └── hotel/hotel.jsonl
│
└── ViTASA/                       # raw source clone (github.com/kh4nh12/ViTASA)
```

> `pytest` → 32 test pass. Chạy demo: `python3 demo_normalizer.py` /
> `python3 demo_losses.py`. Train: `python3 train.py --domain mobile --loss focal --normalize`.

## Cấu trúc dự kiến (khi mở rộng thêm)

```
NLP/
├── PROJECT_STRUCTURE.md
├── train.py                       # ENTRY POINT — DONE
├── demo_normalizer.py             # DONE
├── demo_losses.py                 # DONE
├── run_ablation.py                # chạy batch nhiều config — TODO
│
├── text_normalization/            # Contribution #1 — DONE
├── imbalanced_learning/           # Contribution #2 — DONE
│
├── baseline/                      # Reproduce ViTASD
│   └── data/                      # ViTASA dataset (mobile/restaurant/hotel) — DONE
│
├── experiments/
│   └── results/                   # log, checkpoint, results.json (train.py tự ghi) — TODO
│
└── report/                        # Báo cáo / luận văn — đang viết song song
    ├── report_en.md
    ├── report_vi.md
    └── presentation_script.md
```

## Trạng thái từng phần

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Text Normalization Module | ✅ Done (v1) | Rule-based, đã test, chưa chạy trên dữ liệu ViTASA thật |
| Imbalanced Learning (Focal Loss + Weighted CE) | ✅ Done (v1) | Đã test (13 test), chưa tune gamma/beta trên dữ liệu thật |
| Dataset ViTASA | 🟡 Đang setup | Clone từ [github.com/kh4nh12/ViTASA](https://github.com/kh4nh12/ViTASA), đưa vào `baseline/data/` |
| Baseline reproduction (ViTASD) | 🟡 Đang làm | `train.py` đã viết xong (PhoBERT + BIO tagging), chờ chạy trên Colab (GPU) |
| Ablation study (4 config) | ⬜ Chưa làm | train.py đã hỗ trợ flag `--loss` / `--normalize`; cần `run_ablation.py` để chạy batch |
| Report draft (EN/VI) | 🟡 Đang viết song song | Một số phần đã có placeholder |
| Sarcasm detection | ⛔ Out of scope | Ghi nhận là future work / Module 3 cho luận văn đầy đủ |

## Bước tiếp theo (theo thứ tự ưu tiên)

1. **[ĐANG LÀM]** Clone dataset ViTASA về `baseline/data/`:
   ```bash
   git clone https://github.com/kh4nh12/ViTASA.git
   # Copy thư mục data/ vào baseline/data/
   ```
2. Chạy `text_normalization` trên sample thật, đo % token thay đổi, bổ sung dictionary.
3. Tính `compute_class_weights` trên label distribution thật của ViTASA.
4. Viết `baseline/train.py` + chạy 1 lần trên Google Colab (train overnight).
5. Thiết kế `experiments/run_ablation.py` theo 5 config đã map trong
   `imbalanced_learning/README.md`.
