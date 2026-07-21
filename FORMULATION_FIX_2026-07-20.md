# Phát hiện & sửa lỗi formulation — 2026-07-20

> TL;DR: baseline của đồ án không "yếu" — nó đang bị chấm trên một thang đo
> khắc nghiệt hơn hẳn thang đo của paper. Nguyên nhân là task được mô hình hoá
> sai (BIO span tagging thay vì target-aspect pair classification).

---

## 1. Vấn đề

Suốt từ 2026-07-11 đến 2026-07-16, mọi lần chạy đều cho macro F1 thấp hơn
baseline ViTASD 27–43 điểm, bất kể đã thử:

- fix bug padding/Unicode offset (commit `61a912c`)
- tăng epoch 10 → 20 → 30
- đổi backbone PhoBERT → ViSoBERT
- bật/tắt word segmentation
- đổi class-weighting strategy

| Domain | PhoBERT+CE (baseline tái lập) | ViSoBERT tốt nhất | ViTASD (paper) |
|---|---|---|---|
| Mobile | 28.48% | 34.56% | **61.77%** |
| Restaurant | 5.95% | 13.29% | **41.12%** |
| Hotel | 9.78% | 20.80% | **52.64%** |

Việc gap không nhúc nhích qua từng ấy thay đổi là dấu hiệu vấn đề nằm ở tầng
cao hơn hyperparameter — ở chính cách định nghĩa bài toán.

## 2. Bằng chứng: formulation đúng là pair classification

**(a) Có field trong dataset bị code bỏ qua.** `restaurant.jsonl` và
`hotel.jsonl` chứa key `labels` (số nhiều) mà `train.py` không đọc:

```
"{FACILITIES#DESIGN&FEATURES, negative}, {SERVICE#GENERAL, positive}, ..."
```

Dạng cặp (aspect, sentiment), **không có char offset**. Nếu task gốc là span
tagging thì field này vô nghĩa.

**(b) Số "500,000+ target-aspect pairs" trong abstract không khớp số span.**
Toàn dataset chỉ có ~18,800 span annotation:

| Domain | span annotation | comments × aspects = pairs |
|---|---|---|
| Mobile | 5,091 | 2000 × 10 = 20,000 |
| Restaurant | 7,231 | 2000 × 27 = 54,000 |
| Hotel | 6,461 | 2000 × 54 = 108,000 |

Chỉ cách đếm theo cặp mới ra được bậc độ lớn mà paper báo cáo.

**(c) Các model paper đem ra so sánh đều thuộc họ pair-classification:**
CG-BERT, QACG-BERT, BERT-pair-QA, BERT-pair-NLI — tất cả nhận input (câu,
aspect) và xuất 1 nhãn sentiment (kiểu SentiHood, Saeidi et al. 2016). Không
model nào là sequence labeling.

## 3. Vì sao điều này giải thích được toàn bộ gap

Số lớp dùng để tính macro F1 chênh nhau một trời một vực:

| Domain | Số lớp — BIO (cũ) | Số lớp — pair (đúng) |
|---|---|---|
| Mobile | 59 | 4 (macro tính trên 3) |
| Restaurant | 127 | 4 |
| Hotel | **233** | 4 |

Macro F1 trên 233 lớp span-level exact-match, phần lớn chỉ có vài chục mẫu, là
con số hoàn toàn khác với macro F1 trên 3 lớp sentiment. Để ý domain càng nhiều
lớp thì F1 càng sụp (Hotel 233 lớp → 9.78%) — đúng như dự đoán.

## 4. Thay đổi đã thực hiện

| File | Trạng thái | Nội dung |
|---|---|---|
| `train_pair.py` | **mới** | Entry point theo formulation đúng |
| `run_pair_ablation.sh` | **mới** | Chạy 4 config × 3 domain, skip config đã có kết quả |
| `summarize_pair_results.py` | **mới** | Bảng tổng hợp so với baseline + xuất CSV |
| `train.py` | giữ nguyên | Không xoá — kết quả cũ vẫn dùng được để bàn luận trong report |

### Thiết kế `train_pair.py`

- Mỗi (comment, aspect) → 1 sample, nhãn ∈ {none, positive, negative, neutral}
- Input BERT-pair: `[CLS] câu [SEP] mô tả aspect [SEP]`
- Model: encoder + classifier trên vector `[CLS]` (thay vì per-token như cũ)
- **Metric: macro F1 trên 3 lớp sentiment, LOẠI `none`** — đây là con số so
  sánh trực tiếp được với 61.77 / 41.12 / 52.64
- Split ở **mức comment** (không phải mức cặp) để tránh leak — đã verify leak = 0
- 2 module đóng góp (Normalization + Imbalanced Learning) giữ nguyên, gắn vào
  không sửa gì

### Module 2 giờ hợp lý hơn hẳn

Phân bố lớp thực tế trên train set sau khi đổi formulation:

| Domain | none | positive | negative | neutral |
|---|---|---|---|---|
| Mobile | 76.5% | 14.8% | 7.8% | 1.0% |
| Restaurant | 90.6% | 7.7% | 1.5% | 0.2% |
| Hotel | 94.6% | 3.9% | 1.4% | **0.04%** (28 mẫu) |

Neutral hiếm tới mức 28/75,600 mẫu ở Hotel — đúng bài toán mà Focal Loss và
class weighting sinh ra để giải. Ở formulation BIO cũ, đóng góp của Module 2 bị
lu mờ vì vấn đề chi phối là số lớp quá nhiều, không phải imbalance.

## 5. Trạng thái kiểm thử

| Hạng mục | Trạng thái |
|---|---|
| Syntax `train_pair.py` / `summarize_pair_results.py` | ✅ compile OK |
| Syntax `run_pair_ablation.sh` | ✅ `bash -n` OK |
| Logic tách aspect, gán nhãn cặp, phân bố lớp | ✅ verify trên cả 3 domain |
| Split leak (comment trùng giữa train/dev/test) | ✅ = 0 ở cả 3 domain |
| **Chạy training thật** | ⬜ **CHƯA** — cần chạy trên máy M3 |

Chưa smoke-test được training thật vì môi trường sandbox không cài được
`torch`/`transformers`. Toàn bộ phần logic thuần Python đã verify bằng dữ liệu
thật, nhưng phần forward/backward của model thì chưa chạy lần nào.

## 6. Việc cần làm tiếp

```bash
cd "Documents/Master's Degree IT- UIT/Kì 2/Xử lí ngôn ngữ tự nhiên/VITASA_Enhanced"

# 1. Smoke test (~vài phút) — xác nhận pipeline chạy được
python3 train_pair.py --domain mobile --loss ce --epochs 1 --subsample 0.1

# 2. Baseline thật trên domain nhẹ nhất — đây là số cần nhìn đầu tiên
python3 train_pair.py --domain mobile --loss ce --epochs 3

# 3. Nếu baseline đã về gần 61.77% → chạy full ablation
./run_pair_ablation.sh
```

Ước lượng thời gian trên M3 (MPS): mobile ~14k cặp/epoch nhanh nhất, hotel
~75.6k cặp/epoch nặng nhất (có thể vài tiếng cho 3 epoch) — nên chạy mobile
trước để lấy tín hiệu.

### Nếu baseline vẫn chưa chạm 61.77%

Còn 2 chi tiết chưa xác nhận được vì tác giả không công bố source code
(repo `kh4nh12/ViTASA` chỉ có 3 file dataset, mục Usage/Evaluation để trống):

1. **Cách xây auxiliary sentence.** Hiện dùng mô tả aspect dạng phẳng
   ("room amenities design features"). BERT-pair-QA/NLI dùng câu hỏi tự nhiên
   ("bạn nghĩ gì về tiện nghi phòng?") — có thể cho kết quả khác.
2. **Có tính lớp `none` vào macro F1 hay không.** Hiện loại `none` (3 lớp).
   Nếu paper tính cả 4 lớp thì điểm sẽ khác đáng kể — `results.json` có lưu sẵn
   cả `macro_f1_with_none` để đối chiếu.

Cả hai nên hỏi trực tiếp thầy Kiệt (đồng tác giả ViTASA) để chốt.
