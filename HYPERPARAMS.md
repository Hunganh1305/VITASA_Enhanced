# Bảng siêu tham số — chuẩn bị chạy lại baseline theo feedback của thầy

> Cập nhật: 2026-08-10
> Mục đích: chốt thông số trước khi chạy lại ablation với backbone **PhoBERT**
> (feedback F1 + F2 của thầy Kiệt, xem `SESSION_SUMMARY_2026-08-10.md` mục 3).

---

## ⚠️ Trạng thái: CHƯA lấy được thông số chính thức của tác giả

Đã thử tìm nhưng không lấy được phần Experimental Settings:

| Nguồn | Kết quả |
|---|---|
| ScienceDirect (bản chính thức, CSL 2025) | ❌ Paywall |
| OpenReview `ZdlrKpRg9p` | ❌ Chặn truy cập tự động |
| GitHub `kh4nh12/ViTASA` | ❌ Chỉ có 3 file `.jsonl`, mục Usage/Evaluation để trống |
| ResearchGate | ❌ "Request full-text" |
| OpenReview (mở qua browser thật, 10/8) | ⚠️ Chỉ có abstract + highlights, **không có PDF** (link `/pdf?id=` trả 404) |
| ScienceDirect (mở qua browser thật, 10/8) | ⚠️ Chỉ đọc được preview: Abstract, Introduction, section snippets, References. Full text vẫn khoá |

→ **Cần bạn tải PDF bằng tài khoản UIT**, bỏ vào folder project. Sau đó mình đọc
mục Experimental Settings / Implementation Details và điền cột "Tác giả" bên dưới.

### Thông tin thu được từ phần preview đọc được (10/8)

1. ✅ **Split = 7:1:2 random** — xác nhận từ README repo, trùng khớp code hiện tại.
2. ✅ **Backbone = PhoBERT** — xác nhận từ Section 4: *"instead of BERT (Devlin
   et al., 2018) embeddings, our method utilizes PhoBERT (Nguyen and Tuan Nguyen,
   2020) embeddings"*.
3. 💡 **Kiến trúc ViTASD dựa trên Zhang et al. (2020)**, thay BERT bằng PhoBERT.
   Keywords của paper có *"Multi-head attention"* → ViTASD **không phải** BERT-pair
   thuần như `train_pair.py` đang làm, mà có thêm lớp attention phía trên.
   → Cần đọc Fig. 6 + Section 4 để biết kiến trúc chính xác. Đây có thể là
   nguyên nhân còn lại của gap, ngoài chuyện backbone.
4. ℹ️ Repo `kh4nh12/ViTASD` (ghi trong footnote paper) **redirect về chính repo
   `ViTASA`** — không có repo code riêng. Mục Usage/Evaluation vẫn để trống, và
   "model checkpoints, source code" như abstract hứa thì **không có trên repo**.
   → Đáng hỏi thầy Kiệt trực tiếp.

Trong lúc chờ, cột **"Đề xuất chạy"** là giá trị an toàn theo chuẩn chung của
các paper fine-tune PhoBERT / BERT-pair — đủ để chạy được baseline hợp lý.

---

## 1. Bảng siêu tham số

Ký hiệu: 🔴 = ảnh hưởng LỚN tới kết quả, bắt buộc phải khớp paper |
🟡 = ảnh hưởng vừa | 🟢 = ít ảnh hưởng, giữ nguyên được

| # | Tham số | Hiện tại (code) | **Đề xuất chạy** | Tác giả | Ghi chú |
|---|---|---|---|---|---|
| 🔴 1 | Backbone | `visobert` | **`phobert`** (`vinai/phobert-base-v2`) | PhoBERT ✅ | Paper nói rõ dùng PhoBERT embeddings — đã xác nhận từ abstract |
| ✅ 2 | Train/dev/test split | 70/10/20, random, seed 42 | **giữ nguyên** | **7:1:2 random ✅** | **ĐÃ XÁC NHẬN** từ README repo: *"We divide the dataset randomly into training, development and testing sets with a ratio of 7:1:2"* → code mình đang làm ĐÚNG |
| 🔴 3 | Macro F1 có tính lớp `none`? | KHÔNG (3 lớp) | KHÔNG (giữ) | ❓ | `results.json` lưu sẵn cả `macro_f1_with_none` để đối chiếu 2 chiều |
| 🔴 4 | Auxiliary sentence | Mô tả aspect dạng phẳng | Giữ nguyên | ❓ | BERT-pair-QA/NLI dùng câu hỏi tự nhiên → có thể khác đáng kể |
| 🟡 5 | Learning rate | `2e-5` | **`2e-5`** | ❓ | Chuẩn cho fine-tune BERT-base. Paper thường dùng 2e-5 hoặc 3e-5 |
| 🟡 6 | Batch size | 32 (Colab đang để 64) | **32** | ❓ | Xem mục 3 bên dưới — nên thống nhất 1 giá trị cho mọi config |
| 🟡 7 | Epochs | 5 (đã chạy) / 10 (notebook) | **10** | ❓ | 5 epoch chắc chắn chưa hội tụ (đã verify từ epoch_logs) |
| 🟡 8 | Max sequence length | 160 | **160** | ❓ | PhoBERT-base-v2 tối đa 256 token; 160 phủ ~99% comment ViTASA |
| 🟡 9 | Word segmentation | ON (underthesea) | **ON — bắt buộc** | ✅ ngầm định | PhoBERT pre-train trên text ĐÃ tách từ. Tắt = tụt điểm nặng. **Không truyền `--no-segment`** |
| 🟢 10 | Optimizer | AdamW, weight_decay 0.01 | Giữ | ❓ | Chuẩn của HuggingFace |
| 🟢 11 | LR scheduler | Linear, warmup 10% | Giữ | ❓ | Chuẩn |
| 🟢 12 | Dropout | 0.1 | Giữ | ❓ | Mặc định BERT |
| 🟢 13 | Grad clipping | 1.0 | Giữ | ❓ | Chuẩn |
| 🟢 14 | Seed | 42 | Giữ | — | Cố định để tái lập được |
| 🟢 15 | Focal gamma | 2.0 | Giữ | — | Lin et al. 2017. **Của mình, không phải của tác giả** |
| 🟢 16 | Class weight strategy | `effective_number` | Giữ | — | Cui et al. 2019. **Của mình** |
| 🟢 17 | Mixed precision | `--fp16` (mới thêm) | **BẬT** | — | Chỉ tăng tốc, không đổi kết quả đáng kể |

> Dòng 15-17 là đóng góp của mình (Module 2), **không cần** khớp paper — đó là
> phần cải tiến. Chỉ dòng 1-14 mới cần khớp để so sánh công bằng.

---

## 2. Điều cần đổi so với lần chạy trước

Chỉ **1 thứ bắt buộc đổi**: backbone `visobert` → `phobert`.

Mọi thứ khác giữ nguyên là được. Lý do: mục tiêu chính lúc này là có bộ số liệu
PhoBERT để so với ViTASD, và giữ mọi tham số khác giống hệt lần chạy ViSoBERT
để dòng ablation "+ ViSoBERT backbone" vẫn so sánh được (chỉ khác đúng 1 biến).

---

## 3. Lưu ý về batch size (32 vs 64)

Notebook hiện để `BATCH_SIZE = 64` cho nhanh, nhưng batch size ảnh hưởng tới
kết quả (batch lớn → gradient ít nhiễu → có thể cần LR cao hơn).

Hai lựa chọn, chọn 1 rồi **giữ nguyên cho TẤT CẢ config**:

- **Ưu tiên độ chính xác:** `batch=32` + `--fp16`. fp16 đã bù lại tốc độ, nên
  vẫn nhanh hơn lần chạy 5-epoch cũ (batch 32, không fp16).
- **Ưu tiên tốc độ:** `batch=64` + `--fp16`. Nhanh nhất, nhưng phải ghi rõ trong
  báo cáo là batch 64 (khác giá trị thường thấy 32).

Điều tối kỵ: config này batch 32, config kia batch 64 → bảng ablation vô nghĩa.

---

## 4. File mới: `train_vitasd_baseline.py` — kiểm tra số liệu của tác giả

Vì kiến trúc ViTASD **không chỉ là PhoBERT + linear classifier** (paper có
keyword "Multi-head attention", inspired by Zhang et al. 2020), mình tạo file
riêng `train_vitasd_baseline.py` thay vì chỉ đổi `--model phobert` trong
`train_pair.py`:

- Model: PhoBERT encoder + 1 lớp multi-head self-attention (residual + LayerNorm)
  trên toàn bộ sequence output, rồi phân loại trên `[CLS]` — **best-effort**,
  không phải kiến trúc chính xác (Fig. 6 của paper bị khoá).
- Loss: CE thuần, **không** bật Normalization/Focal — đây là bản baseline gốc
  để kiểm tra số liệu, tách biệt hoàn toàn khỏi 2 module đóng góp.
- Log cả `macro_f1` (loại "none") và `macro_f1_with_none` mỗi epoch, để đối
  chiếu cả 2 khả năng vì chưa biết paper tính `none` hay không.
- File `results.json` xuất ra có 2 field `diff_vs_paper_excl_none` /
  `diff_vs_paper_incl_none` — nhìn thẳng vào đây để biết còn cách baseline
  bao xa.

**Chạy để kiểm tra:**
```bash
python3 train_vitasd_baseline.py --domain mobile --epochs 1 --subsample 0.1   # smoke test
python3 train_vitasd_baseline.py --domain mobile     --epochs 10 --fp16
python3 train_vitasd_baseline.py --domain restaurant --epochs 10 --fp16
python3 train_vitasd_baseline.py --domain hotel      --epochs 10 --fp16
```

**Cách đọc kết quả:**
- Nếu `macro_f1` (hoặc `macro_f1_with_none`) ra **gần** 61.77/41.12/52.64% →
  giả định kiến trúc + hyperparameter hợp lý, dùng file này làm baseline chính
  thức cho report, không cần đợi paper đầy đủ nữa.
- Nếu **vẫn cách xa** → xác nhận vấn đề nằm ở chi tiết chưa biết (kiến trúc
  MHA chính xác, auxiliary sentence, hoặc metric) chứ không phải hyperparameter
  vặt → phải có source code/paper gốc từ thầy Kiệt mới đi tiếp được, không nên
  đoán thêm.

⚠️ Sandbox hiện không cài `torch`/`transformers` nên mình mới chỉ verify được
cú pháp (`py_compile` pass), **chưa chạy thử forward/backward thật**. Bạn chạy
smoke test trước khi chạy full để chắc không có lỗi runtime.

---

## 5. Lệnh chạy (bộ ablation đầy đủ, sau khi baseline ở mục 4 đã khớp)

### Bước 1 — Smoke test (vài phút, xác nhận PhoBERT chạy được)

```bash
python3 train_pair.py --domain mobile --loss ce --model phobert \
    --epochs 1 --subsample 0.1 --batch-size 32 --fp16
```

### Bước 2 — Full ablation PhoBERT, 4 config × 3 domain

```bash
for domain in mobile restaurant hotel; do
  # C1 — Baseline (CE)
  python3 train_pair.py --domain $domain --model phobert --loss ce \
      --epochs 10 --batch-size 32 --fp16
  # C2 — + Text Normalization
  python3 train_pair.py --domain $domain --model phobert --loss ce --normalize \
      --epochs 10 --batch-size 32 --fp16
  # C3 — + Imbalanced Learning
  python3 train_pair.py --domain $domain --model phobert --loss focal \
      --epochs 10 --batch-size 32 --fp16
  # C4 — Full Model
  python3 train_pair.py --domain $domain --model phobert --loss focal --normalize \
      --epochs 10 --batch-size 32 --fp16
done
```

Thứ tự domain: **mobile → restaurant → hotel** (nhẹ → nặng). Mobile chỉ 10
aspect (~14k cặp/epoch), hotel 54 aspect (~75.6k cặp/epoch) nặng nhất — chạy
mobile trước để bắt lỗi sớm.

### Bước 3 — Dòng ablation bổ sung "+ ViSoBERT"

Đã có sẵn kết quả hotel 5 epoch. Chạy lại 10 epoch cho khớp với PhoBERT:

```bash
for domain in mobile restaurant hotel; do
  python3 train_pair.py --domain $domain --model visobert --loss focal --normalize \
      --epochs 10 --batch-size 32 --fp16
done
```

### Bước 4 — Tổng hợp

```bash
python3 summarize_pair_results.py
```

---

## 6. Checklist trước khi chạy

- [ ] **Chạy `train_vitasd_baseline.py` trước** (mục 4) — kiểm tra baseline có
      khớp paper không, trước khi tốn thời gian chạy full ablation ở mục 5
- [ ] Đã tải PDF paper ViTASA → đối chiếu dòng 🔴 3, 4 còn lại trong bảng mục 1
      (auxiliary sentence, macro F1 có tính `none`)
- [ ] Đã cài `underthesea` (bắt buộc cho PhoBERT word segmentation)
- [ ] Đã chốt batch size (32 hoặc 64) và dùng thống nhất mọi config
- [ ] Smoke test PhoBERT chạy không lỗi
- [ ] Có đủ dung lượng Drive (~187MB/checkpoint × số config)
