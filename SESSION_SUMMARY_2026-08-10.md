# Session Summary — 2026-08-10

Đề tài: *Enhancing Vietnamese TASA with Text Normalization and Imbalanced Learning*

> Bản cập nhật cho `SESSION_SUMMARY_2026-07-11.md` — nhiều thứ đã đổi từ 11/7,
> quan trọng nhất là **formulation của task đã được sửa lại** (xem
> `FORMULATION_FIX_2026-07-20.md`), nên toàn bộ kết quả BIO-tagging cũ
> (`train.py`, `experiments/results/`) không còn là hướng chính nữa.

---

## 1. Tóm tắt nhanh trạng thái

| Phần | Trạng thái |
|---|---|
| Text Normalization module | ✅ Done — không đổi từ 11/7 |
| Imbalanced Learning module | ✅ Done — không đổi từ 11/7 |
| `train.py` (BIO tagging — formulation SAI, đã bỏ) | ⚠️ Giữ lại để bàn luận trong report, không dùng số liệu để so baseline |
| `train_pair.py` (pair classification — formulation ĐÚNG) | ✅ Code xong + đã thêm fp16 hôm nay |
| Ablation domain **hotel** (4 config × 5 epoch, ViSoBERT) | 🟡 Có kết quả nhưng **chưa converge** + **sai backbone** so với yêu cầu của thầy (xem mục 3), chưa commit vào repo |
| Ablation domain **mobile / restaurant** (pair formulation) | ⬜ Chưa chạy |
| Ablation với backbone **PhoBERT** (theo yêu cầu của thầy) | ⬜ Chưa chạy — đây là bộ số liệu chính cần có |
| Đối chiếu siêu tham số với paper gốc | ⬜ Chưa làm — **việc cần làm đầu tiên** |
| `colab_pair_ablation.ipynb` | ✅ Sẵn sàng chạy full (10 epoch, batch 64, fp16) |
| Bảng 2 trong `report/paper_vi.md` | ⬜ Vẫn còn trống (`-` toàn bộ) |

---

## 2. Việc đã làm từ 11/7 đến nay (tóm tắt các mốc lớn)

### 20/7 — Phát hiện formulation sai (bước ngoặt quan trọng nhất)
Toàn bộ kết quả BIO-tagging (`train.py`) thấp hơn baseline ViTASD 27-43 điểm dù
đã thử tăng epoch, đổi backbone, đổi class-weighting — vì đang mô hình hoá task
sai (span tagging thay vì target-aspect **pair classification**). Chi tiết đầy
đủ trong `FORMULATION_FIX_2026-07-20.md`. Đã viết `train_pair.py` thay thế,
giữ nguyên 2 module đóng góp (Normalization + Imbalanced Learning), chỉ đổi
cách mô hình hoá bài toán + metric (macro F1 trên 3 lớp sentiment, loại `none`
— so được trực tiếp với 61.77/41.12/52.64% của paper gốc).

### 10/8 (hôm nay)
- Nhận file `results.json` của 4 config ablation trên domain **hotel**
  (CE/Focal × normalize on/off, ViSoBERT, 5 epoch). Kết quả tốt nhất: **Focal +
  Normalize = 36.03% macro F1** (so với baseline ViTASD hotel 52.64%). Xu
  hướng đúng như kỳ vọng (Focal > CE, normalize > không normalize), nhưng
  **dev F1 và train loss vẫn đang cải thiện đều ở epoch 5** → model chưa hội
  tụ, số tuyệt đối chưa dùng để kết luận cuối cùng được, chỉ dùng để xem xu hướng.
- Phát hiện `train_pair.py` **chưa hỗ trợ mixed precision** → đã thêm flag
  `--fp16` (autocast + GradScaler, tự bỏ qua nếu không phải CUDA) để train
  nhanh hơn, cho phép tăng epoch mà không tốn quá nhiều thời gian Colab.
- Cập nhật `colab_pair_ablation.ipynb`: 3 lệnh train (smoke test, baseline-only,
  full ablation) đều đã thêm `--fp16`, giữ `EPOCHS=10`, `BATCH_SIZE=64` sẵn có.
- **Chưa push lên GitHub được** — sandbox hiện tại không có quyền ghi vào
  `.git/` của repo (bị chặn ở tầng mount) và đang dính `index.lock` kẹt. Cần
  chạy `git add/commit/push` từ máy thật (lệnh cụ thể ở cuối file).

---

## 3. Feedback của thầy Kiệt (buổi báo cáo tiến độ) — ƯU TIÊN CAO NHẤT

> Ghi lại từ note của buổi báo cáo. Lúc đó mới chỉ có kết quả mobile, chưa chạy
> restaurant/hotel. 3 ý chính:

### F1 — Đổi backbone chính về PhoBERT (không dùng ViSoBERT làm baseline so sánh)
Tác giả ViTASD dùng **PhoBERT**. Nếu mình dùng ViSoBERT thì không cùng backbone
→ không tách được phần cải thiện nào đến từ 2 module đóng góp, phần nào chỉ đến
từ việc đổi backbone mạnh hơn. Toàn bộ kết quả hiện có (kể cả hotel 4 config)
đều đang chạy trên ViSoBERT → **không dùng để so trực tiếp với 61.77/41.12/52.64%**.

**Cách xử lý đề xuất:** giữ cả hai, nhưng đổi vai trò —
- **PhoBERT = backbone chính**, dùng cho C1-C4 để so sánh công bằng với ViTASD.
- **ViSoBERT = 1 dòng ablation bổ sung** ("+ ViSoBERT backbone"), trình bày như
  một cải tiến riêng chứ không phải điều kiện mặc định.

Code đã hỗ trợ sẵn: `--model phobert` (có trong `MODEL_REGISTRY`), không cần sửa gì.

### F2 — Đọc lại kỹ phần Experimental Settings của paper để lấy đúng siêu tham số
Mình từng nói với thầy là "không tìm được cách chạy baseline giống tác giả",
thầy khẳng định **trong bài báo chắc chắn có hướng dẫn thông số setting** → cần
đọc lại paper thật kỹ, không đoán.

Siêu tham số hiện tại của `train_pair.py` **đều là mình tự chọn**, cần đối chiếu:

| Tham số | Giá trị hiện tại | Nguồn |
|---|---|---|
| Backbone | ViSoBERT (mặc định) | tự chọn → **phải đổi PhoBERT** (F1) |
| Learning rate | 2e-5 | tự chọn |
| Batch size | 32 (Colab dùng 64) | tự chọn |
| Max sequence length | 160 | tự chọn |
| Epochs | 5 (đã chạy) / 10 (notebook) | tự chọn |
| Optimizer | AdamW, weight_decay 0.01 | tự chọn |
| Scheduler | linear warmup 10% | tự chọn |
| Dropout | 0.1 | tự chọn |
| Focal gamma | 2.0 | tự chọn |
| Train/dev/test split | 70/10/20, random, seed 42 | tự chọn |
| Auxiliary sentence | mô tả aspect dạng phẳng | tự chọn |
| Macro F1 có tính lớp `none`? | KHÔNG (3 lớp) | tự chọn |

3 dòng cuối là quan trọng nhất — split khác nhau thì con số không so được, và
cách xây auxiliary sentence + có tính `none` hay không có thể làm lệch điểm rất
nhiều (xem `FORMULATION_FIX_2026-07-20.md` mục 6, đúng 2 câu hỏi này đã ghi từ 20/7).

**Việc cần làm:** lấy full text paper (ScienceDirect qua tài khoản UIT, hoặc
OpenReview) → đọc mục Experimental Settings / Implementation Details → điền cột
"giá trị của tác giả" vào bảng trên → sửa lại default trong `train_pair.py` cho khớp.
Repo GitHub `kh4nh12/ViTASA` chỉ có 3 file dataset, mục Usage để trống → không
lấy được từ code, bắt buộc phải đọc paper.

### F3 — Báo cáo phải show HẾT kết quả trong quá trình training, không chỉ best
Buổi trước mình chỉ show config tốt nhất. Thầy yêu cầu trình bày đầy đủ cả các
kết quả kém, để thấy được quá trình và tính trung thực của thực nghiệm.

Trạng thái hiện tại: `results.json` **đã có sẵn `epoch_logs`** (loss + dev macro
F1 + accuracy từng epoch) → dữ liệu đủ, chỉ thiếu phần trình bày.

**Việc cần làm:**
- Vẽ **training curve** (dev macro F1 theo epoch) cho tất cả config, không chỉ best.
- Bảng ablation đầy đủ: mọi config × mọi domain, kể cả config cho kết quả kém.
- Ghi lại cả các hướng đã thử và **thất bại** (BIO tagging, PhoBERT vs ViSoBERT,
  các lần tăng epoch) — phần này đã có sẵn trong `FORMULATION_FIX_2026-07-20.md`,
  rất hợp để đưa vào báo cáo như một mục "quá trình thực nghiệm".
- Cân nhắc log thêm **test metric theo từng epoch** (hiện chỉ eval test 1 lần ở cuối).

---

## 4. Vấn đề còn tồn đọng

1. **Kết quả hotel chưa được commit vào repo** — 4 file `results.json` đang
   nằm ở nơi khác (bạn gửi qua chat), `experiments/results_pair/` trong repo
   thật hiện **đang trống**. Cần copy vào đúng vị trí rồi commit.
2. **Chưa đủ epoch** — cùng vấn đề như hồi 11/7 (lúc đó là BIO tagging, giờ là
   pair classification), nhưng lần này dễ fix hơn vì đã có `--fp16` + đã biết
   trước cần ít nhất ~10 epoch.
3. **Chưa có kết quả mobile/restaurant** theo formulation mới — chỉ mới có hotel.
4. **Bảng 2 trong `report/paper_vi.md` vẫn trống** — chưa điền được vì chưa có
   bộ số liệu hội tụ đầy đủ 3 domain.
5. **2 câu hỏi chưa chốt với thầy Kiệt** (ghi trong `FORMULATION_FIX_2026-07-20.md`
   mục 6): cách xây auxiliary sentence, và macro F1 có tính lớp `none` không.

---

## 5. Đề xuất thứ tự làm tiếp (đã sắp lại theo feedback của thầy)

**Bước 0 — làm TRƯỚC khi chạy thêm bất kỳ experiment nào (F2):**
Lấy full text paper ViTASA → đọc mục Experimental Settings → điền cột "giá trị
tác giả" vào bảng siêu tham số ở mục 3 → sửa default trong `train_pair.py`.
Lý do phải làm trước: nếu sau này mới phát hiện split/LR/epoch khác tác giả thì
**toàn bộ kết quả đã chạy phải bỏ đi chạy lại**.

1. **Đổi backbone chính sang PhoBERT** (F1) — chạy C1-C4 với `--model phobert`
   trên cả 3 domain. Đây là bộ số liệu chính để so với ViTASD.
2. **Giữ ViSoBERT như 1 dòng ablation bổ sung** — kết quả hotel 5-epoch hiện có
   vẫn dùng được cho phần này (ghi rõ là 5 epoch, chưa converge).
3. Chạy đủ epoch theo thông số tác giả (hoặc ≥10 nếu paper không nêu rõ), dùng
   `--fp16` + batch 64 để rút ngắn thời gian.
4. Copy `experiments/results_pair/*/results.json` về đúng vị trí trong repo,
   commit (lệnh git ở cuối file).
5. Chạy `summarize_pair_results.py` → bảng ablation **đầy đủ mọi config** (F3),
   không lọc bỏ config kém.
6. Vẽ training curve từ `epoch_logs` cho tất cả config (F3).
7. Điền Bảng 2 trong `report/paper_vi.md` + thêm mục "quá trình thực nghiệm"
   (dùng lại nội dung `FORMULATION_FIX_2026-07-20.md`).
8. Hỏi thầy Kiệt 2 câu còn treo nếu paper không nói rõ: cách xây auxiliary
   sentence, và macro F1 có tính lớp `none` không.
9. Dọn file untracked (`.test_write`), xác nhận có giữ
   `SESSION_SUMMARY_2026-07-11.md` không.

### Lệnh git cần chạy trên máy thật (không chạy được từ sandbox này)

```bash
cd "/Users/hunganh130502/Documents/Master's Degree IT- UIT/Kì 2/Xử lí ngôn ngữ tự nhiên/VITASA_Enhanced"
rm -f .git/index.lock
git add train_pair.py colab_pair_ablation.ipynb SESSION_SUMMARY_2026-08-10.md
git commit -m "Add fp16 support to train_pair.py + session summary 2026-08-10"
git push
```
