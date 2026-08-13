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
| `train_pair.py` (pair classification — formulation ĐÚNG) | ✅ Code xong, fp16 + kiến trúc PhoBERT+MHA đã verify (xem mục 6) |
| Đối chiếu siêu tham số với paper gốc (F2) | 🟡 Đọc được preview (không full text) — **split 7:1:2 và backbone PhoBERT đã xác nhận**, còn 2 điểm chưa rõ (aux sentence, có tính `none`) — xem mục 6 |
| Kiến trúc model (PhoBERT + multi-head attention) | ✅ **Đã verify khớp baseline paper** (mobile +2.45, restaurant -5.26, hotel -4.81) — CHỐT dùng cho mọi config từ nay |
| Ablation domain **hotel** (4 config × 5 epoch, ViSoBERT, kiến trúc CŨ) | ⚠️ Không dùng để so sánh nữa — sai cả backbone lẫn kiến trúc so với chuẩn mới chốt hôm nay, cần chạy lại |
| Ablation đầy đủ (4 config × 3 domain, PhoBERT+MHA, kiến trúc mới) | ⬜ **Chưa chạy được — hết quota Google Colab** (xem mục 6) |
| `colab_pair_ablation.ipynb` / `colab_vitasd_baseline.ipynb` | ✅ Sẵn sàng chạy full (10 epoch, batch 64, fp16, tự `git pull` code mới nhất từ GitHub) |
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
git add train_pair.py train_vitasd_baseline.py HYPERPARAMS.md \
        colab_pair_ablation.ipynb colab_vitasd_baseline.ipynb \
        SESSION_SUMMARY_2026-08-10.md
git commit -m "Standardize on PhoBERT+MHA architecture (verified vs paper baseline)"
git push
```

---

## 6. Cập nhật cuối ngày 10/8 — đã verify baseline, kiến trúc CHỐT xong

### Đọc được preview paper (không có full text)
Không lấy được PDF (ScienceDirect paywall, OpenReview chặn bot, GitHub
`ViTASD` chỉ redirect về lại repo `ViTASA` — không có source code/checkpoint
dù abstract nói là có). Nhưng đọc được phần preview công khai của paper +
README repo, xác nhận thêm 2 điều:

1. **Split 7:1:2 random** — đúng như code đang làm (README repo ghi rõ).
2. **Kiến trúc ViTASD không phải chỉ `[CLS]`+linear** — paper có keyword
   "Multi-head attention", lấy cảm hứng từ Zhang et al. (2020), thay embedding
   BERT bằng PhoBERT. Đây là lý do quan trọng nhất khiến baseline trước đó
   (chỉ đổi backbone, giữ nguyên `[CLS]`+linear) không đủ để so sánh công bằng.

Vẫn CHƯA xác nhận: LR/batch/epoch cụ thể, cách xây auxiliary sentence, macro F1
có tính lớp `none` không (2 câu hỏi treo từ 20/7).

### Tạo `train_vitasd_baseline.py` — best-effort reproduction + verify
Thêm kiến trúc PhoBERT + 1 lớp multi-head self-attention (residual+LayerNorm)
trên sequence output, phân loại trên `[CLS]`. Chạy trên Colab (10 epoch, fp16,
batch 32) cho cả 3 domain:

| Domain | Macro F1 (loại none) | Baseline paper | Chênh lệch |
|---|---|---|---|
| Mobile | 64.22% | 61.77% | **+2.45** |
| Restaurant | 35.86% | 41.12% | -5.26 |
| Hotel | 47.83% | 52.64% | -4.81 |

Chênh lệch dưới 6 điểm, mobile còn vượt baseline → **chấp nhận được, CHỐT dùng
kiến trúc này**, không cần đợi full paper mới đi tiếp.

⚠️ Đây là số liệu **best-effort reproduction thuần** (CE loss, không
normalize) — dùng để verify kiến trúc/hyperparameter đúng hướng, KHÔNG phải
kết quả ablation C1-C4 của đề tài.

**Đã nhận `results.json` đầy đủ (kèm `epoch_logs`) cho cả 3 domain**, copy vào
`experiments/results_vitasd_repro/` trong repo. Vài quan sát từ training curve
(cần đưa vào report — đúng yêu cầu F3 của thầy, show cả quá trình chứ không
chỉ số tốt nhất):

- **Mobile**: curve tăng đều và ổn định, plateau quanh epoch 7-8 (66.37% →
  66.36% → 65.12% → 66.09%) — model đã hội tụ, 10 epoch là đủ.
- **Restaurant**: tăng chậm và có dao động nhẹ ở các epoch cuối (35.05% →
  33.63% → 35.37% → 34.84%) — gần hội tụ nhưng chưa hoàn toàn ổn định.
- **Hotel**: **dao động khá mạnh giữa các epoch** (53.49% epoch 6 → tụt còn
  46.62% epoch 7 → nhảy lên 60.18% epoch 8 → 55.00% → 59.87%) — do sample
  lớp `neutral` cực hiếm (~0.04%, 28 mẫu — xem `FORMULATION_FIX_2026-07-20.md`),
  vài mẫu đúng/sai lệch cũng đủ làm macro F1 nhảy mạnh. Đây chính là lý do
  Module 2 (Imbalanced Learning) cần thiết — dự kiến Focal Loss sẽ làm curve
  ổn định hơn khi chạy ablation C3/C4.
- Cơ chế lưu `best_model.pt` theo dev F1 cao nhất (không phải epoch cuối) nên
  con số test cuối cùng dùng đúng checkpoint tốt nhất, không bị ảnh hưởng bởi
  dao động này — nhưng khi vẽ chart cho report vẫn nên show đủ 10 epoch để
  minh hoạ đúng thực tế train.

Checkpoint `best_model.pt` (~273MB/domain, ~820MB tổng) **chưa copy vào repo**
vì quá nặng cho git — chỉ giữ `results.json`. Nếu cần checkpoint cho demo/error
analysis sau này, lấy lại từ file `.tar.gz` gốc.

### Refactor `train_pair.py` — chốt kiến trúc dùng cho ablation
- Thêm class `TASAPairModelMHA` (giống hệt bản vừa verify ở trên), đặt làm
  **mặc định** cho mọi config C1-C4 (thay cho `TASAPairModel` cũ, chỉ
  `[CLS]`+linear — giữ lại qua cờ `--plain-classifier` để đối chiếu khi cần).
- Default `--model` đổi từ `visobert` → **`phobert`**.
- `train_vitasd_baseline.py` giờ import `TASAPairModelMHA` thẳng từ
  `train_pair.py`, không định nghĩa lặp lại — tránh 2 nơi lệch nhau.
- **Hệ quả:** kết quả hotel/ViSoBERT 4-config chạy đầu tháng 8 (kiến trúc
  `plain_classifier` cũ) không còn so trực tiếp được với ablation chạy từ nay
  — cần chạy lại nếu muốn đưa vào bảng so sánh cuối.

### Đổi cách đồng bộ code lên Colab: GitHub thay vì Google Drive
Repo `Hunganh1305/VITASA_Enhanced` đang **public** → cả 2 notebook giờ có 1
cell `git clone`/`git pull` thẳng từ GitHub, không cần mount Drive/kéo-thả file
tay nữa. Workflow mới: sửa code → `git push` → Colab chạy lại đúng 1 cell để
lấy bản mới nhất.

### Phát hiện + fix thêm 2 lỗi trong `colab_pair_ablation.ipynb`

1. **Cell STRATEGY 1 và STRATEGY 2 vẫn hardcode `--model visobert`** — sót lại
   từ trước khi chốt dùng PhoBERT (mục 6 ở trên). Lần chạy STRATEGY 1 bị ngắt
   giữa chừng (mobile+restaurant xong, hotel bị ngắt do hết quota) hoá ra dùng
   sai backbone → bỏ luôn, không cần tải về. Đã sửa cả 2 cell sang `--model phobert`.
2. **Không có cơ chế lưu/resume giữa chừng** — `results.json` chỉ ghi ở cuối
   mỗi lần chạy `train_pair.py` (sau khi xong hết epoch + eval test), nên domain
   nào bị ngắt là mất trắng, không resume được, dễ lãng phí quota đã dùng.
   Đã thêm vào cả 2 cell: **(a)** skip domain/config đã có `results.json` (an
   toàn khi chạy lại nhiều lần), **(b)** `files.download()` ngay sau mỗi
   domain/config chạy xong — không đợi tới cuối mới tải (cell 9 cũ).

### ⚠️ Hết quota Google Colab — chưa chạy được ablation đầy đủ
Sau khi verify baseline xong, hết quota trước khi kịp chạy full ablation 4
config × 3 domain (bước tiếp theo trong kế hoạch). **Đây là việc ưu tiên số 1
khi quota reset** (Colab Free thường reset theo ngày/24h, hoặc cân nhắc Colab
Pro nếu quota hết thường xuyên do khối lượng train nhiều domain × nhiều config).

---

## 7. Việc cần làm tiếp (cập nhật lại theo tiến độ tối 10/8)

1. **Chờ quota Colab reset** → chạy `colab_pair_ablation.ipynb` full: 4 config
   (CE / CE+norm / Focal / Focal+norm) × 3 domain (mobile/restaurant/hotel),
   kiến trúc PhoBERT+MHA đã chốt, 10 epoch, fp16.
2. ✅ ~~Gửi file `.tar.gz` kết quả baseline reproduction~~ — Đã nhận, đã copy
   `results.json` (3 domain, đủ `epoch_logs`) vào
   `experiments/results_vitasd_repro/` trong repo.
3. Sau khi chạy xong bước 1, copy `results.json` của ablation C1-C4 vào đúng
   vị trí trong repo (`experiments/results_pair/`), commit.
4. Chạy `summarize_pair_results.py` → bảng ablation đầy đủ mọi config (F3).
5. Vẽ training curve từ `epoch_logs` cho tất cả config (F3).
6. Cân nhắc chạy lại 4 config hotel/ViSoBERT với kiến trúc mới (PhoBERT+MHA
   → đổi `--model visobert`) để có dòng "+ ViSoBERT backbone" nhất quán.
7. Điền Bảng 2 trong `report/paper_vi.md` + mục "quá trình thực nghiệm".
8. Vẫn nên hỏi thầy Kiệt: auxiliary sentence xây thế nào, macro F1 có tính
   `none` không, và xin source code/PDF đầy đủ nếu có thể — dù baseline đã
   verify tạm ổn, có source code gốc vẫn tốt hơn nhiều so với "best-effort".
9. Dọn file untracked (`.test_write`), xác nhận có giữ
   `SESSION_SUMMARY_2026-07-11.md` không.
