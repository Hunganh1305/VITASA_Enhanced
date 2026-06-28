# Tiến độ đồ án — VITASA_Enhanced
### Enhancing Vietnamese TASA with Text Normalization and Imbalanced Learning

> Cập nhật: 2026-06-28. Review dựa trên code thật trong thư mục
> `Documents/Master's Degree IT- UIT/Kì 2/Xử lí ngôn ngữ tự nhiên/VITASA_Enhanced`,
> đã chạy thử test suite và đọc qua từng module.

## Tóm tắt nhanh

| Phần | Trạng thái | Kết quả kiểm tra |
|---|---|---|
| Text Normalization module | ✅ Hoàn thành (v1) | 19/19 test pass |
| Imbalanced Learning module | ✅ Hoàn thành (v1) | 13/13 test pass |
| Dataset ViTASA | ✅ Đã có | Clone thật từ GitHub + copy vào `baseline/data/` (2000 dòng/domain) |
| `train.py` (fine-tune PhoBERT) | 🟡 Code xong, chưa chạy thật | Bug `best_dev_f1` **đã fix** (`-1.0`), `requirements.txt` đã thêm — sẵn sàng chạy Colab |
| Ablation study (4 config) | ⬜ Chưa làm | `experiments/results/` trống, chưa có `run_ablation.py` |
| Report / luận văn | ⬜ Không thấy trong thư mục này | Có thể đang viết ở nơi khác (Overleaf/Docs) — cần xác nhận |

## 1. Đã hoàn thành

### `text_normalization/` — Contribution #1
- `normalizer.py`: rule-based, pipeline `elongation → phrase lookup → tokenize → dictionary lookup → detokenize`. Code rõ ràng, có docstring giải thích thiết kế và limitation ngay trong file.
- Từ điển `data/teencode_dict.json`: **151 entries** (README ghi 145 — lệch nhẹ do cập nhật dict sau khi viết doc, không phải lỗi, chỉ cần sửa số trong README).
- Test: chạy `pytest text_normalization/tests/test_normalizer.py` → **19/19 pass**.
- README có ghi rõ limitation: không reorder câu, không sửa lỗi chính tả ngoài từ điển, không phục hồi dấu, không xử lý sarcasm (để future work) — phần này tốt, dùng trực tiếp được cho report.

### `imbalanced_learning/` — Contribution #2
- `losses.py`: `FocalLoss`, `WeightedCrossEntropyLoss`, `compute_class_weights` (2 chiến lược: `inverse_frequency` và `effective_number` theo Cui et al. 2019). Implementation đúng công thức, có check input hợp lệ (raise lỗi nếu thiếu class).
- Test: chạy `pytest imbalanced_learning/tests/test_losses.py` → **13/13 pass** (môi trường review ban đầu thiếu `torch`, đã cài bổ sung để chạy được — nên thêm `requirements.txt` cho dự án, xem phần "Cần fix").
- README map sẵn rất rõ 5 config ablation (Baseline / +Normalization / +Weighted CE / +Focal Loss / Full) — sẵn sàng dùng khi chạy `experiments/run_ablation.py`.

### Dataset
- `ViTASA/` là bản clone thật (có `.git`, log commit thật) từ `github.com/kh4nh12/ViTASA`.
- `baseline/data/{hotel,mobile,restaurant}/*.jsonl` — mỗi domain 2000 dòng, format đúng key `data` (text) và `label` (list `[char_start, char_end, ASPECT#POLARITY]`) khớp với code đọc trong `train.py`.

### `train.py` (entry point chính)
- Đã viết đầy đủ: `TASADataset` (BIO tagging theo char offset, có xử lý remap offset khi normalize làm thay đổi độ dài chuỗi bằng `difflib.SequenceMatcher` — chi tiết khá tinh), `TASAModel` (PhoBERT + linear head), loss factory (`ce` / `weighted_ce` / `focal`), training loop + eval macro-F1, lưu `results.json` theo từng config.
- Parse cú pháp và import đều chạy được (đã test bằng cách exec module). Hỗ trợ đúng 4 config ablation qua flag `--loss` + `--normalize`.

## 2. Đang làm / chưa hoàn thành

1. **Chưa train lần nào thật** — toàn bộ phần baseline (PhoBERT fine-tune) chỉ tồn tại ở mức code, chưa chạy trên GPU/Colab nên chưa có số liệu F1 thật để so sánh các config.
2. **`experiments/run_ablation.py` chưa tồn tại** — `experiments/results/` đang trống hoàn toàn. Cần viết script loop qua 5 config (gọi `train.py` với các flag tương ứng) và tổng hợp bảng kết quả.
3. **Class weights / gamma / beta chưa được tune trên label distribution thật** — `compute_class_weights` sẽ tự tính đúng khi `train.py` chạy lần đầu, nhưng chưa có lần chạy nào để biết giá trị thực tế và có cần chỉnh `gamma`/`beta` không.
4. **Report (EN/VI) không thấy trong thư mục `NLP/`** — `PROJECT_STRUCTURE.md` có nhắc tới `report/report_en.md`, `report_vi.md`, `presentation_script.md` nhưng các file này không tồn tại trong project. Nếu đang viết ở Google Docs/Overleaf thì không sao, chỉ cần xác nhận; nếu chưa viết thì đây là phần còn thiếu.

## 3. Cần fix (cụ thể, có vị trí trong code)

1. **Bug nhỏ nhưng thật trong `train.py` (dòng ~329, ~342)**:
   ```python
   best_dev_f1 = 0.0
   ...
   if dev_f1 > best_dev_f1:
       best_dev_f1 = dev_f1
       torch.save(model.state_dict(), output_dir / "best_model.pt")
   ...
   model.load_state_dict(torch.load(output_dir / "best_model.pt", ...))  # crash nếu chưa lưu lần nào
   ```
   Nếu epoch đầu cho `dev_f1 == 0.0` (rất dễ xảy ra ở epoch 1 khi model mới fine-tune, predict toàn `O`), điều kiện `>` không thỏa → **không file nào được lưu** → dòng `torch.load(...best_model.pt)` ở cuối sẽ `FileNotFoundError`. Sửa đơn giản: khởi tạo `best_dev_f1 = -1.0`, hoặc đổi điều kiện thành `>=` ở epoch đầu.

2. **Thiếu `requirements.txt`** — môi trường review không có sẵn `torch`/`transformers`/`scikit-learn`, phải cài tay mới chạy được test. Nên thêm file liệt kê dependency + version (đặc biệt version `torch`/`transformers` dùng trên Colab) để tránh lệch môi trường khi chạy thật.

3. **Lệch giữa `PROJECT_STRUCTURE.md` và code thật**:
   - Doc ghi `cli.py` cho cả 2 module (`text_normalization/cli.py`, `imbalanced_learning/cli.py`), nhưng thực tế là `demo_normalizer.py` và `demo_losses.py` ở **thư mục gốc**. Nên đồng bộ: hoặc đổi tên/di chuyển file cho khớp doc, hoặc sửa doc để khỏi gây nhầm khi viết report.
   - Doc vẽ cấu trúc dự kiến `baseline/train.py`, nhưng file thật là `train.py` ở gốc project. Nên thống nhất 1 vị trí.
   - README `text_normalization` ghi từ điển có 145 entries, thực tế 151 — cập nhật số liệu.

4. **Dataset bị trùng 2 nơi**: `ViTASA/*.jsonl` (bản clone gốc) và `baseline/data/*/*.jsonl` (bản copy dùng để train). Chưa verify 2 bản này có giống nhau 100% không — nên diff lại 1 lần để chắc `baseline/data/` không bị lệch so với bản gốc đã clone.

## 4. Đề xuất thứ tự làm tiếp

1. Fix bug `best_dev_f1` trong `train.py` (5 phút).
2. Thêm `requirements.txt`.
3. Chạy thử `train.py --domain mobile --loss ce --epochs 1` trên CPU/sample nhỏ để bắt lỗi sớm trước khi đưa lên Colab full.
4. Viết `experiments/run_ablation.py`, chạy đủ 5 config trên Colab (GPU), lưu `results.json`.
5. Tổng hợp bảng kết quả ablation → đưa vào report.
6. Dọn lệch tài liệu (`cli.py` vs `demo_*.py`, đường dẫn `train.py`, số entries dictionary).
