# Session Summary — 2026-07-11

Đề tài: *Enhancing Vietnamese TASA with Text Normalization and Imbalanced Learning*

---

## 1. Việc đã làm trong session này

### Hoàn thiện tài liệu (README + PROJECT_STRUCTURE)

- Thêm `cli.py` vào file tree của `text_normalization/README.md` và `imbalanced_learning/README.md`
- Thêm section "Cách chạy CLI" vào cả 2 README (menu 1=sample có sẵn, 2=tự nhập, 0=thoát)
- Cập nhật `PROJECT_STRUCTURE.md` để phản ánh `cli.py` ở cả 2 module

### Khám phá project thực tế (VITASA_Enhanced)

Phát hiện folder làm việc chính là `VITASA_Enhanced/` (khác với `NLP UIT/` dùng ở session trước). Folder này đã có:

| Thành phần | Trạng thái |
|---|---|
| `text_normalization/` | ✅ Done — 19 test pass, 151-entry dict |
| `imbalanced_learning/` | ✅ Done — 13 test pass |
| Dataset ViTASA | ✅ Clone sẵn trong `baseline/data/` (3 domain × 2000 comments) |
| `train.py` | ✅ 399 dòng — PhoBERT fine-tune, BIO tagging, 4 config ablation |
| `demo_normalizer.py` / `demo_losses.py` | ✅ Demo CLI ở root |
| `report/paper_vi.md` | ✅ Draft tiếng Việt đầy đủ — chỉ còn chỗ trống ở Bảng 2 (kết quả thực nghiệm) |
| `colab_run.ipynb` | ✅ Notebook Colab |
| `experiments/results/` | ⬜ Trống (chưa chạy lúc khám phá) |

### Giải thích kỹ thuật (có minh họa visual)

- **Imbalanced Learning hoạt động thế nào**: CE loss trung bình bị dominate bởi token O (80%) → model học ignore B/I tags → F1 = 0. Focal Loss × class weight (Neutral ~6.8×) buộc model học minority class.
- **Text Normalization + dataset gốc**: không cần dataset mới — normalize chạy on-the-fly trên ViTASA, `difflib.SequenceMatcher` remap char offset của label về vị trí mới trong chuỗi normalized. Edge case: từ bị xóa hoàn toàn (vd "kk" → "") có thể mất label, ít xảy ra trong thực tế.
- **Output của pipeline**: không phải "câu đẹp hơn" mà là danh sách `(target, aspect, sentiment)` triples.

### Tổng hợp kết quả training từ Google Colab

Đọc và de-duplicate 12 file `results.json` từ folder `output dataset/`. Tạo 2 file tổng hợp:

- `experiments/results_summary.csv` — bảng 12 config, tiện mở Excel
- `experiments/results_all.json` — đầy đủ epoch logs + baseline ViTASD để so sánh

---

## 2. Kết quả training hiện tại

| Config | Mobile test F1 | Restaurant test F1 | Hotel test F1 |
|---|---|---|---|
| C1 — CE | 0.00% | 0.00% | 0.15% |
| C2 — CE + Norm | 0.00% | 0.00% | 0.13% |
| C3 — Focal | 6.07% | 0.00% | 0.03% |
| **C4 — Focal + Norm** | **9.05%** | **0.71%** | **0.37%** |
| ViTASD (paper gốc) | 61.77% | 41.12% | 52.64% |

**Pattern đúng**: C4 thắng ở cả 3 domain. CE cho F1 = 0 → xác nhận hypothesis class imbalance.

**Vấn đề**: F1 còn thấp hơn baseline rất nhiều. Training curve vẫn tăng ở epoch 10 → chưa hội tụ.

---

## 3. Phân tích nguyên nhân F1 thấp

1. **Chưa đủ epoch** (nguyên nhân chính) — curve mobile focal_norm tăng đều từ 4→10, cần 30+ epoch
2. **O token trong class weight** — `compute_class_weights()` nhận cả O (80% token) → weight không tập trung vào B/I imbalance
3. **Metric khác paper** — paper ViTASD dùng span-level exact match, code hiện tại dùng token-level F1 → không so sánh được trực tiếp

---

## 4. Cải thiện đề xuất (theo ưu tiên)

### Fix 1 — Thêm epoch (dễ nhất, tác động lớn nhất)
```bash
python train.py --domain mobile --loss focal --normalize --epochs 30
```

### Fix 2 — Exclude O khi tính class weight
```python
# Trong train.py, thay:
train_labels_flat = [lbl for item in train_ds for lbl in item["labels"].tolist()]

# Thành:
o_id = label2id["O"]
train_labels_flat = [lbl for item in train_ds
                     for lbl in item["labels"].tolist()
                     if lbl != o_id]
```

### Fix 3 — Thêm MPS support để chạy local (M3 Mac)
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Ước tính tốc độ M3 MPS: ~15–25 phút/epoch (Colab T4: ~5 phút/epoch).

### Fix 4 — Thêm span-level F1 vào evaluate() để align với paper

---

## 5. File được tạo/cập nhật trong session

| File | Thay đổi |
|---|---|
| `text_normalization/README.md` | Thêm `cli.py` vào tree + section "Cách chạy CLI" |
| `imbalanced_learning/README.md` | Thêm `cli.py` vào tree + section "Cách chạy CLI" |
| `PROJECT_STRUCTURE.md` | Thêm `cli.py` cho cả 2 module |
| `experiments/results_summary.csv` | **Mới** — bảng tổng hợp 12 config |
| `experiments/results_all.json` | **Mới** — full epoch logs + metadata |

---

## 6. Bước tiếp theo

1. **Sửa `train.py`**: thêm MPS support + exclude O trong class weight
2. **Chạy 30 epoch** cho C4 (focal + norm) trên cả 3 domain (Colab hoặc local M3)
3. **Thêm span-level F1** để so sánh đúng với ViTASD paper
4. **Điền Bảng 2** trong `report/paper_vi.md` sau khi có kết quả tốt hơn
