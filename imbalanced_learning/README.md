# Imbalanced Learning Module

Module xử lý class imbalance giữa các sentiment polarity (Positive/Negative/
Neutral) — contribution #2 của đề tài *"Cải tiến Vietnamese TASA sử dụng Text
Normalization và Imbalanced Learning"*, xử lý limitation thứ hai mà paper
ViTASA gốc nêu ra.

## Cấu trúc

```
imbalanced_learning/
├── losses.py             # FocalLoss, WeightedCrossEntropyLoss, compute_class_weights
├── cli.py                 # demo CLI (sample distribution có sẵn + tự nhập)
├── tests/
│   └── test_losses.py    # 17 unit test (pytest)
└── README.md
```

## Thành phần

### 1. `FocalLoss`

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

`p_t` là xác suất model gán cho đúng class. Khi sample đã được phân loại
đúng tốt (`p_t` cao), `(1 - p_t)^gamma` nhỏ → loss bị giảm mạnh → model tập
trung học vào sample khó / minority class. `gamma=0` (và không alpha) thì
tương đương Cross-Entropy thường (đã test).

- `gamma`: hệ số focusing, khuyến nghị thử 1.0 - 2.0.
- `alpha`: trọng số theo class (Tensor `[num_classes]`), có thể lấy trực
  tiếp từ `compute_class_weights()` để kết hợp focusing + class weighting.

### 2. `WeightedCrossEntropyLoss`

Wrapper mỏng quanh `nn.CrossEntropyLoss(weight=...)`, cùng interface
`forward(logits, target)` với `FocalLoss` để dễ swap qua lại khi chạy
ablation (so sánh CE thường / Weighted CE / Focal Loss).

### 3. `compute_class_weights(labels, num_classes, strategy, beta)`

Tính trọng số class từ label distribution thực tế của ViTASA dataset.

- `"inverse_frequency"`: `weight_c = N / (num_classes * count_c)`, đơn
  giản, dễ giải thích, nhưng có thể cho weight quá cực đoan khi 1 class rất
  ít sample.
- `"effective_number"` (mặc định, Class-Balanced Loss — Cui et al., CVPR
  2019): `weight_c = (1 - beta) / (1 - beta^count_c)`. Mềm hơn, giả định
  mỗi sample mới có xác suất giảm dần đóng góp thông tin (effective number
  of samples) — phù hợp khi minority class cực ít (case của Neutral trong
  ViTASA, theo paper gốc).
- `"pos_neg_ratio"` (thêm 2026-07-16): `weight_c = (N - count_c) / count_c`
  — công thức `pos_weight` dùng trong paper **ViGoEmotions** (Tran et al.,
  EACL 2026, cùng nhóm tác giả ViTASA) cho `BCEWithLogitsLoss` multilabel.
  Ở đây thích ứng sang multi-class bằng cách coi mỗi class là bài toán
  binary "class c vs phần còn lại" — không chia đều theo `num_classes` như
  `inverse_frequency` nên tăng mạnh hơn cho class rất hiếm (test
  `test_pos_neg_ratio_more_extreme_than_inverse_frequency_for_rare_class`
  minh họa điều này). Chưa biết có tốt hơn 2 chiến lược kia trên ViTASA thật
  hay không — nên đưa vào ablation so sánh, không mặc định thay
  `effective_number`.

Cả 3 chiến lược đều chuẩn hóa để `weights.mean() == 1`, giữ scale loss ổn
định khi so sánh giữa các config trong ablation.

## Cách dùng

```python
from imbalanced_learning.losses import (
    FocalLoss, WeightedCrossEntropyLoss, compute_class_weights,
)

# 1. Tính class weight từ label thật của train set
weights = compute_class_weights(
    labels=train_labels,        # list/array nhãn (0=Positive, 1=Negative, 2=Neutral)
    num_classes=3,
    strategy="effective_number",
)

# 2a. Dùng Focal Loss kết hợp class weight
criterion = FocalLoss(gamma=2.0, alpha=weights)

# 2b. Hoặc Weighted Cross-Entropy thường
criterion = WeightedCrossEntropyLoss(class_weights=weights)

# Trong training loop (sau khi forward qua PhoBERT/ViSoBERT + classification head)
logits = model(input_ids, attention_mask)   # [batch, num_classes]
loss = criterion(logits, target)
loss.backward()
```

## Cách chạy CLI

Demo có giao diện terminal, không cần viết code:

```bash
cd imbalanced_learning
python3 cli.py
```

Menu gồm:
1. **Chạy với sample có sẵn** — 4 distribution dựng sẵn (Mobile/Restaurant/
   Hotel theo style imbalance của ViTASA, và 1 distribution cân bằng để đối
   chứng). Mỗi lựa chọn tính class weight (2 chiến lược) và so sánh CE thường
   / Weighted CE / Focal Loss trên 1 batch giả lập 16 sample lấy theo đúng
   tỉ lệ distribution.
2. **Tự nhập label distribution** — nhập số lượng sample cho từng class
   (Positive/Negative/Neutral), CLI tự tính weight và so sánh loss tương tự.
0. **Thoát**.

## Gợi ý chạy ablation

Kết hợp với Text Normalization Module, thiết kế 5 config đã đề ra cho
ablation study có thể map vào module này như sau:

| Config | Text Normalization | Loss function |
|---|---|---|
| 1. Baseline | ✗ | CE thường |
| 2. +Normalization | ✓ | CE thường |
| 3. +Weighted CE | ✗ | `WeightedCrossEntropyLoss` |
| 4. +Focal Loss | ✗ | `FocalLoss` |
| 5. Full (Normalization + Focal Loss) | ✓ | `FocalLoss` |

## Limitation

- `compute_class_weights` raise lỗi nếu thiếu hẳn 1 class trong `labels`
  truyền vào — cần đảm bảo tính weight trên toàn bộ train set, không tính
  trên 1 batch nhỏ.
- Chưa benchmark `gamma`/`beta` tối ưu trên ViTASA thật — đây là
  hyperparameter cần tune khi có baseline để so sánh (xem mục "Bước tiếp
  theo" trong `PROJECT_STRUCTURE.md`).
- Không xử lý imbalance ở cấp độ aspect/target (chỉ xử lý ở cấp polarity
  label) — nếu sau này thấy aspect cũng imbalance, cần module riêng.
