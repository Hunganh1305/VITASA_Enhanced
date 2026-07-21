# Text Normalization Module

Module chuẩn hóa văn bản mạng xã hội tiếng Việt (teencode, viết tắt, kéo dài
ký tự) — một trong hai contribution chính của đề tài *"Cải tiến Vietnamese
Targeted Aspect Sentiment Analysis sử dụng Text Normalization và Imbalanced
Learning"*, nhằm xử lý limitation về ngôn ngữ phi chuẩn mà paper ViTASA gốc
chưa giải quyết.

## Cấu trúc

```
text_normalization/
├── normalizer.py            # class TextNormalizer (logic chính)
├── cli.py                   # demo CLI (sample có sẵn + tự nhập câu)
├── data/
│   └── teencode_dict.json   # từ điển teencode/viết tắt (145 entries)
├── tests/
│   └── test_normalizer.py   # unit test (pytest)
└── README.md
```

## Thiết kế (v1 — rule-based)

Pipeline gồm 3 bước, áp dụng theo thứ tự trong `TextNormalizer.normalize()`:

1. **Elongation normalization** — thu gọn ký tự/dấu câu bị lặp lại do gõ
   nhấn mạnh cảm xúc (`đẹppppp` → `đẹp`, `!!!!` → `!`). Quy tắc: run ≥3 ký
   tự chữ/số giống nhau liên tiếp thu về 1; run ≥2 dấu câu (`!?.,`) thu về 1.
2. **Phrase lookup** — thay cụm nhiều từ trong từ điển trước khi tách từ
   (vd `gia ca` → `giá cả`, `mn hinh` → `màn hình`), match theo regex với
   word-boundary, ưu tiên cụm dài nhất.
3. **Token-level dictionary lookup** — tách câu thành token (giữ Unicode
   tiếng Việt), tra từng token trong từ điển, không có thì giữ nguyên.

Từ điển `teencode_dict.json` được chia theo nhóm để dễ mở rộng:
đại từ/trợ từ, động từ/tính từ phổ biến, từ vựng domain mobile/restaurant/hotel
(khớp 3 domain của ViTASA dataset), slang tăng cường mức độ (`vcl`, `vl`...).
Key bắt đầu bằng `_` là comment, bị bỏ qua khi load.

## Cách dùng

```python
from text_normalization.normalizer import TextNormalizer

normalizer = TextNormalizer()
normalizer("pyn trâu vcl")
# -> "bạn bền rất"

normalizer("sp ngon wa, dt pin trâu, sạc rất nhanhhhhh!!!!")
# -> "sản phẩm ngon quá, điện thoại pin bền, sạc rất nhanh!"
```

## Cách chạy CLI

Demo có giao diện terminal, không cần viết code:

```bash
cd text_normalization
python3 cli.py
```

Menu gồm:
1. **Chạy với sample có sẵn** — 5 câu mẫu dựng theo style comment thật trên
   mobile/restaurant/hotel, in song song input/output để so sánh nhanh.
2. **Tự nhập câu để test** — nhập câu tuỳ ý, normalize ngay, gõ `back` để
   quay lại menu.
0. **Thoát**.

## Cách mở rộng từ điển

Mở `data/teencode_dict.json`, thêm entry vào nhóm phù hợp:
- Key 1 từ (không có khoảng trắng) → tra theo token, vd `"oder": "đặt hàng"`
- Key nhiều từ (có khoảng trắng) → tra theo cụm trước khi tách từ, vd
  `"giao hang": "giao hàng"`

Không cần sửa code, `TextNormalizer` tự load lại khi khởi tạo instance mới.
Khuyến nghị: khi reproduce baseline và nhìn dataset thật, lọc các comment có
nhiều OOV token (so với PhoBERT vocab) để bổ sung dần — đây nên là một bước
trong pipeline thực nghiệm, không làm thủ công 100%.

## Option ablation: `keep_expressive_markers` (2026-07-16)

Mặc định, các marker biểu cảm/tiếng cười trong từ điển (`hihi`, `haha`,
`huhu`, `kk`, `kkk` — nhóm `_section_misc`) bị **xóa hẳn** (map sang `""`).

Paper **ViGoEmotions** (Tran et al., EACL 2026 — cùng nhóm tác giả với
ViTASA) thử nghiệm giữ nguyên vs. convert vs. loại bỏ **emoji** trên dataset
cảm xúc mạng xã hội tiếng Việt, và thấy giữ nguyên emoji cho Macro-F1 cao
hơn trên hầu hết backbone, kể cả ViSoBERT (62.33% vs các scenario khác).
Marker tiếng cười có thể đóng vai trò tín hiệu cảm xúc tương tự emoji (đặc
biệt cho polarity Positive/amusement), nên việc xóa hẳn có thể đang làm mất
tín hiệu — nhưng finding này chưa được verify trên chính dataset ViTASA.

```python
# Hành vi cũ (mặc định, không đổi)
TextNormalizer()("ngon quá kk")        # -> "ngon quá" (kk bị xóa)

# Ablation mới: giữ nguyên marker biểu cảm
TextNormalizer(keep_expressive_markers=True)("ngon quá kk")  # -> "ngon quá kk"
```

Wire sẵn trong `train.py` qua flag `--keep-expressive` (chỉ có tác dụng khi
kèm `--normalize`) — nên chạy ablation so sánh có/không trước khi chốt default
cho kết quả chính thức, thay vì đoán.

## Limitation (đã biết, ghi cho phần report)

- **Không reorder câu**: word-level replacement giữ nguyên vị trí, nên cụm
  như `"trâu vcl"` → `"bền rất"` (đúng nghĩa nhưng sai trật tự ngữ pháp,
  đúng ra là `"rất bền"`). Đây là lý do hướng seq2seq/model-based được nêu
  là hướng mở rộng tương lai.
- **Không sửa lỗi chính tả ngoài từ điển**: không phải spelling-correction
  model, chỉ xử lý case đã biết.
- **Không phục hồi dấu**: không xử lý input gõ không dấu (`"khong dau"`),
  vì đây là một bài toán khác (Vietnamese diacritics restoration) — nếu cần
  có thể thêm như Module phụ riêng, không trộn vào module này.
- **Sarcasm không nằm trong scope module này** — đã ghi nhận riêng là future
  work của đề tài (theo trao đổi trước đó về Module 3 cho luận văn đầy đủ).

## Bước tiếp theo gợi ý

1. Áp dụng normalizer lên một sample thật từ ViTASA dataset, đếm % token
   được thay đổi và review thủ công 20-30 câu để đánh giá chất lượng.
2. Bổ sung từ điển dựa trên OOV token thực tế (so với PhoBERT/ViSoBERT vocab).
3. Tích hợp normalizer vào pipeline tiền xử lý trước khi fine-tune baseline,
   so sánh kết quả có/không normalization (ablation).
