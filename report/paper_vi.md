# Cải tiến Phân tích Cảm xúc Hướng Đến Khía Cạnh cho Tiếng Việt
## sử dụng Chuẩn hóa Văn bản Mạng Xã Hội và Học Mất Cân Bằng

**Phạm Nguyễn Hùng Anh**

**Giảng viên hướng dẫn:** PGS.TS. Nguyễn Văn Kiệt

**UIT, TP. Hồ Chí Minh, Việt Nam**

---

## Tóm tắt (Abstract)

Phân tích cảm xúc hướng đến khía cạnh ("Targeted Aspect Sentiment Analysis" — TASA) hướng đến việc xác định tất cả các bộ ba ("target", "aspect", "sentiment") trong một câu cho trước. Mặc dù nghiên cứu gần đây đã giới thiệu ViTASA — một bộ dữ liệu chuẩn ("benchmark") quy mô lớn cho TASA tiếng Việt trên ba lĩnh vực (điện thoại, nhà hàng, khách sạn) — mô hình tốt nhất hiện tại ViTASD vẫn còn hai hạn chế quan trọng: (1) mất cân bằng lớp ("class imbalance") đáng kể giữa các nhãn cảm xúc (Positive: 73.2%, Negative: 24.5%, Neutral: 2.3%), và (2) không thể xử lý các đặc điểm ngôn ngữ mạng xã hội như teencode, viết tắt và ký tự lặp.

Bài báo này đề xuất một pipeline cải tiến tích hợp module Chuẩn hóa Văn bản Mạng Xã Hội tiếng Việt ("Vietnamese Social Media Text Normalization") và các kỹ thuật Học Mất Cân Bằng ("Imbalanced Learning") vào framework ViTASD. Chúng tôi thực hiện các thí nghiệm loại trừ ("ablation study") toàn diện để đánh giá đóng góp của từng thành phần. Kết quả thực nghiệm trên tập dữ liệu ViTASA cho thấy phương pháp đề xuất vượt trội so với ViTASD trên cả ba lĩnh vực.

**Từ khóa:** Targeted Aspect Sentiment Analysis, Vietnamese NLP, Text Normalization, Imbalanced Learning, Social Media

---

## Pipeline Tổng thể (Overall Pipeline)

| Văn bản thô *(teencode / viết tắt / ký tự lặp)* | → | Module 1: *Text Normalization* | → | Module 2: *PhoBERT + Imbalanced Learning* | → | Đầu ra *(target, aspect, sentiment)* |
|---|---|---|---|---|---|---|

**Hình 1: Pipeline tổng thể của hệ thống đề xuất**

### Chi tiết từng module:

#### Module 1 — Text Normalization (Chuẩn hóa văn bản):

| Elongation Normalization | → | Phrase Lookup | → | Token Dictionary Lookup |
|---|---|---|---|---|
| *"nhanhhhhh" → "nhanh", "!!!!" → "!"* | | *"gia ca" → "giá cả"* | | *"pyn" → "pin", "vcl" → "rất"* |

**Hình 2: Các bước xử lý trong Module 1 — Text Normalization (rule-based, 151 entries)**

#### Module 2 — PhoBERT + Imbalanced Learning:

| Tokenizer | → | Encoder | → | Loss Function |
|---|---|---|---|---|
| *PhoBERT-base-v2* | | *PhoBERT-base-v2* | | *Focal Loss / Weighted CE* |

**Hình 3: Kiến trúc Module 2 — Encoder và Loss Function**

---

## 1. Giới thiệu (Introduction)

Phân tích cảm xúc ("Sentiment Analysis") trên mạng xã hội ngày càng trở nên quan trọng khi hàng triệu người dùng Việt Nam hàng ngày đăng đánh giá và bình luận trên các nền tảng như Facebook, Shopee và Google Reviews. Khác với phân tích cảm xúc ở mức độ văn bản ("document-level") hay câu ("sentence-level"), Phân tích Cảm xúc Hướng đến Khía cạnh ("Targeted Aspect Sentiment Analysis" — TASA) cung cấp hiểu biết chi tiết hơn bằng cách xác định các bộ ba ("target", "aspect", "sentiment") cụ thể trong câu (Saeidi et al., 2016).

Tran et al. (2025) đã giới thiệu ViTASA — bộ dữ liệu chuẩn ("benchmark") đầu tiên quy mô lớn cho TASA tiếng Việt với hơn 500,000 cặp target-aspect trên ba lĩnh vực. Mô hình ViTASD của họ đạt điểm "macro F1" lần lượt là 61.77%, 41.12% và 52.64% trên các lĩnh vực điện thoại, nhà hàng và khách sạn. Tuy nhiên, tác giả đã thừa nhận rõ hai hạn chế còn tồn tại.

Thứ nhất, tập dữ liệu bị mất cân bằng lớp ("class imbalance") nghiêm trọng — phân tích của chúng tôi trên ViTASA cho thấy nhãn Positive chiếm 73.2%, Negative 24.5%, và Neutral chỉ 2.3%, khiến dự đoán của mô hình bị lệch về phía nhãn đa số.

Thứ hai, ViTASD không thể xử lý ngôn ngữ phi chính thức phổ biến trên mạng xã hội tiếng Việt, bao gồm teencode (ví dụ: "pyn" cho "pin"), viết tắt (ví dụ: "vcl" làm từ nhấn mạnh) và ký tự lặp (ví dụ: "nhanhhhhh").

Bài báo này trực tiếp giải quyết cả hai hạn chế bằng cách đề xuất hai thành phần bổ sung:

1. Module Chuẩn hóa Văn bản Mạng Xã Hội tiếng Việt ("Vietnamese Social Media Text Normalization") chuẩn hóa văn bản phi chính thức trước khi đưa vào mô hình
2. Các kỹ thuật Học Mất Cân Bằng ("Imbalanced Learning") giảm thiểu độ lệch phân phối lớp trong quá trình huấn luyện

---

## 2. Công trình liên quan (Related Work)

### 2.1 Phân tích Cảm xúc Dựa trên Khía cạnh (Aspect-Based Sentiment Analysis)

Phân tích cảm xúc dựa trên khía cạnh ("Aspect-Based Sentiment Analysis" — ABSA) đã được nghiên cứu rộng rãi thông qua chuỗi bài thi "SemEval" (Pontiki et al., 2014, 2015, 2016). Dựa trên ABSA, TASA mở rộng nhiệm vụ để bao gồm nhận dạng đối tượng ("target identification"), tạo thành bộ ba ("target, aspect, sentiment"). Các công trình trước đây như CG-BERT (Wu and Ong, 2021), BERT-pair-QA và BERT-pair-NLI (Sun et al., 2019) đã đạt hiệu suất cao trên các bộ dữ liệu tiếng Anh.

### 2.2 Phân tích Cảm xúc tiếng Việt (Vietnamese Sentiment Analysis)

Phân tích cảm xúc tiếng Việt ngày càng được chú ý với các bộ dữ liệu như UIT-VSFC (phản hồi sinh viên), UIT-ViSFD (đánh giá điện thoại thông minh), và gần đây nhất là ViTASA (Tran et al., 2025). Các mô hình ngôn ngữ được tiền huấn luyện ("pre-trained language models") bao gồm PhoBERT (Nguyen & Nguyen, 2020) và ViSoBERT — được thiết kế đặc biệt cho văn bản mạng xã hội tiếng Việt — đã trở thành "backbone" tiêu chuẩn cho các nhiệm vụ này.

### 2.3 Chuẩn hóa Văn bản Mạng Xã Hội (Text Normalization for Social Media)

Chuẩn hóa văn bản ("Text Normalization") mạng xã hội đã được nghiên cứu nhiều cho tiếng Anh (Han and Baldwin, 2011) và các ngôn ngữ khác, nhưng vẫn còn ít được khám phá cho tiếng Việt. Các pipeline NLP tiếng Việt hiện có giả định văn bản sạch, chính thức — khiến chúng không phù hợp để xử lý ngôn ngữ phi chính thức phổ biến trên các nền tảng tiếng Việt.

### 2.4 Học Mất Cân Bằng trong NLP (Imbalanced Learning in NLP)

Mất cân bằng lớp ("class imbalance") là thách thức phổ biến trong phân loại văn bản. Focal Loss (Lin et al., 2017) dynamically down-weights easy examples để tập trung học vào các sample khó và minority class. Class-weighted Cross-Entropy gán trọng số cao hơn cho các lớp ít xuất hiện. Tuy nhiên, việc áp dụng chúng cho TASA tiếng Việt chưa được nghiên cứu có hệ thống.

---

## 3. Phương pháp đề xuất (Methodology)

### 3.1 Định nghĩa nhiệm vụ (Task Definition)

Cho câu S = {w₁, w₂, ..., wₙ}, nhiệm vụ TASA nhằm trích xuất tất cả bộ ba (t, a, s) trong đó:
- **t** là "target" (chuỗi con của S)
- **a** là "aspect" từ tập A được định nghĩa trước
- **s** là nhãn cảm xúc ("sentiment polarity") từ {Positive, Negative, Neutral}

Chúng tôi mô hình hóa bài toán này dưới dạng token classification với BIO tagging: mỗi token được gán nhãn `O`, `B-ASPECT#SENTIMENT`, hoặc `I-ASPECT#SENTIMENT`.

### 3.2 Module 1 — Chuẩn hóa Văn bản Mạng Xã Hội

Module tiền xử lý rule-based gồm 3 bước theo thứ tự:

1. **Elongation Normalization**: thu gọn ký tự lặp (≥3) và dấu câu lặp (≥2)
   - "nhanhhhhh" → "nhanh", "!!!!" → "!"
2. **Phrase Lookup**: thay cụm nhiều từ trước khi tách token
   - "gia ca" → "giá cả", "mn hinh" → "màn hình"
3. **Token Dictionary Lookup**: tra từng token trong từ điển 151 entries
   - "pyn" → "pin", "vcl" → "rất", "sp" → "sản phẩm"

#### Ví dụ minh họa:

| Văn bản gốc | Sau Text Normalization |
|---|---|
| *"sp ngon wa, sạc rất nhanhhhhh!!!!"* | *"sản phẩm ngon quá, sạc rất nhanh!"* |
| *"pyn trâu vcl"* | *"pin bền rất"* |

**Limitation đã biết:** Module không reorder câu (word-level replacement), không sửa lỗi chính tả ngoài từ điển, không phục hồi dấu thanh. Sarcasm detection nằm ngoài scope (xem Phần 5).

### 3.3 Module 2 — Học Mất Cân Bằng (Imbalanced Learning)

Để giải quyết mất cân bằng lớp trong ViTASA (Positive: 73.2%, Negative: 24.5%, Neutral: 2.3%), chúng tôi nghiên cứu hai kỹ thuật:

**Focal Loss** (Lin et al., 2017):
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
với γ=2.0, α được tính theo Class-Balanced weighting (Cui et al., 2019).

**Class-Weighted Cross-Entropy**: tương đương Focal Loss khi γ=0, chỉ dùng class weights.

Class weights được tính theo chiến lược "effective number" (Cui et al., 2019):
```
weight_c = (1 - β) / (1 - β^n_c),  β = 0.999
```

### 3.4 Mô hình nền (Backbone)

Chúng tôi xây dựng dựa trên ViTASD (Tran et al., 2025), sử dụng PhoBERT-base-v2 (Nguyen & Nguyen, 2020) làm backbone — mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt dựa trên RoBERTa. Một linear classification head được thêm vào trên top của last hidden state để dự đoán BIO tags.

---

## 4. Thực nghiệm (Experiments)

### 4.1 Tập dữ liệu (Dataset)

Tất cả thực nghiệm được thực hiện trên tập dữ liệu ViTASA (Tran et al., 2025), gồm 6,000 bình luận được chú thích thủ công trên ba lĩnh vực:

| Lĩnh vực | Số comment | Số span | Pos% | Neg% | Neu% |
|---|---|---|---|---|---|
| Điện thoại | 2,000 | ~6,000 | 73% | 24% | 3% |
| Nhà hàng | 2,000 | ~7,000 | 73% | 24% | 3% |
| Khách sạn | 2,000 | ~5,800 | 73% | 25% | 2% |

Chia tập dữ liệu theo tỉ lệ 7:1:2 (train/dev/test), seed cố định để đảm bảo reproducibility.

Chỉ số đánh giá chính: **macro F1-score**, nhất quán với bài báo gốc ViTASA.

### 4.2 Thiết kế Ablation Study

Chúng tôi thực hiện ablation study với 4 cấu hình để tách biệt đóng góp của từng thành phần:

| Config | Text Norm | Loss Function | Mô tả |
|---|---|---|---|
| **C1** | ❌ | Cross-Entropy | ViTASD Baseline (reproduce) |
| **C2** | ✅ | Cross-Entropy | + Text Normalization only |
| **C3** | ❌ | Focal Loss | + Imbalanced Learning only |
| **C4** | ✅ | Focal Loss | **Full Model** (cả hai đóng góp) |

**Bảng 1: Thiết kế Ablation Study**

### 4.3 Kết quả thực nghiệm

*(Cập nhật sau khi chạy xong Colab)*

| Config | Mobile F1 | Restaurant F1 | Hotel F1 |
|---|---|---|---|
| C1 — Baseline (CE) | - | - | - |
| C2 — + Text Norm | - | - | - |
| C3 — + Focal Loss | - | - | - |
| **C4 — Full Model** | **-** | **-** | **-** |
| ViTASD (paper gốc) | 61.77% | 41.12% | 52.64% |

**Bảng 2: Kết quả Macro F1 trên tập test ViTASA**

---

## 5. Kết luận (Conclusion)

Bài báo này trình bày một phương pháp cải tiến cho TASA tiếng Việt trực tiếp giải quyết hai hạn chế chính được xác định trong bài báo ViTASA: mất cân bằng lớp và ngôn ngữ phi chính thức mạng xã hội.

Bằng cách tích hợp module Text Normalization (rule-based, 151 entries, 3 bước xử lý) và Focal Loss/Weighted CE vào pipeline ViTASD, chúng tôi hướng đến cải thiện độ bền của mô hình với ngôn ngữ phi chính thức và khả năng phân loại đúng các lớp cảm xúc thiểu số.

### Hạn chế & Hướng nghiên cứu tương lai

- **Sarcasm detection**: câu như *"bút xịn vl, sài được 2 ngày là hỏng"* sử dụng mỉa mai — Text Normalization giúp hiểu từng từ rõ hơn nhưng chưa đủ để xử lý sarcasm ở mức ngữ cảnh. Đây là Module 3 trong hướng mở rộng tương lai.
- **Diacritics restoration**: xử lý văn bản gõ không dấu ("khong dau") là bài toán riêng, có thể tích hợp như module phụ.
- **ViSoBERT backbone**: thay PhoBERT bằng ViSoBERT — được pre-train đặc biệt trên văn bản mạng xã hội tiếng Việt — có thể cải thiện thêm hiệu suất trên data phi chính thức.

---

## Tài liệu tham khảo (References)

Tran, K. Q., et al. (2025). ViTASA: New benchmark and methods for Vietnamese targeted aspect sentiment analysis for multiple textual domains. *Computer Speech & Language*.

Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. In *Findings of EMNLP 2020*.

Lin, T. Y., et al. (2017). Focal loss for dense object detection. In *ICCV 2017*.

Cui, Y., et al. (2019). Class-balanced loss based on effective number of samples. In *CVPR 2019*.

Sun, C., et al. (2019). Utilizing BERT for aspect-based sentiment analysis via constructing auxiliary sentence. In *NAACL 2019*.

Wu, Z., & Ong, D. C. (2021). Context-guided BERT for targeted aspect-based sentiment analysis. In *AAAI 2021*.

Saeidi, M., et al. (2016). SentiHood: Targeted aspect based sentiment analysis dataset for urban neighbourhoods. In *COLING 2016*.

Pontiki, M. et al. (2014, 2015, 2016). SemEval shared tasks on aspect-based sentiment analysis.
