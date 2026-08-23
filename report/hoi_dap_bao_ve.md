# Hỏi đáp dự kiến — Bảo vệ đề tài

Tài liệu tập hợp các câu hỏi thầy/hội đồng có thể hỏi, kèm câu trả lời đã chuẩn bị sẵn. Cập nhật liên tục khi có câu hỏi mới.

---

## 1. Việc thêm và tinh chỉnh gamma/alpha (Focal Loss) có phải "can thiệp" vào phương pháp gốc không?

### Câu hỏi thầy có thể đặt ra
- "Hai trọng số gamma, alpha này có trong công thức gốc của ViTASA không?"
- "Sao lại đi tinh chỉnh, vậy có phải đang cố nặn ra kết quả đẹp không?"
- "Tinh chỉnh nhiều lần rồi test lại — vậy khác gì đang overfit vào tập test?"

### Trả lời

**Bước 1 — Xác nhận: gamma/alpha KHÔNG nằm trong công thức gốc của ViTASA.**

Bài báo gốc (Tran et al., 2025) — mô hình baseline ViTASD — chỉ dùng **Cross-Entropy Loss thông thường**, không hề có gamma hay alpha (đúng như cấu hình C1 trong ablation: Cross-Entropy, tái lập ViTASD).

Hai tham số gamma và alpha đến từ **2 bài báo khác, không liên quan tới ViTASA**:
- **gamma** — Focal Loss (Lin et al., 2017), đề xuất ban đầu cho *object detection* trong ảnh (RetinaNet).
- **alpha** (trọng số lớp) — Class-Balanced Loss dựa trên "effective number of samples" (Cui et al., 2019), cũng thuộc computer vision.

**Bước 2 — Vì sao việc thêm 2 tham số này là hợp lý.**

Tác giả ViTASA tự thừa nhận hạn chế "class imbalance" (mất cân bằng lớp) nhưng KHÔNG tự giải quyết trong bài báo gốc — họ chỉ dùng CE. Do đó gamma/alpha không phải là tham số có sẵn trong công thức của ViTASA để "sửa" — mà là 2 tham số nhóm tự đưa vào từ kỹ thuật bên ngoài, thuộc về **đóng góp mới (Module 2 — Imbalanced Learning)** của đề tài, nhằm giải quyết đúng limitation mà chính tác giả ViTASA đã nêu ra nhưng bỏ ngỏ.

**Bước 3 — Vì sao việc tinh chỉnh (tuning) là hoàn toàn bình thường, không phải "nặn số".**

Vì gamma/alpha mượn từ domain ảnh (object detection), giá trị mặc định (gamma=2.0, kèm alpha) chưa từng được calibrate cho bài toán TASA tiếng Việt — nơi mức mất cân bằng khác hẳn (có lớp chỉ 28 mẫu/hơn 100,000). Việc hiệu chỉnh lại các tham số này khi áp dụng sang domain mới là **bước kỹ thuật bắt buộc, tiêu chuẩn trong mọi nghiên cứu ML** — không phải can thiệp vào phương pháp gốc của ViTASA (vì ViTASA không hề định nghĩa các tham số này), và cũng không phải "sửa" đóng góp của chính đề tài (vì tinh chỉnh diễn ra sau khi đã có giả thuyết rõ ràng, dựa trên quan sát thực nghiệm — không phải dò ngẫu nhiên).

**Bước 4 — Phân biệt rõ "tinh chỉnh" và "rò rỉ dữ liệu test" (2 chuyện khác nhau).**

Tinh chỉnh hyperparameter, kể cả thử nhiều cấu hình rồi so sánh, là quy trình chuẩn của mọi nghiên cứu ML — không có gì sai. Vấn đề chỉ phát sinh nếu **chọn cấu hình cuối cùng dựa trên điểm số của tập test** (leakage). Quy trình đúng — và cũng là quy trình đề tài đang áp dụng — là:
1. Mọi quyết định "chọn cấu hình nào" dựa trên **tập dev** (validation), không nhìn test trước khi chốt.
2. Sau khi chốt cấu hình, tập **test chỉ được đánh giá đúng 1 lần** cho cấu hình đã chọn — không quay lại đổi lựa chọn dựa trên kết quả test.
3. Trong báo cáo ghi rõ: "Hyperparameter được lựa chọn qua exploratory search trên tập dev dựa trên giả thuyết cụ thể (H1–H3); cấu hình cuối cùng được đánh giá 1 lần trên test."

### Câu trả lời ngắn gọn (nếu thầy hỏi trực tiếp, không cần giải thích dài)

> "Gamma và alpha không nằm trong công thức gốc của ViTASD — đó là 2 siêu tham số của Focal Loss (Lin et al. 2017) và Class-Balanced Loss (Cui et al. 2019) mà tụi em đưa vào như một phần đóng góp mới (Module Imbalanced Learning), để giải quyết đúng hạn chế mà chính tác giả ViTASA đã thừa nhận nhưng chưa xử lý trong bài báo gốc. Vì mượn từ domain ảnh, giá trị mặc định chưa chắc phù hợp với mức mất cân bằng cực đoan của dữ liệu tiếng Việt (có lớp chỉ 28 mẫu), nên việc hiệu chỉnh lại là bước kỹ thuật cần thiết — không phải can thiệp vào phương pháp gốc, và toàn bộ quá trình chọn cấu hình đều dựa trên tập dev, tập test chỉ được nhìn đúng 1 lần cuối cùng để tránh rò rỉ."

---

*(Các câu hỏi tiếp theo sẽ được bổ sung vào phần dưới đây.)*
