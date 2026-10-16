# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Đỗ Đình Hoàn |
| MSSV | 02377 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/dinhhoan-04/K4-L3L4-Track2-Day21-DoDinhHoan-02377-CI-CD-for-AI-Systems |
| Ngày nộp | 17/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 150 | 0.15 | 4 | 0.7182 | 0.8760 |

**Bộ siêu tham số đã chọn:** `n_estimators=150`, `learning_rate=0.15`, `max_depth=4`.

**Lý do:** Bộ này đạt F1-score cao nhất (0.7182), cân bằng tốt Precision (0.7719) và Recall (0.6713) trên lớp thu nhập cao (>50K). Lần 1 có Accuracy cao nhất (0.8780) nhưng F1 chỉ 0.7109 do mô hình thiên vị lớp đa số để tăng accuracy, bỏ sót mẫu dương. Về đánh đổi, learning_rate thấp (0.05) khiến 50 cây underfitting (F1=0.6051, bị Quality Gate chặn). Bộ chọn dùng `learning_rate=0.15`, `max_depth=4` và 150 cây giúp học phi tuyến sâu mà không overfitting.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult Income mất cân bằng: lớp dương (>50K) chiếm 24.8%, lớp âm (<=50K) chiếm 75.2%. Mô hình luôn đoán nhãn âm vẫn đạt Accuracy 75.2%, nhưng F1 lớp dương bằng 0.000 nên Accuracy gây hiểu nhầm nghiêm trọng.

F1 lớp dương đo chuẩn xác năng lực nhận diện lớp thiểu số mà không bị lớp âm chi phối. Pipeline không dùng average="macro" hay average="weighted" vì việc tính gộp vẫn bị lớp đa số lấn át, che giấu lỗi trên nhóm mục tiêu.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow lỗi UntrustedTypesFoundException | skops chặn Tree của scikit-learn | Thêm serialization_format="cloudpickle" trong log_model |
| Lỗi UnicodeEncodeError trên terminal | Console cp1252 không hỗ trợ UTF-8 | Thêm sys.stdout.reconfigure(encoding="utf-8") |
| DVC cần quyền Cloud Storage trên CI | Remote gs:// cần xác thực IAM | Cấu hình secret GCP_SA_KEY và nạp credentials |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7182 | 0.8760 |
| Bước 3 (thêm `train_batch2`) | 0.7297 | 0.8800 |

**Nhận xét:** Khi thêm train_batch2 (tổng 44,722 mẫu), F1 tăng từ 0.7182 lên 0.7297 và Accuracy tăng lên 0.8800. Do cùng phân phối, dữ liệu mới giúp mô hình học thêm trường hợp biên của nhóm thu nhập cao, nâng cao độ tổng quát hóa trên tập holdout.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng 0.10-0.95, ngưỡng 0.25 đạt F1 cực đại 0.7447.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Xuất ma trận nhầm lẫn vào outputs/detail.txt và lưu artifact.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Tích hợp hàm kiểm tra phân phối so với mốc 24.8%, cảnh báo data drift.
