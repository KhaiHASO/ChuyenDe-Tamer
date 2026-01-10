# BÁO CÁO TỔNG HỢP & PHÂN TÍCH SOURCE CODE TAMER-GAT

Tài liệu này cung cấp phân tích chi tiết về mã nguồn, cấu trúc dữ liệu, mô hình huấn luyện và đánh giá kết quả dựa trên source code hiện tại và quá trình training của bạn.

---

## 1. Tổng Quan Về Mô Hình (Model Architecture)

Mô hình trong source code này (TAMER) là một kiến trúc **Encoder-Decoder** chuyên dụng cho nhận dạng công thức toán học viết tay (Handwritten Mathematical Expression Recognition - HME).

### Kiến trúc chi tiết:
1.  **Encoder**:
    -   Sử dụng mạng CNN (DenseNet) để trích xuất đặc trưng từ ảnh đầu vào.
    -   **GAT (Graph Attention Network) [Optional]**: Source code có hỗ trợ lớp GAT (`tamer/model/encoder.py`), giúp mô hình học mối quan hệ cấu trúc không gian giữa các nét viết tốt hơn. Tuy nhiên, cần kiểm tra file `config/crohme.yaml` để xem tham số `use_gat` là `true` hay `false`. (Mặc định trong config tôi thấy là `false`).
2.  **Decoder**:
    -   Sử dụng Transformer Decoder để sinh ra chuỗi LaTeX.
    -   Cơ chế **Coverage Attention**: Giúp mô hình "nhớ" những phần nào của ảnh đã được dịch để tránh lặp lại hoặc bỏ sót (`tamer/model/decoder.py`).

---

## 2. Quy Trình Phân Chia Dữ Liệu (Data Splitting)

Trong Machine Learning, việc chia dữ liệu Train/Validation/Test là cực kỳ quan trọng để đánh giá độ chính xác thực tế (Hyperparameter Tuning vs Generalization).

Dựa trên phân tích file `tamer/datamodule/datamodule.py`:

-   **Training Set**: Lấy từ thư mục định nghĩa trong `folder` + subfolder `train`. Dữ liệu này dùng để tính Gradient và cập nhật trọng số mô hình.
-   **Validation Set**: Trong source code này, tác giả cấu hình mặc định Val Set **TRÙNG VỚI** Test Set (thường là thư mục `2014`).
    -   *Logic Code*: `self.val_dataset` gọi tới `build_dataset(..., self.test_folder, ...)` (Dòng 190-197 trong `datamodule.py`).
    -   **Ý nghĩa**: Sau mỗi epoch, mô hình sẽ được kiểm tra ngay trên tập Test 2014 để xem độ chính xác. Điều này giúp bạn chọn được "Best Model" dựa trên khả năng hoạt động trên dữ liệu thực tế.

---

## 3. Các Chỉ Số Đánh Giá (Metrics)

Mô hình sử dụng các chỉ số sau để đo lường hiệu quả (được định nghĩa trong `tamer/utils/utils.py` và `lit_tamer.py`):

1.  **Loss (Hàm mất mát)**:
    -   **`train_loss` / `val_loss`**: Sử dụng Cross Entropy Loss. Đây là thước đo độ sai lệch giữa dự đoán của mô hình và nhãn thực tế. Giá trị càng nhỏ càng tốt (lý tưởng là tiệm cận 0).
    -   **`struct_loss`**: Loss cấu trúc, giúp mô hình học đúng thứ tự của cây cú pháp LaTeX.

2.  **ExpRate (Expression Rate)**:
    -   Đây là chỉ số quan trọng nhất cho bài toán HME.
    -   **Định nghĩa**: Tỷ lệ phần trăm số công thức được dự đoán **CHÍNH XÁC HOÀN TOÀN** (khớp 100% chuỗi LaTeX) so với nhãn gốc.
    -   Trong quá trình Test, code còn tính thêm `ExpRate <= 1` và `ExpRate <= 2` (cho phép sai lệch 1 hoặc 2 ký tự/token theo khoảng cách chỉnh sửa Edit Distance).

---

## 4. Phân Tích Quá Trình Training Của Bạn

Dựa trên biểu đồ bạn cung cấp (Epoch 145/200):

### Quan sát:
1.  **Training Loss (`train_loss`)**: Giảm rất sâu, gần tiệm cận 0. Điều này cho thấy mô hình đã học thuộc lòng tập dữ liệu huấn luyện.
2.  **Validation Loss (`val_loss`)**:
    -   Giảm nhanh ở khoảng 20-40 epoch đầu.
    -   Sau đó bắt đầu dao động (fluctuate) mạnh và có xu hướng đi ngang hoặc tăng nhẹ ở các epoch sau.
3.  **Validation Accuracy (`val_ExpRate`)**:
    -   Tăng nhanh và đạt đỉnh ở khoảng epoch 40-60.
    -   Sau đó đi ngang (plateau) và dao động, không tăng thêm đáng kể dù train loss tiếp tục giảm.

### Kết Luận: OVERFITTING (Quá khớp)
Mô hình hiện tại đang bị **Overfitting**.
-   **Tại sao?** Vì `train_loss` tiếp tục giảm trong khi `val_loss` không giảm nữa (thậm chí nhảy loạn xạ). Mô hình đang "học vẹt" các nhiễu (noise) của tập train thay vì học quy luật chung để áp dụng cho tập test.
-   **Thời điểm dừng tối ưu**: Khoảng epoch **50-80** là lúc mô hình đạt hiệu suất tốt nhất trên tập Validation. Việc train đến 145 epoch là hơi thừa và lãng phí tài nguyên, thậm chí làm giảm khả năng tổng quát hóa của model.

---

## 5. Khuyến Nghị Cải Thiện

1.  **Early Stopping**:
    -   Trong file `config/crohme.yaml`, tham số `early_stopping` đang là `false`. Hãy chỉnh thành `true` hoặc theo dõi thủ công để dừng khi `val_loss` không giảm trong 10-20 epoch liên tiếp.
2.  **Sử Dụng Checkpoint Tốt Nhất**:
    -   Pytorch Lightning tự động lưu checkpoint có `val_ExpRate` cao nhất (do cấu hình `mode: max`, `monitor: val_ExpRate`).
    -   Bạn không cần lấy checkpoint ở epoch 145. Hãy vào thư mục `lightning_logs/.../checkpoints` và lấy file có tên chứa `epoch={khoảng 40-60}` hoặc `val_ExpRate={cao nhất}`. Đó mới là model tốt nhất.
3.  **Data Augmentation**:
    -   Để giảm Overfitting, hãy bật `scale_aug: true` trong file config để tăng cường dữ liệu, giúp mô hình khó "học vẹt" hơn.

---

## 6. Hướng Dẫn Test & Đánh Giá

Sau khi train xong, để đánh giá độ chính xác cuối cùng (Test Accuracy), bạn chạy script `test.py` hoặc dùng `test_step` trong `train.py`.

Kết quả sẽ xuất ra 3 file quan trọng (đã được cấu hình tự động upload lên WandB):
1.  `result.zip`: Chứa từng file text kết quả dự đoán cho từng ảnh.
2.  `predictions.json`: Danh sách chi tiết dự đoán vs thực tế.
3.  `errors.json`: Chỉ chứa các trường hợp dự đoán sai để bạn phân tích lỗi sai (thường do viết xấu, hoặc mô hình nhầm lẫn các ký tự giống nhau như `0` và `o`, `1` và `|`).
4.  `[year].txt`: Report tóm tắt chứa ExpRate.

Tóm lại, quá trình train của bạn đã thành công nhưng nên dừng sớm hơn để tiết kiệm thời gian và tài nguyên, đồng thời chọn checkpoint đúng để sử dụng.
