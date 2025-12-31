# Hướng Dẫn Tích Hợp WandB với TAMER

Repository này hỗ trợ theo dõi quá trình huấn luyện bằng [Weights & Biases (WandB)](https://wandb.ai/). Điều này giúp bạn theo dõi các chỉ số như loss, accuracy theo thời gian thực, xem kết quả dự đoán (hình ảnh và công thức), và giám sát tài nguyên hệ thống.

## Cài đặt

### Cài đặt thư viện
Cài đặt các gói cần thiết:
```bash
pip install -r requirements.txt
```
Lệnh này sẽ cài `wandb` và `pandas`.

## Sử dụng

Có 3 cách để xác thực API Key của WandB, được kiểm tra theo thứ tự ưu tiên:

1. **Tham số dòng lệnh (Khuyên dùng cho Kaggle)**:
   Bạn có thể truyền trực tiếp key khi chạy lệnh train thông qua cờ `--wandb_api_key`. Cách này tiện lợi khi chạy trên Kaggle hoặc Colab.
   
   ```bash
   python train.py --config config/crohme.yaml --wandb_api_key=YOUR_WANDB_API_KEY
   ```
   **Lưu ý**: Key sẽ được tự động ẩn khỏi log của hệ thống để bảo mật (nếu có thể), nhưng hãy cẩn thận khi chia sẻ notebook công khai.

2. **Kaggle Secrets (Khuyên dùng cho Kaggle Script)**:
   - Vào Notebook -> Add-ons -> Secrets.
   - Thêm secret mới với Label: `wandb_api_key` và Value: `YOUR_WANDB_API_KEY`.
   - Script `train.py` (nếu đã được cấu hình thêm) có thể tự động đọc. Tuy nhiên, sử dụng tham số dòng lệnh ở trên là cách trực tiếp nhất nếu bạn không muốn sửa code để đọc secret. *Trong phiên bản hiện tại, chúng tôi ưu tiên dùng tham số dòng lệnh hoặc biến môi trường.*

3. **Biến môi trường**:
   - Thiết lập `WandB_API_KEY` trong môi trường của bạn.
   - Linux/Mac: `export WANDB_API_KEY=YOUR_KEY`
   - Windows: `$env:WANDB_API_KEY="YOUR_KEY"`
   - Sau đó chạy lệnh train bình thường.

### Lệnh chạy mẫu trên Kaggle
```bash
!source /kaggle/working/miniconda/bin/activate tamer && \
python train.py \
--config config/crohme.yaml \
--trainer.max_epochs=200 \
--trainer.gpus=2 \
--wandb_api_key=3010beaefcbb3ca747099418f4dd36cd474cc81c
```

## Các tính năng

- **Metric Logging**: Tự động theo dõi `train_loss`, `val_loss`, `ExpRate` thông qua WandbLogger được cấu hình trong `config/crohme.yaml`.
- **System Monitoring**: Theo dõi CPU, GPU, RAM.
- **Rich Media**: Trong dashboard WandB, mục "Tables" hoặc "Images" sẽ hiển thị 8 mẫu validation mỗi epoch bao gồm:
    - **Image**: Ảnh công thức toán học.
    - **Ground Truth**: Công thức LaTeX đúng.
    - **Prediction**: Công thức LaTeX do mô hình dự đoán.
  Điều này giúp bạn đánh giá trực quan hiệu quả của mô hình.
