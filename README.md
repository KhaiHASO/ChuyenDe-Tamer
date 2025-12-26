# TAMER - Handwritten Mathematical Expression Recognition

Dự án này triển khai mô hình **TAMER** (Two-way Attention-based Model for Expression Recognition) cho nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression - HME).

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Đánh giá](#đánh-giá)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)

## 🎯 Tổng quan

TAMER là mô hình encoder-decoder sử dụng:
- **Encoder**: DenseNet-based để trích xuất đặc trưng từ ảnh biểu thức toán học
- **Decoder**: Transformer-based decoder với cơ chế attention
- **Training**: Bidirectional training (two-way) để cải thiện hiệu suất
- **Inference**: Beam search để tìm kiếm chuỗi LaTeX tốt nhất

Dự án bao gồm 2 phiên bản:
- **0-baseline**: Phiên bản TAMER gốc
- **1-gat**: Phiên bản TAMER tích hợp Graph Attention Network (GAT) để cải thiện khả năng nhận dạng

## 📁 Cấu trúc dự án

```
ChuyenDe-Tamer/
├── 0-baseline/          # Phiên bản TAMER gốc
│   ├── config/          # File cấu hình YAML
│   ├── eval/            # Scripts đánh giá
│   ├── tamer/           # Package chính
│   │   ├── model/       # Kiến trúc mô hình
│   │   ├── datamodule/  # Xử lý dữ liệu
│   │   └── utils/       # Tiện ích
│   ├── train.py         # Script training
│   └── requirements.txt
│
├── 1-gat/               # Phiên bản TAMER với GAT
│   ├── config/
│   ├── eval/
│   ├── tamer/
│   │   ├── model/
│   │   │   └── gat.py   # Graph Attention Network
│   │   ├── datamodule/
│   │   └── utils/
│   ├── train.py
│   └── requirements.txt
│
└── KetQua/              # Kết quả và checkpoints
```

## 🔧 Cài đặt

### Yêu cầu

- Python 3.7+
- PyTorch 1.8+
- PyTorch Lightning

### Cài đặt dependencies

```bash
# Cài đặt cho baseline
cd 0-baseline
pip install -r requirements.txt
pip install -e .

# Hoặc cài đặt cho phiên bản GAT
cd 1-gat
pip install -r requirements.txt
pip install -e .
```

### Cài đặt PyTorch

Đảm bảo cài đặt PyTorch phù hợp với hệ thống của bạn:

```bash
# Ví dụ cho CUDA 11.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu111
```

## 🚀 Sử dụng

### Training

Sử dụng PyTorch Lightning CLI để training:

```bash
cd 0-baseline  # hoặc cd 1-gat

# Training với cấu hình mặc định
python train.py fit --config config/crohme.yaml

# Training với cấu hình debug (số lượng mẫu nhỏ)
python train.py fit --config config/crohme_debug.yaml
```

### Đánh giá (Evaluation)

```bash
cd 0-baseline/eval  # hoặc cd 1-gat/eval

# Đánh giá trên CROHME
bash eval_crohme.sh

# Đánh giá trên HME100K
bash eval_hme100k.sh

# Hoặc chạy trực tiếp
python test.py \
    --folder data/crohme \
    --version 0 \
    --test_year 2014 \
    --max_size 320000 \
    --scale_to_limit true
```

### Sử dụng Jupyter Notebook

Có thể sử dụng các notebook có sẵn:
- `TAMER_Kaggle_Setup.ipynb`: Setup cho Kaggle
- `tamer37_baseline.ipynb`: Notebook training và evaluation

## ⚙️ Cấu hình

Các file cấu hình YAML nằm trong thư mục `config/`:

### Cấu hình mô hình (model)

```yaml
model:
  d_model: 256              # Kích thước embedding
  growth_rate: 24           # Growth rate cho DenseNet encoder
  num_layers: 16            # Số lớp DenseNet
  nhead: 8                 # Số attention heads
  num_decoder_layers: 3    # Số lớp decoder
  dim_feedforward: 1024    # Kích thước feedforward
  dropout: 0.3
  vocab_size: 113          # Kích thước vocabulary
  cross_coverage: true     # Cross attention coverage
  self_coverage: true      # Self attention coverage
  beam_size: 10            # Beam search size
  max_len: 150             # Độ dài tối đa sequence
```

### Cấu hình dữ liệu (data)

```yaml
data:
  folder: data/crohme      # Thư mục dữ liệu
  test_folder: 2014       # Thư mục test
  max_size: 320000        # Kích thước ảnh tối đa
  scale_to_limit: true
  train_batch_size: 8
  eval_batch_size: 2
  num_workers: 5
```

### Cấu hình trainer

```yaml
trainer:
  gpus: 1
  max_epochs: 100
  precision: 16            # Mixed precision training
  check_val_every_n_epoch: 2
  deterministic: true
```

## 📊 Đánh giá

Mô hình được đánh giá bằng **ExpRate** (Expression Rate):
- **ExpRate**: Tỷ lệ biểu thức được nhận dạng chính xác hoàn toàn
- **ExpRate<=1**: Tỷ lệ biểu thức có edit distance <= 1
- **ExpRate<=2**: Tỷ lệ biểu thức có edit distance <= 2

Kết quả được lưu trong:
- `lightning_logs/version_X/`: Logs và checkpoints
- `errors_YEAR.json`: Các lỗi nhận dạng
- `predictions.json`: Tất cả predictions

## 🏗️ Kiến trúc mô hình

### TAMER Baseline

```
Input Image → DenseNet Encoder → Positional Encoding
                                    ↓
                            Transformer Decoder
                                    ↓
                            Bidirectional Training
                                    ↓
                            Beam Search → LaTeX Output
```

### TAMER-GAT

Phiên bản `1-gat` tích hợp Graph Attention Network vào encoder để:
- Mô hình hóa quan hệ không gian giữa các ký tự
- Cải thiện khả năng nhận dạng các biểu thức phức tạp

### Các thành phần chính

1. **Encoder** (`tamer/model/encoder.py`):
   - DenseNet-based feature extraction
   - Positional encoding cho ảnh

2. **Decoder** (`tamer/model/decoder.py`):
   - Transformer decoder với multi-head attention
   - Coverage mechanism để tránh lặp lại

3. **Beam Search** (`tamer/utils/beam_search.py`):
   - Tìm kiếm chuỗi LaTeX tối ưu

4. **Data Module** (`tamer/datamodule/`):
   - Xử lý dữ liệu CROHME và HME100K
   - Chuyển đổi LaTeX sang Ground Truth Data (GTD)

## 📚 Datasets

Dự án hỗ trợ:
- **CROHME**: Competition on Recognition of Online Handwritten Mathematical Expressions
- **HME100K**: Large-scale handwritten math expression dataset

## 🔍 Các tính năng

- ✅ Bidirectional training
- ✅ Coverage mechanism (cross & self)
- ✅ Beam search inference
- ✅ Mixed precision training (FP16)
- ✅ Multi-GPU support (DDP)
- ✅ Graph Attention Network (phiên bản GAT)

## 📝 Ghi chú

- Checkpoints được lưu tự động trong `lightning_logs/`
- Model checkpoint tốt nhất được chọn dựa trên `val_ExpRate`
- Sử dụng seed=7 để đảm bảo reproducibility



