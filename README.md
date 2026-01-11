# TAMER - Handwritten Mathematical Expression Recognition

Dự án này triển khai mô hình **TAMER** (Two-way Attention-based Model for Expression Recognition) cho nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression - HME).

**Tác giả:** Phan Hoàng Khải  
**Đơn vị:** Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE)

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Deep Dive: Graph Attention Networks (GAT)](#deep-dive-graph-attention-networks-gat)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Kết quả](#kết-quả)

## 🎯 Tổng quan

TAMER là một kiến trúc mạnh mẽ kết hợp giữa CNN và Transformer để chuyển đổi hình ảnh biểu thức toán học viết tay thành chuỗi LaTeX. Dự án này bao gồm hai phiên bản chính:

1.  **0-baseline**: Phiên bản chuẩn sử dụng DenseNet làm Encoder và Transformer làm Decoder.
2.  **1-gat**: Phiên bản nâng cấp tích hợp **Graph Attention Networks (GAT)** vào bộ mã hóa (Encoder) để tăng cường khả năng trích xuất đặc trưng không gian và cấu trúc của biểu thức.

## 📁 Cấu trúc dự án

```
ChuyenDe-Tamer/
├── 0-baseline/          # Phiên bản TAMER gốc (DenseNet + Transformer)
├── 1-gat/               # Phiên bản nâng cao (DenseNet + GAT + Transformer)
│   ├── tamer/
│   │   ├── model/
│   │   │   ├── gat.py   # Cài đặt lớp Graph Attention
│   │   │   └── encoder.py # Encoder tích hợp GAT
│   │   └── ...
├── KetQua/              # Lưu trữ kết quả thực nghiệm
└── README.md            # Tài liệu dự án
```

## 🧠 Deep Dive: Graph Attention Networks (GAT)

Điểm nhấn của dự án này là việc tích hợp **Graph Attention Networks (GAT)** vào kiến trúc Encoder. Dưới đây là phân tích chi tiết kỹ thuật về cách GAT hoạt động trong bài toán này:

### Tại sao lại dùng GAT?

Các mạng CNN truyền thống (như DenseNet) rất giỏi trong việc trích xuất đặc trưng cục bộ (local features). Tuy nhiên, đối với biểu thức toán học, mối quan hệ giữa các ký tự không chỉ nằm ở vị trí lân cận mà còn phụ thuộc vào cấu trúc ngữ nghĩa 2D (ví dụ: phân số, số mũ, chỉ số dưới).

GAT cho phép mô hình coi bản đồ đặc trưng (feature map) như một đồ thị, nơi mỗi điểm ảnh (pixel) hoặc vùng đặc trưng là một nút (node). Cơ chế Attention giúp mỗi nút có thể "tập trung" (attend) vào các nút lân cận quan trọng nhất để tổng hợp thông tin, thay vì nhân chập cố định như CNN.

### Kiến trúc chi tiết (Implementation Details)

Module GAT được cài đặt trong `1-gat/tamer/model/gat.py` và `1-gat/tamer/model/encoder.py`.

1.  **Xây dựng Đồ thị (Graph Construction)**:
    *   Feature map đầu ra từ DenseNet có kích thước `[H, W, D]`.
    *   Ta biến đổi feature map này thành một lưới đồ thị (grid graph) với `N = H * W` nút.
    *   **Adjacency Matrix**: Xây dựng ma trận kề dựa trên kết nối 4 hướng (4-connectivity: trên, dưới, trái, phải). Mỗi nút được kết nối với 4 nút lân cận của nó.

2.  **Cơ chế GAT Layer**:
    *   Mỗi lớp GAT (`GATLayer`) sử dụng **Multi-head Attention**.
    *   Đầu vào là các features của nút $h_i$.
    *   Hệ số attention $e_{ij}$ giữa nút $i$ và nút lân cận $j$ được tính toán thông qua một mạng nơ-ron truyền thẳng (feed-forward neural network):
        $$e_{ij} = \text{LeakyReLU}(\vec{a}^T [W\vec{h}_i || W\vec{h}_j])$$
    *   Hệ số này sau đó được chuẩn hóa bằng Softmax để tạo ra trọng số $\alpha_{ij}$.
    *   Đầu ra của nút $i$ là tổng có trọng số của các nút lân cận:
        $$\vec{h}'_i = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\vec{h}_j)$$

3.  **Tích hợp vào Encoder**:
    *   Quy trình xử lý: `Image -> DenseNet -> Feature Map -> Flatten -> GAT Layers -> Reshape -> Feature Map -> Positional Encoding -> Transformer Decoder`.
    *   Việc chèn GAT vào giữa DenseNet và Transformer giúp làm giàu feature map với thông tin ngữ cảnh cấu trúc trước khi giải mã.

## 🔧 Cài đặt

Yêu cầu môi trường:
- Python 3.7+
- PyTorch 1.8+
- CUDA (nếu dùng GPU)

Cài đặt các gói phụ thuộc:

```bash
# Cài đặt cho phiên bản GAT (Khuyên dùng)
cd 1-gat
pip install -r requirements.txt
pip install -e .
```

Nếu muốn chạy baseline:
```bash
cd 0-baseline
pip install -r requirements.txt
pip install -e .
```

## � Sử dụng

### Quá trình Huấn luyện (Training)

Để huấn luyện mô hình, sử dụng script `train.py`. Bạn có thể thay đổi cấu hình trong thư mục `config/`.

```bash
# Di chuyển vào thư mục source code
cd 1-gat

# Chạy huấn luyện với file config mặc định
python train.py fit --config config/crohme.yaml

# Debug nhanh với dữ liệu nhỏ
python train.py fit --config config/crohme_debug.yaml
```

### Đánh giá (Evaluation)

Sử dụng các script trong thư mục `eval/` để đánh giá mô hình đã huấn luyện.

```bash
cd 1-gat/eval

# Đánh giá trên tập dữ liệu CROHME
bash eval_crohme.sh
```

## ⚙️ Cấu hình

Các tham số quan trọng trong `config/crohme.yaml`:

- **model**:
    - `d_model`: 256 (Kích thước vector đặc trưng)
    - `use_gat`: true (Bật tắt module GAT)
    - `gat_num_layers`: 2 (Số lớp GAT chồng lên nhau)
    - `gat_num_heads`: 8 (Số đầu attention trong GAT)
- **data**:
    - `folder`: Đường dẫn đến dữ liệu ảnh
    - `batch_size`: Kích thước batch

## � Kết quả

Kết quả đánh giá được đo bằng **Expression Rate (ExpRate)**. Phiên bản tích hợp GAT được kỳ vọng sẽ xử lý tốt hơn các trường hợp biểu thức có cấu trúc phức tạp hoặc nét viết chồng chéo nhờ khả năng lan truyền tin (message passing) linh hoạt của mạng đồ thị.

---
© 2025 Phan Hoàng Khải - Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE).
