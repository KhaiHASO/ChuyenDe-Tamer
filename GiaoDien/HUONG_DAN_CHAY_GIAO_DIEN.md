# Hướng Dẫn Chạy Giao Diện Pix2Text

Dự án Pix2Text hỗ trợ 2 cách chạy giao diện:

1. **Giao diện Web Streamlit** (GUI đơn giản, dễ sử dụng)
2. **HTTP Service** (API service, phù hợp cho tích hợp)

---

## 1. Cài Đặt Dependencies
```
conda create -n pix2text python=3.9 -y
conda activate pix2text
```
Trước tiên, đảm bảo bạn đã cài đặt các package cần thiết:

```bash
# Cài đặt pix2text

# Nếu cần nhận diện ngôn ngữ khác ngoài tiếng Anh và tiếng Trung

# Cài đặt Streamlit (cho giao diện web)

# Cài đặt FastAPI và uvicorn (cho HTTP service - thường đã có trong pix2text)
pip install pix2text
pip install pix2text[multilingual]
pip install streamlit
pip install fastapi uvicorn
```

---

## 2. Cách 1: Chạy Giao Diện Web Streamlit (GUI)

Đây là giao diện web đơn giản, cho phép upload ảnh và xem kết quả trực tiếp trên trình duyệt.

### Bước 1: Chạy lệnh

```bash
streamlit run pix2text/app.py
```

Hoặc nếu bạn đang ở thư mục gốc của project:

```bash
streamlit run pix2text/app.py
```

### Bước 2: Mở trình duyệt

Lệnh trên sẽ tự động mở trình duyệt tại địa chỉ: `http://localhost:8501`

Nếu không tự động mở, bạn có thể truy cập thủ công:
- URL: `http://localhost:8501`
- Hoặc: `http://127.0.0.1:8501`

### Bước 3: Sử dụng

1. Upload ảnh (hỗ trợ: PNG, JPG, JPEG, WEBP)
2. Hệ thống sẽ tự động nhận diện
3. Xem kết quả ngay trên trang web

### Tùy chỉnh Port (nếu cần)

```bash
streamlit run pix2text/app.py --server.port 8502
```

---

## 3. Cách 2: Chạy HTTP Service (API)

HTTP Service cho phép gọi API từ các ứng dụng khác.

### Bước 1: Chạy lệnh

```bash
# Cách đơn giản nhất
p2t serve -l en,ch_sim -H 0.0.0.0 -p 8503
```

Trong đó:
- `-l en,ch_sim`: Ngôn ngữ nhận diện (English, Simplified Chinese)
- `-H 0.0.0.0`: Host (0.0.0.0 cho phép truy cập từ mọi địa chỉ)
- `-p 8503`: Port (mặc định là 8503)

### Bước 2: Kiểm tra Service

Service sẽ chạy tại: `http://0.0.0.0:8503` hoặc `http://localhost:8503`

Bạn có thể kiểm tra bằng cách mở trình duyệt và truy cập:
```
http://localhost:8503
```

Sẽ thấy thông báo: `{"message": "Welcome to Pix2Text Server!"}`

### Bước 3: Gọi API

#### Python:

```python
import requests

url = 'http://localhost:8503/pix2text'

image_fp = 'docs/examples/page2.png'
data = {
    "file_type": "page",
    "resized_shape": 768,
    "embed_sep": " $,$ ",
    "isolated_sep": "$$\n, \n$$"
}
files = {
    "image": (image_fp, open(image_fp, 'rb'), 'image/jpeg')
}

r = requests.post(url, data=data, files=files)
result = r.json()
print(result['results'])
```

#### Curl:

```bash
curl -X POST \
  -F "file_type=page" \
  -F "resized_shape=768" \
  -F "embed_sep= $,$ " \
  -F "isolated_sep=$$\n, \n$$" \
  -F "image=@docs/examples/page2.png;type=image/jpeg" \
  http://localhost:8503/pix2text
```

### Tùy chỉnh nâng cao:

```bash
# Sử dụng GPU
p2t serve -l en,ch_sim -d cuda:0 -H 0.0.0.0 -p 8503

# Tắt nhận diện công thức toán
p2t serve -l en,ch_sim --disable-formula -H 0.0.0.0 -p 8503

# Tắt nhận diện bảng
p2t serve -l en,ch_sim --disable-table -H 0.0.0.0 -p 8503

# Xem tất cả các tùy chọn
p2t serve -h
```

---

## 4. So Sánh 2 Cách

| Tính năng | Streamlit GUI | HTTP Service |
|-----------|--------------|--------------|
| Giao diện | Web UI đơn giản | API (không có UI) |
| Dễ sử dụng | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Tích hợp | Khó | Dễ |
| Phù hợp | Người dùng cuối | Developer |
| Port mặc định | 8501 | 8503 |

---

## 5. Xử Lý Lỗi Thường Gặp

### Lỗi: TypeError: 'Page' object is not subscriptable

**Nguyên nhân:** Khi bạn sử dụng `p2t(img)` hoặc `p2t.recognize_page()`, nó trả về một đối tượng `Page` (không phải dictionary). Bạn không thể truy cập bằng `page['something']` mà phải dùng thuộc tính như `page.elements` hoặc phương thức `page.to_markdown()`.

**Giải pháp:**

1. **Nếu bạn muốn nhận dictionary với keys 'text', 'image_type':**
   ```python
   # Thay vì: out = p2t(img)
   # Dùng:
   out = p2t.recognize(img, file_type='text_formula', return_text=True)
   # Khi return_text=True, nó trả về string
   print(out)  # Đây là string, không phải dict
   
   # Hoặc nếu muốn dict:
   out = p2t.recognize(img, file_type='text_formula', return_text=False)
   # Trả về list of dicts
   ```

2. **Nếu bạn đã có Page object và muốn lấy text:**
   ```python
   page = p2t.recognize_page(img)
   # Lấy markdown
   md_text = page.to_markdown('output-dir')
   # Hoặc truy cập elements
   for element in page.elements:
       print(element.text)
   ```

3. **File app.py đã được sửa:** Phiên bản mới của `app.py` đã xử lý đúng lỗi này.

### Lỗi: ModuleNotFoundError: No module named 'streamlit'

**Giải pháp:**
```bash
pip install streamlit
```

### Lỗi: Port đã được sử dụng

**Giải pháp:** Đổi port khác
```bash
# Streamlit
streamlit run pix2text/app.py --server.port 8502

# HTTP Service
p2t serve -l en,ch_sim -H 0.0.0.0 -p 8504
```

### Lỗi: Model không tải được

**Giải pháp:** Lần đầu chạy, hệ thống sẽ tự động tải model. Đảm bảo có kết nối internet.

### Lỗi: Out of memory

**Giải pháp:** Sử dụng CPU thay vì GPU, hoặc giảm `resized_shape`:
```bash
p2t serve -l en,ch_sim -d cpu -H 0.0.0.0 -p 8503
```

---

## 6. Tài Liệu Tham Khảo

- Tài liệu chính thức: https://pix2text.readthedocs.io
- GitHub: https://github.com/breezedeus/pix2text
- Hướng dẫn sử dụng CLI: `docs/command.md`

---

## 7. Lưu Ý

- **Lần đầu chạy**: Hệ thống sẽ tự động tải các model cần thiết (có thể mất vài phút)
- **Yêu cầu RAM**: Khuyến nghị ít nhất 4GB RAM
- **GPU**: Tùy chọn, có thể chạy trên CPU nhưng sẽ chậm hơn
- **Internet**: Cần kết nối internet lần đầu để tải model

---

Chúc bạn sử dụng thành công! 🎉

