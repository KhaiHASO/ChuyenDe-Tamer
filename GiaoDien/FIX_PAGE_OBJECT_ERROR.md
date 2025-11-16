# Fix Lỗi: 'Page' object is not subscriptable

## 📋 Tóm Tắt

Lỗi `TypeError: 'Page' object is not subscriptable` xảy ra khi code cố truy cập đối tượng `Page` như một dictionary (dùng dấu ngoặc vuông `[]`), trong khi `Page` là một class object chỉ hỗ trợ truy cập bằng thuộc tính (dot notation).

## 🔍 Nguyên Nhân

1. **`p2t(img)` hoặc `p2t.__call__(img)`** trả về một `Page` object (không phải dictionary)
2. **`p2t.recognize_page(img)`** cũng trả về `Page` object
3. Code cố truy cập như: `page['text']`, `page['image_type']` → **LỖI**

## ✅ Giải Pháp Đã Áp Dụng

### 1. Sửa File `pix2text/app.py`

**Trước:**
```python
out = p2t(img)  # Trả về Page object
out['image_type']  # ❌ Lỗi!
out['text']  # ❌ Lỗi!
```

**Sau:**
```python
# Dùng recognize() với text_formula để nhận string
out = p2t.recognize(img, file_type='text_formula', return_text=True)
# Trả về string, có thể hiển thị trực tiếp
```

### 2. Cách Xử Lý Đúng

#### Option A: Dùng `recognize()` với `text_formula`
```python
# Trả về string khi return_text=True
out = p2t.recognize(img, file_type='text_formula', return_text=True)
print(out)  # Đây là string

# Hoặc trả về list of dicts khi return_text=False
out = p2t.recognize(img, file_type='text_formula', return_text=False)
# out là list of dicts, mỗi dict có keys: 'type', 'text', 'position', etc.
```

#### Option B: Xử Lý Page Object Đúng Cách
```python
page = p2t.recognize_page(img)  # Trả về Page object

# Cách 1: Lấy markdown
md_text = page.to_markdown('output-dir')

# Cách 2: Truy cập elements
for element in page.elements:
    print(f"Type: {element.type}")
    print(f"Text: {element.text}")
    print(f"Position: {element.box}")
```

## 📝 So Sánh Các Phương Thức

| Phương Thức | Return Type | Cách Truy Cập |
|------------|-------------|---------------|
| `p2t(img)` hoặc `p2t.__call__(img)` | `Page` | `page.elements`, `page.to_markdown()` |
| `p2t.recognize_page(img)` | `Page` | `page.elements`, `page.to_markdown()` |
| `p2t.recognize(img, file_type='text_formula', return_text=True)` | `str` | Dùng trực tiếp như string |
| `p2t.recognize(img, file_type='text_formula', return_text=False)` | `List[Dict]` | `out[0]['text']`, `out[0]['type']` |
| `p2t.recognize(img, file_type='formula', return_text=True)` | `str` | Dùng trực tiếp |
| `p2t.recognize(img, file_type='text', return_text=True)` | `str` | Dùng trực tiếp |

## 🎯 Các File Đã Được Sửa

1. ✅ `pix2text/app.py` - Đã sửa để sử dụng `recognize()` với `text_formula`
2. ✅ `HUONG_DAN_CHAY_GIAO_DIEN.md` - Đã thêm phần hướng dẫn xử lý lỗi này

## 🧪 Test

Để test xem fix có hoạt động:

```bash
# Chạy Streamlit app
streamlit run pix2text/app.py

# Upload ảnh và kiểm tra xem có còn lỗi không
```

## 📚 Tham Khảo

- File `pix2text/page_elements.py` - Định nghĩa class `Page`
- File `pix2text/pix_to_text.py` - Các phương thức recognize
- Documentation: https://pix2text.readthedocs.io

---

**Lưu ý:** Nếu bạn gặp lỗi tương tự trong code của mình, hãy kiểm tra xem bạn đang dùng phương thức nào và return type của nó là gì.

