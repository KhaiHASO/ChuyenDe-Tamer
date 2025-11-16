# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from PIL import Image
import streamlit as st

from pix2text import set_logger, Pix2Text

logger = set_logger()
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_model():
    return Pix2Text()


def main():
    p2t = get_model()

    title = 'Công cụ chuyển đổi kí tự toán học viết tay thành Latex'
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

    subtitle = 'Tác giả: <a href="https://github.com/KhaiHASO">Phan Hoàng Khải</a>; '

    st.markdown(f"<div style='text-align: center;'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown('')
    st.subheader('Chọn ảnh cần nhận diện')
    content_file = st.file_uploader('', type=["png", "jpg", "jpeg", "webp"])
    if content_file is None:
        st.stop()

    try:
        img = Image.open(content_file).convert('RGB')
        img.save('ori.jpg')

        # Use recognize() with text_formula to get dictionary result
        # This avoids the 'Page' object is not subscriptable error
        out = p2t.recognize(img, file_type='text_formula', return_text=True)
        logger.info(out)
        st.markdown('##### Ảnh gốc:')
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.image(content_file)

        st.subheader('Kết quả nhận diện:')
        
        # Handle different return types
        if isinstance(out, str):
            # When return_text=True, recognize() returns a string
            st.markdown("* **Nội dung nhận diện:**")
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.text(out)
        elif isinstance(out, dict):
            # Legacy dictionary format (if any)
            image_type = out.get('image_type', 'unknown')
            text = out.get('text', '')
            st.markdown(f"* **Loại ảnh:** {image_type}")
            st.markdown("* **Nội dung nhận diện:**")
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.text(text)
                if image_type == 'formula':
                    st.markdown(f"$${text}$$")
        else:
            # If it's a Page object (shouldn't happen with text_formula, but handle it)
            st.markdown("* **Nội dung nhận diện:**")
            cols = st.columns([1, 3, 1])
            with cols[1]:
                # Convert Page to markdown string
                md_text = out.to_markdown('temp-output') if hasattr(out, 'to_markdown') else str(out)
                st.text(md_text)

    except Exception as e:
        st.error(e)
        import traceback
        st.error(traceback.format_exc())


if __name__ == '__main__':
    main()
