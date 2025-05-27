import streamlit as st
import requests
from PIL import Image
import os


API_URL = "http://localhost:8000/search_cosine"  
MAX_IMAGES = 10
st.set_page_config(page_title="Text-to-Image Retrieval", layout="wide")
st.title("🔍 Text-to-Image Retrieval using CLIP + FastAPI")


query_text = st.text_input("Nhập vào tên con vật muốn kiếm", value="a tiger")

top_k = st.slider("Xuất hiện số lượng k ảnh có điểm tương đồng", min_value=1, max_value=MAX_IMAGES, value=5)

if st.button("Tìm kiếm"):
    if not query_text.strip():
        st.warning("Vui lòng nhập tên con vật")
    else:
        with st.spinner("Tìm kiếm ..."):
            response = requests.post(API_URL, json={"text": query_text, "top_k": top_k})

        if response.status_code == 200:
            result_paths = response.json().get("results", [])
            if result_paths:
                st.success(f"Found {len(result_paths)} results.")
                cols = st.columns(len(result_paths))
                for i, img_path in enumerate(result_paths):
                    try:
                        image = Image.open(img_path)
                        cols[i].image(image, caption=os.path.basename(img_path), use_column_width=True)
                    except Exception as e:
                        cols[i].error(f"Không thể xuất ảnh: {img_path}")
            else:
                st.info("Không có ảnh phù hợp")
        else:
            st.error("Không kết nối được với API")
