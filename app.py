import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI Traffic Monitoring Demo", page_icon="🚗", layout="wide")

st.title("🚗 Intelligent Traffic Monitoring System (YOLOv12)")
st.caption("Proyek Skripsi S1 Matematika Universitas Brawijaya | Deploy Demo")

@st.cache_resource
def load_yolo_model():
    return YOLO("app/models/best.pt") # atau 'best.pt' sesuai lokasi weights Anda

try:
    model = load_yolo_model()
    st.sidebar.success("✅ Model YOLOv12 Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")

st.subheader("🖼️ Deteksi Objek pada Sampel Gambar CCTV")
image_file = st.file_uploader("Unggah sampel citra CCTV...", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Gambar Input", use_container_width=True)
        
    if st.button("🔍 Jalankan Deteksi YOLOv12"):
        img_array = np.array(image)
        results = model(img_array)[0]
        res_plotted = results.plot()
        
        with col2:
            st.image(res_plotted, caption="Hasil Deteksi YOLOv12", use_container_width=True)
          
