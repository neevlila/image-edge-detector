import streamlit as st
from PIL import Image
import numpy as np

from src.edge_detection_algorithms.sobel_edge_detector import sobel_edge_detect

st.title("Sobel Edge Detection Web App (Vector Calculus)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])
threshold = st.slider("Edge threshold", min_value=0, max_value=255, value=50)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(image)

    with st.spinner("Detecting edges… this takes a few seconds ⚙️"):
        edge_rgb, gx_vis, gy_vis, magnitude_vis = sobel_edge_detect(img_arr)

    # Apply threshold to magnitude image for final binary edges
    threshold_mask = (magnitude_vis.astype(np.uint8) >= threshold)
    edge_thresholded = np.zeros_like(magnitude_vis, dtype=np.uint8)
    edge_thresholded[threshold_mask] = 255
    edge_thresholded_rgb = np.stack([edge_thresholded] * 3, axis=-1)

    st.subheader("Input and Outputs")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=350)
        st.subheader("Gradient X")
        st.image(gx_vis, width=350, clamp=True)

    with col2:
        st.subheader("Gradient Y")
        st.image(gy_vis, width=350, clamp=True)
        st.subheader(f"Edge-detected (threshold {threshold})")
        st.image(edge_thresholded_rgb, width=350)

    st.subheader("Gradient Magnitude")
    st.image(edge_rgb, width=350, clamp=True)