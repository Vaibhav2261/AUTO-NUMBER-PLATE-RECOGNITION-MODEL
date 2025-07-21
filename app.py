import cv2
import numpy as np
import streamlit as st
from PIL import Image

from inference.run import infer_on_image

st.title("Automatic Number Plate Recognition (ANPR)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)  # Convert PIL to NumPy array (for OpenCV)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    processed_img, texts = infer_on_image(img_np)

    # Convert BGR back to RGB for display
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    st.image(
        processed_img_rgb, caption="Detected Number Plate(s)", use_column_width=True
    )

    st.markdown("### Extracted Numbers:")
    for i, text in enumerate(texts):
        st.write(f"**Plate {i + 1}:** `{text}`")
