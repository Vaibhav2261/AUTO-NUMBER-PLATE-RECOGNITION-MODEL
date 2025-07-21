import streamlit as st
from inference.run import infer_on_image

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
if uploaded_file:
    img = Image.open(uploaded_file)
    text, processed_img = infer_on_image(img)
    st.image(processed_img, caption="Detected Plate")
    st.write(f"Extracted Number: `{text}`")
