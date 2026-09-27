import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Acne Detector")

model = YOLO("best.pt")

uploaded = st.file_uploader("Image upload karo", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    results = model.predict(img, conf=0.25)
    st.image(results[0].plot()[..., ::-1], caption="Result")
