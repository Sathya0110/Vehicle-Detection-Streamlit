import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚗 Vehicle Detection System")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(np.array(image))

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Vehicles")

    vehicles = ["car","truck","bus","motorcycle"]
    count = 0

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        if label in vehicles:
            count += 1

    st.success(f"Total Vehicles Detected: {count}")
