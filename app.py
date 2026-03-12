import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# --- Streamlit page setup ---
st.set_page_config(page_title="Vehicle Detection from Images", layout="wide")
st.title("🚗 Vehicle Detection System (Images)")

# --- Load YOLO model (auto-downloads weights if missing) ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- Sidebar settings ---
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- Helper functions ---
def draw_boxes(frame, results):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = f"{results[0].names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def count_vehicles(results):
    counts = {}
    for cls in results[0].boxes.cls:
        cls_name = results[0].names[int(cls)]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

# --- Upload multiple images ---
uploaded_files = st.file_uploader("Upload images (jpg, png)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Detection Results")
    cols = st.columns(len(uploaded_files))  # display side by side

    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        frame = np.array(image)

        # --- YOLO Detection ---
        results = model(frame)
        results = results[0].filter(conf=conf_thres)

        # --- Draw boxes ---
        frame = draw_boxes(frame, results)

        # --- Count vehicles ---
        counts = count_vehicles(results)
        df_counts = pd.DataFrame(list(counts.items()), columns=["Vehicle", "Count"])

        # --- Display ---
        with cols[i]:
            st.image(frame, channels="RGB", caption=f"Processed: {file.name}")
            if not df_counts.empty:
                st.table(df_counts)
