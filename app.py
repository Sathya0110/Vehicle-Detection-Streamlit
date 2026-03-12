import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import tempfile

# --- Helper functions ---
def draw_boxes(frame, results):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{results[0].names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return frame

def count_vehicles(results):
    counts = {}
    for cls in results[0].boxes.cls:
        cls_name = results[0].names[int(cls)]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

# --- Streamlit UI ---
st.set_page_config(page_title="Vehicle Detection Dashboard", layout="wide")
st.title("🚗 Vehicle Detection System")

video_file = st.file_uploader("Upload a video file", type=["mp4","avi","mov"])
use_webcam = st.checkbox("Use Webcam Instead of Video", value=False)
conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Auto-downloads in cloud
model = load_model()

stframe = st.empty()
vehicle_count_df = pd.DataFrame(columns=["Vehicle", "Count"])

def process_frame(frame):
    results = model(frame)
    results = results[0].filter(conf=conf_thres)
    frame = draw_boxes(frame, results)
    counts = count_vehicles(results)
    return frame, counts

# Video upload
if video_file is not None and not use_webcam:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame, counts = process_frame(frame)
        if counts: vehicle_count_df = pd.DataFrame(list(counts.items()), columns=["Vehicle","Count"])
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        st.dataframe(vehicle_count_df)
    cap.release()

# Webcam
elif use_webcam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame, counts = process_frame(frame)
        if counts: vehicle_count_df = pd.DataFrame(list(counts.items()), columns=["Vehicle","Count"])
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        st.dataframe(vehicle_count_df)
