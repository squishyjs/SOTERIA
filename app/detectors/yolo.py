from ultralytics import YOLO
import cv2, numpy as np, streamlit as st

CAR_CLASSES = {"car", "truck", "bus", "motorcycle"}

@st.cache_resource
def load(model_name: str = "yolov8n.pt"):
    return YOLO(model_name)

def detect(model, frame_bgr: np.ndarray, conf=0.25):
    """Return list[dict] with xyxy, cls_name, score."""
    results = model(frame_bgr, verbose=False, conf=conf)[0]
    out = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        out.append(
            dict(
                xyxy=box.xyxy[0].cpu().numpy().astype(int),
                cls=cls_name,
                score=float(box.conf[0]),
            )
        )
    return out
