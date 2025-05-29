# detectors/yolo.py
from ultralytics import YOLO
import numpy as np
import streamlit as st

# ⬤ semantic categories
CAR_CLASSES = {"car", "truck", "bus", "motorcycle"}

# ⬤ fixed RGB palette
CLASS_COLOURS = {
    "car":        (  0, 255,   0),   # green
    "truck":      (255, 140,   0),   # orange
    "bus":        (148,   0, 211),   # violet
    "motorcycle": (  0, 255, 255),   # cyan / aqua
}

@st.cache_resource(show_spinner="🔄 Loading YOLO …")
def load(model_name: str = "yolov8n.pt"):
    """Load & cache the Ultralytics YOLO-v8 model."""
    return YOLO(model_name)

def _fallback_colour(seed: int) -> tuple[int, int, int]:
    """Generate a consistent random-ish colour for unknown classes."""
    rng = np.random.default_rng(seed)
    return tuple(int(c) for c in rng.integers(50, 255, size=3))

def detect(model, frame_bgr, conf: float = 0.25) -> list[dict]:
    """
    Run YOLO on a BGR frame and return a list of
    {xyxy, cls, score, colour} dictionaries.
    `colour` is a (B, G, R) tuple ready for OpenCV drawing.
    """
    results = model(frame_bgr, verbose=False, conf=conf)[0]
    out = []

    for box in results.boxes:
        cls_idx  = int(box.cls[0])
        cls_name = model.names[cls_idx]

        colour = CLASS_COLOURS.get(
            cls_name,
            _fallback_colour(cls_idx)
        )

        out.append(
            dict(
                xyxy   = box.xyxy[0].cpu().numpy().astype(int),
                cls    = cls_name,
                score  = float(box.conf[0]),
                colour = colour,
            )
        )
    return out
