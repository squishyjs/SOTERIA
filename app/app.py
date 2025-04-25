#!/usr/bin/env python3
"""
app/app.py – Streamlit demo (realtime-ish version)

run with:
    streamlit run app/app.py
"""

from __future__ import annotations
import time
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageFont   # pillow for overlay text

# ───── MUST be the very first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Crash detector",
    page_icon="🚗",
    layout="wide",
)

# ───── locate newest exports/*/best.onnx ───────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
EXPORTS  = ROOT / "exports"
BESTS    = sorted(EXPORTS.glob("*/best.onnx"),
                  key=lambda p: p.stat().st_mtime)
if not BESTS:
    st.error("❌  No `best.onnx` found under `exports/`. "
             "Run training + `export_model.py` first.")
    st.stop()
BEST     = BESTS[-1]

# ───── constants ───────────────────────────────────────────────────────
IMG_SZ   = 224
NAMES    = ["crash", "normal"]          # [0] is p(crash)

# ───── ONNX session (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def _load_model(path: Path):
    sess = ort.InferenceSession(
        path.as_posix(),
        providers=["CPUExecutionProvider"]
    )
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name

SESSION, IN_NAME, OUT_NAME = _load_model(BEST)

@st.cache_data(show_spinner=False, max_entries=512)
def infer(tensor: np.ndarray) -> float:
    out = SESSION.run([OUT_NAME], {IN_NAME: tensor})[0]
    return float(out.ravel()[0])

def preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (IMG_SZ, IMG_SZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]  # HWC→NCHW + batch dim

# ───── UI header ──────────────────────────────────────────────────────
st.title("Crash Detector 🚗💥")
st.caption(
    f"**{BEST.relative_to(ROOT)}**  – exported "
    f"{datetime.fromtimestamp(BEST.stat().st_mtime):%Y-%m-%d %H:%M}"
)

with st.sidebar:
    fps_target = st.slider("Analyse frames per second", 1, 30, 5)
    st.markdown("---")
    src = st.file_uploader("Upload image (`jpg/png`) or video (`mp4`)",
                           type=["jpg", "jpeg", "png", "mp4"])

if src is None:
    st.info("⬅️  Upload something to begin")
    st.stop()

# write the uploaded file to disk for OpenCV
tmp = tempfile.NamedTemporaryFile(delete=False)
tmp.write(src.read())
tmp.close()
path = tmp.name

# ───── image branch ─────────────────────────────────────────────────
if src.type != "video/mp4":
    frame = cv2.imread(path)
    if frame is None:
        st.error("❌  Could not read the image.")
        st.stop()

    p = infer(preprocess(frame))
    label = NAMES[0] if p > 0.5 else NAMES[1]
    st.image(
        frame[:, :, ::-1],
        caption=f"Prediction: **{label}** – p(crash) = {p:.2%}",
        use_column_width=True
    )
    st.stop()

# ───── video branch (live-ish playback) ──────────────────────────────
cap = cv2.VideoCapture(path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25
step         = max(int(src_fps // fps_target), 1)

progress = st.progress(0.0)
chart    = st.line_chart(y=[])
viewer   = st.empty()

# prepare a font for overlay text
try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()

idx, analysed = 0, 0
last_time = time.perf_counter()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # only run inference every `step` frames
    if idx % step == 0:
        p = infer(preprocess(frame))
        analysed += 1
        chart.add_rows([p])

        # draw overlay text
        overlay = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(overlay)
        draw.text((10, 10),
                  f"p(crash) = {p:.2%}",
                  font=FONT,
                  fill=(0, 255, 0))
        display_frame = np.array(overlay)
    else:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    viewer.image(display_frame, channels="RGB", use_column_width=True)
    progress.progress(idx / total_frames,
                      text=f"{idx}/{total_frames} frames")

    # throttle loop to match original FPS
    frame_duration = 1.0 / src_fps
    elapsed = time.perf_counter() - last_time
    if elapsed < frame_duration:
        time.sleep(frame_duration - elapsed)
    last_time = time.perf_counter()

    idx += 1

cap.release()
st.success(f"Finished 🚀  analysed {analysed}/{total_frames} frames")
