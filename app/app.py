#!/usr/bin/env python3
"""
app/app.py – Streamlit demo for your YOLO-v8 crash-vs-normal classifier.

• looks for the newest  exports/*/best.onnx  checkpoint
• supports single images *and* full MP4s (≈5 fps sampling)
• CPU-only inference → runs anywhere you have Python+ONNXRuntime

run:   streamlit run app/app.py
"""

from __future__ import annotations
import tempfile, cv2, numpy as np, streamlit as st
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from datetime import datetime

# ───────────────────────── locate the model ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]         # <repo>/app/app.py
EXPORTS   = REPO_ROOT / "exports"                       # exports/train*/best.onnx


def newest_best()->Path:
    """Return the newest exports/*/best.onnx (raises if not found)."""
    bests = sorted(EXPORTS.glob("*/best.onnx"),
                   key=lambda p: p.stat().st_mtime)
    if not bests:
        st.stop()  # nicer than raising in Streamlit
    return bests[-1]


# ───────────────────────── constants ─────────────────────────────────
MODEL_PATH  = newest_best()
IMG_SIZE    = 224
CLASS_NAMES = ["crash", "normal"]  # output[0] = p(crash)

# ───────────────────────── ONNX session (cached) ─────────────────────
@st.cache_resource(show_spinner=True)
def load_model(model_path: Path):
    sess = ort.InferenceSession(
        model_path.as_posix(),
        providers=["CPUExecutionProvider"]
    )
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name


sess, IN_NAME, OUT_NAME = load_model(MODEL_PATH)

# ───────────────────────── helpers ───────────────────────────────────
def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → NCHW float32 ∈[0,1] shape (1,3,224,224)"""
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # HWC→NCHW + batch dim
    return img


@st.cache_data(show_spinner=False, max_entries=512)
def predict(img_tensor: np.ndarray) -> float:
    """Return p(crash) as python float."""
    out = sess.run([OUT_NAME], {IN_NAME: img_tensor})[0]
    # ONNX export keeps shape [B,1] or [1] – handle both
    prob_crash = float(out.ravel()[0])
    return prob_crash


# ───────────────────────── UI ────────────────────────────────────────
st.title("Crash Detector 📉 (ONNX demo)")
st.caption(f"Model: **{MODEL_PATH.relative_to(REPO_ROOT)}**  "
           f"(exported {datetime.fromtimestamp(MODEL_PATH.stat().st_mtime):%Y-%m-%d %H:%M})")

src = st.file_uploader(
    "Upload **an image** (`jpg/png`) **or a video** (`mp4`)",
    type=["jpg", "jpeg", "png", "mp4"]
)

if src is None:
    st.info("⬆️  Choose a file to begin.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(src.read())
    local_path = tmp.name                          # path for OpenCV

if src.type != "video/mp4":                        # ── single image ──
    frame = cv2.imread(local_path)
    if frame is None:
        st.error("Could not read the image.")
        st.stop()

    p = predict(preprocess(frame))
    lbl = CLASS_NAMES[0] if p > 0.5 else CLASS_NAMES[1]  # simple arg-max
    st.image(frame[:, :, ::-1],  # BGR→RGB
             caption=f"Prediction: **{lbl}** – p(crash) = {p:.2%}",
             use_column_width=True)

else:                                              # ── video ──────────
    cap = cv2.VideoCapture(local_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    every = max(int(fps // 5), 1)                  # analyse ≈5 fps

    progress = st.progress(0.0, text="Analysing video…")
    chart    = st.line_chart(y=[])
    framebox = st.empty()

    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if idx % every == 0:
            p = predict(preprocess(frame))
            chart.add_rows([p])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            framebox.image(frame_rgb,
                           caption=f"frame {idx}/{total} – p(crash) = {p:.2%}",
                           use_column_width=True)
        idx += 1
        progress.progress(min(idx / total, 1.0),
                          text=f"{idx}/{total} frames")

    cap.release()
    st.success("Finished 🚀")

