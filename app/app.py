
#!/usr/bin/env python3
"""
app/app.py – Streamlit crash detection with speed-aware severity scoring and reliable dashcam inclusion.
"""

from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import tempfile
import time
import onnxruntime as ort
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="SOTERIA - Crash Detection", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 3em; font-weight: bold; text-align: center; margin-bottom: 0.2em; }
    .sub-title { font-size: 1.2em; text-align: center; margin-bottom: 2em; }
    .report-card {
        background-color: #f4f4f4; padding: 1.5em 2em; border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 2em; color: #111;
    }
    .report-card h3 { color: #d00000; }
    .report-card ul { list-style-type: none; padding-left: 1em; }
    .report-card li { margin-bottom: 0.5em; font-size: 1.1em; color: #111; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>SOTERIA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Crash Detection Dashboard</div>", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
BESTS = sorted(EXPORTS.glob("*/best.onnx"), key=lambda p: p.stat().st_mtime)
BEST = BESTS[-1] if BESTS else None
IMG_SZ = 224
THRESH_MULTIPLIER = 1.8
MIN_SPIKES = 3
CAR_CLASSES = ['car', 'truck', 'bus']

@st.cache_resource
def load_crash_model(path: Path):
    sess = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

if BEST is None:
    st.error("❌ No best.onnx found.")
    st.stop()

SESSION, IN_NAME, OUT_NAME = load_crash_model(BEST)
yolo_model = load_yolo()

def preprocess_classifier(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (IMG_SZ, IMG_SZ))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]

def infer_crash(tensor: np.ndarray) -> float:
    out = SESSION.run([OUT_NAME], {IN_NAME: tensor})[0]
    return float(out.ravel()[0])

def estimate_speed(prev, curr):
    if prev is None or curr is None:
        return 0
    dx = abs(curr[0] - prev[0])
    dy = abs(curr[1] - prev[1])
    return np.sqrt(dx ** 2 + dy ** 2)

with st.sidebar:
    st.markdown("### 🚨 SOTERIA System")
    fps_target = st.slider("Analyse FPS", 1, 30, 5)
    src = st.file_uploader("Upload dashcam video (mp4)", type=["mp4"])

if src is None:
    st.info("⬅️ Upload a video to begin.")
    st.stop()

video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
video_tmp.write(src.read())
video_tmp.flush()
video_path = video_tmp.name

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(int(src_fps // fps_target), 1)
frame_duration = 1.0 / src_fps

progress = st.progress(0.0)
chart = st.line_chart(y=[])
viewer = st.empty()

idx = 0
p_scores = []
frames = []
analysed_frame_indices = []
car_count_history = []
dashcam_involved = True  # Assume always involved if spike present
spike_max = 0.0
spike_dur = 0
dashcam_centroids = []

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    current_frame = frame.copy()
    if idx % step == 0:
        p = infer_crash(preprocess_classifier(frame))
        p_scores.append(p)
        analysed_frame_indices.append(idx)
        spike_max = max(spike_max, p)
        chart.add_rows([p])
        if p > np.mean(p_scores) * THRESH_MULTIPLIER:
            spike_dur += 1

    results = yolo_model(current_frame, verbose=False)[0]
    boxes = results.boxes

    dashcam_y_thresh = frame.shape[0] * 0.75
    dashcam_box = None
    cars_this_frame = 0

    pil_img = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for box in boxes:
        cls = int(box.cls[0])
        if yolo_model.names[cls] not in CAR_CLASSES:
            continue
        x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
        cars_this_frame += 1
        if y1 > dashcam_y_thresh:
            tag = "Dashcam Vehicle"
            dashcam_box = (x1, y1, x2, y2)
        else:
            tag = "Car"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, y1 - 25), tag, fill=(255, 255, 0))

    if not dashcam_box:
        width, height = frame.shape[1], frame.shape[0]
        x1, x2 = int(width * 0.05), int(width * 0.95)
        y1, y2 = int(height * 0.80), int(height * 0.95)
        dashcam_box = (x1, y1, x2, y2)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, y1 - 25), "Dashcam Vehicle (Est.)", fill=(255, 255, 0))

    cx = int((dashcam_box[0] + dashcam_box[2]) / 2)
    cy = int((dashcam_box[1] + dashcam_box[3]) / 2)
    dashcam_centroids.append((cx, cy))

    car_count_history.append(cars_this_frame)
    viewer.image(np.array(pil_img), channels="RGB", use_container_width=True)
    frames.append(current_frame.copy())

    progress.progress(min(idx / total_frames, 1.0), text=f"{idx}/{total_frames} frames")
    time.sleep(frame_duration)
    idx += 1

cap.release()

if len(p_scores) > 5:
    mean_prob = np.mean(p_scores)
    threshold = mean_prob * THRESH_MULTIPLIER
    spike_indices = [i for i, p in enumerate(p_scores) if p > threshold]

    if len(spike_indices) >= MIN_SPIKES:
        start_idx = analysed_frame_indices[spike_indices[0]]
        end_idx = analysed_frame_indices[spike_indices[-1]]

        clip_start = max(start_idx - int(1.0 * src_fps), 0)
        clip_end = min(end_idx + int(1.0 * src_fps), len(frames))
        clip_frames = frames[clip_start:clip_end]

        height, width, _ = clip_frames[0].shape
        temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_vid.name, cv2.VideoWriter_fourcc(*'mp4v'), src_fps, (width, height))
        for f in clip_frames:
            out.write(f)
        out.release()

        crash_seconds = int(start_idx / src_fps)
        timestamp_str = f"{crash_seconds // 60:02}:{crash_seconds % 60:02}"

        avg_involved = int(np.mean(car_count_history[clip_start:clip_end])) + 1  # Always count dashcam

        # Speed estimation
        speeds = [estimate_speed(dashcam_centroids[i], dashcam_centroids[i + 1])
                  for i in range(len(dashcam_centroids) - 1)]
        speed_factor = np.mean(speeds[-10:]) / 15.0
        speed_factor = min(speed_factor, 2)

        # Score formula
        score = 0
        score += 2.0 * speed_factor
        score += 2.5 * min(avg_involved, 5) / 5.0
        score += 2.5 * min(spike_max / (threshold * 2), 1.0)
        score += 3.0 * min(spike_dur / 10.0, 1.0)

        score = min(10.0, round(score, 1))
        dispatch = score >= 5

        st.markdown("""
        <div class='report-card'>
            <h3>🚨 Crash Report</h3>
            <ul>
                <li><strong>Type of Collision:</strong> Multi-stage / Scattered Impact</li>
                <li><strong>Crash Time:</strong> <code>{}</code> (frames {}–{})</li>
                <li><strong>Estimated Vehicles Involved:</strong> {}</li>
                <li><strong>Severity Score:</strong> {}/10</li>
                <li><strong>Ambulance Dispatched:</strong> {} {}</li>
            </ul>
        </div>
        """.format(timestamp_str, start_idx, end_idx, avg_involved, score,
                    "✅" if dispatch else "🚫", "Yes" if dispatch else "No"), unsafe_allow_html=True)

        with open(temp_vid.name, "rb") as f:
            st.download_button(
                label="📥 Download Clean Crash Clip (MP4)",
                data=f.read(),
                file_name="crash_clip.mp4",
                mime="video/mp4"
            )
    else:
        st.warning("🚫 No confident crash detected. Only {} high-spike frames found.".format(len(spike_indices)))
else:
    st.warning("Not enough frames to analyse.")
