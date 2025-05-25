#!/usr/bin/env python3
"""
app/app.py – SOTERIA crash-detection demo (2025-05-13, dark-theme refresh v2)

• Dark glass-morphism UI
• 5 : 1 video-first layout
• Altair spark-line with explicit types
• **FIX:** right-side metrics now use single placeholders (no spam)
"""
from __future__ import annotations
import csv, io, zipfile, tempfile, time
from pathlib import Path
import os

import cv2, numpy as np, onnxruntime as ort, streamlit as st, altair as alt
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# 🖼️ GLOBAL THEME & PAGE CONFIG
###############################################################################
st.set_page_config(
    page_title="SOTERIA – Crash Detector",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- custom CSS (dark, glass, neon) ----------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg:        #0d1117;
    --panel:     rgba(255,255,255,0.04);
    --border:    rgba(255,255,255,0.08);
    --fg:        #e6eefb;
    --accent:    #ff595e;   /* critical / primary */
    --success:   #8ac926;   /* safe */
}
/* ===== global ===== */
html, body, [data-testid="stApp"] {
    background: var(--bg);
    color: var(--fg);
    font-family: "Inter", sans-serif;
}
[data-testid="stHeader"]{display:none}
.main{padding-top:2.5rem}
a{color:var(--accent)}
/* ===== sidebar ===== */
section[data-testid="stSidebar"]{
    background: var(--panel);
    backdrop-filter: blur(14px);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSlider>div{margin-top:6px}
/* ===== video card ===== */
.card{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    overflow:hidden;
}
.overlay-text{
    position:absolute;top:12px;left:12px;padding:6px 10px;border-radius:8px;
    font-weight:600;background:rgba(0,0,0,.45);color:#fafafa}
[data-testid="stMetricValue"]{
    font-size:2.2rem;font-weight:600;color:var(--accent);
}
</style>
""",
    unsafe_allow_html=True,
)

PRIMARY, SUCCESS = "#ff595e", "#8ac926"

###############################################################################
# 📦 MODEL LOADING
###############################################################################
ROOT    = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
BEST    = max(EXPORTS.glob("*/best.onnx"), key=lambda p: p.stat().st_mtime)
IMG_SZ, CLASSES = 224, ["Crash", "Normal"]

@st.cache_resource(show_spinner="🔄 Loading ONNX model …")
def load_model(path: Path):
    sess = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name

SESSION, IN_NAME, OUT_NAME = load_model(BEST)

@st.cache_data(max_entries=256, show_spinner=False)
def infer(bgr: np.ndarray) -> float:
    img = cv2.resize(bgr, (IMG_SZ, IMG_SZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    tensor = img.transpose(2,0,1)[None]
    return float(SESSION.run([OUT_NAME], {IN_NAME:tensor})[0].ravel()[0])

###############################################################################
# 🖥️ SIDEBAR – INPUT & SETTINGS
###############################################################################
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    fps_target = st.slider("Analyse FPS", 1, 30, 30)
    crit_th    = st.slider("Critical threshold", 0.5, 1.0, 0.80, 0.01)
    high_th    = st.slider("High-risk threshold", 0.3, crit_th, 0.50, 0.01)
    st.caption("_High-risk < Critical_")
    src_file   = st.file_uploader("📤 Upload image/video", ["jpg","jpeg","png","mp4"])
    st.caption(f"Model: **{BEST.relative_to(ROOT)}**")

if not src_file:
    st.info("⬅️ Upload an image or video to begin")
    st.stop()

###############################################################################
# ⏭️ SAVE UPLOAD TO TEMP
###############################################################################
_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(src_file.name).suffix)
_tmp.write(src_file.getbuffer()); _tmp.close()
SRC = _tmp.name

###############################################################################
# 🖼️ IMAGE BRANCH
###############################################################################
if src_file.type != "video/mp4":
    frame = cv2.imread(SRC)
    if frame is None:
        st.error("❌ Could not read the image.")
        st.stop()
    p = infer(frame)
    label = CLASSES[0] if p > 0.5 else CLASSES[1]

    st.image(frame[:,:,::-1], caption=f"Prediction · {label} ({p:.2%})", use_column_width=True)
    st.metric("Crash probability", f"{p:.1%}")
    st.stop()

###############################################################################
# 🎞️ VIDEO BRANCH
###############################################################################
cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    st.error("❌ Could not open video.")
    st.stop()

src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
step    = max(int(src_fps // fps_target), 1)
total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

# 5 : 1 layout – video dominates
col_vid, col_metrics = st.columns([5,1], gap="medium")

with col_vid:
    vid_ph   = st.empty()
    chart_ph = st.empty()
    progress = st.progress(0.0, text=f"0 / {total_fr}")

# ----- placeholders (fixes metric spam) -----
metric_ph = col_metrics.empty()   # single metric
lat_ph    = col_metrics.empty()   # single caption

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()

crit_frames, high_frames, trend_pts = [], [], []
frames: list[np.ndarray] = []
idx = analysed = 0
prev_p = None
last_time = time.perf_counter()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frames.append(frame.copy())
    start = time.perf_counter()
    if idx % step == 0:
        p = infer(frame)
        analysed += 1
        prev_p = p
        trend_pts.append({"f": idx, "p": p})

        chart = (
            alt.Chart(alt.Data(values=trend_pts))
            .mark_line(strokeWidth=1.5, color=PRIMARY)
            .encode(
                x=alt.X("f:Q", title=None),
                y=alt.Y("p:Q", scale=alt.Scale(domain=[0,1]), title=None),
            )
            .properties(height=100, width="container")
        )
        chart_ph.altair_chart(chart, use_container_width=True)

        if p >= crit_th:
            crit_frames.append((idx,p,frame.copy()))
        elif p >= high_th:
            high_frames.append((idx,p,frame.copy()))
    else:
        p = prev_p or 0.0

    # overlay probability box
    over  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(over)
    colour = SUCCESS if p < 0.5 else PRIMARY
    draw.rectangle([(0,0),(220,45)], fill=colour+"bf")
    draw.text((12,10), f"p = {p:.2%}", font=FONT, fill="white")

    vid_ph.image(over, use_column_width=True)

    metric_ph.metric("Crash probability", f"{p:.1%}")
    progress.progress(idx/total_fr, text=f"{idx}/{total_fr} frames")

    lat_ms = (time.perf_counter()-start)*1000
    lat_ph.caption(f"Latency {lat_ms:.1f} ms · Inference FPS {analysed/(idx+1):.2f}")

    # frame pacing
    delta = 1/src_fps - (time.perf_counter()-last_time)
    if delta > 0:
        time.sleep(delta)
    last_time = time.perf_counter()
    idx += 1

cap.release()
###############################################################################
# 🎬 INCIDENT CLIP  ·  FRAME GALLERIES  ·  EXPORTS
###############################################################################
st.divider()

# --- helper ------------------------------------------------------------------
def export_incident_clip(
    frames: list[np.ndarray],
    first_idx: int,
    last_idx:  int,
    fps:       float,
    pre_sec:   int = 2,
    post_sec:  int = 1,
) -> str | None:
    """
    Returns the path to a short H.264 MP4 clip
    (2 s before first critical frame → 1 s after last).
    """

    if not frames:
        return None

    start = max(first_idx - int(pre_sec * fps), 0)
    end   = min(last_idx  + int(post_sec * fps), len(frames) - 1)

    h,  w, _ = frames[0].shape

    # ---------- create a *closed* temp file (Windows-safe) ----------
    fd, filename = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)                        # release handle so OpenCV can write

    # ---------- use H.264 if available, otherwise fall back ----------
    try_fourcc = ("avc1", "H264", "mp4v")
    for c in try_fourcc:
        fourcc = cv2.VideoWriter_fourcc(*c)
        writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        if writer.isOpened():
            break
    else:                               # couldn’t open any codec
        st.error("❌ OpenCV cannot open an MP4 writer on this system.")
        return None

    # ---------- write frames ----------
    for fr in frames[start : end + 1]:
        writer.write(fr)
    writer.release()

    return filename


# --- tiny image galleries ----------------------------------------------------
def gallery(title: str, data):
    if data:
        with st.expander(title, expanded="Critical" in title):
            cols = st.columns(min(5, len(data)))
            for i, (ix, prob, img) in enumerate(data):
                cols[i % len(cols)].image(
                    img[:, :, ::-1],
                    caption=f"#{ix} • {prob:.2%}",
                    use_column_width=True,
                )


gallery(f"🚨 Critical ({len(crit_frames)})", crit_frames)
gallery(f"⚠️ High-risk ({len(high_frames)})", high_frames)

# --- instant-replay clip -----------------------------------------------------
if crit_frames:                                           # at least one spike
    first_idx = crit_frames[0][0]
    last_idx  = crit_frames[-1][0]

    clip_path = export_incident_clip(
        frames, first_idx, last_idx, src_fps,
        pre_sec=2, post_sec=1
    )

    # inline preview (optional – comment out if not desired)
    st.video(clip_path, start_time=0)

    with open(clip_path, "rb") as f:
        st.download_button(
            "🎬 Download 3-second incident clip",
            f.read(),
            file_name=f"incident_{first_idx:06d}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

# --- zip of all flagged frames + CSV summary ---------------------------------
if crit_frames or high_frames:
    buf, csv_buf = io.BytesIO(), io.StringIO()
    with zipfile.ZipFile(buf, "w") as z:
        w = csv.writer(csv_buf)
        w.writerow(["frame_idx", "prob", "tier"])
        for tier, frameset in (("critical", crit_frames),
                               ("high",     high_frames)):
            for ix, prob, img in frameset:
                _, jpg = cv2.imencode(".jpg", img)
                z.writestr(f"{tier}_{ix}.jpg", jpg.tobytes())
                w.writerow([ix, prob, tier])
        z.writestr("summary.csv", csv_buf.getvalue())

    st.download_button(
        "⬇️ Download flagged frames + CSV",
        buf.getvalue(),
        "crash_frames.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.success(f"✅ Finished · analysed {analysed}/{total_fr} frames")
