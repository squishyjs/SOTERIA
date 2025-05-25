#!/usr/bin/env python3
"""
app/app.py – SOTERIA crash-detection demo (2025-05-13, dark-theme refresh v2)

• Dark glass-morphism UI
• 5 : 1 video-first layout
• Altair spark-line with explicit types
• **FIX:** right-side metrics now use single placeholders (no spam)
"""
from __future__ import annotations


from report import generate_incident_report
import csv, io, zipfile, tempfile, time
from pathlib import Path
import os
import pandas as pd

import cv2, numpy as np, onnxruntime as ort, streamlit as st, altair as alt
from PIL import Image, ImageDraw, ImageFont


# loading themes
from ui.theme import apply_dark_glass, PRIMARY, SUCCESS
apply_dark_glass()

# loading detectors
from detectors.yolo import load as load_yolo, detect, CAR_CLASSES
yolo = load_yolo()                # cached by @st.cache_resource

from severity import SeverityTracker

SCALE = 0.5        # 0.5 ➜ 50 % resolution  (tweak 0.33, 0.25, …)
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

def load_logo():
    return Image.open(ROOT / "app" / "ui" / "SOTERIA_logo.png")

###############################################################################
# 🖥️ SIDEBAR – INPUT & SETTINGS
###############################################################################
with st.sidebar:
    st.image(load_logo(), use_column_width=True)
    st.markdown("## ⚙️ Settings")
    fps_target = st.slider("Analyse FPS", 1, 30, 30)
    crit_th    = st.slider("Critical threshold", 0.5, 1.0, 0.60, 0.01)
    high_th    = st.slider("High-risk threshold", 0.3, crit_th, 0.50, 0.01)
    dispatch_th = st.slider("Dispatch threshold", 0.30, 1.0, 0.60, 0.01)
    st.caption("_High-risk < Critical_")
    src_file   = st.file_uploader("📤 Upload image/video", ["jpg","jpeg","png","mp4"])
    st.caption(f"Model: **{BEST.relative_to(ROOT)}**")
    # NEW – optional webhook (blank = disabled)
    dispatch_url = st.text_input(
        "Dispatch webhook URL (leave blank to disable)",
        value="",                          # "" → does nothing
        placeholder="https://httpbin.org/post"
    )

if not src_file:
    st.info("⬅️ Upload an image or video to begin")
    st.stop()

alert_ph = st.empty()          # stays empty unless we trigger it

# ── NEW helper: render full-width banner ─────────────────────────
def show_severe_banner(msg: str):
    alert_ph.markdown(
        f'<div class="alert-banner">🚑 {msg}</div>',
        unsafe_allow_html=True
    )
###############################################################################
# ⏭️ SAVE UPLOAD TO TEMP
###############################################################################
_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(src_file.name).suffix)
_tmp.write(src_file.getbuffer()); _tmp.close()
SRC = _tmp.name
st.session_state.pop("alert_shown", None)
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
metric2_ph  = col_metrics.empty()   # active vehicles 👈 new
metric3_ph = col_metrics.empty()    #   <<< ADD THIS LINE
lat_ph    = col_metrics.empty()   # single caption
metric_speed_ph = col_metrics.empty()   # ⏩ new line

# ───── finalise severity & show metric ────────────────────────────


try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()

crit_frames, high_frames, trend_pts = [], [], []
car_count_history = []          # 👈 you forgot to re-create this
frames: list[np.ndarray] = []
idx = analysed = 0
prev_p = None
last_time = time.perf_counter()
tracker = SeverityTracker(low_cut=0.30,
                          high_cut=dispatch_th)
sev, sev_cls = 0.0, "Minor"
rel_speed_peak = 0.0


prev_gray = None        # ← for optic-flow speed

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frames.append(frame.copy())
    # 0️⃣ cadence gate FIRST -----------------------------
    analysed_frame = (idx % step == 0)

    # 1️⃣ optic-flow AFTER we know analysed_frame --------
    # --- 1. build small greyscale copies ------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (0, 0), fx=SCALE, fy=SCALE)
    flow = None  # default – no flow this frame

    if analysed_frame and prev_gray is not None:  # run only on gated frames
        small_flow = cv2.calcOpticalFlowFarneback(
            prev_prev_small,  # note the *_small* names
            small_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # map flow back to full-size coordinate system (keep px / frame units)
        flow = small_flow / SCALE

    # keep small + full versions for the next iteration
    prev_prev_small = small_gray
    prev_gray = gray

    # ── YOLO (always) ─────────────────────────────────────────────
    boxes = detect(yolo, frame)
    cars  = [b for b in boxes if b["cls"] in CAR_CLASSES]
    car_count_history.append(len(cars))

    start = time.perf_counter()

    # ── classifier on analysed frames only ───────────────────────
    if analysed_frame:
        p = infer(frame)
        analysed += 1
        prev_p   = p
    else:
        p = prev_p or 0.0                       # reuse last prob


    # ── UI that needs only analysed frames ───────────────────────
    if analysed_frame:
        # 1️⃣  update spark-line ------------------------------------------------
        trend_pts.append({"f": idx, "p": p})

        rules = alt.Chart(
            pd.DataFrame({"y": [high_th, crit_th],
                          "colour": ["amber", "red"]})
        ).mark_rule(strokeDash=[4, 2]).encode(
            y='y:Q', color=alt.Color('colour:N', scale=None))

        chart = (
            alt.Chart(alt.Data(values=trend_pts))
            .mark_line(strokeWidth=1.5, color=PRIMARY)
            .encode(x=alt.X("f:Q", title=None),
                    y=alt.Y("p:Q", scale=alt.Scale(domain=[0, 1]), title=None))
            .properties(height=100, width="container") + rules
        )
        chart_ph.altair_chart(chart, use_container_width=True)

        # 2️⃣  collect key frames ---------------------------------------------
        if p >= crit_th:
            crit_frames.append((idx, p, frame.copy()))
        elif p >= high_th:
            high_frames.append((idx, p, frame.copy()))

        # 3️⃣  severity & real-time actions -----------------------------------
        sev, sev_cls = tracker.result()
        metric3_ph.metric("Severity", f"{sev:.2f}", delta=sev_cls,
                          delta_color="inverse")

        if (sev_cls == "Severe") and ("alert_shown" not in st.session_state):
            show_severe_banner("SEVERE CRASH DETECTED – IMMEDIATE ACTION REQUIRED!")
            st.session_state.alert_shown = True

        # 🔔  dispatch webhook the moment sev ≥ dispatch_th (once per run)
        if (
            dispatch_url
            and sev >= dispatch_th
            and "dispatch_sent" not in st.session_state
        ):
            import requests, json, datetime
            try:
                requests.post(
                    dispatch_url,
                    json=dict(
                        timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        severity  = sev,
                        class_    = sev_cls,
                        vehicles  = int(tracker.stats["cars"] * tracker.car_max),
                        src       = src_file.name,
                    ),
                    timeout=3,
                )
                st.session_state.dispatch_sent = True
                st.success("🚑 Emergency dispatch notified")
            except Exception as e:
                st.error(f"Dispatch failed → {e}")


    # ── draw YOLO boxes ───────────────────────────────────────────
    for det in cars:
        x1, y1, x2, y2 = det["xyxy"]
        colour = det["colour"]

        if flow is not None:
            roi = flow[int(y1*SCALE):int(y2*SCALE), int(x1*SCALE):int(x2*SCALE)]
            mag = np.sqrt(roi[..., 0] ** 2 + roi[..., 1] ** 2)
            det["speed_px"] = float(np.median(mag))  # px / frame
        else:
            det["speed_px"] = 0.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # label shows class + speed in px/s
        label = (f"{det['cls']} "
                 f"{det['score'] * 100:2.0f}%  "  # ← confidence
                 f"{det['speed_px'] * src_fps:4.0f}px/s")  # ← speed
        cv2.putText(frame, label,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    rel_speed = np.median([d["speed_px"] for d in cars]) if cars else 0.0
    rel_speed_peak = max(rel_speed_peak, rel_speed)  # ★ NEW – track the peak
    metric_speed_ph.metric(  # ⏩ new block
        "Relative speed",
        f"{rel_speed * src_fps:0.0f} px/s"
    )

    # 2️⃣ now that rel_speed is known, update tracker --------
    tracker.update(frame, p, len(cars), rel_speed, high_th,
                   analysed_frame=analysed_frame)

    # ── overlay & per-frame metrics ──────────────────────────────
    over   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw   = ImageDraw.Draw(over)
    box_col = SUCCESS if p < 0.5 else PRIMARY
    draw.rectangle([(0, 0), (220, 45)], fill=box_col + "bf")
    draw.text((12, 10), f"p = {p:.2%}", font=FONT, fill="white")
    vid_ph.image(over, use_column_width=True)

    metric_ph.metric("Crash probability", f"{p:.1%}")
    metric2_ph.metric("Active vehicles", f"{len(cars)}")
    progress.progress(idx / total_fr, text=f"{idx}/{total_fr} frames")

    lat_ms = (time.perf_counter() - start) * 1000
    lat_ph.caption(f"Latency {lat_ms:.1f} ms · Inference FPS {analysed / (idx + 1):.2f}")

    # pacing
    delta = 1 / src_fps - (time.perf_counter() - last_time)
    if delta > 0:
        time.sleep(delta)
    last_time = time.perf_counter()
    idx += 1




# ───── optional dispatch hook ─────────────────────────────────────
sev, sev_cls = tracker.result()
if dispatch_url and sev >= dispatch_th:
    import requests
    try:
        r = requests.post(
            dispatch_url,
            json=dict(
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                severity  = sev,
                class_    = sev_cls,
                vehicles  = int(tracker.stats["cars"] * tracker.car_max),
                src       = src_file.name,
            ),
            timeout=3,
        )
        r.raise_for_status()          # surface HTTP errors
        st.success("🚑 Emergency dispatch notified")
    except Exception as e:
        st.error(f"Dispatch failed → {e}")

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

# ──────────────────────────────────────────────────────────────────────────────
# 🎬  INSTANT-REPLAY CLIP  ●  PDF INCIDENT REPORT  ●  FRAMES+CSV ZIP
# ──────────────────────────────────────────────────────────────────────────────
clip_path: str | None = None

# 1️⃣  Build the 3-second “instant-replay” MP4 (2 s pre + 1 s post)
if crit_frames:                               # at least one critical spike
    first_idx = crit_frames[0][0]
    last_idx  = crit_frames[-1][0]

    clip_path = export_incident_clip(
        frames, first_idx, last_idx, src_fps,
        pre_sec=2, post_sec=1
    )

    # inline preview (remove if you’d rather not auto-play)
    st.video(clip_path, start_time=0)

    with open(clip_path, "rb") as f:
        st.download_button(
            "🎬 Download 3-second incident clip",
            f.read(),
            file_name=f"incident_{first_idx:06d}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
# ──────────────────────────────────────────────
# 2️⃣  One-page PDF incident report (optional)
#     • only if we built an incident clip above
#     • hides the button when PDF generation fails
# ──────────────────────────────────────────────
if clip_path:
    from report import generate_incident_report  # imported *inside* the block

    pdf_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    chart_ok = generate_incident_report(

        out_path=pdf_tmp.name,
        clip_path=clip_path,
        critical_frames=crit_frames,
        high_frames=high_frames,
        timeline_df=pd.DataFrame(trend_pts),
        crit_th=crit_th,
        high_th=high_th,
        model_name=BEST.name,
        src_name=src_file.name,
        # ★ NEW fields expected by the updated report.py ★
        tracker_stats=tracker.stats,  # dict with p_peak, dur_high, cars…
        sev=sev,  # final severity score
        sev_cls=sev_cls,  # "Minor" / "Moderate" / "Severe"
        rel_speed_peak=rel_speed_peak,  # px / frame
        src_fps=src_fps,
    )

    if chart_ok:  # timeline successfully embedded
        with open(pdf_tmp.name, "rb") as f:
            st.download_button(
                "📄 Download 1-page Incident Report (PDF)",
                f.read(),
                "incident_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:  # PDF created but without the chart
        st.warning(
            "ℹ️ The PDF was generated, but the timeline chart could not be "
            "embedded (missing **vl-convert** or head-less Chrome)."
        )
        with open(pdf_tmp.name, "rb") as f:
            st.download_button(
                "📄 Download Incident Report (chart-less)",
                f.read(),
                "incident_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ──────────────────────────────────────────────
# 3️⃣  ZIP of all flagged frames + CSV summary
# ──────────────────────────────────────────────
if crit_frames or high_frames:
    buf, csv_buf = io.BytesIO(), io.StringIO()
    with zipfile.ZipFile(buf, "w") as z:
        writer = csv.writer(csv_buf)
        writer.writerow(["frame_idx", "probability", "tier"])
        for tier, frameset in (("critical", crit_frames),
                               ("high",     high_frames)):
            for ix, prob, img in frameset:
                _, jpg = cv2.imencode(".jpg", img)
                z.writestr(f"{tier}_{ix}.jpg", jpg.tobytes())
                writer.writerow([ix, prob, tier])
        z.writestr("summary.csv", csv_buf.getvalue())

    st.download_button(
        "⬇️ Download flagged frames + CSV",
        buf.getvalue(),
        "crash_frames.zip",
        mime="application/zip",
        use_container_width=True,
    )
alert_ph.empty()    # remove banner when run is done
st.success(f"✅ Finished – analysed {analysed}/{total_fr} frames")


