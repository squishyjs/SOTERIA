# ──────────────────────────────────────────────────────────────────────────────
#  report.py   – one-page PDF generator for SOTERIA (robust, no Altair runtime)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple
import numpy as np

import io, base64, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units   import mm
from reportlab.pdfgen      import canvas
from reportlab.lib.utils   import ImageReader

# ──────────────────────────────────────────────────────────────────────────────
# Helper: turn a cv2/BGR image into a ReportLab ImageReader at max size
# ──────────────────────────────────────────────────────────────────────────────

def _embed_image(cv_img, max_w: int, max_h: int) -> Tuple[ImageReader, float, float]:
    """Return an ImageReader + scaled-to-fit width & height (points)."""
    import cv2

    h, w, _ = cv_img.shape
    scale   = min(max_w / w, max_h / h)
    _, png  = cv2.imencode(".png", cv_img[:, :, ::-1])   # BGR ➜ RGB ➜ PNG bytes
    return ImageReader(io.BytesIO(png.tobytes())), w * scale, h * scale

# ──────────────────────────────────────────────────────────────────────────────
# Helper: ALWAYS return PNG bytes of the timeline
# – First tries vl-convert, then Selenium, then Matplotlib fallback
# ──────────────────────────────────────────────────────────────────────────────

def _chart_to_png(timeline_df: "pd.DataFrame", crit_th: float, high_th: float) -> bytes:
    """Return rasterised PNG of the p(crash) timeline (never fails)."""
    import altair as alt, shutil, subprocess, tempfile, matplotlib
    from altair_saver import save as alt_save

    # Build the Vega-Lite spec -------------------------------------------------
    base = (
        alt.Chart(timeline_df)
        .mark_line(strokeWidth=1.2, color="#ff595e")
        .encode(x=alt.X("f:Q", title="Frame"),
                y=alt.Y("p:Q", title="p(crash)", scale=alt.Scale(domain=[0, 1])))
    )
    rules = (
        alt.Chart(pd.DataFrame({"y": [high_th, crit_th],
                                "colour": ["#ffca3a", "#ff595e"]}))
        .mark_rule()
        .encode(y="y:Q", color=alt.Color("colour:N", scale=None))
    )
    chart = alt.layer(base, rules).configure_axis(grid=False)

    # 1️⃣  vl-convert (CLI) ----------------------------------------------------
    vl_bin = shutil.which("vl-convert") or shutil.which("vl2png")
    if vl_bin:
        try:
            return subprocess.check_output(
                [vl_bin, "spec", "--format", "png"],
                input=chart.to_json().encode(),
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass  # fall through

    # 2️⃣  Selenium + head-less Chrome via altair-saver -----------------------
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); tmp.close()
        alt_save(chart, tmp.name, method="selenium")
        with open(tmp.name, "rb") as f:
            return f.read()
    except Exception:
        pass  # fall through

    # 3️⃣  Matplotlib fallback (always available) -----------------------------
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 1.3), dpi=160)
    ax.plot(timeline_df.f, timeline_df.p, lw=1.4, color="#ff595e")
    ax.axhline(high_th, ls="--", lw=0.8, color="#ffca3a")
    ax.axhline(crit_th, ls="--", lw=0.8, color="#ff595e")
    ax.set_ylim(0, 1); ax.set_xlabel("frame"); ax.set_ylabel("p")
    ax.grid(False); fig.tight_layout(pad=0.3)

    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_incident_report(
    out_path:        str | Path,
    clip_path:       str | Path,
    critical_frames: Sequence[Tuple[int, float, "np.ndarray"]],
    high_frames:     Sequence[Tuple[int, float, "np.ndarray"]],
    timeline_df:     "pd.DataFrame",   # columns: f, p
    crit_th:         float,
    high_th:         float,
    model_name:      str,
    src_name:        str,
) -> bool:
    """Build a one-page PDF at *out_path* and return **True** on success."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus   import Paragraph
    from reportlab.lib.enums  import TA_LEFT

    # Canvas ------------------------------------------------------------------
    c, (W, H) = canvas.Canvas(str(out_path), pagesize=A4), A4
    margin, y = 15 * mm, H - 15 * mm

    # Header ------------------------------------------------------------------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "🚨  SOTERIA – Incident Report")
    c.setFont("Helvetica", 10)
    c.drawString(margin,       y - 14, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    c.drawRightString(W-margin, y - 14, f"Model: {model_name}")
    y -= 28

    # Timeline chart ----------------------------------------------------------
    png_bytes = _chart_to_png(timeline_df, crit_th, high_th)
    img = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img, margin, y - 110, width=480, height=100, mask="auto")
    y -= 120

    # Key-frame thumbnails -----------------------------------------------------
    def _thumb_strip(title, frameset, y_pos):
        if not frameset:
            return y_pos
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_pos, title); y_pos -= 16
        x, thumb_h = margin, 60
        for ix, prob, img in frameset[:5]:
            img_r, w_s, h_s = _embed_image(img, 100, thumb_h)
            c.drawImage(img_r, x, y_pos - h_s, width=w_s, height=h_s, mask="auto")
            c.setFont("Helvetica", 7)
            c.drawCentredString(x + w_s/2, y_pos - h_s - 8, f"#{ix} • {prob:.2%}")
            x += 105
        return y_pos - thumb_h - 20

    y = _thumb_strip("Critical frames", critical_frames, y)
    y = _thumb_strip("High-risk frames", high_frames, y)

    # Embedded clip link -------------------------------------------------------
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    link = Paragraph(
        f'<a href="data:video/mp4;base64,{b64}">Download MP4 clip</a>',
        ParagraphStyle(name="Normal", fontSize=8, textColor="#006ddb",
                       leftIndent=0, alignment=TA_LEFT)
    )
    link.wrapOn(c, W - 2*margin, 40)
    link.drawOn(c, margin, y - 15)
    y -= 30

    # Footer ------------------------------------------------------------------
    c.setFont("Helvetica", 8)
    c.drawString(margin, 15, f"Source video: {src_name}")
    c.save()
    return True
