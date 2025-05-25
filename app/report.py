"""
report.py – one-page PDF generator for SOTERIA (stream-safe, no Altair runtime)
2025-05-25  • adds Event-summary block, UUID  • 2025-05-27 • brand logo in header
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple
import uuid, io, base64, shutil, subprocess, tempfile

import numpy as np
import pandas as pd
import cv2  # NEW – used for loading the logo

from reportlab.lib.pagesizes import A4
from reportlab.lib.units   import mm
from reportlab.pdfgen      import canvas
from reportlab.lib.utils   import ImageReader
from reportlab.lib.styles  import ParagraphStyle
from reportlab.platypus    import Paragraph
from reportlab.lib.enums   import TA_LEFT

# ──────────────────────────────────────────────────────────────────────────────
#  Project-wide paths (logo asset)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH    = PROJECT_ROOT / "ui" / "SOTERIA_logo.png"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: small utilities
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_time(sec: float) -> str:            # «0.32 s» ⇢ «320 ms»
    return f"{sec*1000:.0f} ms" if sec < 1 else f"{sec:.1f} s"


def _embed_image(cv_img: np.ndarray, max_w: int, max_h: int) -> Tuple[ImageReader, float, float]:
    """Return an ImageReader + scaled-to-fit width & height (points)."""
    h, w, _ = cv_img.shape
    scale   = min(max_w / w, max_h / h)
    _, png  = cv2.imencode(".png", cv_img[:, :, ::-1])   # BGR → PNG bytes
    return ImageReader(io.BytesIO(png.tobytes())), w * scale, h * scale

# ──────────────────────────────────────────────────────────────────────────────
#   Timeline → PNG  (tries vl-convert → selenium → mpl fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _chart_to_png(timeline_df: "pd.DataFrame", crit_th: float, high_th: float) -> bytes:
    import altair as alt, matplotlib
    from altair_saver import save as alt_save

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

    # 1️⃣  vl-convert
    vl_bin = shutil.which("vl-convert") or shutil.which("vl2png")
    if vl_bin:
        try:
            return subprocess.check_output([vl_bin, "spec", "--format", "png"],
                                            input=chart.to_json().encode(),
                                            stderr=subprocess.DEVNULL)
        except Exception:
            pass

    # 2️⃣  selenium
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); tmp.close()
        alt_save(chart, tmp.name, method="selenium")
        with open(tmp.name, "rb") as f:
            return f.read()
    except Exception:
        pass

    # 3️⃣  mpl fallback (always works)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 1.3), dpi=160)
    ax.plot(timeline_df.f, timeline_df.p, lw=1.4, color="#ff595e")
    ax.axhline(high_th, ls="--", lw=0.8, color="#ffca3a")
    ax.axhline(crit_th, ls="--", lw=0.8, color="#ff595e")
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame"); ax.set_ylabel("p")
    ax.grid(False); fig.tight_layout(pad=0.3)
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
#   Public API  –  returns **True** on success (never throws)
# ──────────────────────────────────────────────────────────────────────────────

def generate_incident_report(
    *,
    out_path:        str | Path,
    clip_path:       str | Path,
    critical_frames: Sequence[Tuple[int, float, "np.ndarray"]],
    high_frames:     Sequence[Tuple[int, float, "np.ndarray"]],
    timeline_df:     "pd.DataFrame",   # columns: f , p
    crit_th:         float,
    high_th:         float,
    model_name:      str,
    src_name:        str,
    tracker_stats:   dict[str, float],  # p_peak  dur_high  cars …
    sev:             float,
    sev_cls:         str,
    rel_speed_peak:  float,             # px / frame
    src_fps:         float,
) -> bool:
    """Build a one-page PDF at *out_path* and return **True** on success."""

    REPORT_ID = uuid.uuid4().hex[:8].upper()

    # Canvas ------------------------------------------------------------------
    c, (W, H) = canvas.Canvas(str(out_path)), A4
    margin, y = 15 * mm, H - 15 * mm

    # ── Logo (if present) ─────────────────────────────────────────────────––
    if LOGO_PATH.exists():
        logo_bgr = cv2.imread(str(LOGO_PATH))
        if logo_bgr is not None:
            logo_img, w_logo, h_logo = _embed_image(logo_bgr, 60, 24)
            c.drawImage(logo_img, W - margin - w_logo, y - h_logo + 6,
                        width=w_logo, height=h_logo, mask="auto")

    # Header ------------------------------------------------------------------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "🚨  SOTERIA – Incident Report")
    c.setFont("Helvetica", 10)
    c.drawRightString(W-margin, y, f"Report ID {REPORT_ID}")
    y -= 14
    c.drawString(margin,       y, f"Generated UTC: {datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    c.drawRightString(W-margin, y, f"Model: {model_name}")
    y -= 20

    # Event summary -----------------------------------------------------------
    summary = (
        f"Peak p(crash) {tracker_stats['p_peak']:.1%}     "
        f"Severity {sev_cls} ({sev:.2f})     "
        f"High-risk {_fmt_time(tracker_stats['dur_high']/src_fps)}     "
        f"Vehicles {int(tracker_stats['cars']*10)}     "
        f"Rel. speed {rel_speed_peak*src_fps:.0f}px/s"
    )
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, summary)
    y -= 18

    # Timeline chart ----------------------------------------------------------
    png_bytes = _chart_to_png(timeline_df, crit_th, high_th)
    img = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img, margin, y - 100, width=480, height=90, mask="auto")
    y -= 110

    # Key-frame strips ---------------------------------------------------------
    def _strip(title: str, frameset, ypos: float) -> float:
        if not frameset:
            return ypos
        c.setFont("Helvetica-Bold", 11); c.drawString(margin, ypos, title)
        ypos -= 14
        x, th = margin, 56
        for ix, prob, img in frameset[:5]:
            img_r, w_s, h_s = _embed_image(img, 100, th)
            c.drawImage(img_r, x, ypos - h_s, width=w_s, height=h_s, mask="auto")
            c.setFont("Helvetica", 7)
            c.drawCentredString(x + w_s/2, ypos - h_s - 6, f"#{ix} • {prob:.1%}")
            x += 105
        return ypos - th - 18

    y = _strip("Critical frames", critical_frames, y)
    y = _strip("High-risk frames", high_frames, y)

    # MP4 download link --------------------------------------------------------
    with open(clip_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    link = Paragraph(
        f'<a href="data:video/mp4;base64,{b64}">Download 3 s replay</a>',
        ParagraphStyle(name="link", fontSize=8, textColor="#006ddb",
                       leftIndent=0, alignment=TA_LEFT)
    )
    link.wrapOn(c, W-2*margin, 30)
    link.drawOn(c, margin, y)

    # Footer ------------------------------------------------------------------
    c.setFont("Helvetica", 7)
    c.drawString(margin, 12, f"Source video: {src_name}")
    c.save()
    return True