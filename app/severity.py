"""
severity.py – shared severity‑scoring utilities for SOTERIA
"""

from __future__ import annotations
import cv2
import numpy as np
from collections import defaultdict

# ───────────────────────────  Tunables  ────────────────────────────
FLOW_MAX: float = 12.0                          # px‑/frame magnitude → 1.0
CAR_MAX:  int   = 10                            # ≥10 vehicles saturates count
WEIGHTS: dict[str, float] = {
    "p_peak":   0.45,
    "dur_high": 0.20,
    "delta_v":  0.25,
    "cars":     0.10,
}
# ───────────────────────────────────────────────────────────────────

def flow_mag(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """Mean dense‑optical‑flow magnitude (pixels) between two BGR frames."""
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow   = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    return float(np.linalg.norm(flow, axis=2).mean())


class SeverityTracker:
    """Collect per-frame features and compute severity score/class."""

    def __init__(
        self,
        *,
        flow_max:  float = FLOW_MAX,
        car_max:   int   = CAR_MAX,
        weights:   dict[str, float] = WEIGHTS,
        low_cut:   float = 0.30,   # Minor  → Moderate  when score ≥ this
        high_cut:  float = 0.60    # Moderate → Severe  when score ≥ this
    ) -> None:
        # --- normalisation constants ---------------------------------------
        self.flow_max   = flow_max
        self.car_max    = car_max

        # --- scoring knobs --------------------------------------------------
        self.weights    = weights
        self.low_cut    = low_cut
        self.high_cut   = high_cut

        # --- running stats --------------------------------------------------
        self.stats      = defaultdict(float)     # p_peak, dur_high, delta_v, cars
        self.prev_frame: np.ndarray | None = None
        self.frames     = 0

    # ──────────────────────────────────────────────────────────────
    def update(
        self,
        frame_bgr: np.ndarray,
        p_crash:   float,
        cars_now:  int,
        rel_speed: float,
        high_th:   float,
            analysed_frame: bool = True  # ← add flag
    ) -> None:
        """Feed one *analysed* frame."""
        if analysed_frame:
            dv = rel_speed / self.flow_max             # 0-1
            self.stats["delta_v"] = max(self.stats["delta_v"], min(dv, 1.0))


        self.prev_frame = frame_bgr.copy()
        self.stats["p_peak"] = max(self.stats["p_peak"], p_crash)
        self.stats["cars"]   = max(
            self.stats["cars"], min(cars_now / self.car_max, 1.0)
        )
        if analysed_frame and p_crash >= high_th:
            self.stats["dur_high"] += 1
        self.frames += 1

    # ──────────────────────────────────────────────────────────────
    def result(self) -> tuple[float, str]:
        """Return (severity_score ∈ [0‑1], class)."""
        if self.frames:
            self.stats["dur_high"] /= self.frames

        score = sum(self.weights[k] * self.stats[k] for k in self.weights)
        score = float(np.clip(score, 0.0, 1.0))
        LOW = self.low_cut
        HIGH = self.high_cut
        cls = ("Minor", "Moderate", "Severe")[(score >= LOW) + (score >= HIGH)]
        return score, cls
