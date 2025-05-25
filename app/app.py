#!/usr/bin/env python3
"""
app/app.py – Streamlit crash detection with simplified, reliable crash type detection
SIMPLIFIED VERSION focusing on bounding box analysis around crash spikes
"""

import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # ✅ Disables torch file watcher error on MacOS

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tempfile
import time
import onnxruntime as ort
import streamlit as st
from ultralytics import YOLO
import re
import urllib.parse
import yt_dlp

st.set_page_config(page_title="LIVE CRASH DETECTION (SOTERIA)", page_icon="🚨", layout="wide")

st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stApp { background-color: #0e1117; }

    .main-title { 
        font-size: 2.5em; 
        font-weight: bold; 
        text-align: center; 
        margin-bottom: 0.5em;
        color: #ffffff;
    }

    .sub-title { 
        font-size: 1.1em; 
        text-align: center; 
        margin-bottom: 1.5em;
        color: #8d9db5;
    }

    .dashboard-card {
        background: linear-gradient(135deg, #1e2329 0%, #262d3a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .video-container {
        background: linear-gradient(135deg, #1e2329 0%, #262d3a 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
    }

    .report-card {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d2626 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a3636;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        color: #ffffff;
    }

    .report-card h3 { 
        color: #ff6b6b; 
        margin-bottom: 1rem;
        font-size: 1.3em;
    }

    .processing-indicator {
        background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #4299e1;
        text-align: center;
        color: #90cdf4;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #262d3a 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2d3748;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ SOTERIA</div>", unsafe_allow_html=True)
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
    """
    CRITICAL: Keep this exactly as the working version to maintain model compatibility
    """
    img = cv2.resize(bgr, (IMG_SZ, IMG_SZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]


def infer_crash(tensor: np.ndarray) -> float:
    out = SESSION.run([OUT_NAME], {IN_NAME: tensor})[0]
    return float(out.ravel()[0])


# SIMPLIFIED CRASH DETECTION ALGORITHMS

def analyze_peak_frame_collision(all_vehicle_boxes, analysed_frame_indices, p_scores, frames, frame_width,
                                 frame_height):
    """
    Analyze the single frame with highest crash probability to determine:
    1. Which specific vehicles are involved in the collision
    2. What type of collision it is
    3. How many vehicles are actually involved
    """
    if not p_scores or not all_vehicle_boxes:
        return "Unknown", 0.1, 1, []

    # Find the frame with the highest crash probability
    max_prob_idx = np.argmax(p_scores)
    max_probability = p_scores[max_prob_idx]
    peak_frame_idx = analysed_frame_indices[max_prob_idx]

    # Get vehicle boxes at the peak crash moment
    if peak_frame_idx >= len(all_vehicle_boxes):
        return "Unknown", 0.1, 1, []

    peak_frame_boxes = list(all_vehicle_boxes[peak_frame_idx])  # Copy the list

    # Define dashcam area (bottom 30% of frame)
    dashcam_threshold = frame_height * 0.7

    # ALWAYS ensure dashcam vehicle has a bounding box
    has_dashcam_box = any(box[1] > dashcam_threshold for box in peak_frame_boxes)

    if not has_dashcam_box:
        # Create estimated dashcam vehicle box - MUCH BIGGER like before
        dashcam_width = int(frame_width * 0.9)  # 90% of frame width (like before)
        dashcam_height = int(frame_height * 0.15)  # 15% of frame height
        dashcam_x1 = int(frame_width * 0.05)  # Start from 5% from left edge
        dashcam_x2 = dashcam_x1 + dashcam_width
        dashcam_y1 = int(frame_height * 0.8)  # Bottom 20% of frame
        dashcam_y2 = dashcam_y1 + dashcam_height

        # Add estimated dashcam box
        estimated_dashcam_box = (dashcam_x1, dashcam_y1, dashcam_x2, dashcam_y2)
        peak_frame_boxes.append(estimated_dashcam_box)

    if not peak_frame_boxes:
        return "Single Vehicle Collision", 0.8, 1, []

    # Find dashcam vehicle box
    dashcam_box = None
    other_boxes = []

    for box in peak_frame_boxes:
        center_y = (box[1] + box[3]) / 2
        if center_y > dashcam_threshold:
            dashcam_box = box
        else:
            other_boxes.append(box)

    # Calculate overlaps with dashcam vehicle
    overlapping_vehicles = []
    collision_detected = False

    if dashcam_box:
        for i, other_box in enumerate(other_boxes):
            # Calculate overlap between dashcam and other vehicle
            overlap_x = max(0, min(dashcam_box[2], other_box[2]) - max(dashcam_box[0], other_box[0]))
            overlap_y = max(0, min(dashcam_box[3], other_box[3]) - max(dashcam_box[1], other_box[1]))
            overlap_area = overlap_x * overlap_y

            # Calculate areas
            dashcam_area = (dashcam_box[2] - dashcam_box[0]) * (dashcam_box[3] - dashcam_box[1])
            other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])

            # Check for meaningful overlap (5% threshold)
            overlap_threshold = min(dashcam_area, other_area) * 0.05

            if overlap_area > overlap_threshold:
                overlapping_vehicles.append(other_box)
                collision_detected = True

    # Also check for overlaps between other vehicles (multi-vehicle without dashcam)
    other_vehicle_overlaps = []
    for i in range(len(other_boxes)):
        for j in range(i + 1, len(other_boxes)):
            box1, box2 = other_boxes[i], other_boxes[j]

            overlap_x = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
            overlap_y = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            overlap_area = overlap_x * overlap_y

            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            overlap_threshold = min(area1, area2) * 0.05

            if overlap_area > overlap_threshold:
                if box1 not in other_vehicle_overlaps:
                    other_vehicle_overlaps.append(box1)
                if box2 not in other_vehicle_overlaps:
                    other_vehicle_overlaps.append(box2)

    # CLEAR COLLISION TYPE LOGIC
    collision_type = "Unknown"
    confidence_score = 0.3
    involved_boxes = []

    if collision_detected:
        # Dashcam vehicle is involved in collision
        involved_boxes = [dashcam_box] + overlapping_vehicles
        involved_count = len(involved_boxes)

        if len(overlapping_vehicles) == 1:
            # Dashcam + 1 other vehicle = 2-vehicle collision
            other_box = overlapping_vehicles[0]

            # Determine collision type based on relative positions
            dashcam_center = ((dashcam_box[0] + dashcam_box[2]) / 2, (dashcam_box[1] + dashcam_box[3]) / 2)
            other_center = ((other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2)

            dx = abs(dashcam_center[0] - other_center[0])
            dy = abs(dashcam_center[1] - other_center[1])

            if dy > dx:  # More vertical separation
                if other_center[1] < dashcam_center[1]:  # Other vehicle is in front
                    collision_type = "Rear-End Collision"
                else:  # Other vehicle is behind
                    collision_type = "Frontal Impact"
                confidence_score = 0.9
            else:  # More horizontal separation
                collision_type = "Side Impact"
                confidence_score = 0.85

        elif len(overlapping_vehicles) > 1:
            # Dashcam + multiple vehicles = Multi-vehicle collision
            collision_type = "Multi-Vehicle Collision"
            confidence_score = 0.9

    elif other_vehicle_overlaps:
        # Other vehicles colliding but not with dashcam
        involved_boxes = other_vehicle_overlaps
        involved_count = len(other_vehicle_overlaps) + 1  # Include dashcam as witness

        if len(other_vehicle_overlaps) == 2:
            collision_type = "Multi-Vehicle Collision (Witnessed)"
            confidence_score = 0.8
        else:
            collision_type = "Multi-Vehicle Collision (Witnessed)"
            confidence_score = 0.75

    else:
        # No overlaps detected = Single vehicle collision
        involved_boxes = [dashcam_box] if dashcam_box else []
        involved_count = 1
        collision_type = "Single Vehicle Collision"
        confidence_score = 0.9

    # Boost confidence based on clear collision indicators
    if collision_detected:
        confidence_score += 0.1  # Direct dashcam involvement

    if max_probability > 0.01:
        confidence_score += 0.1  # High crash probability
    elif max_probability > 0.005:
        confidence_score += 0.05

    # Cap confidence
    confidence_score = min(confidence_score, 1.0)

    return collision_type, confidence_score, involved_count, involved_boxes


def visualize_collision_analysis(frame, involved_boxes, collision_type, confidence_score):
    """
    Create a visualization of the collision analysis showing only involved vehicles
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    frame_height, frame_width = frame.shape[:2]
    dashcam_threshold = frame_height * 0.7

    # Draw involved vehicles with special highlighting
    dashcam_count = 0
    other_count = 0

    for i, box in enumerate(involved_boxes):
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2

        # Determine vehicle type and color
        if center_y > dashcam_threshold:
            color = (255, 0, 0)  # Red for dashcam vehicle
            label = f"Dashcam Vehicle (Involved)"
            dashcam_count += 1
        else:
            color = (255, 165, 0)  # Orange for involved vehicles
            other_count += 1
            label = f"Involved Vehicle {other_count}"

        # Draw thick border for involved vehicles
        draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=color, width=5)
        draw.text((x1, y1 - 30), label, fill=(255, 255, 255))

    # Add collision analysis overlay
    overlay_y = 10
    draw.text((10, overlay_y), f"Collision: {collision_type}", fill=(255, 255, 255))
    overlay_y += 30
    draw.text((10, overlay_y), f"Confidence: {confidence_score:.1%}", fill=(255, 255, 255))
    overlay_y += 30
    draw.text((10, overlay_y), f"Total Involved: {len(involved_boxes)}", fill=(255, 255, 255))
    overlay_y += 30
    draw.text((10, overlay_y), f"Dashcam: {'Yes' if dashcam_count > 0 else 'No'}", fill=(255, 255, 255))
    overlay_y += 30
    draw.text((10, overlay_y), f"Other Vehicles: {other_count}", fill=(255, 255, 255))

    return pil_img


def enhanced_peak_frame_analysis(all_vehicle_boxes, analysed_frame_indices, p_scores, frames, frame_width,
                                 frame_height):
    """
    Enhanced analysis that also considers frames around the peak for better context
    """
    if not p_scores or not all_vehicle_boxes:
        return "Unknown", 0.1, 1, [], None

    # Find peak frame
    max_prob_idx = np.argmax(p_scores)
    peak_frame_idx = analysed_frame_indices[max_prob_idx]

    # Get frames around peak for context (±2 frames)
    context_start = max(0, peak_frame_idx - 2)
    context_end = min(len(all_vehicle_boxes), peak_frame_idx + 3)

    # Analyze the peak frame
    collision_type, confidence_score, involved_count, involved_boxes = analyze_peak_frame_collision(
        all_vehicle_boxes, analysed_frame_indices, p_scores, frames, frame_width, frame_height
    )

    # Create visualization if we have the frame
    visualization = None
    if peak_frame_idx < len(frames):
        visualization = visualize_collision_analysis(
            frames[peak_frame_idx], involved_boxes, collision_type, confidence_score
        )

    # Enhance confidence based on context frames
    if context_end > context_start:
        # Check consistency across nearby frames
        context_vehicle_counts = [len(all_vehicle_boxes[i]) for i in range(context_start, context_end)
                                  if i < len(all_vehicle_boxes)]
        if context_vehicle_counts:
            avg_context_vehicles = np.mean(context_vehicle_counts)
            vehicle_stability = 1.0 - (np.std(context_vehicle_counts) / max(avg_context_vehicles, 1))

            # Boost confidence if vehicle count is stable (more reliable detection)
            if vehicle_stability > 0.8:
                confidence_score += 0.05

    # Final confidence adjustment based on collision type reliability
    type_reliability = {
        "Rear-End Collision": 1.0,  # Most reliable to detect
        "Multi-Vehicle Chain Reaction": 0.95,  # Very reliable pattern
        "Single Vehicle Collision": 0.9,  # Easy to confirm
        "Side Impact": 0.8,  # Moderate reliability
        "Multi-Vehicle Collision": 0.75,  # Can be complex
        "Frontal Impact": 0.7,  # Harder to distinguish
        "Unknown": 0.3  # Low reliability
    }

    confidence_score *= type_reliability.get(collision_type, 0.5)
    confidence_score = min(confidence_score, 1.0)

    return collision_type, confidence_score, involved_count, involved_boxes, visualization


def analyze_bounding_box_patterns(all_vehicle_boxes, crash_start_frame, crash_end_frame, frame_width, frame_height):
    """
    Analyze bounding box patterns around crash to determine collision type
    Returns: collision_type, confidence_score, involved_vehicles
    """

    # Get frames around the crash for analysis
    analysis_start = max(0, crash_start_frame - 10)
    analysis_end = min(len(all_vehicle_boxes), crash_end_frame + 10)

    if analysis_end <= analysis_start:
        return "Unknown", 0.1, 1

    # Count vehicles before, during, and after crash
    pre_crash_frames = all_vehicle_boxes[analysis_start:crash_start_frame]
    crash_frames = all_vehicle_boxes[crash_start_frame:crash_end_frame + 1]
    post_crash_frames = all_vehicle_boxes[crash_end_frame + 1:analysis_end]

    # Calculate average vehicle counts
    pre_crash_count = np.mean([len(frame_boxes) for frame_boxes in pre_crash_frames]) if pre_crash_frames else 0
    crash_count = np.mean([len(frame_boxes) for frame_boxes in crash_frames]) if crash_frames else 0
    post_crash_count = np.mean([len(frame_boxes) for frame_boxes in post_crash_frames]) if post_crash_frames else 0

    # Analyze vehicle count changes
    vehicle_increase = crash_count - pre_crash_count
    max_vehicles_in_crash = max([len(frame_boxes) for frame_boxes in crash_frames]) if crash_frames else 0

    # Analyze bounding box movements and positions
    box_movements = []
    lateral_movements = []
    box_overlaps = []

    # Track how boxes move during crash
    for i in range(len(crash_frames) - 1):
        current_boxes = crash_frames[i]
        next_boxes = crash_frames[i + 1]

        # Simple box matching (closest boxes between frames)
        for curr_box in current_boxes:
            curr_center = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)

            # Find closest box in next frame
            if next_boxes:
                closest_box = min(next_boxes, key=lambda b:
                ((b[0] + b[2]) / 2 - curr_center[0]) ** 2 + ((b[1] + b[3]) / 2 - curr_center[1]) ** 2)
                closest_center = ((closest_box[0] + closest_box[2]) / 2, (closest_box[1] + closest_box[3]) / 2)

                # Calculate movement
                dx = closest_center[0] - curr_center[0]
                dy = closest_center[1] - curr_center[1]
                movement = np.sqrt(dx ** 2 + dy ** 2)

                box_movements.append(movement)
                lateral_movements.append(abs(dx))

        # Check for box overlaps (collision indicator)
        for i, box1 in enumerate(current_boxes):
            for j, box2 in enumerate(current_boxes):
                if i != j:
                    # Calculate overlap
                    overlap_x = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                    overlap_y = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                    overlap_area = overlap_x * overlap_y

                    if overlap_area > 0:
                        box_overlaps.append(overlap_area)

    # Calculate metrics
    avg_movement = np.mean(box_movements) if box_movements else 0
    max_movement = max(box_movements) if box_movements else 0
    avg_lateral = np.mean(lateral_movements) if lateral_movements else 0
    total_overlaps = len(box_overlaps)

    # Determine dashcam vehicle position (usually in bottom portion)
    bottom_threshold = frame_height * 0.6
    dashcam_boxes = []
    other_boxes = []

    for frame_boxes in crash_frames:
        for box in frame_boxes:
            if box[1] > bottom_threshold:  # y1 > threshold (bottom part)
                dashcam_boxes.append(box)
            else:
                other_boxes.append(box)

    # SIMPLIFIED CLASSIFICATION LOGIC
    confidence_score = 0.3  # Base confidence
    collision_type = "Unknown"
    involved_vehicles = max(1, int(max_vehicles_in_crash))

    # 1. SINGLE VEHICLE COLLISION
    # - Low vehicle count throughout
    # - Little movement between vehicles
    # - Low overlaps
    if (max_vehicles_in_crash <= 2 and
            vehicle_increase < 0.5 and
            total_overlaps <= 1):

        collision_type = "Single Vehicle Collision"
        confidence_score = 0.7
        involved_vehicles = 1

    # 2. REAR-END COLLISION
    # - Stable vehicle count (2-4 vehicles)
    # - Moderate movement
    # - Some overlaps
    # - Vehicles mostly in similar lanes (low lateral movement)
    elif (2 <= max_vehicles_in_crash <= 4 and
          abs(vehicle_increase) < 1.0 and
          avg_lateral < 50 and
          total_overlaps >= 1):

        collision_type = "Rear-End Collision"
        confidence_score = 0.8
        involved_vehicles = max(2, int(crash_count))

    # 3. MULTI-VEHICLE COLLISION
    # - High vehicle count (3+ vehicles)
    # - High movement and overlaps
    # - Significant vehicle increase
    elif (max_vehicles_in_crash >= 3 or
          vehicle_increase > 1.0 or
          total_overlaps >= 3):

        collision_type = "Multi-Vehicle Collision"
        confidence_score = 0.7
        involved_vehicles = max(3, int(max_vehicles_in_crash))

    # 4. SIDE IMPACT
    # - Sudden vehicle increase
    # - High lateral movement
    # - Moderate overlaps
    elif (vehicle_increase > 1.5 and
          avg_lateral > 40):

        collision_type = "Side Impact"
        confidence_score = 0.6
        involved_vehicles = max(2, int(crash_count))

    # 5. DEFAULT - UNKNOWN COLLISION
    else:
        if max_vehicles_in_crash >= 2:
            collision_type = "Vehicle Collision"
            involved_vehicles = max(2, int(crash_count))
        else:
            collision_type = "Single Vehicle Collision"
            involved_vehicles = 1
        confidence_score = 0.4

    # Boost confidence based on clear indicators
    if total_overlaps >= 2:
        confidence_score += 0.1
    if max_movement > 30:
        confidence_score += 0.1
    if abs(vehicle_increase) > 0.5:
        confidence_score += 0.1

    # Cap confidence
    confidence_score = min(confidence_score, 1.0)

    return collision_type, confidence_score, involved_vehicles


def calculate_severity(collision_type, max_probability, crash_duration, involved_vehicles, confidence_score):
    """
    Simplified severity calculation
    """
    severity_score = 0.0

    # 1. Base probability factor (0-3 points)
    severity_score += min(max_probability * 300, 3.0)

    # 2. Collision type severity (0-3 points)
    type_severity = {
        "Multi-Vehicle Collision": 3.0,
        "Side Impact": 2.5,
        "Rear-End Collision": 2.0,
        "Vehicle Collision": 2.0,
        "Single Vehicle Collision": 1.5,
        "Unknown": 1.0
    }
    severity_score += type_severity.get(collision_type, 1.0)

    # 3. Vehicle involvement (0-2 points)
    if involved_vehicles >= 3:
        severity_score += 2.0
    elif involved_vehicles == 2:
        severity_score += 1.0
    else:
        severity_score += 0.5

    # 4. Duration factor (0-1 point)
    if crash_duration > 10:
        severity_score += 1.0
    elif crash_duration > 5:
        severity_score += 0.5

    # 5. Confidence boost (0-1 point)
    severity_score += confidence_score

    # Normalize to 0-10 scale
    severity_score = min(severity_score, 10.0)

    # Determine severity level
    if severity_score >= 7.0:
        severity_level = "Critical"
        emergency_dispatch = True
    elif severity_score >= 5.0:
        severity_level = "Severe"
        emergency_dispatch = True
    elif severity_score >= 3.5:
        severity_level = "Moderate"
        emergency_dispatch = True
    elif severity_score >= 2.0:
        severity_level = "Minor"
        emergency_dispatch = False
    else:
        severity_level = "Very Minor"
        emergency_dispatch = False

    # Override for high-risk scenarios
    if collision_type in ["Multi-Vehicle Collision", "Side Impact"] and severity_score >= 3.0:
        emergency_dispatch = True

    return severity_level, round(severity_score, 1), emergency_dispatch


# YouTube and URL handling functions (keep existing ones)
def is_youtube_url(url):
    """Check if the provided URL is a valid YouTube URL including Shorts"""
    if not url:
        return False
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+',
        r'(?:https?://)?(?:m\.)?youtube\.com/shorts/[\w-]+',
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats including Shorts"""
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be/)([0-9A-Za-z_-]{11})',
        r'(?:shorts/)([0-9A-Za-z_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_youtube_video(url):
    """Download YouTube video with improved error handling and format selection"""
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")

        is_short = '/shorts/' in url

        ydl_opts = {
            'outtmpl': output_template,
            'format': (
                'best[height<=1080][ext=mp4]/best[ext=mp4]/best[height<=720]/best'
                if not is_short else
                'best[height<=1920][ext=mp4]/best[ext=mp4]/worst[ext=mp4]/best'
            ),
            'merge_output_format': 'mp4',
            'writesubtitles': False,
            'writeautomaticsub': False,
            'writeinfojson': False,
            'writethumbnail': False,
            'ignoreerrors': False,
            'no_warnings': False,
            'extractaudio': False,
            'audioformat': 'mp3',
            'embed_subs': False,
            'writesubtitles': False,
            'allsubtitles': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
            'retries': 5,
            'fragment_retries': 5,
            'skip_unavailable_fragments': True,
            'socket_timeout': 30,
            'prefer_free_formats': True,
            'youtube_include_dash_manifest': False,
        }

        if is_short:
            ydl_opts.update({
                'extract_flat': False,
                'youtube_include_dash_manifest': False,
                'format': 'best[ext=mp4]/best',
            })

        ffmpeg_locations = [
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',
            'ffmpeg'
        ]

        for ffmpeg_path in ffmpeg_locations:
            if os.path.exists(ffmpeg_path) or ffmpeg_path == 'ffmpeg':
                ydl_opts['ffmpeg_location'] = ffmpeg_path
                break

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)

                max_duration = 180 if is_short else 3600
                if duration and duration > max_duration:
                    return None, f"Video is too long ({duration // 60} minutes). Shorts should be under 3 minutes, regular videos under 1 hour."

                availability = info.get('availability', 'public')
                if availability and availability not in ['public', 'unlisted']:
                    return None, f"Video is not publicly available: {availability}"

            except Exception as info_error:
                return None, f"Failed to get video info: {str(info_error)}"

            ydl.download([url])

            downloaded_files = []
            for file in os.listdir(temp_dir):
                if file.endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov')):
                    downloaded_files.append(os.path.join(temp_dir, file))

            if not downloaded_files:
                return None, "No video file was downloaded"

            video_path = downloaded_files[0]

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return None, "Downloaded file is empty or corrupted"

            return video_path, None

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Video unavailable" in error_msg:
            return None, "Video is unavailable (may be private, deleted, or geo-blocked)"
        elif "Sign in to confirm your age" in error_msg:
            return None, "Video requires age verification and cannot be downloaded"
        elif "This video is not available" in error_msg:
            return None, "Video is not available in your region or has been removed"
        elif "Requested format is not available" in error_msg:
            return None, "Video format not available. Shorts may have limited format options."
        else:
            return None, f"Download failed: {error_msg}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def validate_and_process_youtube_url(video_url):
    """Validate and process YouTube URL with better error handling"""
    if not is_youtube_url(video_url):
        return None, None, "This doesn't appear to be a valid YouTube URL"

    video_id = extract_video_id(video_url)
    if not video_id:
        return None, None, "Could not extract video ID from URL"

    is_short = '/shorts/' in video_url
    video_type = "YouTube Short" if is_short else "YouTube Video"

    progress_msg = "🎥 Downloading YouTube Short..." if is_short else "🎥 Downloading YouTube video..."
    progress_msg += " This may take a few minutes."

    with st.spinner(progress_msg):
        video_path, error = download_youtube_video(video_url)

    if error:
        return None, None, error

    return video_path, f"{video_type} (ID: {video_id})", None


def convert_drive_link(url):
    """Convert a Google Drive share URL into a direct download link."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def is_valid_video_url(url):
    """Check if the URL is a valid direct video URL"""
    if not url:
        return False, "URL is empty"

    invalid_patterns = [
        'google.com/search',
        'youtube.com/results',
        'bing.com/search',
        'search?',
        'results?',
        'google.com',
        'bing.com',
        'yahoo.com'
    ]

    url_lower = url.lower()
    for pattern in invalid_patterns:
        if pattern in url_lower:
            return False, "This appears to be a search URL or webpage, not a direct video URL"

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    if any(ext in url_lower for ext in video_extensions):
        return True, "Valid video file URL"

    streaming_protocols = ['rtmp://', 'rtsp://', 'http://']
    if any(protocol in url_lower for protocol in streaming_protocols):
        return True, "Valid streaming URL"

    return False, "URL doesn't appear to be a direct video file or stream"


def handle_youtube_input(video_url):
    """Main function to handle YouTube URL input"""
    if not video_url or not video_url.strip():
        return None, None, None

    video_url = video_url.strip()

    video_path, video_title, error = validate_and_process_youtube_url(video_url)

    if error:
        return None, None, error

    return video_path, video_title, None


# Streamlit UI Setup
with st.sidebar:
    st.markdown("### 🚨 SOTERIA System")
    fps_target = st.slider("Analysis FPS", 1, 30, 5)

    # Input source selection
    input_type = st.radio(
        "Select Input Source:",
        ["📁 Upload Video File", "🌐 Direct Video URL"]
    )

    if input_type == "📁 Upload Video File":
        src = st.file_uploader("Upload dashcam video (mp4)", type=["mp4"])
        video_url = None
    else:
        src = None
        video_url = st.text_input(
            "Enter Video URL:",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID or http://example.com/video.mp4",
            help="Supports YouTube URLs and direct video file URLs"
        )

        if video_url:
            video_url = video_url.strip()

            if is_youtube_url(video_url):
                video_path, video_title, error = handle_youtube_input(video_url)
                if error:
                    st.error(f"❌ {error}")
                    st.info("💡 Try these troubleshooting steps:")
                    st.info("• Make sure the video is public and not age-restricted")
                    st.info("• Check if the video is available in your region")
                    st.info("• Try copying the URL again from YouTube")
                    if '/shorts/' in video_url:
                        st.info("• YouTube Shorts may have limited availability")
                    video_url = None
                else:
                    st.success(f"✅ Successfully downloaded: {video_title}")

            elif "drive.google.com" in video_url:
                video_url = convert_drive_link(video_url)
                video_path = video_url
                video_title = "Google Drive Video"
                st.success("🔗 Google Drive link converted to direct download.")

            else:
                is_valid, message = is_valid_video_url(video_url)
                if not is_valid:
                    st.error(f"❌ {message}")
                    st.info("💡 Supported formats:")
                    st.info("• YouTube URLs (https://youtube.com/watch?v=...)")
                    st.info("• Direct video files (http://example.com/video.mp4)")
                    st.info("• Google Drive share links")
                    video_url = None
                else:
                    video_path = video_url
                    video_title = "External Video Stream"

    # Show example URLs
    if input_type == "🌐 Direct Video URL":
        with st.expander("ℹ️ Example URLs"):
            st.markdown("""
            **YouTube URLs:**
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - https://www.youtube.com/shorts/VIDEO_ID *(YouTube Shorts)*

            **Direct Video Files:**
            - http://example.com/video.mp4
            - https://sample-videos.com/zip/10/mp4/720p/sample.mp4

            **Live Streams:**
            - rtmp://live.server.com/stream/key
            - rtsp://192.168.1.100:554/stream1

            **Google Drive:**
            - Share links (will be auto-converted)

            **Note:** YouTube Shorts are supported but may have limited format options.
            """)

if src is None and (video_url is None or video_url.strip() == ""):
    st.info("⬅️ Select an input source to begin analysis.")
    st.stop()

# Handle different input sources
video_path = None
video_title = "Video Analysis"
live_mode = False

if src is not None:
    # File upload
    video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_tmp.write(src.read())
    video_tmp.flush()
    video_path = video_tmp.name
    video_title = src.name

elif video_url is not None:
    # URL input - video_path should already be set by the URL handling logic above
    if 'video_path' not in locals() or video_path is None:
        if is_youtube_url(video_url):
            # This should have been handled above, but just in case
            video_path, video_title, error = handle_youtube_input(video_url)
            if error:
                st.error(f"❌ YouTube video failed to process: {error}")
                st.stop()
        elif "drive.google.com" in video_url:
            video_url = convert_drive_link(video_url)
            video_path = video_url
            video_title = "Google Drive Video"
        else:
            video_path = video_url
            video_title = "External Video Stream"

    # Check if it might be a live stream based on URL
    if any(protocol in video_url.lower() for protocol in ['rtmp://', 'rtsp://', 'hls://', 'm3u8']):
        live_mode = True
        st.success(f"🔴 Connecting to live stream...")
    else:
        st.success(f"📹 Loading video from URL...")

# Validate video path before proceeding
if video_path is None:
    st.error("❌ No valid video source provided.")
    st.stop()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("❌ Failed to open video source. Please check the URL or try a different file.")
    st.info("💡 Make sure the URL is a direct link to a video file (not a webpage)")
    st.stop()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(int(src_fps // fps_target), 1)
frame_duration = 1.0 / src_fps

# For live streams, we don't know total frames
if live_mode or total_frames <= 0:
    total_frames = float('inf')
    progress_text = "🔴 Live Analysis"
else:
    progress_text = "📹 Processing Video"

# Create main layout: 60% left for video/chart, 40% right for report
col1, col2 = st.columns([0.6, 0.4])

with col1:
    # Video feed section
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.markdown(f"### 📹 {progress_text}")
    st.markdown(f"**Source:** {video_title}")
    progress = st.progress(0.0)
    viewer = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Chart section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Crash Probability")
    chart = st.line_chart(y=[])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Metrics section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Analysis Metrics")

    metrics_cols = st.columns(2)
    with metrics_cols[0]:
        frames_processed = st.empty()
    with metrics_cols[1]:
        vehicles_detected = st.empty()

    st.markdown('</div>', unsafe_allow_html=True)

    # Report section
    report_placeholder = st.empty()
    with report_placeholder.container():
        st.markdown("""
        <div class='processing-indicator'>
            <h3>🔄 Processing Video</h3>
            <p>Analysis in progress... Report will appear here once complete.</p>
        </div>
        """, unsafe_allow_html=True)

# Initialize tracking variables
idx = 0
p_scores = []
frames = []
analysed_frame_indices = []
all_vehicle_boxes = []  # Store all vehicle bounding boxes for each frame
spike_max = 0.0

# Main video processing loop
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    current_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    if idx % step == 0:
        p = infer_crash(preprocess_classifier(frame))
        p_scores.append(p)
        analysed_frame_indices.append(idx)
        spike_max = max(spike_max, p)
        with col1:
            chart.add_rows([p])

    results = yolo_model(current_frame, verbose=False)[0]
    boxes = results.boxes

    # Store vehicle boxes for this frame
    frame_vehicle_boxes = []
    cars_this_frame = 0

    pil_img = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for box in boxes:
        cls = int(box.cls[0])
        if yolo_model.names[cls] not in CAR_CLASSES:
            continue
        x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
        frame_vehicle_boxes.append((x1, y1, x2, y2))
        cars_this_frame += 1

        # Determine if this is likely the dashcam vehicle (bottom portion of frame)
        if y1 > frame_height * 0.7:
            tag = "Dashcam Vehicle"
            color = (255, 0, 0)  # Red for dashcam
        else:
            tag = "Vehicle"
            color = (0, 255, 0)  # Green for other vehicles

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 25), tag, fill=(255, 255, 0))

    # Store boxes for this frame
    all_vehicle_boxes.append(frame_vehicle_boxes)

    # Update displays
    with col1:
        viewer.image(np.array(pil_img), channels="RGB", use_column_width=True)

        # Different progress display for live vs recorded
        if live_mode or total_frames == float('inf'):
            progress.progress(1.0, text=f"🔴 Live: Frame {idx}")
        else:
            progress.progress(min(idx / total_frames, 1.0), text=f"Processing: {idx}/{total_frames} frames")

    # Update metrics
    with col2:
        with metrics_cols[0]:
            frames_processed.metric("Frames Processed", idx)

        with metrics_cols[1]:
            vehicles_detected.metric("Vehicles Detected", cars_this_frame)

    frames.append(current_frame.copy())

    # For live streams, limit frame storage to prevent memory issues
    if live_mode and len(frames) > 1000:
        frames = frames[-500:]
        analysed_frame_indices = analysed_frame_indices[-500:]
        p_scores = p_scores[-500:]
        all_vehicle_boxes = all_vehicle_boxes[-500:]

    # Adjust sleep for different modes
    if not live_mode:
        time.sleep(frame_duration)
    else:
        time.sleep(0.1)

    idx += 1

    # For live streams, continue indefinitely
    if live_mode and idx > 10000:
        idx = 0

cap.release()

# SIMPLIFIED CRASH ANALYSIS SECTION
if len(p_scores) > 5:
    mean_prob = np.mean(p_scores)
    threshold = mean_prob * THRESH_MULTIPLIER
    spike_indices = [i for i, p in enumerate(p_scores) if p > threshold]

    # High confidence detection
    high_prob_indices = [i for i, p in enumerate(p_scores) if p > 0.4]  # Lowered threshold

    # Multi-spike pattern detection
    multi_spike_detected = len(spike_indices) >= MIN_SPIKES

    # Single high-confidence detection
    high_confidence_detected = len(high_prob_indices) >= 1

    # Combined crash detection logic
    crash_detected = multi_spike_detected or high_confidence_detected

    # Determine which detection method to use for reporting
    detection_method = ""
    if multi_spike_detected and high_confidence_detected:
        detection_method = "multi-spike + high-confidence"
        final_spike_indices = spike_indices
    elif multi_spike_detected:
        detection_method = "multi-spike pattern"
        final_spike_indices = spike_indices
    elif high_confidence_detected:
        detection_method = "high-confidence single event"
        final_spike_indices = high_prob_indices
    else:
        final_spike_indices = []

    if crash_detected and final_spike_indices:
        # Get the actual frame indices for crash analysis
        start_frame_idx = analysed_frame_indices[final_spike_indices[0]]
        end_frame_idx = analysed_frame_indices[final_spike_indices[-1]]

        # Create crash clip
        clip_start = max(start_frame_idx - int(1.0 * src_fps), 0)
        clip_end = min(end_frame_idx + int(1.0 * src_fps), len(frames))
        clip_frames = frames[clip_start:clip_end]

        if clip_frames:
            height, width, _ = clip_frames[0].shape
            temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(temp_vid.name, cv2.VideoWriter_fourcc(*'mp4v'), src_fps, (width, height))
            for f in clip_frames:
                out.write(f)
            out.release()

            crash_seconds = int(start_frame_idx / src_fps)
            timestamp_str = f"{crash_seconds // 60:02}:{crash_seconds % 60:02}"

            # Calculate max_prob for use throughout analysis
            max_prob = max(p_scores) if p_scores else 0

            # PEAK FRAME ANALYSIS - Focus on the highest probability moment
            collision_type, confidence_score, involved_vehicles, involved_boxes, collision_viz = enhanced_peak_frame_analysis(
                all_vehicle_boxes, analysed_frame_indices, p_scores, frames, frame_width, frame_height
            )

            # Calculate severity
            crash_duration = final_spike_indices[-1] - final_spike_indices[0] + 1
            impact_severity, severity_score, dispatch = calculate_severity(
                collision_type, max_prob, crash_duration, involved_vehicles, confidence_score
            )

            # Determine confidence level
            if confidence_score >= 0.8:
                confidence_level = "Very High"
            elif confidence_score >= 0.65:
                confidence_level = "High"
            elif confidence_score >= 0.45:
                confidence_level = "Medium"
            elif confidence_score >= 0.25:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"

            # Display simplified report
            with col2:
                report_placeholder.empty()

                st.markdown("""
                <div class='report-card'>
                    <h3>🚨 Crash Report</h3>
                </div>
                """, unsafe_allow_html=True)

                # Core incident details
                confidence_icons = {
                    "Very High": "🟢", "High": "🟢", "Medium": "🟡",
                    "Low": "🟠", "Very Low": "🔴"
                }
                confidence_icon = confidence_icons.get(confidence_level, "⚪")

                st.markdown(f"**Collision Type:** {collision_type} {confidence_icon}")
                st.markdown(f"**Impact Severity:** {impact_severity}")
                st.markdown(f"**Incident Time:** {timestamp_str}")
                st.markdown(f"**Vehicles Involved:** {involved_vehicles}")
                st.markdown(f"**Analysis Confidence:** {confidence_level} ({confidence_score:.2f})")
                st.markdown(f"**Detection Method:** {detection_method}")
                st.markdown(f"**Max Probability:** {max_prob:.1%}")
                st.markdown(f"**Alert Frames:** {len(final_spike_indices)}")

                # Severity scoring with context
                if severity_score >= 7:
                    st.markdown(f"**Severity Score:** :red[{severity_score}/10]")
                elif severity_score >= 4:
                    st.markdown(f"**Severity Score:** :orange[{severity_score}/10]")
                else:
                    st.markdown(f"**Severity Score:** :green[{severity_score}/10]")

                # Impact assessment
                severity_context = {
                    "Critical": "🔴 Life-threatening injuries likely - immediate response required",
                    "Severe": "🟠 Serious injuries probable - priority response needed",
                    "Moderate": "🟡 Moderate injuries possible - standard response",
                    "Minor": "🟢 Minor injuries likely - low priority response",
                    "Very Minor": "🟢 Minimal injuries expected - monitor situation"
                }

                if impact_severity in severity_context:
                    st.markdown(f"**Impact Assessment:** {severity_context[impact_severity]}")

                # Emergency response
                if dispatch:
                    st.markdown("**Emergency Response:** :green[✅ DISPATCHED]")

                    if involved_vehicles >= 3:
                        st.markdown("*🚑 Multiple vehicles involved - trauma units recommended*")
                    elif severity_score >= 7:
                        st.markdown("*🚨 High severity impact - immediate response required*")
                    elif collision_type in ["Multi-Vehicle Collision", "Side Impact"]:
                        st.markdown("*⚡ High-energy collision - priority response*")
                    else:
                        st.markdown("*⚠️ Significant incident detected - response dispatched*")
                else:
                    st.markdown("**Emergency Response:** :orange[⚠️ Monitoring]")
                    if severity_score < 2.0:
                        st.markdown("*📞 Very low severity - no immediate response needed*")
                    else:
                        st.markdown("*📞 Low-moderate severity - monitor for self-reporting*")

                # Download crash clip
                with open(temp_vid.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Incident Clip",
                        data=f.read(),
                        file_name=f"crash_incident_{timestamp_str.replace(':', '')}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

                # Display collision visualization if available
                if collision_viz:
                    st.markdown("### 🎯 Peak Collision Moment Analysis")
                    st.image(collision_viz, caption="Frame with highest crash probability showing involved vehicles",
                             use_container_width=True)

                # Simplified technical analysis
                with st.expander("🔍 Technical Analysis Details"):
                    st.write(
                        f"**Peak Frame Analysis:** Frame {analysed_frame_indices[np.argmax(p_scores)] if p_scores else 'N/A'}")
                    st.write(f"**Classification Confidence:** {confidence_score:.1%}")
                    st.write(f"**Crash Duration:** {crash_duration} frames ({crash_duration / src_fps:.1f} seconds)")
                    st.write(f"**Peak Probability:** {max_prob:.3f}")
                    st.write(f"**Average Probability:** {np.mean([p_scores[i] for i in final_spike_indices]):.3f}")

                    # Show involved vehicle details
                    st.write(f"**Involved Vehicle Analysis:**")
                    st.write(f"• Detected overlapping/colliding vehicles: {len(involved_boxes)}")
                    st.write(f"• Total involved (including dashcam): {involved_vehicles}")

                    if involved_boxes:
                        st.write(f"• Vehicle positions at peak moment:")
                        for i, box in enumerate(involved_boxes):
                            x1, y1, x2, y2 = box
                            center_y = (y1 + y2) / 2
                            is_dashcam = center_y > frame_height * 0.7
                            vehicle_type = "Dashcam Vehicle" if is_dashcam else f"Other Vehicle {i + 1}"
                            st.write(f"  - {vehicle_type}: Position ({x1}, {y1}) to ({x2}, {y2})")

                    # Vehicle pattern analysis
                    if start_frame_idx < len(all_vehicle_boxes) and end_frame_idx < len(all_vehicle_boxes):
                        pre_crash_vehicles = np.mean([len(boxes) for boxes in all_vehicle_boxes[max(0,
                                                                                                    start_frame_idx - 10):start_frame_idx]]) if start_frame_idx > 10 else 0
                        crash_vehicles = np.mean(
                            [len(boxes) for boxes in all_vehicle_boxes[start_frame_idx:end_frame_idx + 1]])

                        st.write(f"**Scene Vehicle Analysis:**")
                        st.write(f"• Pre-crash total vehicles in scene: {pre_crash_vehicles:.1f}")
                        st.write(f"• During crash total vehicles in scene: {crash_vehicles:.1f}")
                        st.write(f"• Total scene vehicle change: {crash_vehicles - pre_crash_vehicles:+.1f}")
                        st.write(f"• Actual collision participants: {involved_vehicles}")


    else:
        # No crash detected
        with col2:
            report_placeholder.empty()

            max_prob = max(p_scores) if p_scores else 0
            avg_prob = np.mean(p_scores) if p_scores else 0
            threshold = mean_prob * THRESH_MULTIPLIER if p_scores else 0

            st.markdown("""
            <div class='processing-indicator'>
                <h3>✅ Analysis Complete</h3>
                <p>No significant collision detected in video analysis.</p>
            </div>
            """, unsafe_allow_html=True)

            # Analysis summary
            st.info("📊 **Analysis Summary:**")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Peak Probability", f"{max_prob:.1%}")
                st.metric("Detection Threshold", f"{threshold:.3f}")

            with col_b:
                st.metric("Multi-spike Frames", len(spike_indices))
                st.metric("High-confidence Frames", len(high_prob_indices))

            # Analysis quality assessment
            if max_prob > 0.3:
                st.warning(
                    "⚠️ **Note:** Some elevated crash probabilities detected, but below threshold for incident classification.")
                st.info("💡 **Recommendation:** Consider manual review if incident suspected.")
            elif len(p_scores) < 30:
                st.info("ℹ️ **Note:** Short video duration may limit detection accuracy.")
            else:
                st.success("✅ **Confidence:** Normal driving patterns detected throughout analysis.")

            # Recommendations
            with st.expander("💡 Analysis Recommendations"):
                st.write("**For improved detection accuracy:**")
                st.write("• Ensure video shows clear view of road and traffic")
                st.write("• Minimum 10-15 seconds of footage recommended")
                st.write("• Higher resolution videos provide better detection")
                st.write("• Stable camera mounting reduces false positives")

                if max_prob > 0.2:
                    st.write("**Elevated probability notes:**")
                    st.write("• Some frames showed higher crash probability")
                    st.write("• This could indicate: sudden braking, potholes, or minor impacts")
                    st.write("• Consider manual review if incident suspected")

else:
    with col2:
        report_placeholder.empty()
        st.warning("⚠️ Insufficient data for analysis. Video too short or processing failed.")
