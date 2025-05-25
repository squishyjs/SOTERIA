#!/usr/bin/env python3
"""
app/app.py – Streamlit crash detection with professional dashboard layout
FIXED VERSION with improved crash type detection and severity assessment
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


# IMPROVED CRASH DETECTION ALGORITHMS

def calculate_velocity_and_acceleration(centroids, fps):
    """Calculate velocity and acceleration from centroid positions"""
    if len(centroids) < 3:
        return [], []

    velocities = []
    accelerations = []

    # Calculate velocities (pixels per second)
    for i in range(1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            velocities.append(0.0)
            continue

        dx = centroids[i][0] - centroids[i - 1][0]
        dy = centroids[i][1] - centroids[i - 1][1]
        velocity = np.sqrt(dx ** 2 + dy ** 2) * fps
        velocities.append(velocity)

    # Calculate accelerations (pixels per second squared)
    for i in range(1, len(velocities)):
        accel = (velocities[i] - velocities[i - 1]) * fps
        accelerations.append(accel)

    return velocities, accelerations


def detect_sudden_impact(velocities, accelerations, threshold_velocity_drop=50.0, threshold_deceleration=200.0):
    """Detect sudden impact based on velocity and acceleration patterns"""
    if len(velocities) < 5 or len(accelerations) < 3:
        return False, 0.0, 0.0

    # Look for sudden velocity drops
    max_velocity_drop = 0.0
    max_deceleration = 0.0
    sudden_impact = False

    # Check last few frames for impact
    recent_velocities = velocities[-5:]
    recent_accelerations = accelerations[-3:]

    # Velocity drop analysis
    for i in range(1, len(recent_velocities)):
        velocity_drop = recent_velocities[i - 1] - recent_velocities[i]
        if velocity_drop > max_velocity_drop:
            max_velocity_drop = velocity_drop

    # Deceleration analysis
    for accel in recent_accelerations:
        if abs(accel) > abs(max_deceleration):
            max_deceleration = accel

    # Determine if this constitutes sudden impact
    if max_velocity_drop > threshold_velocity_drop or abs(max_deceleration) > threshold_deceleration:
        sudden_impact = True

    return sudden_impact, max_velocity_drop, abs(max_deceleration)


def analyze_movement_direction(centroids, window_size=10):
    """Analyze predominant movement direction"""
    if len(centroids) < window_size:
        return 0.0, 0.0, 0.0  # dx, dy, angle

    recent_centroids = centroids[-window_size:]
    valid_centroids = [c for c in recent_centroids if c is not None]

    if len(valid_centroids) < 3:
        return 0.0, 0.0, 0.0

    # Calculate overall movement vector
    start_point = valid_centroids[0]
    end_point = valid_centroids[-1]

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # Calculate movement angle
    angle = np.arctan2(dy, dx) if dx != 0 or dy != 0 else 0.0

    return dx, dy, angle


def calculate_vehicle_involvement_score(car_count_history, crash_start, crash_end, window_size=30):
    """Calculate more accurate vehicle involvement based on statistical analysis"""
    if crash_start < window_size or crash_end >= len(car_count_history):
        return 1, 0.0  # Default to single vehicle, low confidence

    # Get baseline vehicle count (before crash)
    pre_crash_counts = car_count_history[crash_start - window_size:crash_start]
    crash_counts = car_count_history[crash_start:crash_end + 1]
    post_crash_counts = car_count_history[crash_end + 1:min(len(car_count_history), crash_end + window_size)]

    if not pre_crash_counts or not crash_counts:
        return 1, 0.0

    # Statistical analysis
    baseline_avg = np.mean(pre_crash_counts)
    baseline_std = np.std(pre_crash_counts) if len(pre_crash_counts) > 1 else 0.5
    crash_avg = np.mean(crash_counts)
    crash_max = max(crash_counts)

    # Calculate confidence based on how different crash period is from baseline
    if baseline_std > 0:
        z_score = abs(crash_avg - baseline_avg) / baseline_std
        confidence = min(z_score / 3.0, 1.0)  # Normalize to 0-1
    else:
        confidence = 0.5

    # Estimate involved vehicles
    # Use the maximum increase during crash, but be conservative
    vehicle_increase = max(0, crash_max - baseline_avg)
    involved_vehicles = max(1, int(baseline_avg + vehicle_increase + 1))  # +1 for dashcam

    # Cap at reasonable maximum and adjust based on confidence
    involved_vehicles = min(involved_vehicles, 6)

    # If confidence is low, default to fewer vehicles
    if confidence < 0.3:
        involved_vehicles = min(involved_vehicles, 2)

    return involved_vehicles, confidence


def improved_crash_type_classification(dashcam_centroids, car_count_history, crash_start, crash_end,
                                       p_scores, fps, confidence_threshold=0.05):
    """
    Improved crash type classification - FIXED for dashcam scenarios where dashcam doesn't move much
    """
    if crash_start >= len(dashcam_centroids) or crash_end >= len(dashcam_centroids):
        return "Unknown", "Very Low", 0.1

    # Get probability metrics first - these are our most reliable indicators
    crash_probabilities = p_scores[crash_start:crash_end + 1] if crash_end < len(p_scores) else []
    max_probability = max(crash_probabilities) if crash_probabilities else 0.0
    avg_probability = np.mean(crash_probabilities) if crash_probabilities else 0.0

    # Calculate crash duration and intensity
    crash_duration = crash_end - crash_start + 1

    # Vehicle count analysis - FIXED LOGIC
    if crash_start >= 30 and crash_end < len(car_count_history):
        pre_crash_counts = car_count_history[crash_start - 30:crash_start]
        crash_counts = car_count_history[crash_start:crash_end + 1]

        pre_crash_avg = np.mean(pre_crash_counts) if pre_crash_counts else 0
        crash_avg = np.mean(crash_counts) if crash_counts else 0
        crash_max = max(crash_counts) if crash_counts else 0

        # For dashcam footage, we expect:
        # - Rear-end: vehicle count stays same or increases (other car hits us)
        # - Side impact: vehicle count increases (car comes from side)
        # - Single vehicle: count stays same or decreases (we hit obstacle)

        vehicle_change = crash_avg - pre_crash_avg
        max_vehicle_change = crash_max - pre_crash_avg

        # Involvement calculation
        if max_vehicle_change > 0.5:  # Clear increase in vehicles
            involved_vehicles = int(pre_crash_avg + max_vehicle_change + 1)  # +1 for dashcam
            vehicle_confidence = min(max_vehicle_change / 2.0, 1.0)
        elif abs(vehicle_change) < 0.3:  # Stable count (likely rear-end or we hit them)
            involved_vehicles = max(2, int(pre_crash_avg + 1))  # Assume 2+ vehicles
            vehicle_confidence = 0.6  # Medium confidence for stable count
        else:  # Decrease (we hit obstacle or lost tracking)
            involved_vehicles = 1
            vehicle_confidence = 0.3
    else:
        # Fallback when insufficient data
        involved_vehicles = 2  # Default assumption for crashes
        vehicle_confidence = 0.4
        vehicle_change = 0
        max_vehicle_change = 0

    # Cap involved vehicles
    involved_vehicles = min(involved_vehicles, 6)

    # Movement analysis (may not work well for dashcam)
    crash_centroids = dashcam_centroids[crash_start:crash_end + 5]
    velocities, accelerations = calculate_velocity_and_acceleration(crash_centroids, fps)
    sudden_impact, velocity_drop, max_deceleration = detect_sudden_impact(velocities, accelerations)
    dx, dy, movement_angle = analyze_movement_direction(dashcam_centroids, 15)

    # START CLASSIFICATION - prioritize probability and vehicle patterns over movement
    confidence_score = 0.3  # Higher base confidence
    collision_type = "Unknown"

    # REAR-END COLLISION detection (most common in dashcam footage)
    if crash_duration >= 3 and max_probability > 0.002:  # Very low threshold

        # Pattern 1: Multiple vehicles, stable/increasing count, multi-spike
        if involved_vehicles >= 2 and vehicle_change >= -0.5:  # Allow slight decrease due to tracking issues
            collision_type = "Rear-End Collision"
            confidence_score += 0.4

            # Boost confidence for typical rear-end patterns
            if crash_duration >= 5:  # Multiple impact frames
                confidence_score += 0.2
            if vehicle_confidence > 0.4:
                confidence_score += 0.2
            if max_probability > 0.005:
                confidence_score += 0.1

        # Pattern 2: Clear vehicle increase (side impact)
        elif max_vehicle_change > 1.0:
            collision_type = "Side Impact / T-Bone"
            confidence_score += 0.35

        # Pattern 3: Single vehicle (obstacle/barrier)
        elif involved_vehicles == 1 or vehicle_change < -1.0:
            collision_type = "Single Vehicle / Obstacle Impact"
            confidence_score += 0.25

        # Pattern 4: Movement-based detection (if available)
        elif sudden_impact and velocity_drop > 20:
            if involved_vehicles >= 2:
                collision_type = "Rear-End Collision"
                confidence_score += 0.35
            else:
                collision_type = "Frontal Impact"
                confidence_score += 0.3

        # Pattern 5: Lateral movement
        elif abs(dx) > 25:
            if involved_vehicles >= 2:
                collision_type = "Sideswipe"
                confidence_score += 0.3
            else:
                collision_type = "Lane Departure"
                confidence_score += 0.2

        # Pattern 6: Extended duration
        elif crash_duration > 10:
            if involved_vehicles >= 3:
                collision_type = "Multi-Vehicle Incident"
                confidence_score += 0.3
            else:
                collision_type = "Single Vehicle / Rollover"
                confidence_score += 0.25

        # Default for detected crashes
        else:
            if involved_vehicles >= 2:
                collision_type = "Multi-Vehicle Collision"
                confidence_score += 0.3
            else:
                collision_type = "Single Vehicle Incident"
                confidence_score += 0.2

    # If still unknown but we detected spikes, default to most likely scenario
    if collision_type == "Unknown" and max_probability > 0.001:
        if involved_vehicles >= 2:
            collision_type = "Rear-End Collision"  # Most common dashcam crash
            confidence_score = 0.4
        else:
            collision_type = "Single Vehicle Incident"
            confidence_score = 0.3

    # Probability-based confidence boost
    if max_probability > 0.01:
        confidence_score += 0.3
    elif max_probability > 0.005:
        confidence_score += 0.2
    elif max_probability > 0.002:
        confidence_score += 0.15
    elif max_probability > 0.001:
        confidence_score += 0.1

    # Duration boost
    if crash_duration >= 5:
        confidence_score += 0.1

    # Cap confidence score
    confidence_score = min(confidence_score, 1.0)

    # Determine confidence level
    if confidence_score >= 0.7:
        confidence_level = "Very High"
    elif confidence_score >= 0.55:
        confidence_level = "High"
    elif confidence_score >= 0.35:
        confidence_level = "Medium"
    elif confidence_score >= 0.2:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"

    return collision_type, confidence_level, confidence_score


def enhanced_severity_assessment(collision_type, max_probability, avg_probability,
                                 velocities, accelerations, involved_vehicles,
                                 crash_duration, sudden_impact_data):
    """
    Enhanced severity assessment with proper physics and logic
    """
    severity_score = 0.0

    # Unpack sudden impact data
    sudden_impact, velocity_drop, max_deceleration = sudden_impact_data

    # 1. Base probability factor (0-2.5 points)
    prob_factor = max_probability * 2.5
    severity_score += prob_factor

    # 2. Impact intensity factor (0-2.5 points)
    if sudden_impact:
        # Velocity drop severity
        velocity_severity = min(velocity_drop / 100.0, 1.0) * 1.5
        # Deceleration severity
        decel_severity = min(max_deceleration / 300.0, 1.0) * 1.0
        severity_score += velocity_severity + decel_severity
    else:
        # Gradual impact is less severe
        severity_score += avg_probability * 1.0

    # 3. Collision type severity multiplier (0-2 points)
    type_severity = {
        "High-Speed Frontal Impact": 2.0,
        "Head-On Collision": 2.0,
        "High-Speed Side Impact": 1.8,
        "Multi-Vehicle Pileup": 1.8,
        "Side Impact / T-Bone": 1.6,
        "Frontal Impact / Obstacle": 1.4,
        "Rear-End Collision": 1.2,
        "Complex Multi-Vehicle Event": 1.5,
        "Multi-Vehicle Collision": 1.3,
        "Single Vehicle / Rollover": 1.4,
        "Sideswipe / Lane Departure": 0.8,
        "Minor Multi-Vehicle Contact": 0.6,
        "Single Vehicle Incident": 0.7,
        "Minor Contact / False Positive": 0.2,
        "No Significant Impact": 0.1
    }
    severity_score += type_severity.get(collision_type, 1.0)

    # 4. Vehicle involvement factor (0-1.5 points)
    vehicle_factor = min((involved_vehicles - 1) / 4.0, 1.0) * 1.5
    severity_score += vehicle_factor

    # 5. Duration factor (0-1 point)
    if crash_duration > 10:
        duration_factor = min(crash_duration / 20.0, 1.0)
        severity_score += duration_factor

    # 6. Speed factor (0-1 point)
    if velocities:
        avg_speed = np.mean(velocities[-10:]) if len(velocities) >= 10 else np.mean(velocities)
        speed_factor = min(avg_speed / 150.0, 1.0)
        severity_score += speed_factor

    # Normalize to 0-10 scale
    severity_score = min(severity_score, 10.0)
    severity_score = max(severity_score, 0.0)

    # Determine severity level and emergency dispatch
    if severity_score >= 7.5:
        severity_level = "Critical"
        emergency_dispatch = True
    elif severity_score >= 6.0:
        severity_level = "Severe"
        emergency_dispatch = True
    elif severity_score >= 4.0:
        severity_level = "Moderate"
        emergency_dispatch = True
    elif severity_score >= 2.5:
        severity_level = "Minor"
        emergency_dispatch = False
    elif severity_score >= 1.0:
        severity_level = "Very Minor"
        emergency_dispatch = False
    else:
        severity_level = "Negligible"
        emergency_dispatch = False

    # Override dispatch for high-risk scenarios
    high_risk_types = ["High-Speed Frontal Impact", "Multi-Vehicle Pileup", "Head-On Collision"]
    if collision_type in high_risk_types or involved_vehicles >= 3:
        if severity_score >= 3.0:  # Lower threshold for high-risk scenarios
            emergency_dispatch = True

    # Override for very low probability events
    if max_probability < 0.2 and not sudden_impact:
        emergency_dispatch = False
        if severity_score < 3.0:
            severity_level = "Very Minor"

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


def estimate_speed(prev, curr):
    """Simple speed estimation for compatibility"""
    if prev is None or curr is None:
        return 0
    dx = abs(curr[0] - prev[0])
    dy = abs(curr[1] - prev[1])
    return np.sqrt(dx ** 2 + dy ** 2)


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
car_count_history = []
dashcam_centroids = []
spike_max = 0.0
spike_dur = 0

# Main video processing loop
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
        with col1:
            chart.add_rows([p])

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
        car_count_history = car_count_history[-500:]
        dashcam_centroids = dashcam_centroids[-500:]

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

# CRASH ANALYSIS SECTION - RESTORED ORIGINAL DETECTION LOGIC
if len(p_scores) > 5:
    mean_prob = np.mean(p_scores)
    threshold = mean_prob * THRESH_MULTIPLIER
    spike_indices = [i for i, p in enumerate(p_scores) if p > threshold]

    # High confidence detection (single spike method) - RESTORED ORIGINAL THRESHOLD
    high_prob_indices = [i for i, p in enumerate(p_scores) if p > 0.7]

    # Multi-spike pattern detection (original method)
    multi_spike_detected = len(spike_indices) >= MIN_SPIKES

    # Single high-confidence detection
    high_confidence_detected = len(high_prob_indices) >= 1

    # Combined crash detection logic - RESTORED ORIGINAL LOGIC
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
        # IMPROVED CRASH ANALYSIS
        start_idx = analysed_frame_indices[final_spike_indices[0]]
        end_idx = analysed_frame_indices[final_spike_indices[-1]]

        # Create crash clip
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

        # Calculate max_prob for use throughout analysis
        max_prob = max(p_scores) if p_scores else 0

        # ENHANCED CRASH ANALYSIS
        collision_type, type_confidence, confidence_score = improved_crash_type_classification(
            dashcam_centroids, car_count_history, final_spike_indices[0], final_spike_indices[-1],
            p_scores, src_fps
        )

        # Calculate movement metrics for severity assessment
        crash_centroids = dashcam_centroids[final_spike_indices[0]:final_spike_indices[-1] + 5]
        velocities, accelerations = calculate_velocity_and_acceleration(crash_centroids, src_fps)
        sudden_impact_data = detect_sudden_impact(velocities, accelerations)

        # Get vehicle involvement
        involved_vehicles, vehicle_confidence = calculate_vehicle_involvement_score(
            car_count_history, final_spike_indices[0], final_spike_indices[-1]
        )

        # Enhanced severity assessment
        crash_duration = final_spike_indices[-1] - final_spike_indices[0] + 1
        avg_prob = np.mean([p_scores[i] for i in final_spike_indices])

        impact_severity, severity_score, dispatch = enhanced_severity_assessment(
            collision_type, max_prob, avg_prob, velocities, accelerations,
            involved_vehicles, crash_duration, sudden_impact_data
        )

        # Display enhanced report
        with col2:
            report_placeholder.empty()

            st.markdown("""
            <div class='report-card'>
                <h3>🚨 Crash Report</h3>
            </div>
            """, unsafe_allow_html=True)

            # Core incident details with improved formatting
            confidence_icons = {
                "Very High": "🟢", "High": "🟢", "Medium": "🟡",
                "Low": "🟠", "Very Low": "🔴"
            }
            confidence_icon = confidence_icons.get(type_confidence, "⚪")

            st.markdown(f"**Collision Type:** {collision_type} {confidence_icon}")
            st.markdown(f"**Impact Severity:** {impact_severity}")
            st.markdown(f"**Incident Time:** {timestamp_str}")
            st.markdown(f"**Vehicles Involved:** {involved_vehicles}")
            st.markdown(f"**Analysis Confidence:** {type_confidence} ({confidence_score:.2f})")
            st.markdown(f"**Detection Method:** {detection_method}")
            st.markdown(f"**Max Probability:** {max_prob:.1%}")
            st.markdown(f"**Alert Frames:** {len(final_spike_indices)}")

            # Enhanced severity scoring with context
            if severity_score >= 7:
                st.markdown(f"**Severity Score:** :red[{severity_score}/10]")
            elif severity_score >= 4:
                st.markdown(f"**Severity Score:** :orange[{severity_score}/10]")
            else:
                st.markdown(f"**Severity Score:** :green[{severity_score}/10]")

            # Impact assessment with better descriptions
            severity_context = {
                "Critical": "🔴 Life-threatening injuries likely - immediate response required",
                "Severe": "🟠 Serious injuries probable - priority response needed",
                "Moderate": "🟡 Moderate injuries possible - standard response",
                "Minor": "🟢 Minor injuries likely - low priority response",
                "Very Minor": "🟢 Minimal injuries expected - monitor situation",
                "Negligible": "⚪ No significant impact detected"
            }

            if impact_severity in severity_context:
                st.markdown(f"**Impact Assessment:** {severity_context[impact_severity]}")

            # Emergency response with detailed reasoning
            if dispatch:
                st.markdown("**Emergency Response:** :green[✅ DISPATCHED]")

                # Provide specific reasoning for dispatch
                if involved_vehicles >= 3:
                    st.markdown("*🚑 Multiple vehicles involved - trauma units recommended*")
                elif severity_score >= 7:
                    st.markdown("*🚨 High severity impact - immediate response required*")
                elif "High-Speed" in collision_type or "Head-On" in collision_type:
                    st.markdown("*⚡ High-energy collision - priority response*")
                elif sudden_impact_data[0]:  # sudden_impact is True
                    st.markdown("*💥 Sudden impact detected - medical assessment needed*")
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

            # Detailed analysis expandable section
            with st.expander("🔍 Detailed Technical Analysis"):
                st.write(f"**Classification Confidence:** {confidence_score:.1%}")

                if velocities:
                    st.write(f"**Movement Analysis:**")
                    avg_velocity = np.mean(velocities[-10:]) if len(velocities) >= 10 else np.mean(velocities)
                    max_velocity = max(velocities) if velocities else 0
                    st.write(f"• Average velocity: {avg_velocity:.1f} pixels/second")
                    st.write(f"• Peak velocity: {max_velocity:.1f} pixels/second")

                    if sudden_impact_data[0]:  # sudden_impact
                        st.write(f"• Sudden impact detected: YES")
                        st.write(f"• Velocity drop: {sudden_impact_data[1]:.1f} pixels/second")
                        st.write(f"• Max deceleration: {sudden_impact_data[2]:.1f} pixels/second²")
                    else:
                        st.write(f"• Sudden impact detected: NO")

                st.write(f"**Vehicle Analysis:**")
                st.write(f"• Vehicle involvement confidence: {vehicle_confidence:.2f}")
                pre_crash_avg = np.mean(
                    car_count_history[max(0, final_spike_indices[0] - 30):final_spike_indices[0]]) if \
                final_spike_indices[0] > 0 else 0
                crash_avg = np.mean(car_count_history[final_spike_indices[0]:final_spike_indices[-1] + 1])
                st.write(f"• Pre-crash vehicle average: {pre_crash_avg:.1f}")
                st.write(f"• During crash average: {crash_avg:.1f}")

                st.write(f"**Probability Analysis:**")
                st.write(f"• Peak probability: {max_prob:.3f}")
                st.write(f"• Average probability: {avg_prob:.3f}")
                # Calculate threshold for display
                display_threshold = mean_prob * THRESH_MULTIPLIER if p_scores else 0
                st.write(f"• Detection threshold: {display_threshold:.3f}")
                st.write(f"• Crash duration: {crash_duration} frames ({crash_duration / src_fps:.1f} seconds)")
    else:
        # Enhanced no-crash-detected reporting
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

            # Provide detailed analysis summary
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

            # Provide recommendations
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
