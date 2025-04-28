#!/usr/bin/env python3
"""
extract_frames.py
────────────────────────
Sample each MP4 in data/ccd/videos/Normal/ at 2 fps and save the frames as

    N_<video-id>_<frame-id>.jpg

into data/ccd/frames_normal/  (or a custom --dst path).

Run:
    python src/extract_frames.py            # hard-coded default
    python src/extract_frames.py --dst data/tmp  # custom folder
"""

import argparse, pathlib, subprocess, sys
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ─── CLI ───────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--src", type=pathlib.Path,
               default=ROOT / "data/ccd/videos/Normal",
               help="Folder containing the Normal MP4 clips")
p.add_argument("--dst", type=pathlib.Path,
               default=ROOT / "data/ccd/frames_normal",
               help="Where extracted N_*.jpg frames will be written")
p.add_argument("--fps", type=float, default=2,
               help="Sampling rate (frames per second)")
args = p.parse_args()

SRC, DST, FPS = args.src, args.dst, args.fps

# ─── sanity checks ────────────────────────────────────────────────
if not SRC.exists():
    sys.exit(f"[ERROR] Folder not found: {SRC}")
videos = sorted(SRC.glob("*.mp4"))
if not videos:
    sys.exit(f"[ERROR] No .mp4 files inside {SRC}")

print(f"✓ {len(videos):,} videos detected in {SRC}")
print(f"→ extracting to {DST.resolve()}  ({FPS} fps)\n")

DST.mkdir(parents=True, exist_ok=True)
total = 0

# ─── extraction loop ──────────────────────────────────────────────
for idx, mp4 in enumerate(tqdm(videos, unit='vid'), start=1):
    vid_id  = f"{idx:06d}"
    pattern = DST / f"N_{vid_id}_%02d.jpg"
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(mp4),
           "-r", str(FPS), str(pattern)]

    if idx == 1:
        print("ffmpeg command example:\n", " ".join(cmd), "\n")

    if subprocess.call(cmd) != 0:
        sys.exit(f"[ERROR] ffmpeg failed on {mp4}")

    produced = len(list(DST.glob(f"N_{vid_id}_*.jpg")))
    total += produced

print(f"\n✓ All normal frames extracted: {total:,} jpgs")
