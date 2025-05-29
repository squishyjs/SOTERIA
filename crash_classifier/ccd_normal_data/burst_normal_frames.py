#!/usr/bin/env python3
"""
burst_normal_frames.py  –  super-fast normal-frame burster
───────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import argparse, os, pathlib, shutil, subprocess, sys, uuid, signal
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from tqdm import tqdm

# ───────────────────── defaults ─────────────────────
ROOT        = pathlib.Path(__file__).resolve().parents[2]
DEF_SRC     = ROOT / "data" / "Normal-002"
DEF_DST     = ROOT / "data" / "ccd_frames" / "normal_frames"
DEF_FPS     = 2
DEF_JOBS    = os.cpu_count() or 8
PRESET_Q    = {"high": 2, "med": 4, "low": 7}
DEF_Q       = PRESET_Q["high"]

# locate ffmpeg once (raises if missing)
FFMPEG = shutil.which("ffmpeg") or sys.exit("[ERR] ffmpeg not found in PATH")

CUDA_FLAGS = [
    "-hwaccel", "cuda",
    "-hwaccel_output_format", "cuda",
    "-c:v", "h264_cuvid"
]

# ───────────────────── CLI ───────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Burst Normal driving MP4s into N_*.jpg (parallel + GPU).")
ap.add_argument("--src",       type=pathlib.Path, default=DEF_SRC,
                help="Folder (or tree with --recursive) of MP4 clips")
ap.add_argument("--dst",       type=pathlib.Path, default=DEF_DST,
                help="Where N_*.jpg frames are written")
ap.add_argument("--fps",       type=float, default=DEF_FPS,
                help="Sampling rate")
ap.add_argument("--quality",   default=str(DEF_Q),
                help="JPEG qscale 2-31 or preset high/med/low")
ap.add_argument("--workers",   type=int,   default=DEF_JOBS,
                help="Max parallel ffmpeg processes")
ap.add_argument("--cpu",       action="store_true",
                help="Skip NVDEC and decode on CPU only")
ap.add_argument("--recursive", action="store_true",
                help="Recurse into sub-folders for *.mp4")
args = ap.parse_args()

# quality → int
if args.quality.isdigit():
    q_val = int(args.quality)
    if not 2 <= q_val <= 31:
        sys.exit("[ERR] quality must be 2-31")
else:
    q_val = PRESET_Q.get(args.quality.lower())
    if q_val is None:
        sys.exit("[ERR] unknown preset; use high / med / low")

# collect MP4s
pattern = "**/*.mp4" if args.recursive else "*.mp4"
videos  = sorted(args.src.glob(pattern))
if not videos:
    sys.exit(f"[ERR] no .mp4 found in {args.src}")

args.workers = min(args.workers, len(videos))   # don’t oversubscribe
args.dst.mkdir(parents=True, exist_ok=True)

print(
    f"\033[96m🗂  {len(videos):,} videos\033[0m  |  "
    f"fps={args.fps}  |  q={q_val}  |  "
    f"workers={args.workers}  |  "
    f"{'CPU-only' if args.cpu else 'GPU→CPU fallback'}"
)

# ───────────── Ctrl-C graceful stop ─────────────
stop_flag = False
def _sigint(_s,_f):
    global stop_flag
    stop_flag = True
    print("\n\033[93m[CTRL-C] Finishing current clips then stopping …\033[0m")
signal.signal(signal.SIGINT, _sigint)

# ───────────── helpers ──────────────────────────
def run(cmd: list[str]) -> bool:
    return subprocess.call(cmd, creationflags=subprocess.CREATE_NO_WINDOW) == 0 if os.name=="nt" else subprocess.call(cmd)==0

def process(task) -> int:            # returns #frames saved (0 = fail)
    idx, mp4 = task
    vid_tag  = f"{idx:06d}"
    tmp      = args.dst / f"tmp_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    patt     = tmp / "%04d.jpg"

    core = [
        "-y", "-loglevel", "error",
        "-i", str(mp4),
        "-vf", f"fps={args.fps}",
        "-qscale:v", str(q_val),
        str(patt)
    ]

    cmd_gpu = [FFMPEG, *CUDA_FLAGS, *core] if not args.cpu else None
    cmd_cpu = [FFMPEG, *core]

    if idx == 1:
        print("ffmpeg cmd (GPU attempt):\n ", " ".join(cmd_gpu or cmd_cpu), "\n")

    ok = run(cmd_gpu) if cmd_gpu else False
    if not ok:
        ok = run(cmd_cpu)
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    for j, jpg in enumerate(sorted(tmp.glob("*.jpg")), 1):
        jpg.rename(args.dst / f"N_{vid_tag}_{j:04d}.jpg")
    shutil.rmtree(tmp, ignore_errors=True)
    return j

# ───────────── thread-pool loop ─────────────────
fail = 0
frames_total = 0
t0 = perf_counter()

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for count in tqdm(pool.map(process, enumerate(videos,1)),
                      total=len(videos), unit="vid", dynamic_ncols=True):
        if count == 0:
            fail += 1
        frames_total += count
        if stop_flag:
            break

elapsed = perf_counter() - t0
speed   = frames_total / elapsed if elapsed else 0

# ───────────── summary ──────────────────────────
print("\n\033[95m📊  Summary\033[0m")
print("──────────────────────────────────────────────")
print(f" videos processed : {len(videos)-fail:,}/{len(videos):,}")
print(f" frames extracted : {frames_total:,}")
print(f" failures         : {fail}")
print(f" output folder    : {args.dst}")
print(f" elapsed          : {elapsed:,.1f} s   ({speed:,.1f} img/s)")
print(f" mode             : {'CPU-only' if args.cpu else 'GPU→CPU fallback'}\n")
