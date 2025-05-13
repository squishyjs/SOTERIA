"""
extract_dashcam_frames.py  –  **paper-exact tail (last 10 frames)**
────────────────────────────────────────────────────────────────
This version follows the ACCV-2016 dataset protocol *literally*:
*Always take the last **10 frames** of every positive clip.*

Changes vs. previous build
──────────────────────────
1. **Default `--fps` is now 20** (the paper’s original frame-rate).
2. **`--tail 10` hard-codes the window**; you may override, but the
   default is 10 no matter the FPS.
3. Clips shorter than the tail length fall back to “all remaining
   frames” → positive.

Example (fps = 20)
```
frames/positive/
    accident_000123_091.jpg … accident_000123_100.jpg
frames/negative/
    accident_000123_001.jpg … accident_000123_090.jpg
```
"""
from __future__ import annotations
import argparse, os, pathlib, shutil, subprocess, sys, uuid
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ───────────── defaults ─────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "Dashcam_dataset" / "videos"
DEFAULT_DST = ROOT / "data" / "dashcam_frames"
DEFAULT_FPS = 20                       # paper uses 20 fps
DEFAULT_TAIL = 10                      # always last 10 frames
DEFAULT_JOBS = max(os.cpu_count() // 2, 4)

# ─────────── ffmpeg helper ───────────

def burst(mp4: pathlib.Path, out_dir: pathlib.Path, fps: int, gpu: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    patt = out_dir / "%03d.jpg"
    flags = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"] if gpu else []
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *flags,
           "-i", str(mp4), "-vf", f"fps={fps}", str(patt)]
    if subprocess.call(cmd):
        raise RuntimeError(mp4)
    return sorted(out_dir.glob("*.jpg"))

# ───────────── CLI ─────────────
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Dash-cam clip → JPG extractor (exact last-10 frames for positives).")
cli.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC)
cli.add_argument("--dst", type=pathlib.Path, default=DEFAULT_DST)
cli.add_argument("--fps", type=int, default=DEFAULT_FPS)
cli.add_argument("--tail", type=int, default=DEFAULT_TAIL,
                 help="How many tail frames form the positive label")
cli.add_argument("--threads", type=int, default=DEFAULT_JOBS)
cli.add_argument("--gpu", action="store_true")
args = cli.parse_args()

POS_DST = args.dst / "frames" / "positive"
NEG_DST = args.dst / "frames" / "negative"
for pth in (POS_DST, NEG_DST):
    pth.mkdir(parents=True, exist_ok=True)

splits = [
    ("training", "positive", True), ("training", "negative", False),
    ("testing",  "positive", True), ("testing",  "negative", False)
]
work: list[tuple[pathlib.Path,bool]] = []
for top, sub, isp in splits:
    fold = args.src / top / sub
    if fold.exists():
        work += [(c, isp) for c in fold.glob("*.*")]
if not work:
    sys.exit("[ERR] no clips found – check --src")

print(f"→ Extracting {len(work):,} clips @ {args.fps} fps, tail={args.tail}, workers={args.threads} …")

# ───────────── worker ─────────────

def process(task):
    clip, is_pos = task
    tmp = (POS_DST if is_pos else NEG_DST)/".tmp"/uuid.uuid4().hex[:8]
    try:
        frames = burst(clip, tmp, args.fps, args.gpu)
    except RuntimeError:
        shutil.rmtree(tmp, ignore_errors=True); return 0,0

    stem=clip.stem; p=n=0
    def move(lst, dst):
        nonlocal p,n
        for f in lst:
            dest = dst / f"{stem}_{f.name}"
            if not dest.exists():
                shutil.move(f, dest)
                if dst is POS_DST: p+=1
                else: n+=1

    if is_pos:
        k = args.tail if len(frames) >= args.tail else len(frames)
        move(frames[-k:], POS_DST)
        move(frames[:-k], NEG_DST)
    else:
        move(frames, NEG_DST)

    shutil.rmtree(tmp, ignore_errors=True)
    return p,n

pos=neg=0
with ThreadPoolExecutor(max_workers=args.threads) as pool:
    for p_cnt,n_cnt in tqdm(pool.map(process, work), total=len(work), unit="clip"):
        pos+=p_cnt; neg+=n_cnt

print(f"\n✓ Done. positives: {pos:,} | negatives: {neg:,} | GPU: {'on' if args.gpu else 'off'}")
