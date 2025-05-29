#!/usr/bin/env python3
"""
split_all_frames.py — clip-aware train/val/test splitter

Input  : data/all_frames/{positive,negative}/*.jpg
Output : data/images/{train,val,test}/{crash,normal}/*.jpg (or hard-links)

Guarantees: every dash-cam clip (stem C_000123) is assigned to exactly ONE split,
so the model never sees frames from the same video in both train and val/test.
"""

from __future__ import annotations
import argparse, os, random, shutil, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# ─────────── defaults ────────────
ROOT   = Path(__file__).resolve().parents[2]
DEF_SRC = ROOT / "data" / "all_frames"
DEF_OUT = ROOT / "data" / "images"

# ─────────── CLI ────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--src",        type=Path, default=DEF_SRC,
                help="Source folder with positive/negative sub-folders")
ap.add_argument("--out",        type=Path, default=DEF_OUT,
                help="Destination root for train/val/test")
ap.add_argument("--train",      type=float, default=0.70)
ap.add_argument("--val",        type=float, default=0.15)
ap.add_argument("--test",       type=float, default=0.15)
ap.add_argument("--balance",    choices=["natural", "oversamp"], default="natural",
                help="Oversample crash stems in TRAIN split to balance classes")
ap.add_argument("--link",       action="store_true",
                help="Use hard-links instead of copies (fast, same drive only)")
ap.add_argument("--seed",       type=int, default=42)
ap.add_argument("--workers",    type=int, default=os.cpu_count(),
                help="File-copy threads")
ap.add_argument("--debug",      action="store_true")
args = ap.parse_args()
assert abs(args.train + args.val + args.test - 1) < 1e-6, "Ratios must sum to 1"

copyf = os.link if args.link else shutil.copy2
rng   = random.Random(args.seed)
t0    = time.perf_counter()

print(f"\033[96m▶ Splitting {args.src} → {args.out} "
      f"| ratios {args.train}/{args.val}/{args.test}\033[0m")

# ─────────── 1. group frames by clip-stem ────────────
groups: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: {"positive": [], "negative": []})

for cls in ("positive", "negative"):
    folder = args.src / cls
    if not folder.is_dir():
        raise SystemExit(f"[ERR] Missing {folder}")
    for img in folder.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = "_".join(img.name.split("_")[:2])      # e.g. C_000123_07.jpg → C_000123
        groups[stem][cls].append(img)
    if args.debug:
        print(f"  • {cls:<8}: {sum(len(v[cls]) for v in groups.values()):,} files seen")

tot_stems = len(groups)
print(f"  Total unique clips: {tot_stems:,}")
t_group = time.perf_counter()

# ─────────── 2. assign stems to splits ────────────
stems = list(groups.keys()); rng.shuffle(stems)
splits = {k: [] for k in ("train", "val", "test")}
quota  = {"train": args.train, "val": args.val, "test": args.test}

for stem in stems:
    fill = {s: len(splits[s]) / tot_stems for s in splits}
    target = min(fill, key=lambda s: fill[s] - quota[s] if fill[s] < quota[s] else 1)
    splits[target].append(stem)

# optional oversampling (TRAIN only)
if args.balance == "oversamp":
    crash_stems = [s for s in splits["train"] if groups[s]["positive"]]
    normal_stems = [s for s in splits["train"] if groups[s]["negative"]]
    if len(crash_stems) < len(normal_stems):
        needed = len(normal_stems) - len(crash_stems)
        splits["train"].extend(rng.choices(crash_stems, k=needed))

t_split = time.perf_counter()

# ─────────── 3. create copy/link job list ────────────
jobs: List[tuple[Path, Path]] = []
for split, stems in splits.items():
    for stem in stems:
        for cls, imgs in groups[stem].items():
            label = "crash" if cls == "positive" else "normal"
            dest_dir = args.out / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in imgs:
                dst = dest_dir / src.name
                if dst.exists():
                    # filename collision → append short hash
                    dst = dest_dir / f"{src.stem}_{src.stat().st_mtime_ns & 0xffff:x}{src.suffix}"
                jobs.append((src, dst))

# ─────────── 4. parallel copy/link ────────────
def do_copy(pair):
    src, dst = pair
    try:
        copyf(src, dst)
    except FileExistsError:
        pass  # already copied in oversampling pass

print(f"  I/O jobs: {len(jobs):,} | workers: {args.workers}")
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(tqdm(pool.map(do_copy, jobs, chunksize=256),
              total=len(jobs), unit="file", dynamic_ncols=True))

t_copy = time.perf_counter()

# ─────────── 5. report ────────────
def count(split, lbl):
    d = args.out / split / lbl
    return len(list(d.iterdir())) if d.exists() else 0

print("\n\033[95m📊 Split summary\033[0m")
print("set     |  crash   |  normal  |  total")
print("────────┼──────────┼──────────┼────────")
for split in ("train", "val", "test"):
    c, n = count(split, "crash"), count(split, "normal")
    print(f"{split:<7} | {c:>8,} | {n:>8,} | {(c+n):>8,}")
print("────────┴──────────┴──────────┴────────")

elapsed = time.perf_counter() - t0
print(f"Mode   : {'hard-link' if args.link else 'copy'}  |  "
      f"balance : {args.balance}  |  seed : {args.seed}")
print(f"Timing : group {t_group-t0:,.1f}s  |  "
      f"split {t_split-t_group:,.1f}s  |  "
      f"I/O {t_copy-t_split:,.1f}s  |  "
      f"TOTAL {elapsed:,.1f}s")

if args.debug:
    print("\n[DEBUG] first 5 stems →", stems[:5])
