#!/usr/bin/env python3
"""
split_all_frames.py – clip-aware train/val/test splitter (fast v2)
Same semantics, much faster file I/O.
"""
from __future__ import annotations
import argparse, random, shutil, os, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ───────────── defaults ─────────────
ROOT         = Path(__file__).resolve().parents[2]
DEF_SRC      = ROOT / "data" / "all_frames"
DEF_OUT      = ROOT / "data" / "images"

# ───────────── CLI ─────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--src",  type=Path, default=DEF_SRC)
ap.add_argument("--out",  type=Path, default=DEF_OUT)
ap.add_argument("--train", type=float, default=0.70)
ap.add_argument("--val",   type=float, default=0.15)
ap.add_argument("--test",  type=float, default=0.15)
ap.add_argument("--link",  action="store_true", help="Hard-link instead of copy")
ap.add_argument("--seed",  type=int, default=42)
ap.add_argument("--balance", choices=["natural","oversamp"], default="natural")
ap.add_argument("--workers", type=int, default=os.cpu_count(),
                help="I/O threads (default = all logical cores)")
ap.add_argument("--debug",  action="store_true")
args = ap.parse_args()
assert abs(args.train + args.val + args.test - 1) < 1e-6, "Ratios must sum to 1"

copyf   = os.link if args.link else shutil.copy2
rng     = random.Random(args.seed)
t_start = time.perf_counter()

print(f"\033[96m▶  Splitting {args.src}  →  {args.out}  "
      f"| ratios {args.train}/{args.val}/{args.test}\033[0m")

# ───────────── 1. group by clip-stem ─────────────
groups: dict[tuple[str,str], list[Path]] = defaultdict(list)    # (stem,cls) -> imgs
for cls in ("positive", "negative"):
    folder = args.src / cls
    if not folder.exists():
        raise SystemExit(f"[ERR] Missing {folder}")
    for img in folder.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = "_".join(img.name.split("_")[:2])
        groups[(stem, cls)].append(img)
    if args.debug:
        print(f"   • {cls:<8}: {len(groups)} grouped entries so far…")

tot_clips = len(groups)
print(f"   Total clips grouped: {tot_clips:,}")
t_group = time.perf_counter()

# ───────────── 2. assign clips to splits ─────────────
keys = list(groups.keys()); rng.shuffle(keys)
splits = {"train": [], "val": [], "test": []}
quota  = {"train": args.train, "val": args.val, "test": args.test}

for k in keys:
    fill = {s: len(splits[s])/tot_clips for s in splits}
    target = min(fill, key=lambda s: fill[s]-quota[s] if fill[s] < quota[s] else 1)
    splits[target].append(k)

if args.balance == "oversamp":
    pos = [k for k in splits["train"] if k[1] == "positive"]
    neg = [k for k in splits["train"] if k[1] == "negative"]
    splits["train"].extend(rng.choices(pos, k=len(neg) - len(pos)))

t_split = time.perf_counter()

# ───────────── 3. build global job list ─────────────
jobs: list[tuple[Path, Path]] = []                      # (src, dest)
for split, keys in splits.items():
    for (stem, cls) in keys:
        label   = "crash" if cls == "positive" else "normal"
        destdir = args.out / split / label
        destdir.mkdir(parents=True, exist_ok=True)
        for img in groups[(stem, cls)]:
            tgt = destdir / img.name
            if not tgt.exists():                        # skip if already linked
                jobs.append((img, tgt))

# ───────────── 4. multi-threaded copy/link ─────────────
def do_copy(pair):
    src, dst = pair
    try:
        copyf(src, dst)
    except FileExistsError:
        pass                                            # race – ignore

print(f"   I/O jobs: {len(jobs):,}  |  workers: {args.workers}")
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(tqdm(pool.map(do_copy, jobs, chunksize=256),
              total=len(jobs), unit="file", dynamic_ncols=True))

t_copy = time.perf_counter()

# ───────────── 5. report ─────────────
def cnt(s, lbl): return len(list((args.out / s / lbl).iterdir()))
stats = {s: {'crash': cnt(s, "crash"), 'normal': cnt(s, "normal")}
         for s in splits}

print("\n\033[95m📊  Split summary\033[0m")
print("set     |  crash   |  normal  |  total")
print("────────┼──────────┼──────────┼────────")
for s in ("train", "val", "test"):
    c, n = stats[s]['crash'], stats[s]['normal']
    print(f"{s:<7} | {c:>8,} | {n:>8,} | {(c+n):>8,}")
print("────────┴──────────┴──────────┴────────")
tot_c = sum(d['crash']  for d in stats.values())
tot_n = sum(d['normal'] for d in stats.values())
print(f"TOTAL     {tot_c:>8,}   {tot_n:>8,}")

print(f"\nMode   : {'hard-link' if args.link else 'copy'}"
      f"   |  balance : {args.balance}"
      f"   |  workers : {args.workers}"
      f"   |  seed : {args.seed}")
print(f"Timing : group {t_group-t_start:,.1f}s  | "
      f"split {t_split-t_group:,.1f}s  | "
      f"I/O {t_copy-t_split:,.1f}s  | "
      f"TOTAL {t_copy-t_start:,.1f}s")

if args.debug:
    print("\n[DEBUG] example keys:", keys[:5])
