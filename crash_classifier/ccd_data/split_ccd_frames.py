#!/usr/bin/env python3
"""
split_ccd_frames.py  –  parallel CrashBest splitter
──────────────────────────────────────────────────────────────
Turns the CSV flags into two flat folders:

    data/ccd_frames/pos_neg_split/
        ├─ positive/   (impact, flag == 1)
        └─ negative/   (pre-impact, flag == 0)

Extras
──────
* live ETA / speed in tqdm bar
* per-class counters while running
* pretty summary table
* --dry-run   → analyse only, touch nothing
* --verbose   → log each file op

Example
───────
$ python split_ccd_frames.py --link --workers 24
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import pandas as pd
from tqdm import tqdm

# ───────────────────────── paths ──────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT   = ROOT / "data" / "archive"           # ← your new path
CRASHBEST   = DATA_ROOT / "CrashBest"             # C_*.jpg live here
CSV_PATH    = DATA_ROOT / "Crash_Table.csv"
OUT_ROOT    = ROOT / "data" / "ccd_frames" / "pos_neg_split"

# ───────────────────────── CLI ───────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--link",    action="store_true", help="hard-link instead of copy (same drive only)")
ap.add_argument("--workers", type=int, default=os.cpu_count(), help="max parallel file workers")
ap.add_argument("--dry-run", action="store_true", help="scan & count but DO NOT write anything")
ap.add_argument("--verbose", action="store_true", help="log each individual copy/link")
args = ap.parse_args()

copy_fn = (lambda s, d: os.link(s, d)) if args.link else shutil.copy2
if args.dry_run:
    copy_fn = lambda s, d: None  # no-op

# ───────────────────────── sanity checks ─────────────────────
if not CRASHBEST.exists():
    sys.exit(f"[ERROR] frames not found: {CRASHBEST}")
if not CSV_PATH.exists():
    sys.exit(f"[ERROR] CSV not found: {CSV_PATH}")

# ───────────────────────── load CSV flags ────────────────────
print("⏳  Loading Crash_Table.csv …")
df = pd.read_csv(CSV_PATH)
flag = {
    f"{int(row['vidname']):06d}_{i:02d}": row[f"frame_{i}"]
    for _, row in df.iterrows()
    for i in range(1, 51)
}
print(f"✓ flags loaded for {len(flag):,} frame keys\n")

# ───────────────────────── prepare output ────────────────────
for sub in ("positive", "negative"):
    (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# ───────────────────────── collect frames ────────────────────
pat = re.compile(r"C_(\d{6}_\d{2})")
frames = list(CRASHBEST.glob("C_*.jpg"))
total_files = len(frames)
if total_files == 0:
    sys.exit("[ERROR] No C_*.jpg frames found!")

args.workers = min(args.workers, total_files)  # never spawn more threads than files
mode = "hard-link" if args.link else "copy"
print(f"🚀  Splitting {total_files:,} frames → {OUT_ROOT}  "
      f"({args.workers} workers, {mode}{', dry-run' if args.dry_run else ''})\n")

# ───────────────────────── worker ────────────────────────────
def process(src: pathlib.Path) -> str:   # returns "positive" | "negative"
    key = pat.search(src.name).group(1)
    dst_sub = "positive" if flag.get(key, 0) == 1 else "negative"
    if not args.dry_run:
        dest = OUT_ROOT / dst_sub / src.name
        if not dest.exists():
            copy_fn(src, dest)
            if args.verbose:
                print(f"{src.name} → {dst_sub}")
    return dst_sub

# ───────────────────────── thread-pool loop ──────────────────
counts = {"positive": 0, "negative": 0}
t0 = perf_counter()
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(process, f): f for f in frames}
    pbar = tqdm(total=total_files, unit="img", dynamic_ncols=True)
    for fut in as_completed(futures):
        dst = fut.result()
        counts[dst] += 1
        pbar.set_postfix(pos=counts['positive'], neg=counts['negative'], refresh=False)
        pbar.update()

elapsed = perf_counter() - t0
pbar.close()

# ───────────────────────── summary ───────────────────────────
def fmt(n: int) -> str:
    return f"{n:,}"

print("\n📊  Summary")
print("──────────────────────────────")
print(f" positive : {fmt(counts['positive'])}")
print(f" negative : {fmt(counts['negative'])}")
print(f" total    : {fmt(sum(counts.values()))}")
print(f" mode     : {'dry-run' if args.dry_run else mode}")
print(f" time     : {elapsed:,.1f} s  ({total_files/elapsed:,.1f} img/s)")
