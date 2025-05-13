#!/usr/bin/env python3
"""
consolidate_frames.py  –  merge every positive / negative JPG (or PNG/JPEG)
into one folder each (hard-links by default, copies with --copy).

Additions
─────────
•  colour banner & per-source counts
•  --debug  → show the exact folders scanned + first 3 filenames seen
•  extension-agnostic (.jpg .JPG .jpeg .png)
"""

from pathlib import Path
import os, shutil, sys, argparse
from tqdm import tqdm

# ─────────── CLI ────────────
ap = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Merge all positive / negative frames into data/all_frames/")
ap.add_argument("--copy",  action="store_true", help="Copy instead of hard-link")
ap.add_argument("--debug", action="store_true", help="Verbose per-folder info")
args = ap.parse_args()
link = shutil.copy2 if args.copy else os.link

# ─────────── source & dest folders ────────────
ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    ROOT / "data/ccd_frames/pos_neg_split/positive",
    ROOT / "data/ccd_frames/pos_neg_split/negative",
    ROOT / "data/ccd_frames/normal_frames",
    ROOT / "data/dashcam_frames/frames/positive",
    ROOT / "data/dashcam_frames/frames/negative",
]

DEST_POS = ROOT / "data/all_frames/positive"
DEST_NEG = ROOT / "data/all_frames/negative"
DEST_POS.mkdir(parents=True, exist_ok=True)
DEST_NEG.mkdir(parents=True, exist_ok=True)

exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png")

# ─────────── merge loop ────────────
pos = neg = 0
for src in SOURCES:
    if not src.exists():
        if args.debug:
            print(f"[skip] {src}  (folder not found)")
        continue

    kind = "positive" if "positive" in src.parts else "negative"
    files = [p for e in exts for p in src.glob(e)]
    if args.debug:
        print(f"[scan] {src}  → found {len(files):,} {kind} files")
        for sample in files[:3]:
            print("       ", sample.name)
        if not files:
            print("       (empty!)")

    for f in tqdm(files, desc=src.name, unit="img", leave=False):
        dst = (DEST_POS if kind == "positive" else DEST_NEG) / f.name
        if not dst.exists():
            link(f, dst)

    if kind == "positive":
        pos += len(files)
    else:
        neg += len(files)

# ─────────── summary ────────────
mode = "copied" if args.copy else "hard-linked"
print(f"\n\033[95m📊  Consolidated  {pos:,} positive  |  {neg:,} negative   ({mode})\033[0m")
print(f"Destination : {DEST_POS.parent}/{{positive,negative}}")
print("Tip: run with  --debug  if counts are still zero to see exactly where the script looks.")
