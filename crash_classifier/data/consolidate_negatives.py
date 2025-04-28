#!/usr/bin/env python3
"""
consolidate_negatives.py
──────────────────
Create **one consolidated folder** that holds every negative-class image:
    • all N_*.jpg extracted from the Normal videos
    • all C_*.jpg whose Crash_Table flag == 0   (pre-impact)

Nothing in the originals is modified; the script either **hard-links** (fast,
no extra disk) or copies each file into a new directory:

    data/ccd/negatives_all/
        ├─ N_000001_01.jpg
        ├─ …
        └─ C_001234_27.jpg  (flag 0)

Run after you've executed:
    1. extract_frames.py   (creates frames_normal/)
    2. separate_flag01.py         (creates flag_split/flag0/ flag1/)

Usage
─────
    python src/consolidate_negatives.py            # copies (slower, more space)
    python src/consolidate_negatives.py --link     # hard-links (instant)
"""

import argparse, os, pathlib, shutil, sys
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent

SRC1 = ROOT / "data/ccd/frames_normal"          # N_*.jpg
SRC2 = ROOT / "data/ccd/flag_split/flag0"       # C_*.jpg with flag 0
DEST = ROOT / "data/ccd/negatives_all"          # new merged folder

# ─────────── CLI ────────────
cli = argparse.ArgumentParser()
cli.add_argument("--link", action="store_true", help="use hard-links instead of copies")
args = cli.parse_args()
transfer = (lambda s, d: os.link(s, d)) if args.link else shutil.copy2

# sanity checks
for src in (SRC1, SRC2):
    if not src.exists():
        sys.exit(f"[ERROR] expected folder missing: {src}")

DEST.mkdir(parents=True, exist_ok=True)

# ─────────── merge ───────────
counts = {"frames_normal": 0, "flag0": 0}

for src, key in ((SRC1, "frames_normal"), (SRC2, "flag0")):
    for jpg in tqdm(list(src.glob("*.jpg")), unit="img", desc=key):
        transfer(jpg, DEST / jpg.name)
        counts[key] += 1

print("\n✅  negatives_all built:")
print(f"   from frames_normal : {counts['frames_normal']:,}")
print(f"   from flag0         : {counts['flag0']:,}")
print(f"   total              : {sum(counts.values()):,}")
print("   (hard-linked)" if args.link else "   (copied)")