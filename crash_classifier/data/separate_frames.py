#!/usr/bin/env python3
"""
separate_frames.py
──────────────────────
Read *Crash_Table.csv* and fan-out all CCD frames into three
phys-class folders under **data/ccd/frames_separated/**

    impact/   – C_ frames whose CSV flag == 1  ➜  "crash"
    pre/      – C_ frames whose flag == 0      ➜  optional "pre-crash"
    normal/   – every N_ frame                ➜  "normal"

Run once from the repo root:
    python src/separate_frames.py --link      # instant hard-links
    python src/separate_frames.py            # real copies

Afterwards you can point any ImageFolder or splitter at
`data/ccd/frames_separated/` without worrying about flags.
"""

import argparse, os, pathlib, re, shutil, sys
from typing import Set

import pandas as pd
from tqdm import tqdm

# ─────────── paths ───────────
ROOT   = pathlib.Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data/ccd/frames"          # original C_*.jpg & N_*.jpg
TABLE  = ROOT / "data/ccd/Crash_Table.csv"
OUT    = ROOT / "data/ccd/frames_separated"

# ─────────── CLI ────────────
cli = argparse.ArgumentParser(description="Separate CCD frames into impact / pre / normal folders")
cli.add_argument("--link", action="store_true", help="Use hard-links instead of copies (same drive only)")
args = cli.parse_args()
copy_fn = (lambda s, d: os.link(s, d)) if args.link else shutil.copy2

# ─────────── sanity checks ───────────
if not FRAMES.exists():
    sys.exit(f"[ERROR] source folder missing: {FRAMES}")
if not TABLE.exists():
    sys.exit(f"[ERROR] Crash_Table.csv not found: {TABLE}")

# ─────────── load impact keys ─────────
print("Loading Crash_Table.csv …")
df = pd.read_csv(TABLE)
impact_keys: Set[str] = {
    f"{vid:06d}_{i:02d}"
    for vid, row in df.iterrows()
    for i in range(1, 51)
    if row[f"frame_{i}"] == 1
}
print(f"Impact frames flagged: {len(impact_keys):,}")

# ─────────── prepare output dirs ──────
for sub in ("impact", "pre", "normal"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ─────────── walk frames ──────────────
print("Separating frames …")
counts = {"impact": 0, "pre": 0, "normal": 0}
pattern = re.compile(r"C_(\d{6}_\d{2})")

for img in tqdm(list(FRAMES.rglob("*.jpg")), unit="img"):
    name = img.name
    if name.startswith("N_"):
        dst_sub = "normal"
    else:  # C_ frame
        key = pattern.search(name).group(1)  # 000123_45
        dst_sub = "impact" if key in impact_keys else "pre"
    copy_fn(img, OUT / dst_sub / name)
    counts[dst_sub] += 1

print("\n✅  Finished separating:")
for sub in ("impact", "pre", "normal"):
    print(f"   {sub:<7}: {counts[sub]:,} files → {OUT/sub}")
print("   (hard-linked)" if args.link else "   (copied)")