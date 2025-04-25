#!/usr/bin/env python3
"""
separate_flag01.py  –  split C_*.jpg frames into two folders based on the
per‑frame flag in Crash_Table.csv.

    flag1/  ← frames where the CSV value == 1 (impact)
    flag0/  ← frames where the CSV value == 0 (pre‑impact)

Run once:
    python src/separate_flag01.py            # copies
    python src/separate_flag01.py --link     # hard‑links (instant, same drive)
"""

import argparse, os, pathlib, re, shutil, sys
import pandas as pd
from tqdm import tqdm

ROOT   = pathlib.Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data/ccd/frames"          # C_*.jpg live here
TABLE  = ROOT / "data/ccd/Crash_Table.csv"
OUT    = ROOT / "data/ccd/flag_split"      # new output root

# ─────────── CLI ────────────
cli = argparse.ArgumentParser()
cli.add_argument("--link", action="store_true",
                 help="use hard‑links instead of copies (same drive only)")
args = cli.parse_args()
copy = (lambda s, d: os.link(s, d)) if args.link else shutil.copy2

# ─────────── build {videoId_frameId: flag} mapping ────────────
print("Loading Crash_Table.csv …")
df = pd.read_csv(TABLE)
flag = {
    f"{int(row['vidname']):06d}_{i:02d}": row[f"frame_{i}"]
    for _, row in df.iterrows()
    for i in range(1, 51)
}
print(f"Table entries loaded: {len(flag):,}")

# ─────────── prepare output dirs ────────────
for sub in ("flag0", "flag1"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ─────────── separate frames ────────────────
pat     = re.compile(r"C_(\d{6}_\d{2})")   # extract 000123_45
counts  = {"flag0": 0, "flag1": 0}

print("Separating C_ frames …")
for jpg in tqdm(list(FRAMES.glob("C_*.jpg")), unit="img"):
    key   = pat.search(jpg.name).group(1)          # videoID_frameID
    dst   = "flag1" if flag.get(key, 0) == 1 else "flag0"
    copy(jpg, OUT / dst / jpg.name)
    counts[dst] += 1

print("\n✅  Done:")
print(f"   flag1 (impact) : {counts['flag1']:,}")
print(f"   flag0 (pre)    : {counts['flag0']:,}")
print("   (hard‑linked)" if args.link else "   (copied)")
