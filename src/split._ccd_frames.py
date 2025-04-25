"""
Split CCD Kaggle dump (C_*.jpg & N_*.jpg) into train/val/test folders.

Usage:
    python src/split_ccd_frames.py          # default 70/15/15 split
"""

import argparse, pathlib, random, shutil, re
import pandas as pd

# ------------------ CLI arguments ------------------
p = argparse.ArgumentParser()
p.add_argument("--root", default="data/ccd/frames", help="where the CCD JPGs live")
p.add_argument("--meta", default="data/ccd/Crash_Table.csv")
p.add_argument("--out",  default="data/images")      # output root for model
p.add_argument("--val",  type=float, default=0.15)
p.add_argument("--test", type=float, default=0.15)
p.add_argument("--use_table", action="store_true",
               help="keep only frames flagged as accident in Crash_Table")
args = p.parse_args()

ROOT = pathlib.Path(args.root)
OUT  = pathlib.Path(args.out)
rng  = random.Random(42)

# ------------------ read Crash_Table if needed ------------------
keep = None
if args.use_table:
    df = pd.read_csv(args.meta)
    keep = {f"{vid:06d}_{i:02d}"
            for vid, row in df.iterrows()
            for i in range(1, 51) if row[f"frame_{i}"] == 1}

# ------------------ collect file paths ------------------
crash, normal = [], []
for jpg in ROOT.glob("*.jpg"):
    name = jpg.name
    if   name.startswith("C_"):    crash.append(jpg)
    elif name.startswith("N_"):    normal.append(jpg)

if keep:
    # keep only impact frames inside crash list
    pat = re.compile(r"C_(\d{6}_\d{2})")
    crash = [p for p in crash if pat.search(p.name).group(1) in keep]

print(f"Crash frames:  {len(crash):,}")
print(f"Normal frames: {len(normal):,}")

# ------------------ shuffle & copy ------------------
def split_and_copy(file_list, cls):
    rng.shuffle(file_list)
    n = len(file_list)
    idx = [0,
           int(n*(1-args.val-args.test)),
           int(n*(1-args.test)), n]
    for split, (a,b) in zip(["train","val","test"], zip(idx, idx[1:])):
        dest = OUT/split/cls; dest.mkdir(parents=True, exist_ok=True)
        for src in file_list[a:b]:
            shutil.copy2(src, dest/src.name)

split_and_copy(crash,  "crash")
split_and_copy(normal, "normal")
print("✅  Finished: data/images/{train,val,test}/{crash,normal}")
