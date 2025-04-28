#!/usr/bin/env python3
"""
split_ccd_frames.py
───────────────────
Create the classic train/val/test directory layout
    data/images/{train,val,test}/{crash,normal}/
from **any two folders**:
    • one that contains only positive (crash) JPGs
    • one that contains only negative (normal) JPGs

You may still pass a single folder in legacy mode; the script will fall back
to filename prefixes (`C_` = crash, `N_` = normal).

Quick start (no arguments!)
───────────────────────────
If you keep the recommended folder names produced earlier, you can simply run:

    python src/split_ccd_frames.py --link

The script auto-uses the defaults below.

Explicit example
────────────────
    python src/split_ccd_frames.py \
           --pos data/ccd/flag_split/flag1 \
           --neg data/ccd/negatives_all     \
           --link
"""

import argparse, os, pathlib, random, shutil, sys
from typing import List
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ─────────── defaults so you can just “Run” in the IDE ────────────
DEFAULT_POS = ROOT / "data/ccd/flag_split/flag1"
DEFAULT_NEG = ROOT / "data/ccd/negatives_all"

# ─────────── CLI ────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description="Split positives & negatives into train/val/test folders.")

p.add_argument("--pos", type=pathlib.Path, help="Folder with POSITIVE (crash) JPGs")
p.add_argument("--neg", type=pathlib.Path, help="Folder with NEGATIVE (normal) JPGs")

# legacy single-root mode (prefix C_/N_)
p.add_argument("--root", type=pathlib.Path, default=None,
               help="Single folder containing both C_*.jpg & N_*.jpg (legacy mode)")

p.add_argument("--out",  type=pathlib.Path, default=ROOT / "data/images",
               help="Output root (train/val/test will be created here)")
p.add_argument("--val",  type=float, default=0.15, help="Validation split ratio")
p.add_argument("--test", type=float, default=0.15, help="Test split ratio")
p.add_argument("--link", action="store_true", help="Use hard-links instead of copies")
args = p.parse_args()

# ─────────── argument sanity / auto-defaults ───────────
if args.pos is None and args.neg is None and args.root is None:
    # user clicked Run with no flags → fall back to defaults
    args.pos, args.neg = DEFAULT_POS, DEFAULT_NEG
    print(f"[INFO] Using default folders:\n   pos = {args.pos}\n   neg = {args.neg}")

# allow legacy root OR both pos+neg, but not a single missing flag
if (args.pos is None) ^ (args.neg is None):
    sys.exit("[ERROR] Provide both --pos and --neg, or just --root.")

# choose copy vs link
rng = random.Random(42)
copy_fn = (lambda s, d: os.link(s, d)) if args.link else shutil.copy2

# ─────────── gather image paths ────────────
crash: List[pathlib.Path] = []
normal: List[pathlib.Path] = []

if args.pos and args.neg:                      # explicit mode
    if not args.pos.exists() or not args.neg.exists():
        sys.exit("[ERROR] --pos or --neg folder does not exist.")
    crash  = list(args.pos.rglob("*.jpg"))
    normal = list(args.neg.rglob("*.jpg"))
else:                                          # legacy prefix mode
    if not args.root or not args.root.exists():
        sys.exit(f"[ERROR] --root not found: {args.root}")
    for img in args.root.rglob("*.jpg"):
        if img.name.startswith("C_"):
            crash.append(img)
        elif img.name.startswith("N_"):
            normal.append(img)

print(f"Crash frames : {len(crash):,}")
print(f"Normal frames: {len(normal):,}")
if not crash or not normal:
    sys.exit("[ERROR] One of the classes is empty – check your paths.")

# ─────────── split helper ────────────

def split_and_write(files: List[pathlib.Path], cls: str):
    rng.shuffle(files)
    n = len(files)
    borders = [0, int(n*(1-args.val-args.test)), int(n*(1-args.test)), n]
    for split,(a,b) in zip(("train","val","test"), zip(borders,borders[1:])):
        dest = args.out / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for src in tqdm(files[a:b], desc=f"{cls[:5]}→{split[:3]}", unit="img", leave=False):
            copy_fn(src, dest/src.name)

# clean output dir if rerunning
if args.out.exists(): shutil.rmtree(args.out)

split_and_write(crash,  "crash")
split_and_write(normal, "normal")

print("\n✅  Split complete. Folders created:")
for pth in (args.out / "train").glob("*"):  # crash, normal
    print("   ", pth.relative_to(args.out))
print("   (hard-linked)" if args.link else "   (copied)")