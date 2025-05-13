#!/usr/bin/env python3
"""
train_model.py — one-click YOLO-v8 image-classifier training
Optimised for a single RTX 4090 on Windows 10/11.

❗️ Note: Ultralytics’ *Python* .train() does **not** accept `pin_memory`,
so that flag is **not** passed (even though it exists in the CLI).
"""

from __future__ import annotations
import os, pathlib, sys, torch
from datetime import datetime, timedelta
from ultralytics import YOLO
from torch.multiprocessing import set_start_method

# ─── global bug-workarounds ────────────────────────────────────
os.environ["TORCH_DISABLE_DYNAMO"] = "1"     # avoid Dynamo-spawn bug
set_start_method("spawn", force=True)        # safe multiprocessing start

# ─── CONFIG – tweak here only ──────────────────────────────────
EPOCHS     = 10
IMG_SIZE   = 320          # loader resize
BATCH      = 120          # good for 4090 @224 px
WORKERS    = 16            # raise later (e.g. 12) when stable
PIN_MEMORY = True         # ignored (see note above)
CACHE_DISK = True         # *.cache files next to JPEGs
MODEL_WTS  = "yolov8l-cls.pt"     # or -s/-m/-l
DEVICE     = 0            # GPU index, −1 = CPU
# ───────────────────────────────────────────────────────────────

ROOT       = pathlib.Path(__file__).resolve().parents[2]
IMAGES     = ROOT / "data" / "images"          # produced by split_ccd_frames.py
RUNS_ROOT  = ROOT / "runs" / "classify"


def banner() -> None:
    print("\n────────── environment ──────────")
    print("PyTorch :", torch.__version__)
    if torch.cuda.is_available() and DEVICE != -1:
        print(f"CUDA    : {torch.version.cuda} | GPU {DEVICE}: "
              f"{torch.cuda.get_device_name(DEVICE)}")
    else:
        print("CUDA    : NOT available — training on CPU")
    print()


def train() -> None:
    if not IMAGES.exists():
        sys.exit("[ERROR] data/images not found — run split_ccd_frames.py first.")

    banner()
    start = datetime.now()

    model = YOLO(MODEL_WTS).to(f"cuda:{DEVICE}" if DEVICE != -1 else "cpu")
    print(f"✅ model on {model.device}\n")

    model.train(
        data=str(IMAGES),          # folder root, not YAML
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE_DISK,
        device=DEVICE,
        amp=True,                  # mixed precision
    )

    run = max(RUNS_ROOT.glob("exp*"), key=lambda p: p.stat().st_mtime, default=None)
    elapsed = timedelta(seconds=int((datetime.now() - start).total_seconds()))
    print("\n✅  Training finished in", elapsed)
    if run:
        print("   runs folder :", run)
        print("   best model  :", run / "weights" / "best.pt")
        print("   tensorboard --logdir", run)

    if torch.cuda.is_available() and DEVICE != -1:
        torch.cuda.synchronize()
        vram_gb = torch.cuda.max_memory_reserved() / 1024**3
        print(f"   peak VRAM   : {vram_gb:.2f} GB")
        print("   tip         : run  “nvidia-smi -l 1” in another shell "
              "to watch real-time GPU utilisation.")


if __name__ == "__main__":
    train()
