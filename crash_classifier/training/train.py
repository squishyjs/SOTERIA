#!/usr/bin/env python3
"""
train.py — YOLO-v8 image-classifier
tuned for i9-14900 KF + RTX 4090 (24 GB)

CLI examples
────────────
python train.py                         # full training run
python train.py --epochs 20 --batch 320 # tweak hyper-params
python train.py --debug                 # 1-batch dry-run (no training)
"""
from __future__ import annotations

import argparse, os, pathlib, sys, datetime as dt, torch, shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from torch.multiprocessing import set_start_method

# ───────────────────── one-time perf toggles ──────────────────────
os.environ["TORCH_DISABLE_DYNAMO"] = "1"          # avoids compile overhead
torch.backends.cudnn.benchmark = True             # autotune CuDNN kernels
torch.set_float32_matmul_precision("high")        # enable TF32
set_start_method("spawn", force=True)             # safe mp on Windows

# ───────────────────── repo paths ─────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]      # repo root (two levels up)
IMAGES = ROOT / "data" / "images"                 # created by the splitter
RUNS   = ROOT / "runs"  / "classify"              # Ultralytics output

# ───────────────────── defaults (edit here) ──────────────────────
DEF_EPOCHS   = 15
DEF_IMGSZ    = 224
DEF_BATCH    = 256            # fits comfortably in 24 GB at 224 px
DEF_DEVICE   = 0              # −1 = CPU
DEF_WEIGHTS  = "yolov8l-cls.pt"
EARLY_STOP   = 5              # patience

# ───────────────────── CLI ───────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="One-click YOLO-v8 image-classifier training.")
ap.add_argument("--epochs",  type=int,  default=DEF_EPOCHS)
ap.add_argument("--imgsz",   type=int,  default=DEF_IMGSZ)
ap.add_argument("--batch",   type=int,  default=DEF_BATCH)
ap.add_argument("--device",  type=int,  default=DEF_DEVICE, help="-1 = CPU")
ap.add_argument("--weights", default=DEF_WEIGHTS, help="backbone *.pt")
ap.add_argument("--debug",   action="store_true", help="dry-run a single batch")
args = ap.parse_args()

# ───────────────────── pretty helpers ────────────────────────────
c = lambda text, code="96": f"\033[{code}m{text}\033[0m"      # colour ANSI

def banner() -> None:
    print(c("\n──────── environment ────────", "95"))
    print("torch :", torch.__version__)
    if torch.cuda.is_available() and args.device != -1:
        gpu = torch.cuda.get_device_name(args.device)
        print(f"cuda  : {torch.version.cuda} | GPU {args.device}: {gpu}")
    else:
        print("CUDA  : NOT detected — training on CPU")
    print()

def dataset_stats(root: Path) -> None:
    if not (root / "train").exists():
        sys.exit(c("[ERR] data/images not found – run the splitter first", "91"))

    splits = ("train", "val", "test")
    stats  = {s: (len(list((root/s/"crash" ).glob("*"))),
                  len(list((root/s/"normal").glob("*")))) for s in splits}

    print(c("dataset", "94"))
    print("split |  crash   |  normal  | ratio")
    print("──────┼──────────┼──────────┼──────")
    for s in splits:
        c_cnt, n_cnt = stats[s]
        ratio = f"1:{n_cnt/c_cnt:.1f}" if c_cnt else "∞"
        print(f"{s:<5} | {c_cnt:>8,} | {n_cnt:>8,} | {ratio}")
    print("──────┴──────────┴──────────┴──────")
    tot_c = sum(x for x,_ in stats.values())
    tot_n = sum(y for _,y in stats.values())
    print(f"total   {tot_c:>8,}   {tot_n:>8,}\n")

# ───────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    banner()
    dataset_stats(IMAGES)

    t0 = dt.datetime.now()

    model = YOLO(args.weights)
    device_str = f"cuda:{args.device}" if args.device != -1 else "cpu"
    model.to(device_str)

    # ───── debug: make sure dataloader & model agree ─────
    if args.debug:
        print(c("⚡ debug = TRUE  →  single forward pass only", "93"))
        dl = model.train_loader(data=str(IMAGES),
                                imgsz=args.imgsz,
                                batch=args.batch,
                                shuffle=True)
        batch = next(iter(dl))
        with torch.no_grad():
            _ = model(batch["img"].to(device_str))
        print(c("✓ dataloader + model OK — exiting", "92"))
        sys.exit(0)

    # ───── call .train() with *only* supported keys ──────
    model.train(
        data=str(IMAGES),           # folder root, no YAML needed
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=os.cpu_count()//2,
        device=args.device,
        amp=True,                   # mixed precision
        cache=True,                 # keep samples in RAM (needs ~25 GB)
        patience=EARLY_STOP,
    )

    # ───── summary ─────
    run  = max(RUNS.glob("exp*"), key=lambda p: p.stat().st_mtime)
    best = run / "weights" / "best.pt"

    dur  = dt.datetime.now() - t0
    vram = (torch.cuda.max_memory_reserved()/1024**3
            if torch.cuda.is_available() and args.device != -1 else 0)

    print(c("\n✅ training complete", "92"))
    print("best model :", best.relative_to(ROOT))
    print("logs       :", c(f"tensorboard --logdir {run.relative_to(ROOT)}", "96"))
    print("time       :", str(dur).split('.')[0])
    if vram:
        print(f"peak VRAM  : {vram:.2f} GB")
