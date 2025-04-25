#!/usr/bin/env python3
"""
test_model.py – ultra-simple sanity-check for a YOLOv8-cls checkpoint
(best.pt) produced by train_model.py.

usage:
    python test_model.py <img_or_folder>
"""

from __future__ import annotations
from pathlib import Path
import sys
import cv2
import torch
from ultralytics import YOLO

# ------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[1]          # repo root
WEIGHTS = ROOT / "runs/classify/train8/weights/best.pt"
SAMPLE  = ROOT / "data/images/val/crash"               # folder OR file
# ------------------------------------------------------------------


def collect_images(src: Path) -> list[Path]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if src.is_file():
        return [src] if src.suffix.lower() in img_exts else []
    return [p for p in src.rglob("*") if p.suffix.lower() in img_exts]


def main(source: Path) -> None:
    if not WEIGHTS.is_file():
        sys.exit(f"[ERROR] weights not found: {WEIGHTS}")
    imgs = collect_images(source)
    if not imgs:
        sys.exit(f"[ERROR] no images found at {source}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(str(WEIGHTS)).to(device)
    names  = model.names                      # e.g. {0:'crash', 1:'normal'}

    for img_path in imgs:
        with torch.no_grad():
            res = model(img_path, verbose=False)[0]    # single Result obj

        # res.probs is a Probs object whose .data are already probabilities
        top1_cls   = res.probs.top1          # int index
        top1_conf  = res.probs.top1conf      # float tensor
        print(f"{img_path.name}:  {names[top1_cls]}  {top1_conf*100:.1f}%")

        # ─── optional quick viewer ------------------------------------------------
        img = cv2.imread(str(img_path))
        caption = f"{names[top1_cls]}  {top1_conf*100:.1f}%"
        cv2.putText(img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("prediction – ESC quits", img)
        if cv2.waitKey(0) & 0xFF == 27:      # Esc = quit
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    if not src.exists():
        sys.exit(f"[ERROR] path does not exist: {src}")
    main(src)
