#!/usr/bin/env python3
"""
export_model.py – export a trained YOLO-v8 *classification* checkpoint to:

    • ONNX  – 3 MB, fastest Flutter integration (onnxruntime plugin)
    • TFLite (fp32) – only if TensorFlow is available
    • TorchScript – optional for C++ / LibTorch

Why this layout?
────────────────
• ONNX is universal across all Flutter targets and carries no extra DLLs.
• TFLite still handy for mobile-only builds, but its converter pulls in the
  whole TensorFlow stack.  We try it *only* when those deps are already present.
• Everything runs on CPU → avoids Windows CUDA seg-fault (0xC0000005).

Usage
-----
    python export_model.py                 # auto-detect newest runs/*/best.pt
    python export_model.py path/to/best.pt
"""

from __future__ import annotations
import importlib
import pathlib
import shutil
import sys
from typing import Optional

from ultralytics import YOLO

# ───────────────────────── config ────────────────────────────────
IMGSZ = 224                                # training size
ROOT   = pathlib.Path(__file__).resolve().parents[1]
RUNS   = ROOT / "runs" / "classify"
EXPORT = ROOT / "exports"
# ─────────────────────────────────────────────────────────────────


def latest_best() -> Optional[pathlib.Path]:
    """Return the newest runs/*/best.pt, or None."""
    bests = sorted(RUNS.glob("*/*/best.pt"), key=lambda p: p.stat().st_mtime)
    return bests[-1] if bests else None


def mv(src: pathlib.Path, dst_dir: pathlib.Path, name: str):
    """Move *src* into *dst_dir* under *name* and return the new path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    return pathlib.Path(shutil.move(src, dst))


def has_pkg(pkg: str) -> bool:
    """Return True if *pkg* can be imported."""
    return importlib.util.find_spec(pkg) is not None


def export_all(best_pt: pathlib.Path) -> None:
    run_name = best_pt.parents[1].name        # e.g. train8
    out_dir  = EXPORT / run_name
    print(f"\n▶ exporting  {best_pt.relative_to(ROOT)}  →  {out_dir.relative_to(ROOT)}")

    # ─── load on CPU for safest conversion ──────────────────────
    model  = YOLO(str(best_pt))
    device = "cpu"
    print("✅  checkpoint loaded on CPU")

    # 1️⃣  ONNX – always
    print("• ONNX …")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=True,
        device=device,
    )
    mv(onnx_path, out_dir, "best.onnx")

    # 2️⃣  TFLite fp32 – only if TensorFlow stack present
    if all(has_pkg(p) for p in ("tensorflow", "tflite_support")):
        print("• TFLite (fp32) …")
        tflite_path = model.export(
            format="tflite",
            imgsz=IMGSZ,
            device=device,
            half=False,
            int8=False,          # no INT-8 – avoids seg-fault on Windows
        )
        mv(tflite_path, out_dir, "best.tflite")
    else:
        print("• TFLite skipped (TensorFlow or tflite-support not installed)")

    # 3️⃣  TorchScript – lightweight, no extra deps
    print("• TorchScript …")
    ts_path = model.export(
        format="torchscript",
        imgsz=IMGSZ,
        device=device,
    )
    mv(ts_path, out_dir, "best.torchscript")

    print("\n✅  all exports saved in", out_dir.relative_to(ROOT))


# ────────────────────────── main ─────────────────────────────────
if __name__ == "__main__":
    ckpt = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_best()
    if not ckpt or not ckpt.is_file():
        sys.exit("[ERROR] pass a path to best.pt or run training first.")
    export_all(ckpt)
