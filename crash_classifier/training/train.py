#!/usr/bin/env python3
"""
train_model.py
"""

from __future__ import annotations

# ───────────── imports ─────────────
import argparse
import os
from tqdm import tqdm
import pathlib
import sys
from datetime import datetime, timedelta
from typing import Tuple, List

import torch
from torch.multiprocessing import set_start_method
from ultralytics import YOLO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fpdf import FPDF

# ───────── global work-arounds ─────────
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
set_start_method("spawn", force=True)

# ───────── default CONFIG ─────────
EPOCHS: int = 15
IMG_SIZE: int = 224
BATCH: int = 64
WORKERS: int = 8
CACHE_DISK: bool = False
MODEL_WTS: str = "yolov8l-cls.pt"
DEVICE: int = 0  # GPU index, −1 = CPU

GENERATE_PDF: bool = True
PDF_TITLE: str = "YOLO-v8 Crash-Classifier — Training Report"

# ───────── paths ─────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "images"
RUNS_ROOT = ROOT / "runs" / "classify"
TEST_DIR = IMAGES / "test"

# ═════════ helper functions ═════════
def banner() -> None:
    print("\n────────── environment ──────────")
    print("PyTorch :", torch.__version__)
    if torch.cuda.is_available() and DEVICE != -1:
        name = torch.cuda.get_device_name(DEVICE)
        vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**2
        print(f"CUDA    : {torch.version.cuda} | GPU {DEVICE}: {name} ({vram:.0f} MiB)")
    else:
        print("CUDA    : NOT available — training on CPU")
    print()

# ── curves from Ultralytics CSV
def plot_acc_loss(csv_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    # Accuracy
    if {"metrics/accuracy_top1"}.issubset(df.columns):
        plt.figure()
        plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Top-1")
        if "metrics/accuracy_top5" in df.columns:
            plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Top-5")
        plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend(), plt.tight_layout()
        (out_dir / "accuracy_curve.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
        plt.close()

    # Loss
    if {"train/cls_loss", "val/cls_loss"}.issubset(df.columns):
        plt.figure()
        plt.plot(df["epoch"], df["train/cls_loss"], label="Train")
        plt.plot(df["epoch"], df["val/cls_loss"], label="Val")
        plt.xlabel("Epoch"), plt.ylabel("Cross-entropy"), plt.legend(), plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=300)
        plt.close()

def render_table(text: str, out: pathlib.Path) -> None:
    out.write_text(text)

# ── PDF compilation
def build_pdf(run_dir: pathlib.Path) -> None:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, PDF_TITLE, ln=1)

    charts = [
        "accuracy_curve.png", "loss_curve.png",
        "roc_curve.png", "pr_curve.png",
        "confusion_matrix.png", "conf_hist.png",
        "qual_montage.jpg",
    ]
    w = 90
    for i, img in enumerate(charts):
        p = run_dir / img
        if not p.exists():  # skip missing artefacts
            continue
        if i % 2 == 0: pdf.ln(5)
        pdf.image(str(p), w=w)
        if i % 2 == 0: pdf.set_x(pdf.get_x() + w + 5)
    pdf.output(run_dir / "training_report.pdf")

# ── prediction CSV loader
def _read_predictions(preds_csv: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series | None]:
    df = pd.read_csv(preds_csv)

    # ground-truth
    if "cls" in df.columns:
        y_true = df["cls"].astype(int)
    elif "target" in df.columns:
        y_true = df["target"].astype(int)
    else:
        raise KeyError("Ground-truth column not found")

    # hard predictions
    if "pred" in df.columns:
        y_pred = df["pred"].astype(int)
    elif "prediction" in df.columns:
        y_pred = df["prediction"].astype(int)
    else:
        y_pred = None

    # probabilities
    if "prob" in df.columns:
        y_prob = df["prob"].astype(float)
        if y_pred is None:
            y_pred = (y_prob > 0.5).astype(int)
    else:
        y_prob = None
        if y_pred is None:
            raise KeyError("No predictions found")

    return df, y_true, y_pred, y_prob

# ═════════ evaluation ════════════════════════════════════════════
def evaluate(model: YOLO, run_dir: pathlib.Path) -> None:

    # ── 1. run a single validation pass ───────────────────────────
    print("\n🔎  Evaluating on held-out test set …")
    results = model.val(
        data=str(IMAGES),        # folder with train/val/test
        split="test",
        save=True,
        plots=False,
        device=DEVICE,
    )
    preds_csv = pathlib.Path(results.save_dir) / "predictions.csv"

    # ── 2. obtain frame-level predictions ─────────────────────────
    try:
        df, y_true, y_pred, y_prob = _read_predictions(preds_csv)

    except FileNotFoundError:
        print("[INFO] predictions.csv absent – running manual inference …")

        rows: list[dict] = []
        images = list(TEST_DIR.rglob("*.jpg"))
        for p in tqdm(images, desc="inferring", ncols=90, unit="img"):
            r = model.predict(str(p), imgsz=IMG_SIZE, device=DEVICE,
                              verbose=False)[0]
            prob = float(r.probs.data[1])     # class-index 1 = “crash”
            rows.append({
                "image": str(p.relative_to(TEST_DIR)),
                "cls":   1 if "crash" in p.parts else 0,
                "pred":  int(prob > 0.5),
                "prob":  prob,
            })
        df     = pd.DataFrame(rows)
        y_true = df["cls"]
        y_pred = df["pred"]
        y_prob = df["prob"]

    except Exception as e:
        print(f"[WARN] Could not obtain predictions — {e}")
        return

    # ── 3. frame-level metrics ───────────────────────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa   = cohen_kappa_score      (y_true, y_pred)

    report_txt  = classification_report(
        y_true, y_pred, target_names=["Normal", "Crash"], digits=3
    )
    report_txt += f"\nBalanced Acc : {bal_acc:.3f}"
    report_txt += f"\nCohen κ      : {kappa:.3f}"
    render_table(report_txt, run_dir / "classification_report.txt")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Crash"]).plot(
        cmap="Blues", colorbar=False
    )
    plt.title("Confusion Matrix (test)")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC (AUROC = {roc_auc_score(y_true, y_prob):.3f})")
        plt.tight_layout(); plt.savefig(run_dir / "roc_curve.png", dpi=300); plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
        plt.tight_layout(); plt.savefig(run_dir / "pr_curve.png", dpi=300); plt.close()

        plt.figure(); plt.hist(y_prob, bins=40)
        plt.xlabel("Predicted crash-probability"); plt.ylabel("Count")
        plt.title("Prediction Confidence Histogram")
        plt.tight_layout(); plt.savefig(run_dir / "conf_hist.png", dpi=300); plt.close()
    else:
        print("[INFO] Probability column absent — ROC/PR skipped")

    plot_acc_loss(run_dir / "results.csv", run_dir)

    def to_clip_id(p: str) -> str:
        """C_000123_07.jpg → '000123' (fallback 'unknown')."""
        parts = pathlib.Path(p).name.split("_")
        return parts[1] if len(parts) > 1 else "unknown"

    df["clip_id"]   = df["image"].apply(to_clip_id)
    df["hard_pred"] = y_pred

    clip_df = (df.groupby("clip_id")
                 .agg(gt=("cls", "max"), pred=("hard_pred", "max")))

    clip_report = classification_report(
        clip_df["gt"], clip_df["pred"],
        target_names=["Normal-clip", "Crash-clip"], digits=3
    )
    (run_dir / "clip_level_report.txt").write_text(clip_report)

    def _sample(mask: np.ndarray, k: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return np.empty(0, dtype=int)
        return np.random.choice(idx, size=min(k, len(idx)), replace=False)

    tp_i = _sample((y_true == 1) & (y_pred == 1), 4)
    fp_i = _sample((y_true == 0) & (y_pred == 1), 4)
    fn_i = _sample((y_true == 1) & (y_pred == 0), 4)

    tiles: list[Image.Image] = []
    for i in np.concatenate([tp_i, fp_i, fn_i]):
        p = TEST_DIR / df.loc[i, "image"]
        if p.exists():
            img   = Image.open(p).convert("RGB").resize((224, 224))
            colour = ("green"  if i in tp_i else
                      "orange" if i in fp_i else
                      "red")
            ImageDraw.Draw(img).rectangle([(0, 0), (223, 223)],
                                          outline=colour, width=3)
            tiles.append(img)

    if tiles:
        grid = Image.new("RGB", (224 * 4, 224 * 3), "white")
        for n, t in enumerate(tiles):
            grid.paste(t, (224 * (n % 4), 224 * (n // 4)))
        grid.save(run_dir / "qual_montage.jpg")

    if GENERATE_PDF:
        try:
            build_pdf(run_dir)
        except Exception as e:
            print("[WARN] PDF generation failed:", e)

    # ── 8. console summary ───────────────────────────────────────
    print("\n── Frame-level metrics ──\n", report_txt)
    print("── Clip-level  metrics ──\n", clip_report)

def main() -> None:
    if not IMAGES.exists():
        sys.exit("[ERROR] data/images not found — run split_all_frames.py first.")

    banner()
    start = datetime.now()

    model = YOLO(MODEL_WTS).to(f"cuda:{DEVICE}" if DEVICE != -1 else "cpu")
    print(f"✅ model on {model.device}\n")

    results = model.train(
        data=str(IMAGES),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE_DISK,
        device=DEVICE,
        amp=True,
        save=True,
        plots=True,
    )
    run_dir = pathlib.Path(results.save_dir)

    elapsed = timedelta(seconds=int((datetime.now() - start).total_seconds()))
    print("\n✅  Training finished in", elapsed)
    print("   runs folder :", run_dir)
    print("   best model  :", run_dir / "weights" / "best.pt")
    print("   tensorboard :  tensorboard --logdir", run_dir)

    if torch.cuda.is_available() and DEVICE != -1:
        torch.cuda.synchronize()
        vram_gb = torch.cuda.max_memory_reserved() / 1024**3
        print(f"   peak VRAM   : {vram_gb:.2f} GB")

    best_ckpt = run_dir / "weights" / "best.pt"
    model = YOLO(str(best_ckpt)).to(f"cuda:{DEVICE}" if DEVICE != -1 else "cpu")
    evaluate(model, run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-v8 crash-classifier trainer")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of epochs")
    parser.add_argument("--model", type=str, default=MODEL_WTS, help="initial weights .pt")
    parser.add_argument("--device", type=int, default=DEVICE, help="GPU index or −1 (CPU)")
    args = parser.parse_args()

    EPOCHS, MODEL_WTS, DEVICE = args.epochs, args.model, args.device
    main()
