"""
train_model.py — one‑click YOLO‑v8 image‑classifier training **plus** full
post‑training evaluation pack (curves, confusion matrix, PDF summary).

Key upgrades in this revision (2025‑05‑25)
──────────────────────────────────────────
• Uses the `results.save_dir` returned by Ultralytics to reliably locate the
  run directory (avoids the *run directory not found* error on Windows).
• Works no matter how you customised `project=` or `name=` arguments.
• Robust reading of `predictions.csv` (handles column‑name changes across
  Ultralytics versions).
• Generates classification report, confusion matrix, ROC & PR curves even if
  the probability column is absent (falls back to hard predictions).
• Slightly cleaner terminal banner and GPU stats.

Dependencies (extra):
────────────────────
    pip install pandas scikit-learn matplotlib fpdf2

Folder structure expected (unchanged):
    data/images/{train,val,test}/{crash,normal}/*.jpg

Outputs written to the new run folder under runs/classify/…
    • accuracy_curve.png / loss_curve.png
    • roc_curve.png / pr_curve.png / confusion_matrix.png (sk‑learn)
    • classification_report.txt  (tables)
    • training_report.pdf        (one‑page summary)
"""
from __future__ import annotations

# ───────────────────────── imports ────────────────────────────
import os
import pathlib
import sys
from datetime import datetime, timedelta
from typing import Tuple

import torch
from torch.multiprocessing import set_start_method
from ultralytics import YOLO

# post‑training extras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from fpdf import FPDF

# ─── global bug‑workarounds ───────────────────────────────────
os.environ["TORCH_DISABLE_DYNAMO"] = "1"  # avoid Dynamo‑spawn bug
set_start_method("spawn", force=True)     # safe multiprocessing start

# ─── CONFIG — tweak here only ─────────────────────────────────
EPOCHS: int   = 2
IMG_SIZE: int = 224          # loader resize
BATCH: int    = 64           # good for 4090 @224 px
WORKERS: int  = 8            # raise later (e.g. 12) when stable
CACHE_DISK: bool = False     # *.cache files next to JPEGs
MODEL_WTS: str = "yolov8l-cls.pt"  # or -s/-m/-l
DEVICE: int = 0              # GPU index, −1 = CPU

# report options
GENERATE_PDF: bool = True
PDF_TITLE: str = "YOLO‑v8 Crash‑Classifier — Training Report"

# ───────────────────────── paths ──────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "images"        # produced by split_all_frames.py
RUNS_ROOT = ROOT / "runs" / "classify"
TEST_DIR = IMAGES / "test"               # held‑out split

# ───────────────────────── helpers ────────────────────────────

def banner() -> None:
    print("\n────────── environment ──────────")
    print("PyTorch :", torch.__version__)
    if torch.cuda.is_available() and DEVICE != -1:
        name = torch.cuda.get_device_name(DEVICE)
        vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1024 ** 2
        print(f"CUDA    : {torch.version.cuda} | GPU {DEVICE}: {name} ({vram:.0f} MiB)")
    else:
        print("CUDA    : NOT available — training on CPU")
    print()


def plot_acc_loss(csv_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Generate accuracy & loss curves from Ultralytics results.csv"""
    df = pd.read_csv(csv_path)

    # accuracy
    plt.figure()
    if "metrics/accuracy_top1" in df.columns:
        plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="Top‑1")
    if "metrics/accuracy_top5" in df.columns:
        plt.plot(df["epoch"], df["metrics/accuracy_top5"], label="Top‑5")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    (out_dir / "accuracy_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    # loss
    plt.figure()
    if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="Train")
        plt.plot(df["epoch"], df["val/cls_loss"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Cross‑entropy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=300)
    plt.close()


def render_table(report_text: str, out_path: pathlib.Path) -> None:
    out_path.write_text(report_text)


def build_pdf(run_dir: pathlib.Path) -> None:
    """Compile all PNGs + table into a one‑page PDF."""
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, PDF_TITLE, ln=1)

    # images in a 2×3 grid
    charts = [
        "accuracy_curve.png",
        "loss_curve.png",
        "roc_curve.png",
        "pr_curve.png",
        "confusion_matrix.png",
    ]
    w = 90  # two charts per row
    for i, img in enumerate(charts):
        path = run_dir / img
        if not path.exists():
            continue
        if i % 2 == 0:
            pdf.ln(5)
        pdf.image(str(path), w=w)
        if i % 2 == 0:
            pdf.set_x(pdf.get_x() + w + 5)
    pdf.output(run_dir / "training_report.pdf")


def _read_predictions(preds_csv: pathlib.Path) -> Tuple[pd.Series, pd.Series, pd.Series | None]:
    """Return (y_true, y_pred, y_prob?) handling column‑name differences."""
    df = pd.read_csv(preds_csv)

    # ground truth
    if "cls" in df.columns:
        y_true = df["cls"].astype(int)
    elif "target" in df.columns:
        y_true = df["target"].astype(int)
    else:
        raise KeyError("Ground‑truth column not found in predictions.csv")

    # hard predictions
    if "pred" in df.columns:
        y_pred = df["pred"].astype(int)
    elif "prediction" in df.columns:
        y_pred = df["prediction"].astype(int)
    else:
        y_pred = None  # will derive from prob if possible

    # probabilities (for ROC/PR)
    if "prob" in df.columns:
        y_prob = df["prob"].astype(float)
        if y_pred is None:
            y_pred = (y_prob > 0.5).astype(int)
    else:
        y_prob = None
        if y_pred is None:
            raise KeyError("Neither hard predictions nor probabilities found.")

    return y_true, y_pred, y_prob


def evaluate(model: YOLO, run_dir: pathlib.Path) -> None:
    """Run test‑set evaluation + charts + tables."""
    print("\n🔎  Evaluating on held‑out test set …")
    metrics = model.val(
        data=str(TEST_DIR),
        split="test",
        save=True,
        plots=False,  # we generate our own nicer plots
        device=DEVICE,
    )
    val_dir = run_dir / "val"  # created by Ultralytics

    preds_csv = val_dir / "predictions.csv"
    try:
        y_true, y_pred, y_prob = _read_predictions(preds_csv)
    except Exception as e:
        print("[WARN] Failed to parse predictions.csv —", e)
        return

    # table
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Crash"],
        digits=3,
    )
    render_table(report_txt, run_dir / "classification_report.txt")

    # confusion matrix heat‑map (sklearn)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Crash"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (test)")
    plt.tight_layout(); plt.savefig(run_dir / "confusion_matrix.png", dpi=300); plt.close()

    # ROC & PR curves if we have probabilities
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC (AUROC = {roc_auc_score(y_true, y_prob):.3f})")
        plt.tight_layout(); plt.savefig(run_dir / "roc_curve.png", dpi=300); plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision‑Recall")
        plt.tight_layout(); plt.savefig(run_dir / "pr_curve.png", dpi=300); plt.close()
    else:
        print("[INFO] Probability column absent — skipping ROC & PR curves.")

    # Accuracy / loss curves from training csv
    plot_acc_loss(run_dir / "results.csv", run_dir)

    # Optional PDF summary
    if GENERATE_PDF:
        try:
            build_pdf(run_dir)
            print("📝  training_report.pdf generated →", run_dir)
        except Exception as e:
            print("[WARN] PDF generation failed:", e)

    print("✅  Evaluation complete. Key metrics:\n" + report_txt)

# ───────────────────────── main ───────────────────────────────

def main() -> None:
    if not IMAGES.exists():
        sys.exit("[ERROR] data/images not found — run split_all_frames.py first.")

    banner()
    start = datetime.now()

    model = YOLO(MODEL_WTS).to(f"cuda:{DEVICE}" if DEVICE != -1 else "cpu")
    print(f"✅ model on {model.device}\n")

    # ─── training ────────────────────────────────────────────
    results = model.train(
        data=str(IMAGES),   # folder root; Ultralytics infers splits
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        cache=CACHE_DISK,
        device=DEVICE,
        amp=True,
        save=True,          # keep final checkpoint + confusion matrix
        plots=True,         # Ultralytics PNGs
    )

    run_dir = pathlib.Path(results.save_dir)

    elapsed = timedelta(seconds=int((datetime.now() - start).total_seconds()))
    print("\n✅  Training finished in", elapsed)
    print("   runs folder :", run_dir)
    print("   best model  :", run_dir / "weights" / "best.pt")
    print("   tensorboard :  tensorboard --logdir", run_dir)

    # GPU stats
    if torch.cuda.is_available() and DEVICE != -1:
        torch.cuda.synchronize()
        vram_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"   peak VRAM   : {vram_gb:.2f} GB")

    # ─── evaluation ──────────────────────────────────────────
    best_ckpt = run_dir / "weights" / "best.pt"
    model = YOLO(str(best_ckpt)).to(f"cuda:{DEVICE}" if DEVICE != -1 else "cpu")
    evaluate(model, run_dir)


if __name__ == "__main__":
    main()
