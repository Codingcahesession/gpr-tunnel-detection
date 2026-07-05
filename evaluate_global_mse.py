r"""
eval_global_mse.py
GPR Autoencoder Evaluation - GLOBAL MSE BASELINE (ablation row 1).

This is the naive baseline: the anomaly score is the mean squared
reconstruction error over the WHOLE image (all 128x128 pixels), with no
top-k selection, no depth restriction, and no spatial aggregation.

It exists to provide the first row of the ablation table, showing what
plain global-MSE scoring achieves before any of the improvements.

Reads the same labels CSV and survey-line folder layout as the other
scripts so the numbers are directly comparable.

Results go to: ./Results/Inference_GlobalMSE
"""

from pathlib import Path
from typing import Dict, Tuple
from math import sqrt
import shutil
import random
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GPRBottleneckAE


# =============================================================================
# BLOCK 1: SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INFERENCE_ROOT = Path("data") / "Inference_with_line_folders"
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / "Inference_GlobalMSE"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

THRESHOLD_K = 2.5


evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# BLOCK 2: DATA
# =============================================================================
def load_labels_csv(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {csv_path}\nRun build_labels_csv.py first."
        )
    mapping = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"].strip().lower()] = (
                row["survey_line"].strip(), int(row["label"])
            )
    return mapping


def collect_samples(inference_root: Path, label_map):
    if not inference_root.exists():
        raise FileNotFoundError(f"Inference root does not exist: {inference_root}")
    samples = []
    unmatched = []
    for p in sorted(inference_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = p.name.lower()
        if key not in label_map:
            unmatched.append(p)
            continue
        _, label = label_map[key]
        samples.append((p, p.parent.name, label))
    return samples, unmatched


class LabelledLineDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, survey_line, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path), survey_line


def load_model() -> GPRBottleneckAE:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = GPRBottleneckAE().to(DEVICE)
    weights = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        model.load_state_dict(weights["model_state_dict"])
    else:
        model.load_state_dict(weights)
    model.eval()
    return model


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - half, centre + half


def compute_full_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn_count, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    n_total = tn + fp + fn_count + tp
    precision_v   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_v      = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
    specificity_v = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv_v         = tn / (tn + fn_count) if (tn + fn_count) > 0 else 0.0
    f1_v = (2 * precision_v * recall_v / (precision_v + recall_v)
            if (precision_v + recall_v) > 0 else 0.0)
    f2_v = (5 * precision_v * recall_v / (4 * precision_v + recall_v)
            if (4 * precision_v + recall_v) > 0 else 0.0)
    balanced_acc = (recall_v + specificity_v) / 2.0
    g_mean = sqrt(max(0.0, recall_v * specificity_v))
    mcc_num = tp * tn - fp * fn_count
    mcc_den = sqrt((tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count))
    mcc_v = mcc_num / mcc_den if mcc_den > 0 else 0.0
    po = (tp + tn) / n_total
    pe = (((tp + fp) / n_total) * ((tp + fn_count) / n_total)
          + ((tn + fn_count) / n_total) * ((tn + fp) / n_total))
    kappa_v = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    acc = (tp + tn) / n_total
    return {
        "cm": cm, "tn": tn, "fp": fp, "fn": fn_count, "tp": tp,
        "accuracy": acc, "precision": precision_v, "recall": recall_v,
        "specificity": specificity_v, "npv": npv_v, "f1": f1_v, "f2": f2_v,
        "balanced_acc": balanced_acc, "g_mean": g_mean, "mcc": mcc_v,
        "kappa": kappa_v, "fpr": 1.0 - specificity_v, "fnr": 1.0 - recall_v,
    }


# =============================================================================
# BLOCK 3: MAIN
# =============================================================================
def run_evaluation() -> None:
    print("=" * 70)
    print("GPR AUTOENCODER EVALUATION - GLOBAL MSE BASELINE")
    print("Score = mean squared error over the WHOLE image (no top-k)")
    print("=" * 70)
    print(f"Device      : {DEVICE}")
    print(f"Model       : {MODEL_PATH}")
    print(f"Inference   : {INFERENCE_ROOT}")
    print(f"Labels CSV  : {LABELS_CSV}")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"THRESHOLD_K : {THRESHOLD_K}")
    print("=" * 70)

    label_map = load_labels_csv(LABELS_CSV)
    samples, unmatched = collect_samples(INFERENCE_ROOT, label_map)
    print(f"\nMatched {len(samples)} images to labels.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} images had no label and were skipped.")

    n_normal = sum(1 for _, _, l in samples if l == 0)
    n_tunnel = sum(1 for _, _, l in samples if l == 1)
    print(f"Class balance: normal={n_normal}, tunnel={n_tunnel}")

    model = load_model()
    dataset = LabelledLineDataset(samples, transform=evaluation_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    errors = []
    y_true = []
    paths = []
    origs = []
    recons = []

    print("\nProcessing all images (global MSE)...")
    with torch.no_grad():
        for imgs, labels, path_list, _ in loader:
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            # GLOBAL MSE: mean over ALL pixels (no top-k, no depth restriction)
            mse = torch.mean((imgs - output) ** 2, dim=(1, 2, 3))
            errors.extend(mse.cpu().numpy())
            y_true.extend(labels.numpy())
            paths.extend(path_list)
            origs.extend(imgs.cpu().numpy())
            recons.extend(output.cpu().numpy())

    errors = np.array(errors)
    y_true = np.array(y_true)

    normal_scores = errors[y_true == 0]
    tunnel_scores = errors[y_true == 1]

    threshold = float(normal_scores.mean() + THRESHOLD_K * normal_scores.std())
    y_pred = (errors >= threshold).astype(int)

    auc = roc_auc_score(y_true, errors)
    ap = average_precision_score(y_true, errors)
    full = compute_full_metrics(y_true, y_pred)

    tn, fp, fn_count, tp = full["tn"], full["fp"], full["fn"], full["tp"]
    n_total = tn + fp + fn_count + tp
    acc_lo, acc_hi = wilson_ci(tp + tn, n_total)
    rec_lo, rec_hi = wilson_ci(tp, tp + fn_count)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    spec_lo, spec_hi = wilson_ci(tn, tn + fp)

    print("\n--- Global MSE baseline metrics ---")
    print(f"Threshold value : {threshold:.6f}")
    print(f"AUC-ROC         : {auc:.4f}")
    print(f"AP              : {ap:.4f}")
    print(f"Accuracy        : {acc:=full['accuracy']:.4f}" if False else f"Accuracy        : {full['accuracy']:.4f}")
    print(f"Precision       : {full['precision']:.4f}")
    print(f"Recall          : {full['recall']:.4f}")
    print(f"Specificity     : {full['specificity']:.4f}")
    print(f"F1              : {full['f1']:.4f}")
    print(f"F2              : {full['f2']:.4f}")
    print(f"MCC             : {full['mcc']:.4f}")
    print(f"Confusion (TN, FP, FN, TP): {tn}, {fp}, {fn_count}, {tp}")
    print(f"\nScore distributions:")
    print(f"  normal : mean={normal_scores.mean():.6f}  std={normal_scores.std():.6f}")
    print(f"  tunnel : mean={tunnel_scores.mean():.6f}  std={tunnel_scores.std():.6f}")

    # ---- Threshold sweep ----
    print("\n--- Threshold sweep ---")
    sweep_csv = RESULTS_DIR / "threshold_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "threshold", "TN", "FP", "FN", "TP", "F1", "accuracy"])
        rows_sweep = [(f"mean+{THRESHOLD_K}std", threshold)]
        for pct in [95, 97, 98, 99, 99.5]:
            rows_sweep.append((f"p{pct}", float(np.percentile(normal_scores, pct))))
        for name, t in rows_sweep:
            yp = (errors >= t).astype(int)
            tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
            print(f"{name:>14}: threshold={t:.6f} | FP={fp_s:4d} FN={fn_s:4d} | "
                  f"F1={f1_score(y_true, yp):.4f} | Acc={accuracy_score(y_true, yp):.4f}")
            writer.writerow([name, f"{t:.6f}", tn_s, fp_s, fn_s, tp_s,
                             f"{f1_score(y_true, yp):.4f}",
                             f"{accuracy_score(y_true, yp):.4f}"])
    print(f"Saved sweep CSV: {sweep_csv}")

    # ---- Metrics report (txt + csv) ----
    txt = RESULTS_DIR / "metrics_report.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("GPR ANOMALY DETECTION - GLOBAL MSE BASELINE\n")
        f.write("=" * 55 + "\n\n")
        f.write("Scoring: mean squared error over the whole image (no top-k, "
                "no depth restriction, no aggregation).\n")
        f.write(f"Threshold rule: mean + {THRESHOLD_K}*std on normal scores.\n")
        f.write(f"Threshold value: {threshold:.6f}\n\n")
        f.write(f"Test samples: N = {n_total}  (normal = {tn + fp}, tunnel = {fn_count + tp})\n")
        f.write(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn_count}  TP={tp}\n\n")
        f.write("Threshold-free metrics\n")
        f.write(f"  AUC-ROC              : {auc:.4f}\n")
        f.write(f"  Average Precision    : {ap:.4f}\n\n")
        f.write("Classification metrics (95% Wilson CI)\n")
        f.write(f"  Accuracy             : {full['accuracy']:.4f}  [{acc_lo:.4f}, {acc_hi:.4f}]\n")
        f.write(f"  Precision (PPV)      : {full['precision']:.4f}  [{prec_lo:.4f}, {prec_hi:.4f}]\n")
        f.write(f"  Recall / TPR         : {full['recall']:.4f}  [{rec_lo:.4f}, {rec_hi:.4f}]\n")
        f.write(f"  Specificity / TNR    : {full['specificity']:.4f}  [{spec_lo:.4f}, {spec_hi:.4f}]\n")
        f.write(f"  F1                   : {full['f1']:.4f}\n")
        f.write(f"  F2                   : {full['f2']:.4f}\n")
        f.write(f"  Balanced accuracy    : {full['balanced_acc']:.4f}\n")
        f.write(f"  MCC                  : {full['mcc']:.4f}\n")
        f.write(f"  Cohen's kappa        : {full['kappa']:.4f}\n")
        f.write(f"  G-mean               : {full['g_mean']:.4f}\n\n")
        f.write("Error rates\n")
        f.write(f"  FPR                  : {full['fpr']:.4f}  ({fp}/{tn + fp})\n")
        f.write(f"  FNR                  : {full['fnr']:.4f}  ({fn_count}/{fn_count + tp})\n")
        f.write(f"  NPV                  : {full['npv']:.4f}\n")

    csv_path = RESULTS_DIR / "metrics_report.csv"
    rows = [
        ("AUC-ROC",           f"{auc:.4f}",              "", ""),
        ("Average Precision", f"{ap:.4f}",               "", ""),
        ("Accuracy",          f"{full['accuracy']:.4f}", f"{acc_lo:.4f}",  f"{acc_hi:.4f}"),
        ("Precision",         f"{full['precision']:.4f}",f"{prec_lo:.4f}", f"{prec_hi:.4f}"),
        ("Recall",            f"{full['recall']:.4f}",   f"{rec_lo:.4f}",  f"{rec_hi:.4f}"),
        ("Specificity",       f"{full['specificity']:.4f}", f"{spec_lo:.4f}", f"{spec_hi:.4f}"),
        ("F1",                f"{full['f1']:.4f}",       "", ""),
        ("F2",                f"{full['f2']:.4f}",       "", ""),
        ("Balanced accuracy", f"{full['balanced_acc']:.4f}", "", ""),
        ("MCC",               f"{full['mcc']:.4f}",      "", ""),
        ("Cohen kappa",       f"{full['kappa']:.4f}",    "", ""),
        ("G-mean",            f"{full['g_mean']:.4f}",   "", ""),
        ("FPR",               f"{full['fpr']:.4f}",      "", ""),
        ("FNR",               f"{full['fnr']:.4f}",      "", ""),
        ("NPV",               f"{full['npv']:.4f}",      "", ""),
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "ci_lower_95", "ci_upper_95"])
        writer.writerows(rows)
    print(f"Saved metrics report: {txt}")
    print(f"Saved metrics CSV: {csv_path}")

    # ---- ROC / PR / CM ----
    fpr_arr, tpr_arr, _ = roc_curve(y_true, errors)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_arr, tpr_arr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve (global MSE)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150); plt.close()

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, errors)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_arr, precision_arr, linewidth=2, label=f"AP = {ap:.3f}")
    plt.title("Precision-Recall Curve (global MSE)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    cm = full["cm"]
    plt.imshow(cm)
    plt.title("Confusion Matrix (global MSE)")
    plt.xticks([0, 1], ["Pred Normal", "Pred Tunnel"])
    plt.yticks([0, 1], ["True Normal", "True Tunnel"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150); plt.close()

    # ---- Per-image CSV ----
    per_image_csv = RESULTS_DIR / "per_image_scores.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "filename", "label", "score", "predicted"])
        for i in range(len(errors)):
            writer.writerow([paths[i], Path(paths[i]).name,
                             int(y_true[i]), f"{errors[i]:.6f}", int(y_pred[i])])
    print(f"Saved per-image CSV: {per_image_csv}")

    print(f"\nDone. Global MSE baseline results in: {RESULTS_DIR}")


if __name__ == "__main__":
    run_evaluation()
