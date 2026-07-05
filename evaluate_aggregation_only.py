r"""
eval_idea2_only.py
GPR Autoencoder Evaluation - IDEA 2 ONLY (window aggregation).

Scoring is the ORIGINAL full-image top-5% MSE (no depth restriction).
This isolates idea 2 so you can measure what spatial voting alone
contributes, using the clean survey-line grouping from the CSV.

Reads:
    Inference_with_line_folders/  (images organised by survey line)
    inference_labels.csv          (survey_line, filename, label)

Aggregation rule (no-demote):
    aggregated[i] = 1  if raw_pred[i] == 1
                        OR  >= N of M neighbours in the same survey line
                        were raw-positive
    aggregated[i] = 0  otherwise

Confirmed detections are never demoted. Missed tunnels can be recovered
if enough of their spatially-adjacent neighbours were detected.

Results go to: ./Results/Inference_Idea2_Only
"""

from pathlib import Path
from typing import List, Tuple, Dict
from math import sqrt, ceil
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
# BLOCK 1: PATHS AND MAIN SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INFERENCE_ROOT = Path("data") / "Inference_with_line_folders"
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / "Inference_Idea2_Only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Scoring (idea 2 uses the ORIGINAL full-image top-k rule).
TOP_FRACTION = 0.05

# Threshold rule: mean + K * std of normal-class scores.
THRESHOLD_K = 2.5

# Aggregation configurations (idea 2 knobs).
WINDOW_CONFIGS: List[Tuple[int, int]] = [
    (3, 2),
    (5, 2),
    (5, 3),
    (7, 2),
    (7, 3),
    (7, 4),
    (9, 3),
    (9, 4),
    (9, 5),
    (11, 4),
]

SCALE_RULE_AT_EDGES = True


# =============================================================================
# BLOCK 2: EVALUATION TRANSFORM
# =============================================================================
evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# BLOCK 3: DATASET AND LABEL LOADING
# =============================================================================
def load_labels_csv(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {csv_path}\n"
            "Run build_labels_csv.py first."
        )
    mapping = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["filename"].strip().lower()
            mapping[key] = (row["survey_line"].strip(), int(row["label"]))
    return mapping


def collect_samples(inference_root: Path, label_map):
    if not inference_root.exists():
        raise FileNotFoundError(f"Inference root does not exist: {inference_root}")
    samples = []
    unmatched = []
    for p in sorted(inference_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = p.name.lower()
        if key not in label_map:
            unmatched.append(p)
            continue
        _, label = label_map[key]
        survey_line = p.parent.name
        samples.append((p, survey_line, label))
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


# =============================================================================
# BLOCK 4: MODEL LOADING
# =============================================================================
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


# =============================================================================
# BLOCK 5: VISUALIZATION HELPERS
# =============================================================================
def save_reconstruction_panel(path_str, orig, recon, score, out_dir, tag):
    p = Path(path_str)
    x = orig.squeeze()
    xh = recon.squeeze()
    heatmap = (x - xh) ** 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap="gray")
    plt.title(f"{tag} Input"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(xh, cmap="gray")
    plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(heatmap, cmap="magma")
    plt.title(f"Heatmap | score: {score:.6f}"); plt.axis("off")
    plt.suptitle(f"{tag} Analysis | {p.name}", y=0.98)
    plt.tight_layout()
    safe_stem = p.stem.replace(" ", "_")
    out_path = out_dir / f"{tag.lower()}_{safe_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0, 1], ["Pred Normal", "Pred Tunnel"])
    plt.yticks([0, 1], ["True Normal", "True Tunnel"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - half, centre + half


# =============================================================================
# BLOCK 6: AGGREGATION (no-demote, grouped by survey_line)
# =============================================================================
def natkey(s: str):
    out, buf = [], ""
    for ch in s:
        if ch.isdigit():
            buf += ch
        else:
            if buf:
                out.append(int(buf)); buf = ""
            out.append(ch.lower())
    if buf:
        out.append(int(buf))
    return out


def build_groups(paths, survey_lines):
    by_line: Dict[str, List[Tuple[list, int]]] = {}
    for i, (p, sl) in enumerate(zip(paths, survey_lines)):
        key = natkey(Path(p).name)
        by_line.setdefault(sl, []).append((key, i))
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for sl, items in by_line.items():
        items.sort(key=lambda t: t[0])
        groups[sl] = [(pos, orig_i) for pos, (_, orig_i) in enumerate(items)]
    return groups


def apply_window_voting_no_demote(groups, raw_predictions, M, N, scale_at_edges=True):
    aggregated = raw_predictions.copy()
    half = M // 2
    n_recovered = 0
    n_protected = 0
    for sl, members in groups.items():
        orig_idx = [t[1] for t in members]
        preds_in_group = raw_predictions[orig_idx]
        g_len = len(members)
        for pos in range(g_len):
            lo = max(0, pos - half)
            hi = min(g_len, pos + half + 1)
            window_preds = preds_in_group[lo:hi]
            actual_size = len(window_preds)
            n_pos = int(window_preds.sum())
            if scale_at_edges and actual_size < M:
                required = max(1, ceil(actual_size * (N / M)))
            else:
                required = N

            gi = orig_idx[pos]
            raw = int(raw_predictions[gi])
            if raw == 1:
                aggregated[gi] = 1
                if n_pos < required:
                    n_protected += 1
            else:
                if n_pos >= required:
                    aggregated[gi] = 1
                    n_recovered += 1
                else:
                    aggregated[gi] = 0
    return aggregated, n_recovered, n_protected


# =============================================================================
# BLOCK 7: METRICS
# =============================================================================
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
        "accuracy": acc,
        "precision": precision_v, "recall": recall_v,
        "specificity": specificity_v, "npv": npv_v,
        "f1": f1_v, "f2": f2_v,
        "balanced_acc": balanced_acc, "g_mean": g_mean,
        "mcc": mcc_v, "kappa": kappa_v,
        "fpr": 1.0 - specificity_v, "fnr": 1.0 - recall_v,
    }


def write_metrics_report(name_suffix, y_true, y_pred, errors, auc, ap,
                         threshold, results_dir, header_note=""):
    full = compute_full_metrics(y_true, y_pred)
    tn, fp, fn_count, tp = full["tn"], full["fp"], full["fn"], full["tp"]
    n_total = tn + fp + fn_count + tp
    n_normal = tn + fp
    n_tunnel = fn_count + tp

    acc_lo,  acc_hi  = wilson_ci(tp + tn, n_total)
    rec_lo,  rec_hi  = wilson_ci(tp, tp + fn_count)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    spec_lo, spec_hi = wilson_ci(tn, tn + fp)

    txt = results_dir / f"metrics_report_{name_suffix}.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"GPR ANOMALY DETECTION - {name_suffix.upper()}\n")
        f.write("=" * 60 + "\n\n")
        if header_note:
            f.write(header_note + "\n\n")
        f.write(f"Threshold value: {threshold:.6f}\n")
        f.write(f"Test samples: N = {n_total}  (normal = {n_normal}, tunnel = {n_tunnel})\n")
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
        f.write(f"  FPR                  : {full['fpr']:.4f}  ({fp}/{n_normal})\n")
        f.write(f"  FNR                  : {full['fnr']:.4f}  ({fn_count}/{n_tunnel})\n")
        f.write(f"  NPV                  : {full['npv']:.4f}\n")

    csv_path = results_dir / f"metrics_report_{name_suffix}.csv"
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
    return full


def print_metrics_block(header, full, threshold, auc, ap):
    print(f"\n--- {header} ---")
    print(f"Threshold value : {threshold:.6f}")
    print(f"AUC-ROC         : {auc:.4f}")
    print(f"AP              : {ap:.4f}")
    print(f"Accuracy        : {full['accuracy']:.4f}")
    print(f"Precision       : {full['precision']:.4f}")
    print(f"Recall          : {full['recall']:.4f}")
    print(f"Specificity     : {full['specificity']:.4f}")
    print(f"F1              : {full['f1']:.4f}")
    print(f"MCC             : {full['mcc']:.4f}")
    print(f"Confusion (TN, FP, FN, TP): "
          f"{full['tn']}, {full['fp']}, {full['fn']}, {full['tp']}")


# =============================================================================
# BLOCK 8: MAIN PIPELINE
# =============================================================================
def run_evaluation() -> None:
    print("=" * 70)
    print("GPR AUTOENCODER EVALUATION - IDEA 2 ONLY")
    print("Full-image top-5% MSE scoring + window aggregation (no-demote)")
    print("=" * 70)
    print(f"Device            : {DEVICE}")
    print(f"Model             : {MODEL_PATH}")
    print(f"Inference root    : {INFERENCE_ROOT}")
    print(f"Labels CSV        : {LABELS_CSV}")
    print(f"Results dir       : {RESULTS_DIR}")
    print(f"TOP_FRACTION      : {TOP_FRACTION}")
    print(f"THRESHOLD_K       : {THRESHOLD_K}")
    print(f"Window configs    : {WINDOW_CONFIGS}")
    print("=" * 70)

    label_map = load_labels_csv(LABELS_CSV)
    print(f"\nLoaded {len(label_map)} label entries from CSV.")

    samples, unmatched = collect_samples(INFERENCE_ROOT, label_map)
    print(f"Matched {len(samples)} images to labels.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} images had no label in the CSV and were skipped.")
        for u in unmatched[:5]:
            print(f"  {u.name}")

    n_normal = sum(1 for _, _, l in samples if l == 0)
    n_tunnel = sum(1 for _, _, l in samples if l == 1)
    print(f"Class balance: normal={n_normal}, tunnel={n_tunnel}")
    print(f"Distinct survey lines: {len(set(sl for _, sl, _ in samples))}")

    model = load_model()
    dataset = LabelledLineDataset(samples, transform=evaluation_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    errors = []
    y_true = []
    paths = []
    survey_lines = []
    origs = []
    recons = []

    print("\nProcessing all survey-line images...")
    with torch.no_grad():
        for imgs, labels, path_list, sl_list in loader:
            imgs = imgs.to(DEVICE)
            output = model(imgs)

            # Full-image top-5% MSE (idea 2 alone, no depth restriction)
            err = ((imgs - output) ** 2).flatten(start_dim=1)
            k = max(1, int(TOP_FRACTION * err.shape[1]))
            mse = torch.topk(err, k, dim=1).values.mean(dim=1)

            errors.extend(mse.cpu().numpy())
            y_true.extend(labels.numpy())
            paths.extend(path_list)
            survey_lines.extend(list(sl_list))
            origs.extend(imgs.cpu().numpy())
            recons.extend(output.cpu().numpy())

    errors = np.array(errors)
    y_true = np.array(y_true)

    normal_scores = errors[y_true == 0]
    tunnel_scores = errors[y_true == 1]

    threshold = float(normal_scores.mean() + THRESHOLD_K * normal_scores.std())
    raw_pred = (errors >= threshold).astype(int)

    auc = roc_auc_score(y_true, errors)
    ap = average_precision_score(y_true, errors)

    base = compute_full_metrics(y_true, raw_pred)
    print_metrics_block("Per-image (full-image top-5% MSE) metrics", base, threshold, auc, ap)
    print(f"\nScore distributions:")
    print(f"  normal : mean={normal_scores.mean():.6f}  std={normal_scores.std():.6f}")
    print(f"  tunnel : mean={tunnel_scores.mean():.6f}  std={tunnel_scores.std():.6f}")

    # Threshold sweep
    print("\n--- Threshold sweep ---")
    sweep_csv = RESULTS_DIR / "threshold_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "threshold", "TN", "FP", "FN", "TP", "F1", "accuracy"])
        sweep_rows = [(f"mean+{THRESHOLD_K}std", threshold)]
        for pct in [95, 97, 98, 99, 99.5]:
            sweep_rows.append((f"p{pct}", float(np.percentile(normal_scores, pct))))
        for name, t in sweep_rows:
            yp = (errors >= t).astype(int)
            tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
            print(f"{name:>14}: threshold={t:.6f} | FP={fp_s:4d} FN={fn_s:4d} | "
                  f"F1={f1_score(y_true, yp):.4f} | Acc={accuracy_score(y_true, yp):.4f}")
            writer.writerow([name, f"{t:.6f}", tn_s, fp_s, fn_s, tp_s,
                             f"{f1_score(y_true, yp):.4f}",
                             f"{accuracy_score(y_true, yp):.4f}"])
    print(f"Saved sweep CSV: {sweep_csv}")

    # Groups
    groups = build_groups(paths, survey_lines)
    print(f"\n--- Survey-line groups ---")
    sizes = [len(v) for v in groups.values()]
    print(f"Distinct survey lines: {len(groups)}")
    print(f"Group sizes -- min: {min(sizes)}, median: {int(np.median(sizes))}, "
          f"max: {max(sizes)}, mean: {np.mean(sizes):.1f}")
    groups_csv = RESULTS_DIR / "survey_groups.csv"
    with open(groups_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["survey_line", "n_images", "n_normal", "n_tunnel"])
        for sl in sorted(groups.keys()):
            members = groups[sl]
            oi = [t[1] for t in members]
            nn = int((y_true[oi] == 0).sum())
            nt = int((y_true[oi] == 1).sum())
            writer.writerow([sl, len(members), nn, nt])
    print(f"Saved groups CSV: {groups_csv}")

    # Aggregation comparison
    print("\n--- Window-aggregation comparison (NO-DEMOTE rule) ---")
    print("Rec+ = FNs promoted to TPs   |   Prot = TPs old rule would have demoted")

    comparison_rows = [{
        "config": "baseline (per-image, full-image scoring)",
        "M": 1, "N": 1, "recovered": 0, "protected": 0, **base,
    }]
    best_f1 = base["f1"]
    best_cfg = (1, 1)
    best_pred = raw_pred.copy()
    best_recovered = 0

    for (M, N) in WINDOW_CONFIGS:
        agg_pred, n_recovered, n_protected = apply_window_voting_no_demote(
            groups=groups,
            raw_predictions=raw_pred,
            M=M, N=N,
            scale_at_edges=SCALE_RULE_AT_EDGES,
        )
        m = compute_full_metrics(y_true, agg_pred)
        comparison_rows.append({
            "config": f"aggregated M={M}, N={N}",
            "M": M, "N": N,
            "recovered": n_recovered,
            "protected": n_protected,
            **m,
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_cfg = (M, N)
            best_pred = agg_pred
            best_recovered = n_recovered

    print(f"\n{'config':<42} {'Rec+':>5} {'Prot':>5} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'MCC':>7}")
    print("-" * 118)
    for r in comparison_rows:
        print(f"{r['config']:<42} {r['recovered']:>5} {r['protected']:>5} "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {r['tn']:>5} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {r['mcc']:>7.4f}")
    if best_cfg == (1, 1):
        print("\nNo aggregation configuration improved F1 over the per-image baseline.")
    else:
        print(f"\nBest by F1: M={best_cfg[0]}, N={best_cfg[1]} "
              f"(F1={best_f1:.4f}, recovered {best_recovered} FN)")

    compare_csv = RESULTS_DIR / "window_comparison.csv"
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "M", "N", "recovered", "protected",
            "TN", "FP", "FN", "TP",
            "accuracy", "precision", "recall", "specificity",
            "F1", "F2", "MCC", "balanced_accuracy", "kappa", "FPR", "FNR",
        ])
        for r in comparison_rows:
            writer.writerow([
                r["config"], r["M"], r["N"], r["recovered"], r["protected"],
                r["tn"], r["fp"], r["fn"], r["tp"],
                f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                f"{r['recall']:.4f}", f"{r['specificity']:.4f}",
                f"{r['f1']:.4f}", f"{r['f2']:.4f}", f"{r['mcc']:.4f}",
                f"{r['balanced_acc']:.4f}", f"{r['kappa']:.4f}",
                f"{r['fpr']:.4f}", f"{r['fnr']:.4f}",
            ])
    print(f"Saved comparison CSV: {compare_csv}")

    # Full metrics reports (baseline and best-aggregated)
    write_metrics_report(
        "baseline_per_image", y_true, raw_pred, errors, auc, ap, threshold,
        RESULTS_DIR,
        header_note=(
            "Per-image classification (no aggregation).\n"
            f"Scoring: top-{TOP_FRACTION*100:.1f}% MSE over full image.\n"
            f"Threshold rule: mean + {THRESHOLD_K}*std on normal scores."
        ),
    )
    Mb, Nb = best_cfg
    write_metrics_report(
        f"aggregated_M{Mb}_N{Nb}", y_true, best_pred, errors, auc, ap, threshold,
        RESULTS_DIR,
        header_note=(
            "Aggregated classification (no-demote spatial voting).\n"
            f"Per-image score: top-{TOP_FRACTION*100:.1f}% MSE over full image.\n"
            f"Threshold rule: mean + {THRESHOLD_K}*std on normal scores.\n"
            f"Aggregation: M={Mb}, N={Nb} within survey-line groups; "
            f"no-demote rule."
        ),
    )
    print(f"Saved metrics reports (baseline + best) in: {RESULTS_DIR}")

    # ROC / PR / CMs
    fpr_arr, tpr_arr, _ = roc_curve(y_true, errors)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_arr, tpr_arr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve (per-image, full-image top-5% MSE)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150); plt.close()

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, errors)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_arr, precision_arr, linewidth=2, label=f"AP = {ap:.3f}")
    plt.title("Precision-Recall Curve (per-image, full-image top-5% MSE)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=150); plt.close()

    save_confusion_matrix_plot(
        confusion_matrix(y_true, raw_pred, labels=[0, 1]),
        RESULTS_DIR / "confusion_matrix_baseline.png",
        title="Confusion Matrix (per-image, full-image top-5% MSE)",
    )
    save_confusion_matrix_plot(
        confusion_matrix(y_true, best_pred, labels=[0, 1]),
        RESULTS_DIR / f"confusion_matrix_aggregated_M{Mb}_N{Nb}.png",
        title=f"Confusion Matrix (aggregated, M={Mb}, N={Nb}, no-demote)",
    )

    # FN/FP folders for baseline AND best aggregated
    def dump_fn_fp(pred, label_suffix):
        FN_DIR = RESULTS_DIR / f"false_negatives_{label_suffix}"
        FP_DIR = RESULTS_DIR / f"false_positives_{label_suffix}"
        for d in (FN_DIR, FP_DIR):
            if d.exists():
                shutil.rmtree(d)
            (d / "panels").mkdir(parents=True, exist_ok=True)
            (d / "original_images").mkdir(parents=True, exist_ok=True)

        fn_c = 0
        fp_c = 0
        for i, score in enumerate(errors):
            is_fn = (y_true[i] == 1) and (pred[i] == 0)
            is_fp = (y_true[i] == 0) and (pred[i] == 1)
            if not (is_fn or is_fp):
                continue
            target_dir = FN_DIR if is_fn else FP_DIR
            tag = "FN_Tunnel" if is_fn else "FP_Normal"
            save_reconstruction_panel(paths[i], origs[i], recons[i], float(score),
                                      target_dir / "panels", tag)
            src = Path(paths[i])
            dst = target_dir / "original_images" / src.name
            shutil.copy2(src, dst)
            if is_fn:
                fn_c += 1
            else:
                fp_c += 1
        print(f"  {label_suffix}: FN={fn_c}, FP={fp_c}  (in {FN_DIR.name}, {FP_DIR.name})")

    print("\nSaving FN/FP folders...")
    dump_fn_fp(raw_pred, "baseline_per_image")
    dump_fn_fp(best_pred, f"aggregated_M{Mb}_N{Nb}")

    # Thesis grid figures for baseline AND best aggregated
    N_SAMPLES_TP_TN = 12
    GRID_COLS = 5
    FIG_SEED = 42

    def index_label(n):
        s, n = "", n + 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            s = chr(97 + rem) + s
        return s

    def save_grid_figure(indices, title, out_path):
        if not indices:
            print(f"No images for: {title} - skipped.")
            return
        indices = sorted(indices, key=lambda i: errors[i])
        cols = min(GRID_COLS, len(indices))
        rows_g = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows_g, cols, figsize=(3 * cols, 3.4 * rows_g))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for j, idx in enumerate(indices):
            axes[j].imshow(origs[idx].squeeze(), cmap="gray")
            axes[j].set_title(f"({index_label(j)}) score = {errors[idx]:.4f}", fontsize=10)
            axes[j].axis("off")
        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_path.name}  ({len(indices)} images)")

    print("\nSaving thesis grid figures...")
    rng = random.Random(FIG_SEED)

    def make_thesis_figs(pred, label_suffix, title_extra):
        fn_i = [i for i in range(len(errors)) if y_true[i] == 1 and pred[i] == 0]
        fp_i = [i for i in range(len(errors)) if y_true[i] == 0 and pred[i] == 1]
        tp_i = [i for i in range(len(errors)) if y_true[i] == 1 and pred[i] == 1]
        tn_i = [i for i in range(len(errors)) if y_true[i] == 0 and pred[i] == 0]
        tp_s = rng.sample(tp_i, k=min(N_SAMPLES_TP_TN, len(tp_i)))
        tn_s = rng.sample(tn_i, k=min(N_SAMPLES_TP_TN, len(tn_i)))
        save_grid_figure(fn_i, f"False Negatives (n={len(fn_i)}) | {title_extra}",
                         RESULTS_DIR / f"fig_false_negatives_{label_suffix}.png")
        save_grid_figure(fp_i, f"False Positives (n={len(fp_i)}) | {title_extra}",
                         RESULTS_DIR / f"fig_false_positives_{label_suffix}.png")
        save_grid_figure(tp_s, f"True Positives (sample of {len(tp_s)} from {len(tp_i)})",
                         RESULTS_DIR / f"fig_true_positives_sample_{label_suffix}.png")
        save_grid_figure(tn_s, f"True Negatives (sample of {len(tn_s)} from {len(tn_i)})",
                         RESULTS_DIR / f"fig_true_negatives_sample_{label_suffix}.png")

    make_thesis_figs(raw_pred, "baseline", f"per-image, threshold = {threshold:.4f}")
    make_thesis_figs(best_pred, f"agg_M{Mb}_N{Nb}", f"aggregated M={Mb} N={Nb} no-demote")

    # Per-image CSV with all predictions
    per_image_csv = RESULTS_DIR / "per_image_scores.csv"
    all_agg = {}
    for (M, N) in WINDOW_CONFIGS:
        agg, _, _ = apply_window_voting_no_demote(
            groups=groups, raw_predictions=raw_pred,
            M=M, N=N, scale_at_edges=SCALE_RULE_AT_EDGES,
        )
        all_agg[(M, N)] = agg
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "survey_line", "filename", "label", "score", "pred_raw"]
        for (M, N) in WINDOW_CONFIGS:
            header.append(f"pred_M{M}_N{N}")
        writer.writerow(header)
        for i in range(len(errors)):
            row = [
                paths[i], survey_lines[i], Path(paths[i]).name,
                int(y_true[i]), f"{errors[i]:.6f}", int(raw_pred[i]),
            ]
            for (M, N) in WINDOW_CONFIGS:
                row.append(int(all_agg[(M, N)][i]))
            writer.writerow(row)
    print(f"\nSaved per-image CSV: {per_image_csv}")

    print(f"\nDone. All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    run_evaluation()
