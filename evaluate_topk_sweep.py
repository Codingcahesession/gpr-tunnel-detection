r"""
eval_topk_sweep.py
Compare top-k% reconstruction-error pooling fractions on the CURRENT,
final test set (survey-line layout + labels CSV), so the top-k selection
reported in the thesis is consistent with the dataset composition and all
other ablation numbers.

This is FULL-IMAGE scoring (no depth restriction) at each fraction, matching
what was originally compared when 5% was selected. Depth restriction is a
separate, later refinement applied on top of the chosen fraction.

Results go to: ./Results/TopK_Sweep/
"""

from pathlib import Path
from typing import Dict, Tuple
import csv

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GPRBottleneckAE


# =============================================================================
# SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = Path("data") / "Inference_with_line_folders"
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / "TopK_Sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

THRESHOLD_K = 2.5

# Fractions to compare. 5% is the one ultimately selected; the others are
# reported to justify that choice.
FRACTIONS = [0.01, 0.02, 0.05, 0.10]

evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# DATA
# =============================================================================
def load_labels_csv(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    mapping = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"].strip().lower()] = (
                row["survey_line"].strip(), int(row["label"])
            )
    return mapping


def collect_samples(inference_root: Path, label_map):
    samples, unmatched = [], []
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


class LabelledDataset(Dataset):
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
        return image, label, str(path)


def load_model():
    model = GPRBottleneckAE().to(DEVICE)
    weights = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        model.load_state_dict(weights["model_state_dict"])
    else:
        model.load_state_dict(weights)
    model.eval()
    return model


# =============================================================================
# MAIN
# =============================================================================
def run():
    print("=" * 70)
    print("TOP-K FRACTION SWEEP (full-image scoring, current final test set)")
    print("=" * 70)

    label_map = load_labels_csv(LABELS_CSV)
    samples, unmatched = collect_samples(INFERENCE_ROOT, label_map)
    n_normal = sum(1 for _, _, l in samples if l == 0)
    n_tunnel = sum(1 for _, _, l in samples if l == 1)
    print(f"Test set: {len(samples)} images (normal={n_normal}, tunnel={n_tunnel})")
    if unmatched:
        print(f"WARNING: {len(unmatched)} unmatched files skipped.")

    model = load_model()
    dataset = LabelledDataset(samples, transform=evaluation_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    all_errs = {frac: [] for frac in FRACTIONS}
    y_true = []

    print("\nRunning model once, computing all top-k fractions per image...")
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            err = ((imgs - output) ** 2).flatten(start_dim=1)  # [B, 16384]
            n_pixels = err.shape[1]

            for frac in FRACTIONS:
                k = max(1, int(frac * n_pixels))
                score = torch.topk(err, k, dim=1).values.mean(dim=1)
                all_errs[frac].extend(score.cpu().numpy())

            y_true.extend(labels.numpy())

    y_true = np.array(y_true)

    print("\n--- Top-k Fraction Comparison ---")
    print(f"{'Fraction':>10} {'AUC':>8} {'AP':>8} {'Threshold':>10} "
          f"{'TN':>6} {'FP':>5} {'FN':>5} {'TP':>6} {'F1':>8}")
    print("-" * 85)

    rows = []
    for frac in FRACTIONS:
        errors = np.array(all_errs[frac])
        normal_scores = errors[y_true == 0]
        threshold = float(normal_scores.mean() + THRESHOLD_K * normal_scores.std())
        y_pred = (errors >= threshold).astype(int)

        auc = roc_auc_score(y_true, errors)
        ap = average_precision_score(y_true, errors)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        f1 = f1_score(y_true, y_pred)

        print(f"{frac*100:>9.0f}% {auc:>8.4f} {ap:>8.4f} {threshold:>10.6f} "
              f"{tn:>6} {fp:>5} {fn:>5} {tp:>6} {f1:>8.4f}")

        rows.append({
            "fraction_pct": frac * 100, "AUC": auc, "AP": ap,
            "threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "F1": f1,
        })

    csv_path = RESULTS_DIR / "topk_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction_pct", "AUC", "AP", "threshold",
                         "TN", "FP", "FN", "TP", "F1"])
        for r in rows:
            writer.writerow([
                f"{r['fraction_pct']:.0f}", f"{r['AUC']:.4f}", f"{r['AP']:.4f}",
                f"{r['threshold']:.6f}", r["TN"], r["FP"], r["FN"], r["TP"],
                f"{r['F1']:.4f}",
            ])
    print(f"\n(Columns already include TN and TP alongside FP and FN.)")
    print(f"\nSaved: {csv_path}")

    best = max(rows, key=lambda r: r["AUC"])
    print(f"\nBest AUC: {best['fraction_pct']:.0f}% (AUC={best['AUC']:.4f})")


if __name__ == "__main__":
    run()
