r"""
panel_all_categories.py
Generate reconstruction panels for ALL images, tagged into six categories,
for ONE method at a time. Run it twice (change METHOD below).

METHOD options:
  "final"      -> depth-restricted top-5% MSE + M=7 N=3 aggregation
                  (matches eval_final.py)
  "idea2_only" -> full-image top-5% MSE + M=7 N=3 aggregation
                  (matches eval_idea2_only.py)

Six categories (every image lands in exactly one):
  1_tunnel_correct          TP throughout            (label 1, raw 1)
  2_recovered_tunnel        FN -> TP via aggregation  (label 1, raw 0, agg 1)
  3_missed_tunnel           FN -> FN still missed     (label 1, raw 0, agg 0)
  4_normal_correct          TN throughout            (label 0, raw 0, agg 0)
  5_recovered_normal_to_FP  TN -> FP via aggregation  (label 0, raw 0, agg 1)
  6_false_alarm             FP baseline              (label 0, raw 1)

Each image is saved as:
  panels/           3-panel input | reconstruction | error-heatmap, titled
  original_images/  untouched copy with original filename
organized by category -> survey line.

A verification block at the end cross-checks the six category counts against
the baseline and aggregated confusion matrices, so you can confirm the
categories reproduce your reported numbers exactly.

Results go to: ./Results/Panels_<METHOD>/
"""

from pathlib import Path
from typing import Dict, Tuple, List
from math import ceil
import shutil
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GPRBottleneckAE


# =============================================================================
# CHOOSE METHOD HERE
# =============================================================================
METHOD = "idea2_only"          # "final"  or  "idea2_only"

# Aggregation config (matches the M=7 N=3 used in the reports).
M = 7
N = 3
SCALE_RULE_AT_EDGES = True


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = Path("data") / "Inference_with_line_folders"
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / f"Panels_{METHOD}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TOP_FRACTION = 0.05
THRESHOLD_K = 2.5
DEPTH_START_FRACTION = 0.5   # only used when METHOD == "final"

evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# DATA
# =============================================================================
def load_labels_csv(csv_path: Path) -> Dict[str, Tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
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


def load_model():
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
# AGGREGATION (no-demote, grouped by survey line)
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
        by_line.setdefault(sl, []).append((natkey(Path(p).name), i))
    groups = {}
    for sl, items in by_line.items():
        items.sort(key=lambda t: t[0])
        groups[sl] = [(pos, orig_i) for pos, (_, orig_i) in enumerate(items)]
    return groups


def apply_window_voting_no_demote(groups, raw_predictions, M, N, scale_at_edges=True):
    aggregated = raw_predictions.copy()
    half = M // 2
    for sl, members in groups.items():
        orig_idx = [t[1] for t in members]
        preds = raw_predictions[orig_idx]
        g_len = len(members)
        for pos in range(g_len):
            lo = max(0, pos - half)
            hi = min(g_len, pos + half + 1)
            window = preds[lo:hi]
            actual = len(window)
            n_pos = int(window.sum())
            required = max(1, ceil(actual * (N / M))) if (scale_at_edges and actual < M) else N
            gi = orig_idx[pos]
            if int(raw_predictions[gi]) == 1:
                aggregated[gi] = 1
            else:
                aggregated[gi] = 1 if n_pos >= required else 0
    return aggregated


# =============================================================================
# PANEL SAVING
# =============================================================================
def save_panel(path_str, img_np, recon_np, score, out_dir, tag, depth_start_row=None):
    p = Path(path_str)
    x = img_np.squeeze()
    xh = recon_np.squeeze()
    heatmap = (x - xh) ** 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap="gray")
    plt.title(f"{tag} Input"); plt.axis("off")
    if depth_start_row is not None:
        plt.axhline(depth_start_row, color="cyan", linewidth=1, linestyle="--")
    plt.subplot(1, 3, 2); plt.imshow(xh, cmap="gray")
    plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(heatmap, cmap="magma")
    plt.title(f"score: {score:.6f}"); plt.axis("off")
    if depth_start_row is not None:
        plt.axhline(depth_start_row, color="cyan", linewidth=1, linestyle="--")
    plt.suptitle(f"{tag} | {p.name}", y=0.98)
    plt.tight_layout()
    safe_stem = p.stem.replace(" ", "_")
    plt.savefig(out_dir / f"{safe_stem}.png", dpi=120, bbox_inches="tight")
    plt.close()


def safe_line_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip() or "_unknown_line_"


# =============================================================================
# MAIN
# =============================================================================
def run():
    if METHOD not in ("final", "idea2_only"):
        raise ValueError("METHOD must be 'final' or 'idea2_only'")

    print("=" * 70)
    print(f"PANEL GENERATION - METHOD = {METHOD}  (aggregation M={M}, N={N})")
    print("=" * 70)
    depth_start_row = int(IMG_SIZE * DEPTH_START_FRACTION) if METHOD == "final" else None
    if METHOD == "final":
        print(f"Scoring: depth-restricted top-{TOP_FRACTION*100:.0f}% MSE "
              f"(rows {depth_start_row}..{IMG_SIZE-1})")
    else:
        print(f"Scoring: full-image top-{TOP_FRACTION*100:.0f}% MSE")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    label_map = load_labels_csv(LABELS_CSV)
    samples, unmatched = collect_samples(INFERENCE_ROOT, label_map)
    print(f"\nMatched {len(samples)} images. Unmatched: {len(unmatched)}")

    model = load_model()
    dataset = LabelledLineDataset(samples, transform=evaluation_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    errors, y_true, paths, survey_lines, origs, recons = [], [], [], [], [], []

    print("Running model over all images...")
    with torch.no_grad():
        for imgs, labels, path_list, sl_list in loader:
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            if METHOD == "final":
                err_map = (imgs - output) ** 2
                restricted = err_map[:, :, depth_start_row:, :]
                err = restricted.flatten(start_dim=1)
            else:
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
    threshold = float(normal_scores.mean() + THRESHOLD_K * normal_scores.std())
    raw_pred = (errors >= threshold).astype(int)

    groups = build_groups(paths, survey_lines)
    agg_pred = apply_window_voting_no_demote(groups, raw_pred, M, N, SCALE_RULE_AT_EDGES)

    # ---- Confusion matrices (for verification) ----
    cm_base = confusion_matrix(y_true, raw_pred, labels=[0, 1])
    cm_agg = confusion_matrix(y_true, agg_pred, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_a, fp_a, fn_a, tp_a = cm_agg.ravel()

    print(f"\nThreshold: {threshold:.6f}")
    print(f"Baseline   confusion  TN={tn_b} FP={fp_b} FN={fn_b} TP={tp_b}")
    print(f"Aggregated confusion  TN={tn_a} FP={fp_a} FN={fn_a} TP={tp_a}")

    # ---- Assign every image to exactly one of six categories ----
    categories = {
        "1_tunnel_correct": [],
        "2_recovered_tunnel": [],
        "3_missed_tunnel": [],
        "4_normal_correct": [],
        "5_recovered_normal_to_FP": [],
        "6_false_alarm": [],
    }
    for i in range(len(errors)):
        lab = int(y_true[i])
        raw = int(raw_pred[i])
        agg = int(agg_pred[i])
        if lab == 1:
            if raw == 1:
                categories["1_tunnel_correct"].append(i)
            elif agg == 1:
                categories["2_recovered_tunnel"].append(i)
            else:
                categories["3_missed_tunnel"].append(i)
        else:
            if raw == 1:
                categories["6_false_alarm"].append(i)
            elif agg == 1:
                categories["5_recovered_normal_to_FP"].append(i)
            else:
                categories["4_normal_correct"].append(i)

    counts = {k: len(v) for k, v in categories.items()}
    print("\nCategory counts:")
    for k, c in counts.items():
        print(f"  {k:<28}: {c}")

    # =========================================================================
    # VERIFICATION: category counts must reproduce the confusion matrices
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION (category counts vs confusion matrices)")
    print("=" * 70)

    checks = []

    # Baseline TP = tunnels with raw==1 = category 1
    checks.append(("Baseline TP", counts["1_tunnel_correct"], tp_b))
    # Baseline FN = tunnels with raw==0 = categories 2 + 3
    checks.append(("Baseline FN",
                   counts["2_recovered_tunnel"] + counts["3_missed_tunnel"], fn_b))
    # Baseline FP = normals with raw==1 = category 6
    checks.append(("Baseline FP", counts["6_false_alarm"], fp_b))
    # Baseline TN = normals with raw==0 = categories 4 + 5
    checks.append(("Baseline TN",
                   counts["4_normal_correct"] + counts["5_recovered_normal_to_FP"], tn_b))

    # Aggregated TP = tunnels ending at 1 = categories 1 + 2
    checks.append(("Aggregated TP",
                   counts["1_tunnel_correct"] + counts["2_recovered_tunnel"], tp_a))
    # Aggregated FN = tunnels ending at 0 = category 3
    checks.append(("Aggregated FN", counts["3_missed_tunnel"], fn_a))
    # Aggregated FP = normals ending at 1 = categories 5 + 6
    checks.append(("Aggregated FP",
                   counts["5_recovered_normal_to_FP"] + counts["6_false_alarm"], fp_a))
    # Aggregated TN = normals ending at 0 = category 4
    checks.append(("Aggregated TN", counts["4_normal_correct"], tn_a))

    all_ok = True
    for name, from_categories, from_cm in checks:
        ok = (from_categories == from_cm)
        all_ok = all_ok and ok
        flag = "OK " if ok else "MISMATCH"
        print(f"  [{flag}] {name:<16} categories={from_categories:<6} cm={from_cm}")

    total_cat = sum(counts.values())
    print(f"\n  Total images in categories: {total_cat}  (expected {len(errors)})")
    if total_cat != len(errors):
        all_ok = False
        print("  [MISMATCH] total image count")

    if all_ok:
        print("\n[VERIFIED] All category counts reproduce the reported confusion "
              "matrices exactly.")
    else:
        print("\n[FAILED] Category counts do NOT match. Do not trust the panels "
              "until this is resolved.")

    # Write verification to a file too
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "verification.txt", "w", encoding="utf-8") as f:
        f.write(f"METHOD = {METHOD}, aggregation M={M} N={N}\n")
        f.write(f"Threshold = {threshold:.6f}\n\n")
        f.write(f"Baseline   confusion  TN={tn_b} FP={fp_b} FN={fn_b} TP={tp_b}\n")
        f.write(f"Aggregated confusion  TN={tn_a} FP={fp_a} FN={fn_a} TP={tp_a}\n\n")
        f.write("Category counts:\n")
        for k, c in counts.items():
            f.write(f"  {k}: {c}\n")
        f.write("\nChecks (category-derived vs confusion-matrix):\n")
        for name, fc, cmv in checks:
            f.write(f"  {name}: categories={fc} cm={cmv} "
                    f"{'OK' if fc == cmv else 'MISMATCH'}\n")
        f.write(f"\nTotal images: {total_cat} (expected {len(errors)})\n")
        f.write(f"\nRESULT: {'VERIFIED' if all_ok else 'FAILED'}\n")
    print(f"\nSaved verification to: {RESULTS_DIR / 'verification.txt'}")

    # =========================================================================
    # SAVE ALL PANELS (only if verification passed)
    # =========================================================================
    if not all_ok:
        print("\nVerification failed - panels NOT generated. Fix mismatch first.")
        return

    print("\nGenerating panels for all 1600 images (this takes a while)...")
    # Wipe old run
    for sub in RESULTS_DIR.iterdir() if RESULTS_DIR.exists() else []:
        if sub.is_dir():
            shutil.rmtree(sub)

    saved = 0
    for cat_name, idxs in categories.items():
        for i in idxs:
            sl = safe_line_name(survey_lines[i])
            line_dir = RESULTS_DIR / cat_name / sl
            panels_dir = line_dir / "panels"
            orig_dir = line_dir / "original_images"
            panels_dir.mkdir(parents=True, exist_ok=True)
            orig_dir.mkdir(parents=True, exist_ok=True)

            # Human-readable tag on the panel
            tag_map = {
                "1_tunnel_correct": "TUNNEL",
                "2_recovered_tunnel": "RECOVERED TUNNEL",
                "3_missed_tunnel": "MISSED TUNNEL",
                "4_normal_correct": "NORMAL",
                "5_recovered_normal_to_FP": "RECOVERED NORMAL (FP)",
                "6_false_alarm": "FALSE ALARM",
            }
            save_panel(paths[i], origs[i], recons[i], float(errors[i]),
                       panels_dir, tag_map[cat_name], depth_start_row=depth_start_row)
            src = Path(paths[i])
            shutil.copy2(src, orig_dir / src.name)
            saved += 1
            if saved % 200 == 0:
                print(f"  ... {saved}/{len(errors)} panels saved")

    # Per-line summary CSV
    summary = {}
    for cat_name, idxs in categories.items():
        for i in idxs:
            sl = survey_lines[i]
            summary.setdefault(sl, {k: 0 for k in categories})
            summary[sl][cat_name] += 1
    with open(RESULTS_DIR / "summary_by_line.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["survey_line"] + list(categories.keys()))
        for sl in sorted(summary.keys()):
            writer.writerow([sl] + [summary[sl][k] for k in categories])

    print(f"\nSaved {saved} panels + originals.")
    print(f"Per-line summary: {RESULTS_DIR / 'summary_by_line.csv'}")
    print(f"\nDone. Explore: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
