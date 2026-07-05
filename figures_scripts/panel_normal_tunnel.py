r"""
panel_normal_tunnel.py
Reconstruction panels for ALL images, saved into normal/ and tunnel/ output
folders (mirroring the separate-folder layout), with the six-way category
written on each panel below the score. No original-image copies.

Run once per METHOD ("final" or "idea2_only").

Reads images from the SEPARATE-folder layout (normal/ and tunnel/), but takes
the survey_line for each image from inference_labels.csv so that spatial
aggregation (which needs neighbours) can still run. Categories therefore
remain valid.

Six categories (each image lands in exactly one):
  TUNNEL                  true tunnel, detected at baseline           (lab1 raw1)
  RECOVERED TUNNEL        true tunnel, missed at baseline, agg-rescued (lab1 raw0 agg1)
  MISSED TUNNEL           true tunnel, missed by both                  (lab1 raw0 agg0)
  NORMAL                  true normal, correct                        (lab0 raw0 agg0)
  RECOVERED NORMAL (FP)   true normal, wrongly promoted by agg         (lab0 raw0 agg1)
  FALSE ALARM             true normal, wrongly flagged at baseline     (lab0 raw1)

Output:
  Results/Panels_NT_<METHOD>/
      normal/   <one panel png per normal image>
      tunnel/   <one panel png per tunnel image>
      verification.txt
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

M = 7
N = 3
SCALE_RULE_AT_EDGES = True


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Read images from the SEPARATE-folder layout.
SEPARATE_ROOT = Path("data") / "Inference_with_separate_folders"
NORMAL_DIR = SEPARATE_ROOT / "normal"
TUNNEL_DIR = SEPARATE_ROOT / "tunnel"

# survey_line lookup comes from the CSV.
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / f"Panels_NT_{METHOD}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TOP_FRACTION = 0.05
THRESHOLD_K = 2.5
DEPTH_START_FRACTION = 0.5

evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# DATA
# =============================================================================
def load_survey_line_map(csv_path: Path) -> Dict[str, str]:
    """filename_lower -> survey_line"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    mapping = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"].strip().lower()] = row["survey_line"].strip()
    return mapping


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_samples(survey_map: Dict[str, str]):
    """Return list of (path, survey_line, label). label from folder."""
    samples = []
    missing_survey = 0
    for label, folder in [(0, NORMAL_DIR), (1, TUNNEL_DIR)]:
        for p in collect_images(folder):
            key = p.name.lower()
            sl = survey_map.get(key)
            if sl is None:
                sl = f"_unknown_{p.parent.name}_"
                missing_survey += 1
            samples.append((p, sl, label))
    return samples, missing_survey


class SampleDataset(Dataset):
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
# AGGREGATION
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
# PANEL
# =============================================================================
def save_panel(path_str, img_np, recon_np, score, category, out_dir, depth_start_row=None):
    p = Path(path_str)
    x = img_np.squeeze()
    xh = recon_np.squeeze()
    heatmap = (x - xh) ** 2

    plt.figure(figsize=(12, 4.4))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap="gray")
    plt.title("Input"); plt.axis("off")
    if depth_start_row is not None:
        plt.axhline(depth_start_row, color="cyan", linewidth=1, linestyle="--")
    plt.subplot(1, 3, 2); plt.imshow(xh, cmap="gray")
    plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(heatmap, cmap="magma")
    # score on first line, category on second line (below the score)
    plt.title(f"score: {score:.6f}\n{category}"); plt.axis("off")
    if depth_start_row is not None:
        plt.axhline(depth_start_row, color="cyan", linewidth=1, linestyle="--")
    plt.suptitle(p.name, y=1.02, fontsize=10)
    plt.tight_layout()
    safe_stem = p.stem.replace(" ", "_")
    plt.savefig(out_dir / f"{safe_stem}.png", dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def run():
    if METHOD not in ("final", "idea2_only"):
        raise ValueError("METHOD must be 'final' or 'idea2_only'")

    print("=" * 70)
    print(f"NORMAL/TUNNEL PANELS - METHOD = {METHOD} (aggregation M={M}, N={N})")
    print("=" * 70)
    depth_start_row = int(IMG_SIZE * DEPTH_START_FRACTION) if METHOD == "final" else None
    if METHOD == "final":
        print(f"Scoring: depth-restricted top-{TOP_FRACTION*100:.0f}% MSE "
              f"(rows {depth_start_row}..{IMG_SIZE-1})")
    else:
        print(f"Scoring: full-image top-{TOP_FRACTION*100:.0f}% MSE")
    print(f"Reading images from: {SEPARATE_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    survey_map = load_survey_line_map(LABELS_CSV)
    samples, missing_survey = build_samples(survey_map)
    print(f"\nCollected {len(samples)} images from normal/ and tunnel/.")
    if missing_survey:
        print(f"WARNING: {missing_survey} images had no survey_line in the CSV; "
              f"they were placed in singleton groups and cannot be aggregated.")

    model = load_model()
    dataset = SampleDataset(samples, transform=evaluation_transform)
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

    cm_base = confusion_matrix(y_true, raw_pred, labels=[0, 1])
    cm_agg = confusion_matrix(y_true, agg_pred, labels=[0, 1])
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_a, fp_a, fn_a, tp_a = cm_agg.ravel()

    print(f"\nThreshold: {threshold:.6f}")
    print(f"Baseline   confusion  TN={tn_b} FP={fp_b} FN={fn_b} TP={tp_b}")
    print(f"Aggregated confusion  TN={tn_a} FP={fp_a} FN={fn_a} TP={tp_a}")

    # Category per image
    CAT_TUNNEL = "TUNNEL"
    CAT_REC_TUN = "RECOVERED TUNNEL"
    CAT_MISS_TUN = "MISSED TUNNEL"
    CAT_NORMAL = "NORMAL"
    CAT_REC_NORM = "RECOVERED NORMAL (FP)"
    CAT_FALSE = "FALSE ALARM"

    category = [None] * len(errors)
    counts = {CAT_TUNNEL: 0, CAT_REC_TUN: 0, CAT_MISS_TUN: 0,
              CAT_NORMAL: 0, CAT_REC_NORM: 0, CAT_FALSE: 0}
    for i in range(len(errors)):
        lab, raw, agg = int(y_true[i]), int(raw_pred[i]), int(agg_pred[i])
        if lab == 1:
            if raw == 1:
                c = CAT_TUNNEL
            elif agg == 1:
                c = CAT_REC_TUN
            else:
                c = CAT_MISS_TUN
        else:
            if raw == 1:
                c = CAT_FALSE
            elif agg == 1:
                c = CAT_REC_NORM
            else:
                c = CAT_NORMAL
        category[i] = c
        counts[c] += 1

    print("\nCategory counts:")
    for k, v in counts.items():
        print(f"  {k:<24}: {v}")

    # ---- Verification vs confusion matrices ----
    print("\n" + "=" * 70)
    print("VERIFICATION (category counts vs confusion matrices)")
    print("=" * 70)
    checks = [
        ("Baseline TP", counts[CAT_TUNNEL], tp_b),
        ("Baseline FN", counts[CAT_REC_TUN] + counts[CAT_MISS_TUN], fn_b),
        ("Baseline FP", counts[CAT_FALSE], fp_b),
        ("Baseline TN", counts[CAT_NORMAL] + counts[CAT_REC_NORM], tn_b),
        ("Aggregated TP", counts[CAT_TUNNEL] + counts[CAT_REC_TUN], tp_a),
        ("Aggregated FN", counts[CAT_MISS_TUN], fn_a),
        ("Aggregated FP", counts[CAT_REC_NORM] + counts[CAT_FALSE], fp_a),
        ("Aggregated TN", counts[CAT_NORMAL], tn_a),
    ]
    all_ok = True
    for name, fc, cmv in checks:
        ok = (fc == cmv)
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'MISMATCH'}] {name:<16} categories={fc:<6} cm={cmv}")
    total_cat = sum(counts.values())
    print(f"\n  Total images: {total_cat} (expected {len(errors)})")
    if total_cat != len(errors):
        all_ok = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "verification.txt", "w", encoding="utf-8") as f:
        f.write(f"METHOD = {METHOD}, aggregation M={M} N={N}\n")
        f.write(f"Threshold = {threshold:.6f}\n\n")
        f.write(f"Baseline   TN={tn_b} FP={fp_b} FN={fn_b} TP={tp_b}\n")
        f.write(f"Aggregated TN={tn_a} FP={fp_a} FN={fn_a} TP={tp_a}\n\n")
        for k, v in counts.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nChecks:\n")
        for name, fc, cmv in checks:
            f.write(f"  {name}: categories={fc} cm={cmv} {'OK' if fc == cmv else 'MISMATCH'}\n")
        f.write(f"\nTotal images: {total_cat} (expected {len(errors)})\n")
        f.write(f"RESULT: {'VERIFIED' if all_ok else 'FAILED'}\n")

    if all_ok:
        print("\n[VERIFIED] Category counts reproduce the confusion matrices exactly.")
    else:
        print("\n[FAILED] Mismatch - panels NOT generated. Resolve first.")
        return

    # ---- Save panels into normal/ and tunnel/ ----
    out_normal = RESULTS_DIR / "normal"
    out_tunnel = RESULTS_DIR / "tunnel"
    for d in (out_normal, out_tunnel):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("\nGenerating panels for all images...")
    saved = 0
    for i in range(len(errors)):
        out_dir = out_tunnel if int(y_true[i]) == 1 else out_normal
        save_panel(paths[i], origs[i], recons[i], float(errors[i]),
                   category[i], out_dir, depth_start_row=depth_start_row)
        saved += 1
        if saved % 200 == 0:
            print(f"  ... {saved}/{len(errors)} panels saved")

    print(f"\nSaved {saved} panels.")
    print(f"  normal panels: {counts[CAT_NORMAL] + counts[CAT_REC_NORM] + counts[CAT_FALSE]}")
    print(f"  tunnel panels: {counts[CAT_TUNNEL] + counts[CAT_REC_TUN] + counts[CAT_MISS_TUN]}")
    print(f"\nDone. Explore: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
