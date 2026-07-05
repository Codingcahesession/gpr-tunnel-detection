r"""
organize_errors_linewise.py
Organize the error analysis for ONE aggregation config (default M=7, N=3)
into a category -> survey-line folder tree.

Reads:
    Results/Inference_Idea2_Only/per_image_scores.csv   (already generated)
    gpr_best_model.pt                                    (to regenerate panels)

Builds, inside Results/Inference_Idea2_Only/error_analysis_M{M}_N{N}/ :
    1_recovered_FN_to_TP/   <line>/ panels/ + original_images/
        tunnels the baseline MISSED but aggregation RECOVERED
    2_still_missed_FN_to_FN/ <line>/ ...
        tunnels MISSED by both baseline and aggregation
    3_new_FP_TN_to_FP/      <line>/ ...
        normals correctly passed by baseline but WRONGLY promoted by aggregation
    4_baseline_FP/          <line>/ ...
        normals the baseline wrongly flagged (aggregation leaves them, no-demote)

Each per-image entry is saved as:
    panels/           -> 3-panel input | reconstruction | error-heatmap
    original_images/  -> untouched copy with original filename

The model is only run on the images that fall into one of the four buckets,
so this is fast.
"""

from pathlib import Path
import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GPRBottleneckAE


# =============================================================================
# SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = SCRIPT_DIR / "Results" / "Inference_Idea2_Only"
PER_IMAGE_CSV = RESULTS_DIR / "per_image_scores.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

# Which aggregation config to analyse. Must be a column that exists in the CSV
# (pred_M7_N3 etc). Change these two if you want a different config.
M = 7
N = 3
PRED_COLUMN = f"pred_M{M}_N{N}"

OUT_ROOT = RESULTS_DIR / f"error_analysis_M{M}_N{N}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

evaluation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =============================================================================
# HELPERS
# =============================================================================
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


def safe_line_name(name: str) -> str:
    """Make a survey-line name safe to use as a folder name."""
    return name.replace("/", "_").replace("\\", "_").strip() or "_unknown_line_"


def save_panel(path_str, img_tensor, recon_tensor, score, out_dir, tag):
    p = Path(path_str)
    x = img_tensor.squeeze().cpu().numpy()
    xh = recon_tensor.squeeze().cpu().numpy()
    heatmap = (x - xh) ** 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(x, cmap="gray")
    plt.title(f"{tag} Input"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(xh, cmap="gray")
    plt.title("Reconstruction"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(heatmap, cmap="magma")
    plt.title(f"Heatmap | score: {score:.6f}"); plt.axis("off")
    plt.suptitle(f"{tag} | {p.name}", y=0.98)
    plt.tight_layout()
    safe_stem = p.stem.replace(" ", "_")
    out_path = out_dir / f"{tag.lower()}_{safe_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(f"ORGANIZE ERROR ANALYSIS - config M={M}, N={N}")
    print("=" * 70)

    if not PER_IMAGE_CSV.exists():
        raise FileNotFoundError(
            f"Per-image CSV not found: {PER_IMAGE_CSV}\n"
            "Run eval_idea2_only.py first."
        )

    # ---- Read the per-image CSV ----
    rows = []
    with open(PER_IMAGE_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if PRED_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Column '{PRED_COLUMN}' not in CSV. Available: {reader.fieldnames}\n"
                f"Change M/N at the top to match one of the pred_* columns."
            )
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} rows from {PER_IMAGE_CSV.name}")

    # ---- Categorise each image ----
    # label: 1=tunnel, 0=normal
    # pred_raw: baseline per-image prediction
    # agg: aggregated prediction for this config
    buckets = {
        "1_recovered_FN_to_TP": [],
        "2_still_missed_FN_to_FN": [],
        "3_new_FP_TN_to_FP": [],
        "4_baseline_FP": [],
    }

    for r in rows:
        label = int(r["label"])
        raw = int(r["pred_raw"])
        agg = int(r[PRED_COLUMN])

        if label == 1:  # true tunnel
            if raw == 0 and agg == 1:
                buckets["1_recovered_FN_to_TP"].append(r)
            elif raw == 0 and agg == 0:
                buckets["2_still_missed_FN_to_FN"].append(r)
            # raw==1 (already detected) -> not an error, skip
        else:  # true normal
            if raw == 0 and agg == 1:
                buckets["3_new_FP_TN_to_FP"].append(r)
            elif raw == 1:
                # baseline FP; under no-demote, agg stays 1 as well
                buckets["4_baseline_FP"].append(r)
            # raw==0 and agg==0 -> correct normal, skip

    print("\nCategory counts:")
    for name, items in buckets.items():
        print(f"  {name:<28}: {len(items)}")

    # ---- Collect all images we need to run through the model ----
    needed = []
    for name, items in buckets.items():
        for r in items:
            needed.append((r["path"], name, r))
    print(f"\nImages to generate panels for: {len(needed)}")

    if not needed:
        print("Nothing to save. Done.")
        return

    # ---- Prepare output tree (wipe old run) ----
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Model pass on just the needed images ----
    model = load_model()

    # Simple sequential loop (few hundred images, no need for a DataLoader)
    print("\nGenerating panels and copying originals...")
    saved = 0
    missing_files = 0
    for path_str, bucket_name, r in needed:
        src = Path(path_str)
        if not src.exists():
            missing_files += 1
            continue

        survey_line = safe_line_name(r["survey_line"])
        line_dir = OUT_ROOT / bucket_name / survey_line
        panels_dir = line_dir / "panels"
        orig_dir = line_dir / "original_images"
        panels_dir.mkdir(parents=True, exist_ok=True)
        orig_dir.mkdir(parents=True, exist_ok=True)

        # Load + run model
        image = Image.open(src).convert("L")
        x = evaluation_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = model(x)

        score = float(r["score"])
        # Tag reflects the true class for readability
        tag = "Tunnel" if int(r["label"]) == 1 else "Normal"
        save_panel(path_str, x[0], recon[0], score, panels_dir, tag)

        # Copy original with its real filename
        shutil.copy2(src, orig_dir / src.name)
        saved += 1

    print(f"\nSaved {saved} images across the category/line tree.")
    if missing_files:
        print(f"WARNING: {missing_files} source files listed in the CSV were not found on disk.")

    # ---- Write a small summary CSV: per line, per bucket counts ----
    summary_csv = OUT_ROOT / "summary_by_line.csv"
    # Build {line: {bucket: count}}
    line_bucket = {}
    for name, items in buckets.items():
        for r in items:
            sl = r["survey_line"]
            line_bucket.setdefault(sl, {k: 0 for k in buckets})
            line_bucket[sl][name] += 1

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["survey_line",
                         "recovered_FN_to_TP",
                         "still_missed_FN_to_FN",
                         "new_FP_TN_to_FP",
                         "baseline_FP"])
        for sl in sorted(line_bucket.keys()):
            b = line_bucket[sl]
            writer.writerow([sl,
                             b["1_recovered_FN_to_TP"],
                             b["2_still_missed_FN_to_FN"],
                             b["3_new_FP_TN_to_FP"],
                             b["4_baseline_FP"]])
    print(f"Saved per-line summary: {summary_csv}")

    print(f"\nDone. Explore results in:\n  {OUT_ROOT}")
    print("\nFolder meaning:")
    print("  1_recovered_FN_to_TP    = tunnels the window RESCUED (baseline missed -> now caught)")
    print("  2_still_missed_FN_to_FN = tunnels STILL missed by both")
    print("  3_new_FP_TN_to_FP       = normals the window WRONGLY flagged (the cost)")
    print("  4_baseline_FP           = normals the baseline wrongly flagged (unchanged)")


if __name__ == "__main__":
    main()
