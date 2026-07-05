r"""
show_denoising_examples.py
For TRUE TUNNEL images only, show what the denoising process looks like:
    Panel 1: Clean input (what's actually on disk)
    Panel 2: Noisy input (clean + Gaussian noise, same as used in training)
    Panel 3: Reconstruction (model output when fed the NOISY input)
    Panel 4: Error heatmap (|clean - reconstruction|^2), with depth-restriction
             line marked (this script uses the eval_final scoring convention)

This uses the SAME noise level as training (NOISE_STD) so what you see here
is representative of what the model learned to undo, NOT what happens at
normal evaluation time (evaluation always uses clean, noise-free input -
this script is for illustration/explanation purposes only).

Reads from the survey-line inference layout + labels CSV (same as
eval_final.py) so true-tunnel status is known.

Output: Results/Denoising_Examples/
    One 4-panel PNG per true-tunnel image (or a sample, see SAMPLE_SIZE below)
"""

from pathlib import Path
from typing import Dict, Tuple, List
import csv
import random

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

INFERENCE_ROOT = Path("data") / "Inference_with_line_folders"
LABELS_CSV = SCRIPT_DIR / "inference_labels.csv"
MODEL_PATH = SCRIPT_DIR / "gpr_best_model.pt"

RESULTS_DIR = SCRIPT_DIR / "Results" / "Denoising_Examples"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Same noise level used during training (see untitled.py NOISE_STD).
NOISE_STD = 0.05

# Depth restriction used by eval_final.py, only for the illustrative marker line.
DEPTH_START_FRACTION = 0.5

# Set to None to generate for ALL true tunnels (could be 600+ images).
# Set to an integer (e.g. 30) to save a random reproducible sample instead.
SAMPLE_SIZE = 30
SAMPLE_SEED = 42

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


def collect_true_tunnels(inference_root: Path, label_map) -> List[Path]:
    tunnels = []
    for p in sorted(inference_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = p.name.lower()
        if key not in label_map:
            continue
        _, label = label_map[key]
        if label == 1:
            tunnels.append(p)
    return tunnels


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
# PANEL
# =============================================================================
def save_denoising_panel(path_str, clean_np, noisy_np, recon_np, out_dir,
                         depth_start_row=None):
    p = Path(path_str)
    clean = clean_np.squeeze()
    noisy = noisy_np.squeeze()
    recon = recon_np.squeeze()
    heatmap = (clean - recon) ** 2

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(clean, cmap="gray")
    plt.title("Clean Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(noisy, cmap="gray")
    plt.title(f"Noisy Input (sigma={NOISE_STD})")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(recon, cmap="gray")
    plt.title("Reconstruction\n(from noisy input)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(heatmap, cmap="magma")
    plt.title("Error Heatmap\n(clean vs reconstruction)")
    plt.axis("off")
    if depth_start_row is not None:
        plt.axhline(depth_start_row, color="cyan", linewidth=1, linestyle="--")

    plt.suptitle(f"Denoising Illustration | {p.name}", y=1.03, fontsize=10)
    plt.tight_layout()
    safe_stem = p.stem.replace(" ", "_")
    plt.savefig(out_dir / f"denoise_{safe_stem}.png", dpi=130, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def run():
    print("=" * 70)
    print("DENOISING ILLUSTRATION - TRUE TUNNEL IMAGES ONLY")
    print("=" * 70)
    print(f"Model      : {MODEL_PATH}")
    print(f"Inference  : {INFERENCE_ROOT}")
    print(f"Noise std  : {NOISE_STD}")
    print(f"Sample size: {'ALL' if SAMPLE_SIZE is None else SAMPLE_SIZE}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    label_map = load_labels_csv(LABELS_CSV)
    tunnel_paths = collect_true_tunnels(INFERENCE_ROOT, label_map)
    print(f"\nFound {len(tunnel_paths)} true-tunnel images.")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(tunnel_paths):
        rng = random.Random(SAMPLE_SEED)
        tunnel_paths = rng.sample(tunnel_paths, k=SAMPLE_SIZE)
        print(f"Sampled {len(tunnel_paths)} for illustration (seed={SAMPLE_SEED}).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    depth_start_row = int(IMG_SIZE * DEPTH_START_FRACTION)

    print("\nGenerating denoising illustration panels...")
    saved = 0
    with torch.no_grad():
        for p in tunnel_paths:
            image = Image.open(p).convert("L")
            clean = evaluation_transform(image).unsqueeze(0).to(DEVICE)  # [1,1,128,128]

            # Add the SAME noise process used in training.
            noisy = (clean + NOISE_STD * torch.randn_like(clean)).clamp(0.0, 1.0)

            # Feed the NOISY version into the model, exactly like training does.
            recon = model(noisy)

            save_denoising_panel(
                str(p),
                clean[0].cpu().numpy(),
                noisy[0].cpu().numpy(),
                recon[0].cpu().numpy(),
                RESULTS_DIR,
                depth_start_row=depth_start_row,
            )
            saved += 1
            if saved % 20 == 0:
                print(f"  ... {saved}/{len(tunnel_paths)} saved")

    print(f"\nSaved {saved} denoising illustration panels to: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
