"""
journal_common.py
Shared settings and helpers for the journal experiments (step1 ... step10).
You do not run this file yourself; the step scripts import it.

Data: the Kaggle dataset (version 4), unzipped into  <repo>/data/
      (or set the environment variable GPR_DATA to another folder).
      It must contain labels.csv and the folders train_normal_siteA/,
      test_siteA/, unseen_normal_siteB/, unseen_normal_siteC/.
Model: gpr_best_model.pt from the GitHub release, placed in the repo root.
Output: <repo>/Results/Journal/   (nothing outside this folder is written)
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import (GPRBottleneckAE, SEED, TRAIN_RATIO, basic_transform,  # noqa: E402
                   split_train_validation)

# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT = Path(os.environ.get("GPR_DATA", REPO_ROOT / "data"))
LABELS_CSV = DATA_ROOT / "labels.csv"
SPLITS_DIR = REPO_ROOT / "splits"
MODEL_PATH = REPO_ROOT / "gpr_best_model.pt"          # the published model (seed 42)
UNSEEN_SITES = ("Site B", "Site C")                     # normal ground never used for training
OUT_DIR = REPO_ROOT / "Results" / "Journal"
MODELS_DIR = OUT_DIR / "models"

# =============================================================================
# SURVEY GEOMETRY (measured in this project)
# =============================================================================
STEP_M = 0.149          # ground distance between consecutive windows (most lines; some 0.075 m)
WINDOW_M = 2.45         # ground width of one 128x128 window
TIME_WINDOW_NS = 40.0   # time range of one window (top to bottom)
VELOCITY_M_NS = 0.085   # from hyperbola fitting in EKKO_Project

# =============================================================================
# SCORING SETTINGS (as in the paper)
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
DEPTH_START_ROW = 64          # rows 64-127 = 20-40 ns (about 0.85-1.7 m)
TOP_FRACTION = 0.05
THRESHOLD_K = 2.5
BATCH_SIZE = 64
EXPECTED_TEST = (1600, 966, 634)            # windows, normal, tunnel
MERGED_GROUPS = {"line_01": "line_01+02", "line_02": "line_01+02"}   # two segments of one scan

# extra settings tried in the sensitivity study
SWEEP_DEPTH_ROWS = (0, 32, 48, 64, 80, 96)
SWEEP_FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def row_to_depth_m(row):
    """Image row -> approximate depth in metres (two-way time, v = 0.085 m/ns)."""
    t_ns = row / IMG_SIZE * TIME_WINDOW_NS
    return VELOCITY_M_NS * t_ns / 2.0


def natkey(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def calibration_group(line):
    return MERGED_GROUPS.get(line, line)


# =============================================================================
# DATA LOADING
# =============================================================================
_ROWS = None


def label_rows():
    """All rows of the dataset's labels.csv (file, split, site, survey_line, label)."""
    global _ROWS
    if _ROWS is None:
        if not LABELS_CSV.exists():
            raise FileNotFoundError(
                f"{LABELS_CSV} not found. Download the Kaggle dataset and unzip it into "
                f"{DATA_ROOT} (or set the GPR_DATA environment variable).")
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            _ROWS = list(csv.DictReader(f))
    return _ROWS


def load_images(paths, desc):
    """Same preprocessing as training (resize to 128x128, scale to 0..1)."""
    out = np.empty((len(paths), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        out[i] = basic_transform(Image.open(p).convert("L")).squeeze(0).numpy()
        if (i + 1) % 1000 == 0 or i + 1 == len(paths):
            print(f"  {desc}: {i + 1}/{len(paths)} images loaded ({time.time() - t0:.0f}s)")
    return out


def _read_list(path):
    with open(path, encoding="utf-8") as f:
        return [DATA_ROOT / line.strip() for line in f if line.strip()]


def fixed_split(prefix, files):
    """Use splits/<prefix>_train.txt and _val.txt when present (the exact split of the
    paper); otherwise fall back to a seed-42 random 80/20 split of `files`."""
    tr, va = SPLITS_DIR / f"{prefix}_train.txt", SPLITS_DIR / f"{prefix}_val.txt"
    if tr.exists() and va.exists():
        train_files, val_files = _read_list(tr), _read_list(va)
        print(f"{prefix}: using the paper's split from {SPLITS_DIR.name}/ "
              f"({len(train_files)} train, {len(val_files)} validation)")
        return train_files, val_files
    print(f"{prefix}: {SPLITS_DIR.name}/{prefix}_*.txt not found - using a new seed-42 split "
          "(results will differ slightly from the paper)")
    return split_train_validation(files, TRAIN_RATIO, SEED)


def site_images(name):
    """All images of one site's normal data (e.g. 'Site C'), in scan order."""
    letter = name.split()[-1]
    files = [r["file"] for r in label_rows()
             if r["site"] == letter and r["split"] in ("train_val", "unseen_normal")]
    return [DATA_ROOT / f for f in sorted(files, key=natkey)]


def training_split():
    """Site A train/validation split used for every model in the paper."""
    return fixed_split("siteA", site_images("Site A"))


def load_test_set():
    """The 1,600 labelled test windows, in scan order inside each survey line."""
    rows = [r for r in label_rows() if r["split"] == "test"]
    rows.sort(key=lambda r: (natkey(r["survey_line"]), natkey(r["file"])))
    found = [DATA_ROOT / r["file"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    lines = [r["survey_line"] for r in rows]
    counts = (len(y), int((y == 0).sum()), int((y == 1).sum()))
    print(f"Test set: {counts[0]} windows (normal {counts[1]}, tunnel {counts[2]})")
    if counts != EXPECTED_TEST:
        raise RuntimeError(f"Test set has {counts}, expected {EXPECTED_TEST}. Check {LABELS_CSV}.")
    return found, y, lines


def load_other_sites():
    """Normal windows from the unseen sites (Site B and Site C).
    Returns paths, site names, survey-line groups."""
    paths, sites, groups = [], [], []
    for name in UNSEEN_SITES:
        letter = name.split()[-1]
        rows = sorted((r for r in label_rows() if r["site"] == letter and r["split"] == "unseen_normal"),
                      key=lambda r: natkey(r["file"]))
        for r in rows:
            paths.append(DATA_ROOT / r["file"])
            sites.append(name)
            groups.append(f"{name}/{r['survey_line']}")
        print(f"  unseen site '{name}': {len(rows)} normal windows")
    return paths, sites, groups


# =============================================================================
# MODEL + SCORES
# =============================================================================
def load_ae(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path} (download gpr_best_model.pt from the "
                                "GitHub release into the repository folder)")
    model = GPRBottleneckAE().to(DEVICE)
    weights = torch.load(path, map_location=DEVICE)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()
    return model


def ae_error_maps(model, images):
    maps = np.empty_like(images)
    with torch.no_grad():
        for s in range(0, len(images), BATCH_SIZE):
            x = torch.from_numpy(images[s:s + BATCH_SIZE]).unsqueeze(1).to(DEVICE)
            maps[s:s + BATCH_SIZE] = ((x - model(x)) ** 2).squeeze(1).cpu().numpy()
    return maps


def topk_mean(flat, fraction):
    k = max(1, int(fraction * flat.shape[1]))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.astype(np.float64).mean(axis=1)


def pool_scores(err_maps):
    """The three standard scores used in the paper."""
    n = len(err_maps)
    full = err_maps.reshape(n, -1)
    depth = err_maps[:, DEPTH_START_ROW:, :].reshape(n, -1)
    return {"global": full.astype(np.float64).mean(axis=1),
            "top5_full": topk_mean(full, TOP_FRACTION),
            "top5_depth": topk_mean(depth, TOP_FRACTION)}


def sweep_scores(err_maps):
    """Same idea with other depth limits and top-k fractions (sensitivity study)."""
    n = len(err_maps)
    out = {}
    for row in SWEEP_DEPTH_ROWS:
        flat = err_maps[:, row:, :].reshape(n, -1)
        for frac in SWEEP_FRACTIONS:
            out[f"top{int(round(frac * 100))}_row{row}"] = topk_mean(flat, frac)
    return out


def write_scores(path, win_paths, groups, labels, scores, group_header="survey_line"):
    names = list(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", group_header, "label"] + names)
        for i, p in enumerate(win_paths):
            w.writerow([str(p), groups[i], int(labels[i])] + [f"{scores[n][i]:.8g}" for n in names])
    print(f"Saved: {path}")


def free_gpu():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
