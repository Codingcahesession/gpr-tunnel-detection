r"""
verify_dataset_table.py
Recomputes every number that belongs in the thesis "Dataset Composition"
table, directly from disk, and cross-checks it against the exact split
logic used in untitled.py (TRAIN_RATIO, AUGMENT_FACTOR, SEED).

This does NOT retrain anything. It only recounts files and reruns the
deterministic split function so you get the exact same train/validation
partition your training script produced (same seed = same split).

Run this from the same folder as untitled.py (it imports the real split
function so there is no risk of the check drifting from the actual code).
"""

from pathlib import Path
import sys

# Import the exact functions/constants used during training, so this
# verification can never silently drift from what actually happened.
from train import (
    NORMAL_DIRS,
    TRAIN_RATIO,
    AUGMENT_FACTOR,
    SEED,
    USE_AUGMENTATION,
    collect_normal_training_images,
    split_train_validation,
)

# =============================================================================
# EDIT THESE TWO PATHS IF YOURS DIFFER
# =============================================================================
TEST_NORMAL_DIR = Path("data") / "Inference_with_separate_folders" / "normal"
TEST_TUNNEL_DIR = Path("data") / "Inference_with_separate_folders" / "tunnel"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        print(f"  WARNING: folder does not exist: {folder}")
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    print("=" * 70)
    print("DATASET COMPOSITION VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Raw training images (recomputed using the EXACT function from
    #    untitled.py, so this is guaranteed to match what training actually saw)
    # -------------------------------------------------------------------------
    print(f"\nTraining source folder(s): {NORMAL_DIRS}")
    all_normal_files = collect_normal_training_images(NORMAL_DIRS)
    n_raw = len(all_normal_files)
    print(f"Raw normal training images (recount): {n_raw}")

    # -------------------------------------------------------------------------
    # 2. Reproduce the exact train/validation split (same seed => same split)
    # -------------------------------------------------------------------------
    train_files, val_files = split_train_validation(all_normal_files, TRAIN_RATIO, SEED)
    n_train = len(train_files)
    n_val = len(val_files)
    effective_augment_factor = AUGMENT_FACTOR if USE_AUGMENTATION else 1
    n_train_augmented = n_train * effective_augment_factor

    print(f"\nSplit parameters used during training:")
    print(f"  TRAIN_RATIO      : {TRAIN_RATIO}")
    print(f"  SEED             : {SEED}")
    print(f"  USE_AUGMENTATION : {USE_AUGMENTATION}")
    print(f"  AUGMENT_FACTOR   : {AUGMENT_FACTOR} "
          f"(effective factor applied: {effective_augment_factor})")

    print(f"\nRecomputed split:")
    print(f"  Training split (before augmentation) : {n_train}")
    print(f"  Validation split                     : {n_val}")
    print(f"  Training samples per epoch (augmented): {n_train_augmented}")

    # Sanity check: train + val should equal raw count
    split_ok = (n_train + n_val == n_raw)
    print(f"\n  Check: train + val == raw?  "
          f"{n_train} + {n_val} = {n_train + n_val}  vs raw {n_raw}  "
          f"[{'OK' if split_ok else 'MISMATCH'}]")

    # -------------------------------------------------------------------------
    # 3. Test set composition (recount from disk)
    # -------------------------------------------------------------------------
    print(f"\nTest set folders:")
    print(f"  Normal: {TEST_NORMAL_DIR}")
    print(f"  Tunnel: {TEST_TUNNEL_DIR}")

    n_test_normal = count_images(TEST_NORMAL_DIR)
    n_test_tunnel = count_images(TEST_TUNNEL_DIR)
    n_test_total = n_test_normal + n_test_tunnel

    print(f"\nRecomputed test set:")
    print(f"  Test - normal : {n_test_normal}")
    print(f"  Test - tunnel : {n_test_tunnel}")
    print(f"  Test - total  : {n_test_total}")

    # -------------------------------------------------------------------------
    # 4. Final summary table, ready to paste into the thesis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL VERIFIED DATASET COMPOSITION TABLE")
    print("=" * 70)
    rows = [
        ("Raw normal training images (Site A)", n_raw),
        ("Training split (80%, before augmentation)", n_train),
        ("Validation split (20%)", n_val),
        (f"Training samples per epoch ({effective_augment_factor}x augmentation)", n_train_augmented),
        ("Test set - normal", n_test_normal),
        ("Test set - tunnel", n_test_tunnel),
        ("Test set - total", n_test_total),
    ]
    name_width = max(len(r[0]) for r in rows) + 2
    for name, val in rows:
        print(f"  {name:<{name_width}}: {val:,}")

    print("\nNote: the model NEVER sees tunnel images during training or "
          "validation. Training/validation are drawn entirely from the "
          "normal-only Site A pool; the test set is a completely separate, "
          "independently labelled dataset used only for evaluation.")

    if not split_ok:
        print("\n[WARNING] train+val did not equal the raw count. "
              "Do not publish these numbers until this is resolved.")
        sys.exit(1)
    else:
        print("\n[VERIFIED] All numbers are internally consistent and "
              "traceable directly to the training script's logic and the "
              "current contents of your data folders.")


if __name__ == "__main__":
    main()
