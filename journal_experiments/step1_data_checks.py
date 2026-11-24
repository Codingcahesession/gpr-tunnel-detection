r"""
STEP 1 - data checks (about 5-10 minutes). Read-only.

  1. Exact copies: is any test window (or unseen-site window) an exact copy of a
     training/validation window?
  2. Closest training window: for every test window, how similar is the most
     similar training window? (The training scans were taken on the same strips
     before the tunnel was dug, so normal test windows are expected to look
     similar - this measures how similar, honestly, for the paper.)
  3. Window overlap: how far apart consecutive windows are, measured directly
     from the images of every survey line (expected about 8 pixels = 0.149 m,
     i.e. about 94% overlap).

Run:    python -u journal_experiments/step1_data_checks.py   (from the repository folder)
Writes: Results/Journal/step1_data_checks.md
"""

import hashlib
from collections import Counter

import numpy as np

from journal_common import (OUT_DIR, STEP_M, WINDOW_M, IMG_SIZE, load_images,
                            load_other_sites, load_test_set, training_split)

REPORT = OUT_DIR / "step1_data_checks.md"
SMALL = 32                       # images are compared at 32x32 for the similarity search
MAX_SHIFT = 32                   # largest window-to-window shift searched (pixels)


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def small(images):
    n = len(images)
    f = IMG_SIZE // SMALL
    s = images.reshape(n, SMALL, f, SMALL, f).mean(axis=(2, 4)).reshape(n, -1)
    s = s - s.mean(axis=1, keepdims=True)
    s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-8
    return s.astype(np.float32)


def best_shift(a, b):
    """Size of the horizontal shift (pixels) that best aligns window b onto window a
    (either scroll direction)."""
    best, best_c = 0, -2.0
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        if s >= 0:
            x, y = a[:, s:].ravel(), b[:, :IMG_SIZE - s].ravel()
        else:
            x, y = a[:, :IMG_SIZE + s].ravel(), b[:, -s:].ravel()
        c = np.corrcoef(x, y)[0, 1]
        if c > best_c:
            best, best_c = s, c
    return abs(best), best_c


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = ["# Step 1 - data checks", ""]

    train_files, val_files = training_split()
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, _ = load_other_sites()
    seen = list(train_files) + list(val_files)

    # ---------------- 1. exact copies ----------------
    print("\n[1/3] Exact-copy check (file contents)")
    seen_hash = Counter(file_hash(p) for p in seen)
    dup_test = [p for p in test_paths if file_hash(p) in seen_hash]
    dup_other = [p for p in other_paths if file_hash(p) in seen_hash]
    within = sum(c - 1 for c in seen_hash.values() if c > 1)
    rep += ["## 1. Exact copies", "",
            f"- Training+validation windows: {len(seen)} (exact repeats among them: {within})",
            f"- Test windows that are exact copies of a training/validation window: **{len(dup_test)}** of {len(test_paths)}",
            f"- Unseen-site windows that are exact copies of a training/validation window: **{len(dup_other)}** of {len(other_paths)}", ""]
    for p in dup_test[:20]:
        rep.append(f"  - {p}")

    # ---------------- 2. closest training window ----------------
    print("\n[2/3] Closest-training-window search")
    tr = small(load_images(seen, "train+val"))
    te = small(load_images(test_paths, "test"))
    nn = np.empty(len(te), dtype=np.float32)
    for s in range(0, len(te), 256):
        nn[s:s + 256] = (te[s:s + 256] @ tr.T).max(axis=1)
    rep += ["## 2. Similarity of each test window to its closest training window",
            "(correlation at 32x32 resolution; 1.0 = identical)", "",
            "| Test windows | Median | 90th percentile | Max | Share above 0.95 | Share above 0.99 |",
            "|---|---|---|---|---|---|"]
    for name, mask in (("normal", y == 0), ("tunnel", y == 1)):
        v = nn[mask]
        rep.append(f"| {name} | {np.median(v):.3f} | {np.percentile(v, 90):.3f} | {v.max():.3f} | "
                   f"{(v > 0.95).mean():.1%} | {(v > 0.99).mean():.1%} |")
    if other_paths:
        ot = small(load_images(other_paths, "unseen sites"))
        nn_o = np.concatenate([(ot[s:s + 256] @ tr.T).max(axis=1) for s in range(0, len(ot), 256)])
        rep.append(f"| unseen sites (normal) | {np.median(nn_o):.3f} | {np.percentile(nn_o, 90):.3f} | "
                   f"{nn_o.max():.3f} | {(nn_o > 0.95).mean():.1%} | {(nn_o > 0.99).mean():.1%} |")
    rep.append("")

    # ---------------- 3. window overlap ----------------
    print("\n[3/3] Window-to-window shift along every survey line")
    test_imgs = load_images(test_paths, "test (full size)")
    shifts, per_line = [], {}
    for i in range(len(test_paths) - 1):
        if lines[i] != lines[i + 1]:
            continue
        s, c = best_shift(test_imgs[i], test_imgs[i + 1])
        if c > 0.5:                         # skip pairs that are not neighbours (gaps)
            shifts.append(s)
            per_line.setdefault(lines[i], []).append(s)
    shifts = np.array(shifts)
    if len(shifts):
        med = float(np.median(shifts))
        step_m = med / IMG_SIZE * WINDOW_M
        rep += ["## 3. Window overlap measured from the images", "",
                f"- Consecutive pairs measured: {len(shifts)}",
                f"- Median shift: **{med:.1f} pixels** (middle 50%: {np.percentile(shifts, 25):.0f}-{np.percentile(shifts, 75):.0f} px)",
                f"- = {step_m:.3f} m per step if a window is {WINDOW_M} m wide (expected {STEP_M} m)",
                f"- Overlap between consecutive windows: **{1 - med / IMG_SIZE:.1%}**",
                f"- Pairs with zero shift (repeated screenshot): {(shifts == 0).sum()}", "",
                "| Survey line | Pairs | Median shift (px) |", "|---|---|---|"]
        for line, v in per_line.items():
            rep.append(f"| {line} | {len(v)} | {np.median(v):.0f} |")
    else:
        rep += ["## 3. Window overlap", "", "No neighbouring pairs found."]

    REPORT.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[:30]))
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
