"""
STEP 13 - where the variation in normal-ground scores comes from (Table 12 of the paper).
Reads only the score files written by step 2; no images or models are needed (seconds).

For the autoencoder with three poolings (global mean, top 5% of the whole window,
top 5% of the depth band), all on a logarithmic scale:
  - Separation: median tunnel score minus mean normal score of the Site A test set,
    in units of the standard deviation (SD) of the normal test scores.
  - Between-line share: fraction of the variance of the normal test scores that lies
    between survey lines (lines 1 and 2 are one scan and are grouped together).
  - Shift (spread): mean score of the pre-construction validation windows, and of the
    unseen Sites B and C, minus the mean normal test score, in SD; spread is the ratio
    of their SD to that of the normal test scores.

Run:    python -u journal_experiments/step13_score_variation.py   (from the repository folder)
Writes: Results/Journal/score_variation.md
"""

import csv
from collections import defaultdict

import numpy as np

from journal_common import OUT_DIR, calibration_group

POOLINGS = [("Global mean", "AE | global"),
            ("Top 5%, whole window", "AE | top5_full"),
            ("Top 5%, depth band", "AE | top5_depth")]


def read(name):
    with open(OUT_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    test, val, unseen = (read("scores_test_classical.csv"), read("scores_val_classical.csv"),
                         read("scores_unseen_classical.csv"))
    y = np.array([int(r["label"]) for r in test])
    groups = np.array([calibration_group(r["survey_line"]) for r in test])
    rows = ["| Pooling | Separation | Between-line share | Validation windows | Site B | Site C |",
            "|---|---|---|---|---|---|"]
    for title, col in POOLINGS:
        s = np.log([float(r[col]) for r in test])
        normal, tunnel = s[y == 0], s[y == 1]
        mu, sd = normal.mean(), normal.std()
        by_line = defaultdict(list)
        for v, g in zip(normal, groups[y == 0]):
            by_line[g].append(v)
        between = sum(len(v) * (np.mean(v) - mu) ** 2 for v in by_line.values())
        share = between / ((normal - mu) ** 2).sum()

        def shift(values):
            v = np.log(values)
            return f"{(v.mean() - mu) / sd:+.2f} ({v.std() / sd:.2f})"

        val_s = [float(r[col]) for r in val]
        site = {k: [float(r[col]) for r in unseen if r["site"] == k] for k in ("Site B", "Site C")}
        rows.append(f"| {title} | {(np.median(tunnel) - mu) / sd:.2f} | {share:.2f} | {shift(val_s)} | "
                    f"{shift(site['Site B'])} | {shift(site['Site C'])} |")
    text = "\n".join(["# Variation of normal-ground scores (autoencoder, log scale)", "",
                      f"Normal test windows: {int((y == 0).sum())} in {len(set(groups[y == 0]))} line groups; "
                      f"tunnel test windows: {int(y.sum())}.", "", *rows, ""])
    out = OUT_DIR / "score_variation.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
