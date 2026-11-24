r"""
simple_baseline_check.py - can a detector with NO learning do the same job?

It scores every window with four very simple measures, all taken only in the
depth band (rows 64-127), exactly where our method looks:

  brightness   average pixel value in the band
  contrast     spread (standard deviation) of pixel values in the band
  edges        average change between neighbouring pixels in the band
  template     difference from the AVERAGE training window (top 5% of squared
               differences in the band) - the simplest possible "model of normal"

Each is judged exactly like the autoencoder in the paper: threshold for each
survey line = mean + 2.5 SD of the normal windows of the OTHER lines; false alarms
on Sites B and C with the threshold from the Site A test normal windows.

Run from the repository folder:

    python -u journal_experiments/step11_no_learning_rules.py

Takes a few minutes (it only reads images; nothing is trained or changed).
Result: Results/Journal/simple_baseline_check.md

"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from journal_common import (DEPTH_START_ROW, OUT_DIR, THRESHOLD_K, TOP_FRACTION,
                            calibration_group, load_ae, load_images, load_other_sites,
                            load_test_set, natkey, ae_error_maps, pool_scores,
                            topk_mean, training_split)


def simple_scores(images, mean_train):
    band = images[:, DEPTH_START_ROW:, :].astype(np.float64)
    n = len(band)
    dv = np.abs(np.diff(band, axis=1)).reshape(n, -1).mean(axis=1)
    dh = np.abs(np.diff(band, axis=2)).reshape(n, -1).mean(axis=1)
    diff = (band - mean_train[DEPTH_START_ROW:, :]) ** 2
    return {"brightness": band.reshape(n, -1).mean(axis=1),
            "contrast": band.reshape(n, -1).std(axis=1),
            "edges": dv + dh,
            "template": topk_mean(diff.reshape(n, -1).astype(np.float32), TOP_FRACTION)}


def main():
    train_files, _ = training_split()
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, _ = load_other_sites()
    other_sites = np.array(other_sites)

    train = load_images(train_files, "training windows")
    mean_train = train.mean(axis=0).astype(np.float64)
    del train
    test = load_images(test_paths, "test windows")
    other = load_images(other_paths, "Sites B and C")

    S_test, S_other = simple_scores(test, mean_train), simple_scores(other, mean_train)
    model = load_ae()                                   # the paper's detector, for comparison
    S_test["AUTOENCODER (paper)"] = pool_scores(ae_error_maps(model, test))["top5_depth"]
    S_other["AUTOENCODER (paper)"] = pool_scores(ae_error_maps(model, other))["top5_depth"]

    groups = np.array([calibration_group(l) for l in lines])
    ug = sorted(set(groups), key=natkey)

    # tunnel crossings: runs of tunnel windows per line (gap of one normal window allowed)
    cross = []
    for l in dict.fromkeys(lines):
        ix = [i for i in range(len(y)) if lines[i] == l]
        runs, k = [], 0
        while k < len(ix):
            if y[ix[k]] == 1:
                a = k
                while k + 1 < len(ix) and y[ix[k + 1]] == 1:
                    k += 1
                runs.append([a, k])
            k += 1
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= 2:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        cross += [[ix[j] for j in range(a, b + 1) if y[ix[j]] == 1] for a, b in merged]

    rep = ["# Simple (no-learning) baselines in the depth band", "",
           "| Score | AUC | Recall at 1% false alarms | F1 | Missed | False positives | Crossings found "
           "| Site B false alarms | Site C false alarms |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name in S_test:
        best = None
        for sign in (1, -1):                            # a tunnel may make the band brighter OR darker
            s = sign * S_test[name]
            auc = roc_auc_score(y, s)
            if best is None or auc > best[0]:
                best = (auc, sign)
        auc, sign = best
        s, so = sign * S_test[name], sign * S_other[name]
        pred = np.zeros_like(y)
        for g in ug:
            fit = (groups != g) & (y == 0)
            pred[groups == g] = s[groups == g] >= s[fit].mean() + THRESHOLD_K * s[fit].std()
        rec1 = (s[y == 1] > np.percentile(s[y == 0], 99)).mean()
        thr = s[y == 0].mean() + THRESHOLD_K * s[y == 0].std()
        fa = {site: 100 * (so[other_sites == site] >= thr).mean() for site in ("Site B", "Site C")}
        label = name + (" (reversed: lower = tunnel)" if sign == -1 else "")
        rep.append(f"| {label} | {auc:.4f} | {rec1:.3f} | {f1_score(y, pred):.4f} "
                   f"| {int(((pred == 0) & (y == 1)).sum())} | {int(((pred == 1) & (y == 0)).sum())} "
                   f"| {sum(1 for c in cross if pred[c].any())}/{len(cross)} "
                   f"| {fa['Site B']:.1f}% | {fa['Site C']:.1f}% |")

    out = OUT_DIR / "simple_baseline_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n" + "\n".join(rep))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
