r"""
STEP 12 - Site B's own model, repeated over ALL Site B lines (about 3 hours: 4 models are trained)

Step 11 held out one fifth of Site B's survey lines once. This step rotates the
five groups so that every Site B line takes a turn as a test line:

  round k (k = 0..4):  TEST lines         = every 5th line, starting at line k
                       CALIBRATION lines  = every 5th line, starting at line k+1
                       TRAINING lines     = the other 3 in 5
  Round 0 corresponds to the single split reported first; all five rounds are trained here.
  Every model: same autoencoder and settings as the paper, seed 7, same number
  of training steps; the epoch with the lowest validation loss is kept.

For every Site B window (all 136 lines) we then have one honest test score from:
  - the Site A model with the Site A threshold         (as in the paper)
  - the Site A model with a threshold from Site B calibration lines
  - Site B's own model with a threshold from Site B calibration lines
False alarms are counted over all Site B lines, with 95% confidence intervals from
resampling whole survey lines (2,000 times), including the difference between the
Site A model (Site B threshold) and Site B's own model.
Each Site B model is also tested on the Site A tunnel test set.

Run from the repository folder (step 9 must be present; it provides the training loop):
    python -u journal_experiments/step12_siteB_rotation.py
If the computer restarts, run it again: finished models are skipped and an unfinished
one continues where it stopped.

Writes: Results/Journal/step12_siteB_rotation.md
        Results/Journal/step12_siteB_scores.csv
        Results/Journal/models/exp_B_own.pt, exp_B_fold1.pt ... exp_B_fold4.pt
"""

import csv
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from journal_common import (DATA_ROOT, DEVICE, MODELS_DIR, OUT_DIR, SEED, THRESHOLD_K,
                            TRAIN_RATIO, ae_error_maps, calibration_group, free_gpu, label_rows,
                            load_ae, load_images, load_test_set, natkey, pool_scores,
                            training_split)
from step9_multisite_and_datasize import BASE_AUGMENT, to_tensor, train_model
from train import split_train_validation

SITE = "Site B"
ROUNDS = 5
N_BOOT = 2000


def depth_scores(model, images):
    return pool_scores(ae_error_maps(model, images))["top5_depth"]


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    rows = sorted((r for r in label_rows() if r["site"] == "B" and r["split"] == "unseen_normal"),
                  key=lambda r: (natkey(r["survey_line"]), natkey(r["file"])))
    if not rows:
        raise SystemExit("No Site B rows in labels.csv")
    files = [DATA_ROOT / r["file"] for r in rows]
    line_of = [r["survey_line"] for r in rows]
    lines = sorted(set(line_of), key=natkey)
    idx = {l: i for i, l in enumerate(lines)}
    line_no = np.array([idx[l] for l in line_of])
    print(f"{SITE}: {len(files)} windows in {len(lines)} survey lines")

    a_train, _ = training_split()
    steps = len(a_train) * BASE_AUGMENT

    # ---- load everything once ----
    b_all = load_images(files, "all Site B windows")
    a_test_paths, y, a_lines = load_test_set()
    a_test = load_images(a_test_paths, "Site A test set")
    groups = np.array([calibration_group(l) for l in a_lines])

    model_a = load_ae()
    a_on_b = depth_scores(model_a, b_all)
    a_on_a = depth_scores(model_a, a_test)
    del model_a
    free_gpu()
    pred_a = np.zeros_like(y)
    for g in set(groups):
        fit = (groups != g) & (y == 0)
        pred_a[groups == g] = a_on_a[groups == g] >= a_on_a[fit].mean() + THRESHOLD_K * a_on_a[fit].std()
    f1_a = f1_score(y, pred_a)
    normal_a = a_on_a[y == 0]
    thr_paper = normal_a.mean() + THRESHOLD_K * normal_a.std()

    fold_of = np.full(len(files), -1)
    b_score = np.zeros(len(files))
    flag_paper = a_on_b >= thr_paper
    flag_a_recal = np.zeros(len(files), bool)
    flag_b_own = np.zeros(len(files), bool)
    fold_rows, side_rows = [], []

    for k in range(ROUNDS):
        role = line_no % ROUNDS
        test_m = role == k
        cal_m = role == (k + 1) % ROUNDS
        train_m = ~test_m & ~cal_m
        train_files = [files[i] for i in np.where(train_m)[0]]
        b_tr, b_va = split_train_validation(train_files, TRAIN_RATIO, SEED)
        augment = max(1, int(round(steps / len(b_tr))))
        name = "B_own" if k == 0 else f"B_fold{k}"
        print(f"\nRound {k}: training {int(train_m.sum())} windows, calibration {int(cal_m.sum())}, "
              f"test {int(test_m.sum())}")
        tr, va = to_tensor(b_tr, f"round {k} training"), to_tensor(b_va, f"round {k} validation")
        path = train_model(name, tr, va, augment)
        del tr, va
        free_gpu()

        model_b = load_ae(path)
        s = depth_scores(model_b, b_all)
        s_a = depth_scores(model_b, a_test)
        del model_b
        free_gpu()

        thr_a = a_on_b[cal_m].mean() + THRESHOLD_K * a_on_b[cal_m].std()
        thr_b = s[cal_m].mean() + THRESHOLD_K * s[cal_m].std()
        fold_of[test_m] = k
        b_score[test_m] = s[test_m]
        flag_a_recal[test_m] = a_on_b[test_m] >= thr_a
        flag_b_own[test_m] = s[test_m] >= thr_b
        n = int(test_m.sum())
        fold_rows.append(f"| {k} | {len(set(line_no[test_m]))} | {n} | "
                         f"{100 * flag_paper[test_m].mean():.1f}% | {100 * flag_a_recal[test_m].mean():.1f}% | "
                         f"{100 * flag_b_own[test_m].mean():.1f}% |")

        pred = np.zeros_like(y)
        for g in set(groups):
            fit = (groups != g) & (y == 0)
            pred[groups == g] = s_a[groups == g] >= s_a[fit].mean() + THRESHOLD_K * s_a[fit].std()
        side_rows.append((roc_auc_score(y, s_a), f1_score(y, pred)))

    # ---- totals over all Site B lines, with line-resampling intervals ----
    L = len(lines)
    win = np.bincount(line_no, minlength=L)
    cnt = {n: np.bincount(line_no, weights=f.astype(float), minlength=L)
           for n, f in (("paper", flag_paper), ("a_recal", flag_a_recal), ("b_own", flag_b_own))}
    rng = np.random.default_rng(42)
    boot = {n: [] for n in ("paper", "a_recal", "b_own", "diff_recal", "diff_paper")}
    for _ in range(N_BOOT):
        pick = rng.integers(0, L, L)
        w = win[pick].sum()
        r = {n: cnt[n][pick].sum() / w for n in cnt}
        for n in cnt:
            boot[n].append(r[n])
        boot["diff_recal"].append(r["a_recal"] - r["b_own"])
        boot["diff_paper"].append(r["paper"] - r["b_own"])
    ci = lambda v: "[{:.1f}%, {:.1f}%]".format(*(100 * np.percentile(v, [2.5, 97.5])))
    tot = {n: cnt[n].sum() / win.sum() for n in cnt}
    worse = {n: int((cnt[n] > cnt["b_own"]).sum()) for n in ("paper", "a_recal")}
    better = {n: int((cnt[n] < cnt["b_own"]).sum()) for n in ("paper", "a_recal")}

    with open(OUT_DIR / "step12_siteB_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "survey_line", "round", "siteA_model_score", "siteB_model_score",
                    "flag_siteA_threshold", "flag_siteA_model_siteB_threshold", "flag_siteB_model"])
        for i, p in enumerate(files):
            w.writerow([str(p), line_of[i], fold_of[i], f"{a_on_b[i]:.8g}", f"{b_score[i]:.8g}",
                        int(flag_paper[i]), int(flag_a_recal[i]), int(flag_b_own[i])])

    aucs, f1s = np.array([a for a, _ in side_rows]), np.array([b for _, b in side_rows])
    rep = ["# Step 12 - Site B's own model, all Site B lines (5 rounds)", "",
           f"Site B: {len(files)} windows in {L} survey lines; every window is a test window exactly once.", "",
           "## False alarms over all Site B lines (normal ground; lower is better)", "",
           "| Detector | Threshold from | Windows flagged | False-alarm rate [95% CI] |", "|---|---|---|---|",
           f"| Site A model | Site A (as in the paper) | {int(cnt['paper'].sum())} of {win.sum()} | "
           f"{100 * tot['paper']:.1f}% {ci(boot['paper'])} |",
           f"| Site A model | Site B calibration lines | {int(cnt['a_recal'].sum())} of {win.sum()} | "
           f"{100 * tot['a_recal']:.1f}% {ci(boot['a_recal'])} |",
           f"| **Site B's own model** | Site B calibration lines | {int(cnt['b_own'].sum())} of {win.sum()} | "
           f"**{100 * tot['b_own']:.1f}%** {ci(boot['b_own'])} |", "",
           "## Difference (Site A model minus Site B's own model; above zero = own model better)", "",
           "| Comparison | Difference [95% CI] | Lines where own model is better / worse |", "|---|---|---|",
           f"| Site A model with Site B threshold | {100 * (tot['a_recal'] - tot['b_own']):.1f} points "
           f"{ci(boot['diff_recal'])} | {worse['a_recal']} / {better['a_recal']} |",
           f"| Site A model with Site A threshold | {100 * (tot['paper'] - tot['b_own']):.1f} points "
           f"{ci(boot['diff_paper'])} | {worse['paper']} / {better['paper']} |", "",
           "## Each round", "",
           "| Round | Test lines | Test windows | Site A model, Site A threshold | Site A model, Site B threshold "
           "| Site B's own model |", "|---|---|---|---|---|---|"] + fold_rows + [
           "", "## Site B models on the Site A tunnel test set (depth band; leave-one-line-out threshold)", "",
           "| Model | AUC | F1 |", "|---|---|---|",
           f"| Site A model (paper) | {roc_auc_score(y, a_on_a):.4f} | {f1_a:.4f} |"] + [
           f"| Site B model, round {k} | {a:.4f} | {b:.4f} |" for k, (a, b) in enumerate(side_rows)] + [
           f"| Site B models, mean ± SD | {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f} | "
           f"{f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} |"]
    out = OUT_DIR / "step12_siteB_rotation.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n" + "\n".join(rep))
    print(f"\nSaved: {out}\nStep 12 finished in {(time.time() - t0) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
