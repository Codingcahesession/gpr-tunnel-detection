r"""
evaluate.py
Reproduces the paper's headline result with the published model in one command.

  1. Scores every Site A test window: squared reconstruction error, averaged over
     the largest 5% of pixels in rows 64-127 (20-40 ns, about 0.85-1.7 m depth).
  2. Sets the threshold for each survey line from the OTHER lines' normal windows
     (mean + 2.5 SD; leave-one-line-out, lines 1 and 2 held out together because
     they are two segments of one scan).
  3. Reports AUC, average precision, precision, recall, F1, the confusion counts
     and how many of the 29 tunnel crossings were detected.
  4. With --unseen: false-alarm rate on the unseen normal ground of Sites B and C,
     using one threshold from all Site A test normal windows.

Expected with gpr_best_model.pt: AUC 0.9936, F1 0.972, 18 missed tunnel windows,
18 false positives, 29/29 crossings.
(Confidence intervals, baselines and all other experiments: journal_experiments/.)

Data: the Kaggle dataset (version 4) unzipped into data/ (or set GPR_DATA, or pass --data).
Run:  python -u evaluate.py            or     python -u evaluate.py --unseen
Writes: Results/Evaluation/scores_test.csv, scores_unseen.csv, summary.txt
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from train import GPRBottleneckAE, basic_transform

SCRIPT_DIR = Path(__file__).resolve().parent
DEPTH_START_ROW = 64       # rows 64-127 of 128 = 20-40 ns
TOP_FRACTION = 0.05        # top 5% of pixels in the band
THRESHOLD_K = 2.5          # threshold = mean + 2.5 SD of normal scores
BATCH = 64
MERGE = {"line_01": "line_01+02", "line_02": "line_01+02"}


def natkey(t):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(t))]


def depth_band_scores(model, files, device, desc):
    scores = np.empty(len(files))
    with torch.no_grad():
        for s in range(0, len(files), BATCH):
            x = torch.stack([basic_transform(Image.open(p).convert("L")) for p in files[s:s + BATCH]]).to(device)
            err = ((x - model(x)) ** 2)[:, 0, DEPTH_START_ROW:, :].reshape(len(x), -1)
            k = max(1, int(TOP_FRACTION * err.shape[1]))
            scores[s:s + len(x)] = err.topk(k, dim=1).values.double().mean(dim=1).cpu().numpy()
            print(f"\r  {desc}: {min(s + BATCH, len(files))}/{len(files)}", end="", flush=True)
    print()
    return scores


def leave_one_line_out(scores, y, lines):
    groups = np.array([MERGE.get(l, l) for l in lines])
    pred, thr = np.zeros(len(y), dtype=int), {}
    for g in sorted(set(groups), key=natkey):
        fit = (groups != g) & (y == 0)
        thr[g] = scores[fit].mean() + THRESHOLD_K * scores[fit].std()
        pred[groups == g] = scores[groups == g] >= thr[g]
    return pred, thr


def crossings(y, lines):
    """Runs of tunnel windows in each line; runs separated by one normal window are one crossing."""
    out = []
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
        out += [(l, [ix[j] for j in range(a, b + 1) if y[ix[j]] == 1]) for a, b in merged]
    return out


def summarize(scores, y, lines):
    pred, _ = leave_one_line_out(scores, y, lines)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    cr = crossings(y, lines)
    found = [l for l, ix in cr if pred[ix].any()]
    missed = [l for l, ix in cr if not pred[ix].any()]
    return [f"Test windows: {len(y)} ({int((y == 0).sum())} normal, {int(y.sum())} tunnel)",
            f"AUC:       {roc_auc_score(y, scores):.4f}",
            f"AP:        {average_precision_score(y, scores):.4f}",
            "Leave-one-line-out threshold (mean + 2.5 SD of the other lines' normal windows):",
            f"Precision: {precision_score(y, pred):.4f}",
            f"Recall:    {recall_score(y, pred):.4f}",
            f"F1:        {f1_score(y, pred):.4f}",
            f"Tunnel windows missed (FN): {fn}   False positives (FP): {fp}   TP: {tp}   TN: {tn}",
            f"Tunnel crossings detected: {len(found)}/{len(cr)}" + (f" (missed: {', '.join(missed)})" if missed else "")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GPR_DATA", str(SCRIPT_DIR / "data")),
                    help="folder with labels.csv (Kaggle v4); default: GPR_DATA or ./data")
    ap.add_argument("--model", default=str(SCRIPT_DIR / "gpr_best_model.pt"))
    ap.add_argument("--unseen", action="store_true", help="also score unseen Sites B and C")
    args = ap.parse_args()
    data, out = Path(args.data), SCRIPT_DIR / "Results" / "Evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with open(data / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPRBottleneckAE().to(device)
    w = torch.load(args.model, map_location=device)
    model.load_state_dict(w.get("model_state_dict", w) if isinstance(w, dict) else w)
    model.eval()
    print(f"Model: {args.model}   Device: {device}")

    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: (natkey(r["survey_line"]), natkey(r["file"])))
    y = np.array([int(r["label"]) for r in test])
    lines = [r["survey_line"] for r in test]
    s = depth_band_scores(model, [data / r["file"] for r in test], device, "test windows")
    pred, _ = leave_one_line_out(s, y, lines)
    with open(out / "scores_test.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "survey_line", "label", "score", "flagged"])
        wr.writerows([r["file"], r["survey_line"], r["label"], f"{v:.8g}", int(p)] for r, v, p in zip(test, s, pred))
    report = summarize(s, y, lines)

    if args.unseen:
        thr = s[y == 0].mean() + THRESHOLD_K * s[y == 0].std()
        report += ["", f"Unseen normal ground (threshold {thr:.6g} from all Site A test normal windows):"]
        with open(out / "scores_unseen.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "site", "survey_line", "score", "flagged"])
            for site in ("B", "C"):
                sub = [r for r in rows if r["split"] == "unseen_normal" and r["site"] == site]
                if not sub:
                    continue
                su = depth_band_scores(model, [data / r["file"] for r in sub], device, f"Site {site}")
                wr.writerows([r["file"], site, r["survey_line"], f"{v:.8g}", int(v >= thr)] for r, v in zip(sub, su))
                report.append(f"  Site {site}: {int((su >= thr).sum())} of {len(su)} windows flagged "
                              f"({100 * (su >= thr).mean():.1f}% false alarms)")

    (out / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
