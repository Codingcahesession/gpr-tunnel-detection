"""
STEP 10 - the remaining numbers and two figures of the paper (about 1 minute).
Reads only the score files written by steps 1-9; no images or models are needed.

  1. Isolated false-alarm events per km of normal ground (each line's real window
     spacing, measured in step 1, converts windows to metres).
  2. Recall at a 1% false-positive rate, with 95% confidence intervals from
     resampling whole survey lines (2,000 resamples).
  3. Paired differences (ours minus each other method) in F1 and in recall at 1%
     false-positive rate, with 95% intervals from the same resamples.
  4. Tunnel crossings detected: a crossing counts as found when at least one of
     its tunnel windows is flagged. Runs of tunnel windows separated by at most
     one normal window are treated as one crossing.
  5. fig_roc.png (ROC curves at low false-positive rates) and fig_datasize.png
     (F1 versus amount of training data; needs steps 4 and 9).

All thresholds: mean + 2.5 SD of the normal windows of the OTHER survey lines
(leave-one-line-out; lines 1 and 2 are one scan and are held out together).

Run:    python -u journal_experiments/step10_paper_numbers.py   (from the repository folder)
Writes: Results/Journal/paper_numbers.md, fig_roc.png, fig_datasize.png
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_curve

from journal_common import OUT_DIR, THRESHOLD_K, WINDOW_M, natkey

R = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR
OURS = "AE | top5_depth"
OTHERS = ["OCSVM | depth std", "IF | depth", "PCA | top5_depth", "PaDiM | top5_depth",
          "PatchCore | top5_depth", "AE | top5_full", "AE | global"]
NEAR_TUNNEL_WINDOWS = 17          # a false alarm within 17 windows of a tunnel window counts as "near"
N_BOOT = 2000
MERGE = {"Line 1", "Line 2", "line_01", "line_02"}


def fname(p):
    return p.replace("\\", "/").split("/")[-1]


def read(name):
    path = R / name
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    base = read("scores_test_classical.csv")
    if base is None:
        raise SystemExit(f"scores_test_classical.csv not found in {R} - run steps 2 and 3 first.")
    base.sort(key=lambda r: (natkey(r["survey_line"]), natkey(fname(r["path"]))))
    paths = [r["path"] for r in base]
    lines = [r["survey_line"] for r in base]
    y = np.array([int(r["label"]) for r in base])

    S = {}
    for f in ("scores_test_classical.csv", "scores_test_deep.csv", "scores_test_seeds.csv",
              "scores_test_exp.csv", "scores_test_ablation.csv"):
        rows = read(f)
        if rows is None:
            print(f"  ({f} not found - skipped)")
            continue
        d = {r["path"]: r for r in rows}
        for m in list(rows[0])[3:]:
            S[m] = np.array([float(d[p][m]) for p in paths])

    grp = np.array([("Line 1+2" if l.startswith("Line") else "line_01+02") if l in MERGE else l for l in lines])
    ug = sorted(set(grp), key=natkey)
    gid = np.array([ug.index(g) for g in grp])
    members = [np.where(gid == g)[0] for g in range(len(ug))]

    def lolo(s):
        p = np.zeros_like(y)
        for g in range(len(ug)):
            fit = gid != g
            n = s[fit][y[fit] == 0]
            p[gid == g] = s[gid == g] >= n.mean() + THRESHOLD_K * n.std()
        return p

    # window spacing per line, from step 1 (median shift in pixels of a 128-px window)
    shift = {}
    md = R / "step1_data_checks.md"
    if md.exists():
        for l in open(md, encoding="utf-8"):
            m = re.match(r"\| ((?:Line \d+)|(?:line_\d+)) \| \d+ \| (\d+) \|", l)
            if m:
                shift[m.group(1)] = int(m.group(2))
    step_m = np.array([(4 if shift.get(l, 8) <= 5 else 8) / 128 * WINDOW_M for l in lines])
    km = step_m[y == 0].sum() / 1000

    def events(p):
        far = near = 0
        for l in dict.fromkeys(lines):
            ix = [i for i in range(len(y)) if lines[i] == l]
            lab = y[ix]
            tp = np.where(lab == 1)[0]
            k = 0
            while k < len(ix):
                if p[ix[k]] and lab[k] == 0:
                    a = k
                    while k + 1 < len(ix) and p[ix[k + 1]] and lab[k + 1] == 0:
                        k += 1
                    d = np.min(np.abs(tp[:, None] - np.arange(a, k + 1)[None, :])) if len(tp) else 999
                    if d <= NEAR_TUNNEL_WINDOWS:
                        near += 1
                    else:
                        far += 1
                k += 1
        return near, far

    def rec1(s, idx):
        yy, ss = y[idx], s[idx]
        return (ss[yy == 1] > np.percentile(ss[yy == 0], 99)).mean()

    # crossings
    cross = []
    for l in dict.fromkeys(lines):
        ix = [i for i in range(len(y)) if lines[i] == l]
        lab = [y[i] for i in ix]
        runs, k = [], 0
        while k < len(ix):
            if lab[k] == 1:
                a = k
                while k + 1 < len(ix) and lab[k + 1] == 1:
                    k += 1
                runs.append([a, k])
            k += 1
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= 2:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        cross += [(l, [ix[j] for j in range(a, b + 1) if lab[j] == 1]) for a, b in merged]

    methods = [m for m in [OURS] + OTHERS if m in S]
    P = {m: lolo(S[m]) for m in methods}
    allidx = np.arange(len(y))
    rng = np.random.default_rng(42)
    G = len(ug)
    boot = {m: {"r1": [], "dF": [], "dR": []} for m in methods}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[g] for g in rng.integers(0, G, G)])
        if not 0 < y[idx].sum() < len(idx):
            continue
        fo, ro = f1_score(y[idx], P[OURS][idx]), rec1(S[OURS], idx)
        boot[OURS]["r1"].append(ro)
        for m in methods[1:]:
            r = rec1(S[m], idx)
            boot[m]["r1"].append(r)
            boot[m]["dF"].append(fo - f1_score(y[idx], P[m][idx]))
            boot[m]["dR"].append(ro - r)

    ci = lambda v: "[{:.3f}, {:.3f}]".format(*np.percentile(v, [2.5, 97.5]))
    rep = ["# Paper numbers (step 10)", "",
           f"Test windows: {len(y)} ({int((y == 0).sum())} normal, {int(y.sum())} tunnel); "
           f"survey-line groups: {G}; tunnel crossings: {len(cross)}; "
           f"normal ground covered: {km * 1000:.0f} m", "",
           "| Method | F1 | Crossings found | False alarms near tunnel | Isolated false alarms | per km "
           "| Recall at 1% FPR [95% CI] | F1 difference, ours minus this [95% CI] "
           "| Recall@1% difference [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        near, far = events(P[m])
        found = sum(1 for _, ix in cross if P[m][ix].any())
        missed = [l for l, ix in cross if not P[m][ix].any()]
        rep.append(f"| {m.replace(' | ', ' / ')} | {f1_score(y, P[m]):.4f} | {found}/{len(cross)}"
                   f"{' (missed ' + ', '.join(missed) + ')' if missed else ''} | {near} | {far} | {far / km:.0f} "
                   f"| {rec1(S[m], allidx):.3f} {ci(boot[m]['r1'])} "
                   f"| {ci(boot[m]['dF']) if m != OURS else '-'} | {ci(boot[m]['dR']) if m != OURS else '-'} |")

    extra = [m for m in S if m.endswith("top5_depth") and (" seed" in m or m.startswith("ABL") or m.startswith("EXP"))]
    if extra:
        rep += ["", "## Other trained models (depth band, top 5%)", "",
                "| Model | F1 | Crossings found |", "|---|---|---|"]
        for m in extra:
            p = lolo(S[m])
            rep.append(f"| {m.split(' | ')[0]} | {f1_score(y, p):.4f} | "
                       f"{sum(1 for _, ix in cross if p[ix].any())}/{len(cross)} |")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#52514e"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    sets = [[("Average error, whole window", "AE | global", C[5], ":"),
             ("Top-5%, whole window", "AE | top5_full", C[1], "--"),
             ("Top-5%, depth band (ours)", OURS, C[0], "-")],
            [("Autoencoder (ours)", OURS, C[0], "-"), ("One-Class SVM", "OCSVM | depth std", C[1], "--"),
             ("Isolation Forest", "IF | depth", C[2], "-."), ("PCA", "PCA | top5_depth", C[3], (0, (5, 1, 1, 1))),
             ("PaDiM", "PaDiM | top5_depth", C[4], ":")]]
    titles = ["(a) Autoencoder: how the error map is pooled", "(b) All methods with the depth band"]
    for a, ss, t in zip(ax, sets, titles):
        for lab, m, c, ls in ss:
            if m in S:
                fpr, tpr, _ = roc_curve(y, S[m])
                a.plot(fpr * 100, tpr, color=c, ls=ls, lw=1.6, label=lab)
        a.axvline(1, color="#a3a29d", lw=0.8)
        a.set_xlim(0, 10)
        a.set_ylim(0.4, 1.005)
        a.grid(True, color="#e4e3df", lw=0.6)
        a.set_title(t, fontsize=8.5, loc="left")
        a.set_xlabel("False-positive rate (%)")
        a.set_ylabel("True-positive rate (recall)")
        a.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "fig_roc.png", dpi=300)
    plt.close(fig)

    size_cols = [("EXP data 25%", 25), ("EXP data 50%", 50), ("AE seed7", 100)]   # all trained with seed 7
    if all(f"{c} | top5_depth" in S for c, _ in size_cols):
        frac = [v for _, v in size_cols]
        fd = [f1_score(y, lolo(S[f"{c} | top5_depth"])) for c, _ in size_cols]
        ff = [f1_score(y, lolo(S[f"{c} | top5_full"])) for c, _ in size_cols]
        rep += ["", "## Training-data size (seed 7)", "", "| Share of Site A training data | F1 depth band | F1 whole window |",
                "|---|---|---|"] + [f"| {a}% | {b:.4f} | {c:.4f} |" for a, b, c in zip(frac, fd, ff)]
        fig, a = plt.subplots(figsize=(3.5, 2.6))
        a.plot(frac, fd, "-o", color=C[0], lw=2, ms=5, label="Depth band (top 5%)")
        a.plot(frac, ff, "--s", color=C[1], lw=2, ms=5, label="Whole window (top 5%)")
        for x, v in zip(frac, fd):
            a.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
        for x, v in zip(frac, ff):
            a.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6.5)
        a.set_xticks(frac)
        a.set_xticklabels([f"{v}%" for v in frac])
        a.set_xlim(15, 110)
        a.set_ylim(0.65, 1.0)
        a.set_xlabel("Share of Site A training windows used")
        a.set_ylabel("F1 (leave-one-line-out threshold)")
        a.grid(True, axis="y", color="#e4e3df", lw=0.6)
        a.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(R / "fig_datasize.png", dpi=300)
        plt.close(fig)

    (R / "paper_numbers.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))
    print(f"\nSaved: {R / 'paper_numbers.md'}, fig_roc.png" + (", fig_datasize.png" if extra else ""))


if __name__ == "__main__":
    main()
