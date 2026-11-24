r"""
STEP 5 - final analysis (about 2-5 minutes). Uses only the score files saved by
steps 2-4; no model is run. Works with whatever steps have finished (step 2 is
required; steps 3 and 4 are added automatically when their files exist).

What it produces, in Results/Journal/final_report.md:
  1. Main table: every method, threshold set fairly (leave-one-line-out: each
     survey line is judged by a threshold built only from the OTHER lines'
     normal windows), with 95% confidence intervals from resampling whole
     survey lines.
  2. Is ours significantly better? (AUC difference, line-resampled CIs)
  3. Tunnel passes: on how many of the 30 passes over the tunnel was it caught,
     and where the missed windows sit (edges of a pass or middle).
  4. False alarms per km of normal ground (isolated ones vs next to the tunnel).
  5. Threshold protocols for our method (fair ones vs the original paper's).
  6. Within-line detection (tunnel vs normal windows of the SAME line, same
     processing setting), so the processing choice cannot explain the result.
  7. Sensitivity: other depth limits and top-k fractions.
  8. Training-seed variation (published model + 2 retrained models).
  9. Unseen sites B and C: false alarms on ground the model never saw.

Run:    python -u journal_experiments/step5_final_analysis.py   (from the repository folder)
Writes: Results/Journal/final_report.md, per_pass_detection.csv, within_line_auc.csv,
        main_table.csv
"""

import csv
import math
from collections import OrderedDict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from journal_common import (DEPTH_START_ROW, OUT_DIR, STEP_M, SWEEP_DEPTH_ROWS,
                            SWEEP_FRACTIONS, THRESHOLD_K, WINDOW_M,
                            calibration_group, natkey, row_to_depth_m)

OURS = "AE | top5_depth"
N_BOOT = 2000
SEED = 42
NEAR_WINDOWS = math.ceil(WINDOW_M / STEP_M)     # a normal window this close to a tunnel window can contain hyperbola tails
EDGE_WINDOWS = 3                                # "edge of a pass" = within 3 windows (~0.45 m) of its start/end
REPORT = OUT_DIR / "final_report.md"


def show(m):
    """Method name for markdown tables (a '|' would break the table)."""
    return m.replace(" | ", " / ")


# =============================================================================
# LOADING
# =============================================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_scores(kind, required=False):
    """Merge Results/Journal/scores_<kind>_*.csv by window path. Only windows present in
    every file are kept (e.g. if a folder was moved after an earlier step was run)."""
    files = sorted(OUT_DIR.glob(f"scores_{kind}_*.csv"))
    files = [f for f in files if not f.name.endswith("_sweep.csv")]
    if not files:
        if required:
            raise FileNotFoundError(f"No scores_{kind}_*.csv in {OUT_DIR}. Run step 2 first.")
        return None
    tables = []
    for f in files:
        rows = read_csv(f)
        gcol = [c for c in rows[0] if c not in ("path", "label")][0]
        tables.append((f, rows, gcol, {r["path"]: r for r in rows}))
    common = set(tables[0][3])
    for _, _, _, by in tables[1:]:
        common &= set(by)
    base_rows = [r for r in tables[0][1] if r["path"] in common]
    group_col = tables[0][2]
    scores = OrderedDict()
    for f, rows, gcol, by in tables:
        dropped = len(rows) - len(common)
        for m in [c for c in rows[0] if c not in ("path", "label", gcol)]:
            scores[m] = np.array([float(by[r["path"]][m]) for r in base_rows])
        print(f"  read {f.name}" + (f"  ({dropped} windows not in every file - left out)" if dropped else ""))
    paths = [r["path"] for r in base_rows]
    groups = [r[group_col] for r in base_rows]
    y = np.array([int(r["label"]) for r in base_rows])
    return paths, groups, y, scores


def order_by_line(paths, lines):
    names = [p.replace("\\", "/").split("/")[-1] for p in paths]
    return sorted(range(len(paths)), key=lambda i: (natkey(lines[i]), natkey(names[i])))


# =============================================================================
# METRICS
# =============================================================================
def thr_rule(normals, rule, k=THRESHOLD_K):
    return normals.mean() + k * normals.std() if rule == "musigma" else np.percentile(normals, 99)


def grouped_pred(s, y, gid, splits, rule="musigma", k=THRESHOLD_K):
    pred = np.zeros_like(y)
    for fit, held in splits:
        pred[held] = (s[held] >= thr_rule(s[fit][y[fit] == 0], rule, k)).astype(int)
    return pred


def basic(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred), "TP": tp, "FP": fp, "FN": fn}


def runs(mask):
    """(start, end) of consecutive True stretches."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


class Lines:
    """Window order along each survey line, tunnel passes, distance bookkeeping."""

    def __init__(self, order, lines, y):
        self.per_line = OrderedDict()
        for i in order:
            self.per_line.setdefault(lines[i], []).append(i)
        self.passes = []
        for line, idx in self.per_line.items():
            for a, b in runs([y[i] == 1 for i in idx]):
                self.passes.append((line, idx[a:b + 1]))
        self.normal_km = (y == 0).sum() * STEP_M / 1000.0

    def pass_detection(self, pred):
        return [bool(pred[np.array(ix)].any()) for _, ix in self.passes]

    def fp_events(self, pred, y):
        near = far = 0
        for line, idx in self.per_line.items():
            lab = np.array([y[i] for i in idx])
            tunnel_pos = np.where(lab == 1)[0]
            for a, b in runs([(pred[i] == 1 and y[i] == 0) for i in idx]):
                if len(tunnel_pos) and np.min(np.abs(tunnel_pos[:, None] - np.arange(a, b + 1)[None, :])) <= NEAR_WINDOWS:
                    near += 1
                else:
                    far += 1
        return near, far

    def fn_positions(self, pred):
        edge = middle = 0
        for _, ix in self.passes:
            for k, i in enumerate(ix):
                if pred[i] == 0:
                    if min(k, len(ix) - 1 - k) < EDGE_WINDOWS:
                        edge += 1
                    else:
                        middle += 1
        return edge, middle


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading score files from", OUT_DIR)
    paths, lines, y, test = load_scores("test", required=True)
    val = load_scores("val")
    unseen = load_scores("unseen")
    sweep = None
    if (OUT_DIR / "scores_test_sweep.csv").exists():
        rows = read_csv(OUT_DIR / "scores_test_sweep.csv")
        by = {r["path"]: r for r in rows}
        sweep = {m: np.array([float(by[p][m]) for p in paths])
                 for m in rows[0] if m not in ("path", "label", "survey_line")}

    methods = [m for m in test if " seed" not in m and not m.startswith(("ABL ", "EXP "))]
    abl_methods = [m for m in test if m.startswith("ABL ") and m.endswith("top5_depth")]
    seed_methods = [m for m in test if " seed" in m and m.endswith("top5_depth")]
    if OURS not in test:
        raise RuntimeError(f"'{OURS}' not found in the score files.")

    groups = [calibration_group(l) for l in lines]
    ug = sorted(set(groups), key=natkey)
    gid = np.array([ug.index(g) for g in groups])
    members = [np.where(gid == g)[0] for g in range(len(ug))]
    lolo = [(np.where(gid != g)[0], np.where(gid == g)[0]) for g in range(len(ug))]
    k5 = list(GroupKFold(n_splits=5).split(y, y, groups=gid))
    L = Lines(order_by_line(paths, lines), lines, y)
    n_pass = len(L.passes)
    print(f"{len(y)} windows | {len(set(lines))} survey lines ({len(ug)} calibration groups) | "
          f"{n_pass} tunnel passes | {L.normal_km:.3f} km of normal ground")

    # ---------- predictions + point metrics ----------
    pred_lolo = {m: grouped_pred(test[m], y, gid, lolo) for m in list(test)}
    point = {}
    for m in test:
        r = basic(y, pred_lolo[m])
        near, far = L.fp_events(pred_lolo[m], y)
        thr1 = np.percentile(test[m][y == 0], 99)
        r.update({"AUC": roc_auc_score(y, test[m]), "AP": average_precision_score(y, test[m]),
                  "passes": sum(L.pass_detection(pred_lolo[m])),
                  "fa_far_km": far / L.normal_km, "fa_all_km": (near + far) / L.normal_km,
                  "fa_near": near, "fa_far": far,
                  "rec_at_1fpr": float((test[m][y == 1] > thr1).mean())})
        point[m] = r

    # ---------- line bootstrap (whole survey lines resampled) ----------
    # Fast exact form: a resample is a vector of how many times each line was drawn,
    # so AUC and F1 can be computed from per-line pair counts instead of re-sorting.
    print(f"Bootstrap over survey lines ({N_BOOT} resamples) ...")
    G = len(ug)
    rng = np.random.default_rng(SEED)
    pos_g = np.array([int(y[members[g]].sum()) for g in range(G)])
    neg_g = np.array([int((y[members[g]] == 0).sum()) for g in range(G)])
    counts = []
    while len(counts) < N_BOOT:
        c = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        if (c @ pos_g) > 0 and (c @ neg_g) > 0:
            counts.append(c)
    C = np.array(counts)                                  # (N_BOOT, G)
    onehot_pos = np.zeros((int(y.sum()), G))
    onehot_neg = np.zeros((int((y == 0).sum()), G))
    onehot_pos[np.arange(len(onehot_pos)), gid[y == 1]] = 1
    onehot_neg[np.arange(len(onehot_neg)), gid[y == 0]] = 1

    def boot_auc(s):
        sp, sn = s[y == 1], s[y == 0]
        wins = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
        pair = onehot_pos.T @ wins @ onehot_neg           # (G, G) wins between line g (tunnel) and line h (normal)
        return np.einsum("bg,gh,bh->b", C, pair, C) / ((C @ pos_g) * (C @ neg_g))

    def boot_f1(pred):
        tp = np.array([int(((pred == 1) & (y == 1))[members[g]].sum()) for g in range(G)])
        fp = np.array([int(((pred == 1) & (y == 0))[members[g]].sum()) for g in range(G)])
        fn = np.array([int(((pred == 0) & (y == 1))[members[g]].sum()) for g in range(G)])
        TP, FP, FN = C @ tp, C @ fp, C @ fn
        den = 2 * TP + FP + FN
        return np.where(den > 0, 2 * TP / np.maximum(den, 1e-12), 0.0)

    ours_b = boot_auc(test[OURS])
    ci = {}
    for m in test:
        a_b = ours_b if m == OURS else boot_auc(test[m])
        ci[m] = {"AUC": np.percentile(a_b, [2.5, 97.5]),
                 "F1": np.percentile(boot_f1(pred_lolo[m]), [2.5, 97.5]),
                 "dAUC": np.percentile(ours_b - a_b, [2.5, 97.5])}

    rep = ["# Final analysis for the journal version", "",
           f"Test set: {len(y)} windows ({int((y == 0).sum())} normal, {int(y.sum())} tunnel), "
           f"{len(set(lines))} survey lines, **{n_pass} tunnel passes** over the same tunnel. "
           f"Normal ground covered: {L.normal_km * 1000:.0f} m (one window step = {STEP_M} m). "
           f"Depth band used by '_depth' scores: rows {DEPTH_START_ROW}-127 = "
           f"{row_to_depth_m(DEPTH_START_ROW):.2f}-{row_to_depth_m(128):.2f} m at 0.085 m/ns.", "",
           "Threshold for every method: **leave-one-line-out** (each line judged by mean + 2.5 SD of the "
           "normal windows of the other lines; Lines 1 and 2 treated as one scan). 95% CIs resample whole "
           "survey lines.", ""]

    # 1. main table
    rep += ["## 1. Main comparison", "",
            "| Method | AUC [95% CI] | AP | Precision | Recall | F1 [95% CI] | FN | FP | Passes caught | Isolated false alarms / km | Recall at 1% false-positive rate |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(methods, key=lambda m: -point[m]["AUC"])
    for m in ranked:
        p, c = point[m], ci[m]
        bold = "**" if m == OURS else ""
        rep.append(f"| {bold}{show(m)}{bold} | {p['AUC']:.4f} [{c['AUC'][0]:.4f}, {c['AUC'][1]:.4f}] | {p['AP']:.4f} | "
                   f"{p['Precision']:.4f} | {p['Recall']:.4f} | {p['F1']:.4f} [{c['F1'][0]:.4f}, {c['F1'][1]:.4f}] | "
                   f"{p['FN']} | {p['FP']} | {p['passes']}/{n_pass} | {p['fa_far_km']:.1f} | {p['rec_at_1fpr']:.3f} |")
    rep.append("")

    # 2. significance
    rep += [f"## 2. AUC difference to {show(OURS)} (positive = ours better)", "",
            "| Method | dAUC | 95% CI (line resampling) | Significant? |", "|---|---|---|---|"]
    for m in ranked:
        if m == OURS:
            continue
        lo, hi = ci[m]["dAUC"]
        rep.append(f"| {show(m)} | {point[OURS]['AUC'] - point[m]['AUC']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                   f"{'yes' if lo > 0 or hi < 0 else 'no'} |")
    rep.append("")

    # 3. passes
    det = L.pass_detection(pred_lolo[OURS])
    edge, middle = L.fn_positions(pred_lolo[OURS])
    rep += ["## 3. Tunnel passes (our method, leave-one-line-out threshold)", "",
            f"- Passes in which the tunnel was flagged in at least one window: **{sum(det)} of {n_pass}**",
            f"- Missed tunnel windows: {edge + middle} - at the edge of a pass (first/last {EDGE_WINDOWS} windows, "
            f"~{EDGE_WINDOWS * STEP_M:.2f} m): **{edge}**, in the middle: **{middle}**", ""]
    pass_rows = []
    for (line, ix), d in zip(L.passes, det):
        flagged = int(pred_lolo[OURS][np.array(ix)].sum())
        pass_rows.append([line, len(ix), flagged, f"{flagged / len(ix):.2f}", "yes" if d else "NO"])
    rep += ["| Survey line | Tunnel windows | Flagged | Share | Caught |", "|---|---|---|---|---|"]
    rep += [f"| {' | '.join(map(str, r))} |" for r in pass_rows]
    rep.append("")
    with open(OUT_DIR / "per_pass_detection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["survey_line", "tunnel_windows", "flagged", "share", "caught"])
        w.writerows(pass_rows)

    # 4. false alarms
    rep += ["## 4. False alarms (our method)", "",
            f"- False-alarm windows: {point[OURS]['FP']} of {int((y == 0).sum())} normal windows "
            f"({point[OURS]['FP'] / (y == 0).sum():.1%})",
            f"- False-alarm events (consecutive flagged normal windows count as one): "
            f"{point[OURS]['fa_near'] + point[OURS]['fa_far']}",
            f"  - next to the tunnel (within {NEAR_WINDOWS} windows = one window width, may contain hyperbola tails): "
            f"{point[OURS]['fa_near']}",
            f"  - isolated (away from the tunnel): **{point[OURS]['fa_far']}** = "
            f"**{point[OURS]['fa_far_km']:.1f} per km** of normal ground", ""]

    # 5. threshold protocols
    rep += ["## 5. Threshold protocols for our method", "",
            "| Threshold set from | Fair? | Precision | Recall | F1 | FN | FP |", "|---|---|---|---|---|---|---|"]
    prot = OrderedDict()
    prot["Other survey lines, mean + 2.5 SD (leave-one-line-out) - headline"] = ("yes", pred_lolo[OURS])
    prot["Other survey lines, 99th percentile (leave-one-line-out)"] = ("yes", grouped_pred(test[OURS], y, gid, lolo, "p99"))
    prot["5 groups of survey lines, mean + 2.5 SD"] = ("yes", grouped_pred(test[OURS], y, gid, k5))
    if val is not None and OURS in val[3]:
        v = val[3][OURS]
        prot["Validation windows only (Site A, before the tunnel)"] = ("yes", (test[OURS] >= v.mean() + THRESHOLD_K * v.std()).astype(int))
    s0 = test[OURS][y == 0]
    prot["All test normal windows (original paper; optimistic)"] = ("no - reference only", (test[OURS] >= s0.mean() + THRESHOLD_K * s0.std()).astype(int))
    for name, (fair, pr) in prot.items():
        b = basic(y, pr)
        rep.append(f"| {name} | {fair} | {b['Precision']:.4f} | {b['Recall']:.4f} | {b['F1']:.4f} | {b['FN']} | {b['FP']} |")
    rep.append("")

    rep += ["**Sensitivity to k (threshold = mean + k SD of the other lines' normal windows)**", "",
            "| k | Precision | Recall | F1 | FN | FP | Passes caught | Isolated false alarms / km |",
            "|---|---|---|---|---|---|---|---|"]
    for k in (2.0, 2.5, 3.0, 3.5):
        pr = grouped_pred(test[OURS], y, gid, lolo, k=k)
        b = basic(y, pr)
        near, far = L.fp_events(pr, y)
        rep.append(f"| {k} | {b['Precision']:.4f} | {b['Recall']:.4f} | {b['F1']:.4f} | {b['FN']} | {b['FP']} | "
                   f"{sum(L.pass_detection(pr))}/{n_pass} | {far / L.normal_km:.1f} |")
    rep.append("")

    # 6. within-line AUC
    wl_rows = []
    for line, idx in L.per_line.items():
        ix = np.array(idx)
        if 0 < y[ix].sum() < len(ix):
            wl_rows.append([line, int((y[ix] == 0).sum()), int(y[ix].sum())] +
                           [roc_auc_score(y[ix], test[m][ix]) for m in ranked])
    if wl_rows:
        arr = np.array([r[3:] for r in wl_rows], dtype=float)
        rep += ["## 6. Within-line detection (tunnel vs normal windows of the same line and processing setting)", "",
                f"Lines containing both classes: {len(wl_rows)}", "",
                "| Method | Median within-line AUC | Lowest | Lines with AUC >= 0.95 |", "|---|---|---|---|"]
        for j, m in enumerate(ranked):
            rep.append(f"| {show(m)} | {np.median(arr[:, j]):.4f} | {arr[:, j].min():.4f} | "
                       f"{int((arr[:, j] >= 0.95).sum())}/{len(wl_rows)} |")
        rep.append("")
        with open(OUT_DIR / "within_line_auc.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["survey_line", "normal", "tunnel"] + ranked)
            for r in wl_rows:
                w.writerow(r[:3] + [f"{v:.4f}" for v in r[3:]])

    # 7. sweep
    if sweep:
        rep += ["## 7. Sensitivity to the depth limit and the top-k fraction (AUC)", ""]
        for model in ("AE", "PCA"):
            rep += [f"**{model}**", "",
                    "| Rows used (depth) | " + " | ".join(f"top {int(f * 100)}%" for f in SWEEP_FRACTIONS) + " |",
                    "|---|" + "---|" * len(SWEEP_FRACTIONS)]
            for row in SWEEP_DEPTH_ROWS:
                cells = []
                for frac in SWEEP_FRACTIONS:
                    key = f"{model} | top{int(round(frac * 100))}_row{row}"
                    cells.append(f"{roc_auc_score(y, sweep[key]):.4f}" if key in sweep else "-")
                rep.append(f"| {row}-127 ({row_to_depth_m(row):.2f}-{row_to_depth_m(128):.2f} m) | " + " | ".join(cells) + " |")
            rep.append("")

    # 8. seeds
    if seed_methods:
        rows = [("published model (seed 42)", OURS)] + [(m.split(" | ")[0].replace("AE ", ""), m) for m in seed_methods]
        rep += ["## 8. Training-seed variation (top5_depth, leave-one-line-out threshold)", "",
                "| Model | AUC | AP | F1 | FN | FP | Passes caught |", "|---|---|---|---|---|---|---|"]
        vals = []
        for name, m in rows:
            pr = pred_lolo[m]
            b = basic(y, pr)
            a, ap = roc_auc_score(y, test[m]), average_precision_score(y, test[m])
            pc = sum(L.pass_detection(pr))
            vals.append([a, ap, b["F1"], b["FN"], b["FP"], pc])
            rep.append(f"| {name} | {a:.4f} | {ap:.4f} | {b['F1']:.4f} | {b['FN']} | {b['FP']} | {pc}/{n_pass} |")
        v = np.array(vals, dtype=float)
        rep.append(f"| **mean +/- SD** | {v[:, 0].mean():.4f} +/- {v[:, 0].std(ddof=1):.4f} | "
                   f"{v[:, 1].mean():.4f} +/- {v[:, 1].std(ddof=1):.4f} | {v[:, 2].mean():.4f} +/- {v[:, 2].std(ddof=1):.4f} | "
                   f"{v[:, 3].mean():.1f} | {v[:, 4].mean():.1f} | {v[:, 5].mean():.1f} |")
        rep.append("")

    # 8b. ablation
    if abl_methods:
        ref = "AE seed7 | top5_depth"
        rows = ([("full model (seed 7)", ref)] if ref in test else []) + \
               [(m.split(" | ")[0].replace("ABL ", ""), m) for m in abl_methods]
        labels = {"no_noise": "without input noise (not denoising)", "no_flip": "without horizontal flip",
                  "latent128": "bottleneck 128 instead of 256", "latent512": "bottleneck 512 instead of 256"}
        rep += ["## 8b. Ablation: one design choice changed at a time (training seed 7)", "",
                "| Model | AUC | AP | AUC with global score | F1 | FN | FP | Passes caught |",
                "|---|---|---|---|---|---|---|---|"]
        for name, m in rows:
            pr = pred_lolo[m]
            b = basic(y, pr)
            g = m.replace("top5_depth", "global")
            ga = f"{roc_auc_score(y, test[g]):.4f}" if g in test else "-"
            rep.append(f"| {labels.get(name, name)} | {roc_auc_score(y, test[m]):.4f} | "
                       f"{average_precision_score(y, test[m]):.4f} | {ga} | {b['F1']:.4f} | {b['FN']} | {b['FP']} | "
                       f"{sum(L.pass_detection(pr))}/{n_pass} |")
        if seed_methods:
            rep.append("")
            rep.append("Compare differences with the seed-to-seed spread in section 8: a change smaller "
                       "than that spread is not a real effect.")
        rep.append("")

    # 10-11. extra training experiments (step 9)
    def unseen_rate(m, site):
        if unseen is None or m not in unseen[3]:
            return None
        u_sites = np.array(unseen[1])
        s0 = test[m][y == 0]
        flag = unseen[3][m][u_sites == site] >= s0.mean() + THRESHOLD_K * s0.std()
        return flag.mean() if len(flag) else None

    def det_row(m):
        b = basic(y, pred_lolo[m])
        return roc_auc_score(y, test[m]), b["F1"], b["FN"], b["FP"], sum(L.pass_detection(pred_lolo[m]))

    fmt = lambda v: "-" if v is None else f"{v:.1%}"
    if any(m.startswith("EXP sites") for m in test):
        rep += ["## 8c. Training on more sites (Site A + Site C; Site B never seen)", "",
                "Same model and settings, training seed 7. False-alarm rate = share of normal screenshots "
                "flagged, threshold from Site A test normal screenshots.", "",
                "| Trained on | Score | Site A AUC | Site A F1 | FN | FP | Passes caught | Site B false alarms (unseen) | Site C false alarms |",
                "|---|---|---|---|---|---|---|---|---|"]
        for label, pre, c_note in (("Site A only", "AE seed7", ""), ("Site A + Site C", "EXP sites A+C", " (trained on)")):
            for kind in ("top5_depth", "top5_full"):
                m = f"{pre} | {kind}"
                if m not in test:
                    continue
                a_, f1, fn, fp, pc = det_row(m)
                rep.append(f"| {label} | {kind} | {a_:.4f} | {f1:.4f} | {fn} | {fp} | {pc}/{n_pass} | "
                           f"{fmt(unseen_rate(m, 'Site B'))} | {fmt(unseen_rate(m, 'Site C'))}{c_note if unseen_rate(m, 'Site C') is not None else ''} |")
        rep.append("")
    if any(m.startswith("EXP data") for m in test):
        rep += ["## 8d. Less training data (training seed 7; same number of training steps)", "",
                "| Site A training data | AUC depth band | F1 depth band | FN depth band | AUC whole picture | F1 whole picture | FN whole picture | AUC average error |",
                "|---|---|---|---|---|---|---|---|"]
        for label, pre in (("25%", "EXP data 25%"), ("50%", "EXP data 50%"), ("100%", "AE seed7")):
            md, mf, mg = f"{pre} | top5_depth", f"{pre} | top5_full", f"{pre} | global"
            if md not in test:
                continue
            ad, f1d, fnd, _, _ = det_row(md)
            af, f1f, fnf, _, _ = det_row(mf) if mf in test else (np.nan,) * 5
            ag = roc_auc_score(y, test[mg]) if mg in test else np.nan
            rep.append(f"| {label} | {ad:.4f} | {f1d:.4f} | {fnd} | {af:.4f} | {f1f:.4f} | {fnf} | {ag:.4f} |")
        rep.append("")

    # 9. unseen sites
    if unseen is not None:
        u_paths, u_sites, _, u_scores = unseen
        sites = sorted(set(u_sites), key=natkey)
        rep += ["## 9. Unseen sites (normal ground the model never saw)", "",
                "Threshold: mean + 2.5 SD of all Site A test normal windows (no unseen-site data used). "
                "Distance = windows x 0.149 m.", "",
                "| Method | " + " | ".join(f"{s}: flagged windows" for s in sites) + " | All unseen: false-alarm rate | Flagged windows per km |",
                "|---|" + "---|" * (len(sites) + 2)]
        for m in ranked + [m for m in seed_methods]:
            if m not in u_scores:
                continue
            s0 = test[m][y == 0]
            thr = s0.mean() + THRESHOLD_K * s0.std()
            flag = u_scores[m] >= thr
            cells = []
            for s in sites:
                mask = np.array([x == s for x in u_sites])
                cells.append(f"{int(flag[mask].sum())}/{int(mask.sum())} ({flag[mask].mean():.1%})")
            rep.append(f"| {show(m)} | " + " | ".join(cells) + f" | {flag.mean():.1%} | {flag.sum() / (len(flag) * STEP_M / 1000):.1f} |")
        rep.append("")

    with open(OUT_DIR / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["AUC", "AP", "Precision", "Recall", "F1", "MCC", "FN", "FP", "passes", "fa_far_km", "rec_at_1fpr"]
        w.writerow(["method"] + keys + ["AUC_lo", "AUC_hi", "F1_lo", "F1_hi", "dAUC_lo", "dAUC_hi"])
        for m in ranked + seed_methods + abl_methods:
            w.writerow([m] + [round(float(point[m][k]), 6) for k in keys] +
                       [round(float(x), 6) for x in (*ci[m]["AUC"], *ci[m]["F1"], *ci[m]["dAUC"])])

    REPORT.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[:40]))
    print(f"\n... full report: {REPORT}")
    print("Next: python -u journal_experiments/step10_paper_numbers.py")


if __name__ == "__main__":
    main()
