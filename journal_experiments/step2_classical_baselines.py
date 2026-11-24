r"""
STEP 2 - our autoencoder + classical baselines (about 15-40 minutes).

Methods (all fitted on the same Site A normal training windows only):
  AE    our autoencoder (gpr_best_model.pt, the published model; no retraining)
  PCA   PCA with 256 components (same size as the AE's 256-D code)
  OCSVM One-Class SVM on 256-D PCA features (raw, and standardised)
  IF    Isolation Forest on 256-D PCA features
Each is scored on the whole window ("full") and on rows 64-127 only ("depth").

Also saved:
  - the sensitivity study (other depth limits and top-k fractions) for AE and PCA
  - every method's scores on the unseen sites (B, C), for the false-alarm check
  - every method's scores on the validation windows (for the validation-threshold row)

Run:    python -u journal_experiments/step2_classical_baselines.py   (from the repository folder)
Writes: Results/Journal/scores_test_classical.csv, scores_test_sweep.csv,
        scores_val_classical.csv, scores_unseen_classical.csv
"""

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from journal_common import (DEPTH_START_ROW, OUT_DIR, SEED, ae_error_maps,
                            free_gpu, load_ae, load_images, load_other_sites,
                            load_test_set, pool_scores, sweep_scores,
                            training_split, write_scores)

PCA_COMPONENTS = 256
OCSVM_NU = 0.05
IF_TREES = 300


def pca_error_maps(pca, images):
    out = np.empty_like(images)
    for s in range(0, len(images), 1000):
        flat = images[s:s + 1000].reshape(-1, images.shape[1] * images.shape[2])
        rec = pca.inverse_transform(pca.transform(flat))
        out[s:s + 1000] = ((flat - rec) ** 2).reshape(-1, images.shape[1], images.shape[2])
    return out


def main():
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train_files, val_files = training_split()
    print(f"Training windows: {len(train_files)} | validation windows: {len(val_files)}")
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, other_groups = load_other_sites()

    train = load_images(train_files, "train")
    val = load_images(val_files, "validation")
    test = load_images(test_paths, "test")
    other = load_images(other_paths, "unseen sites") if other_paths else None
    sets = {"val": val, "test": test}
    if other is not None:
        sets["other"] = other

    scores = {k: {} for k in sets}
    sweep = {}

    def add(method, per_set):
        for k, v in per_set.items():
            scores[k][method] = v

    # ---- 1. our autoencoder ----
    print("\n[1/4] Autoencoder (published model)")
    model = load_ae()
    for k, imgs in sets.items():
        maps = ae_error_maps(model, imgs)
        for name, v in pool_scores(maps).items():
            scores[k][f"AE | {name}"] = v
        if k == "test":
            sweep.update({f"AE | {n}": v for n, v in sweep_scores(maps).items()})
    del model
    free_gpu()

    # ---- 2. PCA reconstruction ----
    print("\n[2/4] PCA reconstruction")
    flat = lambda a: a.reshape(len(a), -1)
    lower = lambda a: a[:, DEPTH_START_ROW:, :].reshape(len(a), -1)
    pca_full = PCA(PCA_COMPONENTS, svd_solver="randomized", random_state=SEED).fit(flat(train))
    print(f"  explained variance: {pca_full.explained_variance_ratio_.sum():.3f}")
    for k, imgs in sets.items():
        maps = pca_error_maps(pca_full, imgs)
        for name, v in pool_scores(maps).items():
            scores[k][f"PCA | {name}"] = v
        if k == "test":
            sweep.update({f"PCA | {n}": v for n, v in sweep_scores(maps).items()})

    # ---- features for OCSVM / IF ----
    pca_depth = PCA(PCA_COMPONENTS, svd_solver="randomized", random_state=SEED).fit(lower(train))
    feats = {"full": (pca_full.transform(flat(train)), {k: pca_full.transform(flat(v)) for k, v in sets.items()}),
             "depth": (pca_depth.transform(lower(train)), {k: pca_depth.transform(lower(v)) for k, v in sets.items()})}

    # ---- 3. One-Class SVM / 4. Isolation Forest ----
    for label, make, score_fn in (
            ("OCSVM", lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=OCSVM_NU), lambda m, f: -m.decision_function(f)),
            ("IF", lambda: IsolationForest(n_estimators=IF_TREES, random_state=SEED, n_jobs=-1), lambda m, f: -m.score_samples(f))):
        print(f"\n[{'3' if label == 'OCSVM' else '4'}/4] {label}")
        for region, (ftr, fsets) in feats.items():
            for scaled in ((False, True) if label == "OCSVM" else (False,)):   # trees ignore scaling
                t0 = time.time()
                scaler = StandardScaler().fit(ftr) if scaled else None
                tf = (lambda a: scaler.transform(a)) if scaled else (lambda a: a)
                m = make().fit(tf(ftr))
                name = f"{label} | {region}{' std' if scaled else ''}"
                add(name, {k: score_fn(m, tf(f)) for k, f in fsets.items()})
                print(f"  {name}: AUC {roc_auc_score(y, scores['test'][name]):.4f} ({time.time() - t0:.0f}s)")

    # ---- save ----
    print("\nAUC on the test windows:")
    for name, v in scores["test"].items():
        print(f"  {name:22s} {roc_auc_score(y, v):.4f}")
    write_scores(OUT_DIR / "scores_test_classical.csv", test_paths, lines, y, scores["test"])
    write_scores(OUT_DIR / "scores_test_sweep.csv", test_paths, lines, y, sweep)
    write_scores(OUT_DIR / "scores_val_classical.csv", val_files, ["validation"] * len(val_files),
                 np.zeros(len(val_files), int), scores["val"])
    if other is not None:
        write_scores(OUT_DIR / "scores_unseen_classical.csv", other_paths, other_sites,
                     np.zeros(len(other_paths), int), scores["other"], group_header="site")
    print(f"\nStep 2 finished in {(time.time() - t_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
