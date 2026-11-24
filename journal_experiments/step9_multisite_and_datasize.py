r"""
STEP 9 - two training experiments (run overnight, about 3-4 hours).

  A. MORE SITES: train the same autoencoder on Site A + Site C normal ground.
     Site B stays completely unseen. Questions:
       - does a model that has seen more kinds of ground give fewer false alarms
         on a new site (Site B)?
       - does it still detect the Site A tunnel as well?
  B. LESS DATA: train on only 25% and 50% of the Site A training data.
     Question: does the depth band still work with less data, and does scoring
     the whole picture need much more data to catch up?

All models: same autoencoder and settings as train.py, training seed 7, so
they compare directly with the seed-7 model from step 4. Each model makes the
same number of training steps as the full model, so smaller data sets are not
simply trained less. The 25% / 50% subsets keep whole blocks of consecutive
screenshots (neighbouring screenshots overlap ~95%, so picking single random
screenshots would keep almost all the information).

The published model and earlier models are NOT touched. If the computer restarts,
run it again: finished models are skipped and an unfinished one continues.

Run:    python -u journal_experiments/step9_multisite_and_datasize.py   (from the repository folder)
Writes: Results/Journal/models/exp_*.pt, training_log_exp.csv,
        scores_test_exp.csv, scores_unseen_exp.csv
"""

import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from journal_common import (DEVICE, MODELS_DIR, OUT_DIR, SEED,
                            ae_error_maps, fixed_split, free_gpu, load_ae, load_images,
                            load_other_sites, load_test_set, natkey, pool_scores,
                            site_images, training_split, write_scores)
from train import BATCH_SIZE, EPOCHS, GPRBottleneckAE, LEARNING_RATE, NOISE_STD

TRAIN_SEED = 7
BASE_AUGMENT = 10              # as in train.py
EXTRA_SITE = "Site C"          # added to training in experiment A (Site B stays unseen)
FRACTIONS = (0.25, 0.50)       # experiment B
BLOCK = 64                     # consecutive screenshots kept together when sub-sampling


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_epoch(row):
    path = OUT_DIR / "training_log_exp.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "epoch", "train_loss", "val_loss", "new_best"])
        w.writerow(row)


def block_subset(files, fraction, seed):
    """Keep `fraction` of the screenshots as whole blocks of consecutive screenshots."""
    ordered = sorted(files, key=lambda p: (natkey(p.parent), natkey(p.name)))
    blocks = [ordered[i:i + BLOCK] for i in range(0, len(ordered), BLOCK)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    target = int(round(fraction * len(ordered)))
    keep = []
    for b in blocks:
        if len(keep) >= target:
            break
        keep.extend(b)
    return keep


def train_model(name, train_t, val_t, augment):
    best_path = MODELS_DIR / f"exp_{name}.pt"
    ckpt_path = MODELS_DIR / f"checkpoint_exp_{name}.pth"
    done_flag = MODELS_DIR / f"done_exp_{name}.txt"
    if done_flag.exists() and best_path.exists():
        print(f"\n{name}: already finished - skipped.")
        return best_path

    set_seed(TRAIN_SEED)
    model = GPRBottleneckAE().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    start, best_val = 1, float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best_val = ck["epoch"] + 1, ck["best_val"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if DEVICE.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        print(f"\n{name}: continuing from epoch {start}")
    else:
        print(f"\n{name}: starting fresh ({len(train_t)} training screenshots, each used {augment}x per epoch)")

    n = len(train_t)
    for epoch in range(start, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = (torch.randperm(n * augment) % n).to(train_t.device)
        total, steps = 0.0, 0
        for s in range(0, len(order), BATCH_SIZE):
            imgs = train_t[order[s:s + BATCH_SIZE]].to(DEVICE)
            flip = torch.rand(len(imgs), device=DEVICE) < 0.5
            imgs = torch.where(flip.view(-1, 1, 1, 1), imgs.flip(-1), imgs)
            noisy = (imgs + NOISE_STD * torch.randn_like(imgs)).clamp(0.0, 1.0)
            loss = loss_fn(model(noisy), imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        model.eval()
        with torch.no_grad():
            vl = [float(loss_fn(model(val_t[s:s + BATCH_SIZE].to(DEVICE)), val_t[s:s + BATCH_SIZE].to(DEVICE)))
                  for s in range(0, len(val_t), BATCH_SIZE)]
        val_loss = sum(vl) / len(vl)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        log_epoch([name, epoch, round(total / steps, 8), round(val_loss, 8), int(improved)])
        print(f"  {name} | epoch {epoch}/{EPOCHS} | train {total / steps:.6f} | "
              f"val {val_loss:.6f}{'  <- best' if improved else ''} | {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best_val": best_val, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None},
                   ckpt_path)
    done_flag.write_text(f"finished, best validation loss {best_val:.6f}\n")
    del model, opt
    free_gpu()
    return best_path


def to_tensor(files, desc):
    t = torch.from_numpy(load_images(files, desc)).unsqueeze(1)
    return t.to(DEVICE) if DEVICE.type == "cuda" else t


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"Device: {DEVICE}")
    a_train, a_val = training_split()
    full_steps = len(a_train) * BASE_AUGMENT            # training steps per epoch of the full model (x batch)
    paths = {}

    # ---- A. Site A + Site C ----
    c_files = site_images(EXTRA_SITE)
    if not c_files:
        print(f"! {EXTRA_SITE} folder not found - experiment A skipped.")
    else:
        c_train, c_val = fixed_split("siteC", c_files)
        print(f"Experiment A: Site A {len(a_train)} + {EXTRA_SITE} {len(c_train)} training screenshots")
        tr = to_tensor(list(a_train) + list(c_train), "train A+C")
        va = to_tensor(list(a_val) + list(c_val), "validation A+C")
        paths["A+C"] = train_model("A_plus_C", tr, va, BASE_AUGMENT)
        del tr, va
        free_gpu()

    # ---- B. 25% / 50% of Site A ----
    va = to_tensor(a_val, "validation A")
    for frac in FRACTIONS:
        sub = block_subset(a_train, frac, SEED)
        augment = max(1, int(round(full_steps / len(sub))))          # same number of steps as the full model
        tag = f"A_{int(frac * 100)}pct"
        print(f"Experiment B: {tag} = {len(sub)} training screenshots")
        tr = to_tensor(sub, f"train {tag}")
        paths[f"{int(frac * 100)}%"] = train_model(tag, tr, va, augment)
        del tr
        free_gpu()
    del va
    free_gpu()

    # ---- scoring ----
    print("\nScoring Site A test screenshots and unseen sites with each model")
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, _ = load_other_sites()
    test = load_images(test_paths, "test")
    other = load_images(other_paths, "unseen sites") if other_paths else None
    s_test, s_other = {}, {}
    for label, path in paths.items():
        prefix = "EXP sites A+C" if label == "A+C" else f"EXP data {label}"
        model = load_ae(path)
        for kind, v in pool_scores(ae_error_maps(model, test)).items():
            s_test[f"{prefix} | {kind}"] = v
        if other is not None:
            for kind, v in pool_scores(ae_error_maps(model, other)).items():
                s_other[f"{prefix} | {kind}"] = v
        print(f"  {prefix}: AUC top5_depth {roc_auc_score(y, s_test[f'{prefix} | top5_depth']):.4f} | "
              f"top5_full {roc_auc_score(y, s_test[f'{prefix} | top5_full']):.4f}")
        del model
        free_gpu()
    write_scores(OUT_DIR / "scores_test_exp.csv", test_paths, lines, y, s_test)
    if other is not None:
        write_scores(OUT_DIR / "scores_unseen_exp.csv", other_paths, other_sites,
                     np.zeros(len(other_paths), int), s_other, group_header="site")
    print(f"\nStep 9 finished in {(time.time() - t_start) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
