r"""
STEP 4 - train the same autoencoder again from scratch with two other random
seeds (run overnight). Shows the result does not depend on one lucky training run.

Identical to train.py in every setting that matters:
  same Site A train/validation split (seed 42), same model, Adam lr 5e-4,
  batch 64, 50 epochs, each training image seen 10x per epoch with random
  horizontal flip, input noise sigma 0.05, loss against the clean image,
  best epoch chosen by validation loss.
Only the training seed changes (7 and 123). To make it faster, all images are
loaded into memory once instead of being re-read from disk every epoch.

The published model files are NOT touched. New files go to:
  Results/Journal/models/gpr_best_model_seed7.pt, ..._seed123.pt
If the computer restarts, just run the script again: a finished seed is skipped
and an unfinished one continues from its last completed epoch.

Run:    python -u journal_experiments/step4_retrain_seeds.py   (from the repository folder)
Writes: Results/Journal/models/*.pt, training_log_seeds.csv,
        scores_test_seeds.csv, scores_unseen_seeds.csv
"""

import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from journal_common import (DEVICE, MODELS_DIR, OUT_DIR, ae_error_maps, free_gpu,
                            load_ae, load_images, load_other_sites, load_test_set,
                            pool_scores, training_split, write_scores)
from train import BATCH_SIZE, EPOCHS, GPRBottleneckAE, LEARNING_RATE, NOISE_STD

SEEDS = (7, 123)
AUGMENT_FACTOR = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_epoch(row):
    path = OUT_DIR / "training_log_seeds.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seed", "epoch", "train_loss", "val_loss", "new_best"])
        w.writerow(row)


def train_one(seed, train_t, val_t):
    best_path = MODELS_DIR / f"gpr_best_model_seed{seed}.pt"
    ckpt_path = MODELS_DIR / f"checkpoint_seed{seed}.pth"
    done_flag = MODELS_DIR / f"done_seed{seed}.txt"
    if done_flag.exists() and best_path.exists():
        print(f"\nSeed {seed}: already finished - skipped.")
        return best_path

    set_seed(seed)
    model = GPRBottleneckAE().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    start, best_val = 1, float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")          # RNG states must stay on the CPU
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best_val = ck["epoch"] + 1, ck["best_val"]
        torch.set_rng_state(ck["rng_cpu"].cpu())
        if DEVICE.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        print(f"\nSeed {seed}: continuing from epoch {start}")
    else:
        print(f"\nSeed {seed}: starting fresh")

    n = len(train_t)
    for epoch in range(start, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = (torch.randperm(n * AUGMENT_FACTOR) % n).to(train_t.device)
        total, steps = 0.0, 0
        for s in range(0, len(order), BATCH_SIZE):
            imgs = train_t[order[s:s + BATCH_SIZE]].to(DEVICE)
            flip = torch.rand(len(imgs), device=DEVICE) < 0.5          # random horizontal flip
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
        log_epoch([seed, epoch, round(total / steps, 8), round(val_loss, 8), int(improved)])
        print(f"  seed {seed} | epoch {epoch}/{EPOCHS} | train {total / steps:.6f} | "
              f"val {val_loss:.6f}{'  <- best' if improved else ''} | {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best_val": best_val, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None},
                   ckpt_path)
    done_flag.write_text(f"finished, best validation loss {best_val:.6f}\n")
    del model, opt
    free_gpu()
    return best_path


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"Device: {DEVICE}")
    train_files, val_files = training_split()
    print(f"Training windows: {len(train_files)} | validation windows: {len(val_files)}")
    train_t = torch.from_numpy(load_images(train_files, "train")).unsqueeze(1)
    val_t = torch.from_numpy(load_images(val_files, "validation")).unsqueeze(1)
    if DEVICE.type == "cuda":
        train_t, val_t = train_t.to(DEVICE), val_t.to(DEVICE)

    best_paths = {seed: train_one(seed, train_t, val_t) for seed in SEEDS}
    del train_t, val_t
    free_gpu()

    print("\nScoring the test windows and unseen sites with each new model")
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, _ = load_other_sites()
    test = load_images(test_paths, "test")
    other = load_images(other_paths, "unseen sites") if other_paths else None
    s_test, s_other = {}, {}
    for seed, path in best_paths.items():
        model = load_ae(path)
        for name, v in pool_scores(ae_error_maps(model, test)).items():
            s_test[f"AE seed{seed} | {name}"] = v
        if other is not None:
            for name, v in pool_scores(ae_error_maps(model, other)).items():
                s_other[f"AE seed{seed} | {name}"] = v
        print(f"  seed {seed}: AUC (top5_depth) = {roc_auc_score(y, s_test[f'AE seed{seed} | top5_depth']):.4f}")
        del model
        free_gpu()
    write_scores(OUT_DIR / "scores_test_seeds.csv", test_paths, lines, y, s_test)
    if other is not None:
        write_scores(OUT_DIR / "scores_unseen_seeds.csv", other_paths, other_sites,
                     np.zeros(len(other_paths), int), s_other, group_header="site")
    print(f"\nStep 4 finished in {(time.time() - t_start) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
