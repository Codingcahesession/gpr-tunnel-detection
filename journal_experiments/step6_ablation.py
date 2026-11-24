r"""
STEP 6 - ablation study (run on the second night, after step 4 has finished).

Question answered: which design choices actually matter?
Each variant below is the SAME autoencoder and training as in the paper, with
exactly ONE thing changed. All use training seed 7, so each can be compared
directly with the full model trained with seed 7 in step 4.

  no_noise   : no input noise during training (plain autoencoder, not denoising)
  no_flip    : no random horizontal flip during training
  latent128  : bottleneck of 128 numbers instead of 256 (stronger compression)
  latent512  : bottleneck of 512 numbers instead of 256 (weaker compression)

The published model and the step 4 models are NOT touched. New files go to
Results/Journal/models/ablation_*.pt. If the computer restarts, run the script
again: finished variants are skipped and an unfinished one continues.

Run:    python -u journal_experiments/step6_ablation.py   (from the repository folder)
Writes: Results/Journal/models/ablation_*.pt, training_log_ablation.csv,
        scores_test_ablation.csv
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
                            load_images, load_test_set, pool_scores, training_split,
                            write_scores)
from train import BATCH_SIZE, EPOCHS, GPRBottleneckAE, LEARNING_RATE, NOISE_STD

TRAIN_SEED = 7
AUGMENT_FACTOR = 10
VARIANTS = {                      # name: (noise_std, use_flip, latent_size)
    "no_noise": (0.0, True, 256),
    "no_flip": (NOISE_STD, False, 256),
    "latent128": (NOISE_STD, True, 128),
    "latent512": (NOISE_STD, True, 512),
}


def make_ae(latent):
    """The paper's autoencoder; only the bottleneck size can differ."""
    model = GPRBottleneckAE()
    if latent != 256:
        assert isinstance(model.encoder[13], nn.Linear) and isinstance(model.decoder[0], nn.Linear)
        model.encoder[13] = nn.Linear(256 * 8 * 8, latent)
        model.decoder[0] = nn.Linear(latent, 256 * 8 * 8)
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_epoch(row):
    path = OUT_DIR / "training_log_ablation.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["variant", "epoch", "train_loss", "val_loss", "new_best"])
        w.writerow(row)


def train_variant(name, noise, use_flip, latent, train_t, val_t):
    best_path = MODELS_DIR / f"ablation_{name}.pt"
    ckpt_path = MODELS_DIR / f"checkpoint_ablation_{name}.pth"
    done_flag = MODELS_DIR / f"done_ablation_{name}.txt"
    if done_flag.exists() and best_path.exists():
        print(f"\n{name}: already finished - skipped.")
        return best_path

    set_seed(TRAIN_SEED)
    model = make_ae(latent).to(DEVICE)
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
        print(f"\n{name}: continuing from epoch {start}")
    else:
        print(f"\n{name}: starting fresh (noise {noise}, flip {use_flip}, bottleneck {latent})")

    n = len(train_t)
    for epoch in range(start, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = (torch.randperm(n * AUGMENT_FACTOR) % n).to(train_t.device)
        total, steps = 0.0, 0
        for s in range(0, len(order), BATCH_SIZE):
            imgs = train_t[order[s:s + BATCH_SIZE]].to(DEVICE)
            if use_flip:
                flip = torch.rand(len(imgs), device=DEVICE) < 0.5
                imgs = torch.where(flip.view(-1, 1, 1, 1), imgs.flip(-1), imgs)
            inp = (imgs + noise * torch.randn_like(imgs)).clamp(0.0, 1.0) if noise > 0 else imgs
            loss = loss_fn(model(inp), imgs)
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


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"Device: {DEVICE}")
    train_files, val_files = training_split()
    train_t = torch.from_numpy(load_images(train_files, "train")).unsqueeze(1)
    val_t = torch.from_numpy(load_images(val_files, "validation")).unsqueeze(1)
    if DEVICE.type == "cuda":
        train_t, val_t = train_t.to(DEVICE), val_t.to(DEVICE)

    paths = {name: train_variant(name, *cfg, train_t, val_t) for name, cfg in VARIANTS.items()}
    del train_t, val_t
    free_gpu()

    print("\nScoring the test windows with each variant")
    test_paths, y, lines = load_test_set()
    test = load_images(test_paths, "test")
    scores = {}
    for name, path in paths.items():
        model = make_ae(VARIANTS[name][2]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        for kind, v in pool_scores(ae_error_maps(model, test)).items():
            scores[f"ABL {name} | {kind}"] = v
        print(f"  {name}: AUC (top5_depth) = {roc_auc_score(y, scores[f'ABL {name} | top5_depth']):.4f}")
        del model
        free_gpu()
    write_scores(OUT_DIR / "scores_test_ablation.csv", test_paths, lines, y, scores)
    print(f"\nStep 6 finished in {(time.time() - t_start) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
