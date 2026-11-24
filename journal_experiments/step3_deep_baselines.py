r"""
STEP 3 - modern deep anomaly-detection baselines (about 30-90 minutes on the GTX 1660 Ti).

  PatchCore (Roth et al., CVPR 2022): WideResNet-50 features + coreset memory bank
  PaDiM     (Defard et al., ICPR 2021): ResNet-18 features + Gaussian per position

Both are standard "industrial" anomaly detectors. Like our autoencoder they see
only the Site A normal training windows. Their anomaly maps are smoothed with a
Gaussian (sigma = 4 px at 224x224 input), as in the original papers, and scored
four ways: max_full, max_depth (rows 64-127), top5_full, top5_depth.

The ImageNet weights download once on the first run (~270 MB + ~45 MB).

Run:    python -u journal_experiments/step3_deep_baselines.py   (from the repository folder)
Writes: Results/Journal/scores_test_deep.csv, scores_val_deep.csv, scores_unseen_deep.csv
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from journal_common import (DEPTH_START_ROW, DEVICE, OUT_DIR, SEED, TOP_FRACTION,
                            free_gpu, load_images, load_other_sites, load_test_set,
                            topk_mean, training_split, write_scores)

INPUT_SIZE = 224
BATCH_SIZE = 16
SMOOTH_SIGMA = 4.0 * 128 / INPUT_SIZE     # sigma 4 at 224 px = 2.3 px on the 128 px map

PC_EMB_DIM = 512
PC_MEMORY_STRIDE = 10      # every 10th training window (neighbours overlap ~94%)
PC_CORESET_RATIO = 0.02
PC_PROJ_DIM = 128
PC_QUERY_CHUNK = 4096

PD_DIMS = 100
PD_EPS = 0.01

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_backbone(name):
    from torchvision import models
    if name == "wide_resnet50_2":
        m = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return m.to(DEVICE).eval()


def to_input(batch):
    x = torch.from_numpy(batch).unsqueeze(1).to(DEVICE)
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    return (x.repeat(1, 3, 1, 1) - MEAN.to(DEVICE)) / STD.to(DEVICE)


def features(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    l1 = model.layer1(x)
    l2 = model.layer2(l1)
    return l1, l2, model.layer3(l2)


def batches(images):
    for s in range(0, len(images), BATCH_SIZE):
        yield images[s:s + BATCH_SIZE]


def maps_to_scores(small_maps):
    t = torch.from_numpy(small_maps).unsqueeze(1)
    big = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False).squeeze(1).numpy()
    big = np.stack([gaussian_filter(m, SMOOTH_SIGMA) for m in big])
    n = len(big)
    full = big.reshape(n, -1)
    depth = big[:, DEPTH_START_ROW:, :].reshape(n, -1)
    return {"max_full": full.max(axis=1).astype(np.float64),
            "max_depth": depth.max(axis=1).astype(np.float64),
            "top5_full": topk_mean(full, TOP_FRACTION),
            "top5_depth": topk_mean(depth, TOP_FRACTION)}


# ----------------------------------------------------------------------------- PatchCore
def pc_embed(model, x):
    _, l2, l3 = features(model, x)
    l2 = F.avg_pool2d(l2, 3, 1, 1)
    l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
    e = torch.cat([l2, l3], dim=1)
    b, c, h, w = e.shape
    e = F.adaptive_avg_pool1d(e.permute(0, 2, 3, 1).reshape(-1, 1, c), PC_EMB_DIM)
    return e.reshape(b, h, w, PC_EMB_DIM)


def coreset(feats, n_select):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    proj = torch.randn(feats.shape[1], PC_PROJ_DIM, generator=gen).to(DEVICE)
    z = torch.cat([feats[s:s + 65536].to(DEVICE).float() @ proj for s in range(0, len(feats), 65536)])
    chosen = [0]
    dist = ((z - z[0]) ** 2).sum(1)
    for i in range(1, n_select):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((z - z[nxt]) ** 2).sum(1))
        if (i + 1) % 2000 == 0:
            print(f"    coreset {i + 1}/{n_select}")
    return torch.tensor(chosen)


def run_patchcore(train, sets):
    print("\n[PatchCore] WideResNet-50")
    model = load_backbone("wide_resnet50_2")
    t0 = time.time()
    with torch.no_grad():
        feats = torch.cat([pc_embed(model, to_input(b)).reshape(-1, PC_EMB_DIM).half().cpu()
                           for b in batches(train[::PC_MEMORY_STRIDE])])
        n_sel = max(1, int(PC_CORESET_RATIO * len(feats)))
        print(f"  {len(feats):,} training patches -> memory bank of {n_sel:,}")
        bank = feats[coreset(feats, n_sel)].to(DEVICE).float()
    del feats
    print(f"  memory bank ready ({time.time() - t0:.0f}s)")
    out = {}
    with torch.no_grad():
        for k, imgs in sets.items():
            maps = []
            for b in batches(imgs):
                e = pc_embed(model, to_input(b))
                bs, h, w, c = e.shape
                q = e.reshape(-1, c)
                d = torch.cat([torch.cdist(q[s:s + PC_QUERY_CHUNK], bank).min(dim=1).values
                               for s in range(0, len(q), PC_QUERY_CHUNK)])
                maps.append(d.reshape(bs, h, w).cpu().numpy())
            out[k] = maps_to_scores(np.concatenate(maps).astype(np.float32))
            print(f"  scored {k} ({time.time() - t0:.0f}s)")
    del model, bank
    free_gpu()
    return out


# ----------------------------------------------------------------------------- PaDiM
def pd_embed(model, x, sel):
    l1, l2, l3 = features(model, x)
    size = l1.shape[-2:]
    e = torch.cat([l1, F.interpolate(l2, size=size, mode="nearest"),
                   F.interpolate(l3, size=size, mode="nearest")], dim=1)[:, sel]
    b, d, h, w = e.shape
    return e.reshape(b, d, h * w), (h, w)


def run_padim(train, sets):
    print("\n[PaDiM] ResNet-18")
    model = load_backbone("resnet18")
    sel = torch.randperm(448, generator=torch.Generator(device="cpu").manual_seed(SEED))[:PD_DIMS].to(DEVICE)
    t0 = time.time()
    s1 = s2 = None
    n = 0
    with torch.no_grad():
        for b in batches(train):
            e = pd_embed(model, to_input(b), sel)[0].double()
            if s1 is None:
                p = e.shape[2]
                s1 = torch.zeros(PD_DIMS, p, dtype=torch.float64, device=DEVICE)
                s2 = torch.zeros(p, PD_DIMS, PD_DIMS, dtype=torch.float64, device=DEVICE)
            s1 += e.sum(0)
            s2 += torch.einsum("bdp,bep->pde", e, e)
            n += e.shape[0]
        mean = (s1 / n).T
        cov = (s2 / n - torch.einsum("pd,pe->pde", mean, mean)) * n / max(1, n - 1)
        cov += PD_EPS * torch.eye(PD_DIMS, dtype=torch.float64, device=DEVICE)
        inv_cov, mean = torch.linalg.inv(cov).float(), mean.float()
        del s1, s2, cov
    print(f"  Gaussians fitted on {n} windows ({time.time() - t0:.0f}s)")
    out = {}
    with torch.no_grad():
        for k, imgs in sets.items():
            maps = []
            for b in batches(imgs):
                e, (h, w) = pd_embed(model, to_input(b), sel)
                diff = e.permute(0, 2, 1) - mean.unsqueeze(0)
                m2 = torch.einsum("bpd,pde,bpe->bp", diff, inv_cov, diff).clamp_min(0)
                maps.append(m2.sqrt().reshape(-1, h, w).cpu().numpy())
            out[k] = maps_to_scores(np.concatenate(maps).astype(np.float32))
            print(f"  scored {k} ({time.time() - t0:.0f}s)")
    del model, inv_cov, mean
    free_gpu()
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"Device: {DEVICE}")

    train_files, val_files = training_split()
    test_paths, y, lines = load_test_set()
    other_paths, other_sites, _ = load_other_sites()
    train = load_images(train_files, "train")
    sets = {"val": load_images(val_files, "validation"), "test": load_images(test_paths, "test")}
    if other_paths:
        sets["other"] = load_images(other_paths, "unseen sites")

    scores = {k: {} for k in sets}
    for name, fn in (("PatchCore", run_patchcore), ("PaDiM", run_padim)):
        for k, variants in fn(train, sets).items():
            for v, s in variants.items():
                scores[k][f"{name} | {v}"] = s

    print("\nAUC on the test windows:")
    for name, v in scores["test"].items():
        print(f"  {name:24s} {roc_auc_score(y, v):.4f}")
    write_scores(OUT_DIR / "scores_test_deep.csv", test_paths, lines, y, scores["test"])
    write_scores(OUT_DIR / "scores_val_deep.csv", val_files, ["validation"] * len(val_files),
                 np.zeros(len(val_files), int), scores["val"])
    if other_paths:
        write_scores(OUT_DIR / "scores_unseen_deep.csv", other_paths, other_sites,
                     np.zeros(len(other_paths), int), scores["other"], group_header="site")
    print(f"\nStep 3 finished in {(time.time() - t_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
