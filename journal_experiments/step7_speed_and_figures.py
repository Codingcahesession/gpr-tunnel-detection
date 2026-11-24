r"""
STEP 7 - model size, speed, and two figures for the paper (about 5 minutes).
Uses the published model (gpr_best_model.pt). Run any time after step 1.

  1. Model size and speed: number of parameters, file size, time per window
     on this computer (GPU and CPU), and how many km of survey per hour that is.
  2. figure_error_maps.png: for typical tunnel and normal windows, and for
     the hardest cases, shows input -> reconstruction -> error map, with the
     depth band (rows 64-127) marked. Axes in metres and nanoseconds.
  3. figure_same_tunnel.png: one window from the middle of 12 different passes
     over the same tunnel - shows how much its appearance changes between surveys.

Run:    python -u journal_experiments/step7_speed_and_figures.py   (from the repository folder)
Writes: Results/Journal/speed_and_size.md, figure_error_maps.png, figure_same_tunnel.png
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from journal_common import (DEPTH_START_ROW, DEVICE, MODEL_PATH, OUT_DIR, STEP_M,
                            TIME_WINDOW_NS, VELOCITY_M_NS, WINDOW_M, ae_error_maps,
                            load_ae, load_images, load_test_set, pool_scores)

EXTENT = [0, WINDOW_M, TIME_WINDOW_NS, 0]          # x in metres, y in ns


def timing(model, device, images, batch):
    model = model.to(device).eval()
    x = torch.from_numpy(images).unsqueeze(1)
    with torch.no_grad():
        for _ in range(3):                           # warm-up
            model(x[:batch].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, len(x), batch):
            out = model(x[s:s + batch].to(device))
            ((x[s:s + batch].to(device) - out) ** 2).mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(x) * 1000.0      # ms per window


def runs(mask):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_paths, y, lines = load_test_set()
    test = load_images(test_paths, "test")

    # ---------------- 1. size and speed ----------------
    model = load_ae()
    n_params = sum(p.numel() for p in model.parameters())
    rep = ["# Model size and speed", "",
           f"- Parameters: **{n_params:,}** ({n_params / 1e6:.2f} million)",
           f"- Model file: {MODEL_PATH.stat().st_size / 1e6:.1f} MB", ""]
    sample = test[:min(512, len(test))]
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    rep += ["| Hardware | Batch | ms per window | Windows per second | Survey km per hour of processing |",
            "|---|---|---|---|---|"]
    for dev in devices:
        name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
        for batch in (1, 64):
            ms = timing(load_ae(), dev, sample, batch)
            wps = 1000.0 / ms
            rep.append(f"| {name} | {batch} | {ms:.2f} | {wps:.0f} | {wps * STEP_M * 3600 / 1000:.0f} |")
    rep += ["", f"(km per hour = windows per second x {STEP_M} m per window step x 3600.)"]
    (OUT_DIR / "speed_and_size.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))

    # ---------------- scores + maps ----------------
    model = load_ae()
    maps = ae_error_maps(model, test)
    score = pool_scores(maps)["top5_depth"]
    with torch.no_grad():
        recon = np.concatenate([model(torch.from_numpy(test[s:s + 64]).unsqueeze(1).to(DEVICE)).squeeze(1).cpu().numpy()
                                for s in range(0, len(test), 64)])

    s0 = score[y == 0]
    thr = s0.mean() + 2.5 * s0.std()
    tun, nor = np.where(y == 1)[0], np.where(y == 0)[0]
    pick = [("Tunnel (typical)", tun[np.argsort(score[tun])[len(tun) // 2]]),
            ("Tunnel (strong)", tun[np.argsort(score[tun])[int(len(tun) * 0.9)]]),
            ("Tunnel (hardest)", tun[np.argmin(score[tun])]),
            ("Normal (typical)", nor[np.argsort(score[nor])[len(nor) // 2]]),
            ("Normal (hardest)", nor[np.argmax(score[nor])])]

    vmax = np.percentile(maps[[i for _, i in pick]], 99.5)
    fig, ax = plt.subplots(len(pick), 3, figsize=(9.5, 2.7 * len(pick)))
    for r, (title, i) in enumerate(pick):
        for c, (img, cmap, lab) in enumerate(((test[i], "gray", "Input"), (recon[i], "gray", "Reconstruction"),
                                               (maps[i], "inferno", "Squared error"))):
            a = ax[r, c]
            a.imshow(img, cmap=cmap, extent=EXTENT, aspect="auto",
                     vmin=0, vmax=1 if c < 2 else vmax)
            a.axhline(DEPTH_START_ROW / 128 * TIME_WINDOW_NS, color="cyan", lw=1, ls="--")
            if r == 0:
                a.set_title(lab, fontsize=10)
            if c == 0:
                a.set_ylabel(f"{title}\nTime (ns)", fontsize=9)
            if r == len(pick) - 1:
                a.set_xlabel("Distance (m)", fontsize=9)
            a.tick_params(labelsize=8)
        ax[r, 2].text(0.02, 0.95, f"score {score[i]:.4f}\n{'above' if score[i] >= thr else 'below'} threshold",
                      transform=ax[r, 2].transAxes, color="white", fontsize=7, va="top")
        sec = ax[r, 2].secondary_yaxis("right", functions=(lambda t: VELOCITY_M_NS * t / 2,
                                                            lambda d: 2 * d / VELOCITY_M_NS))
        sec.set_ylabel("Depth (m)", fontsize=8)
        sec.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_error_maps.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'figure_error_maps.png'}")

    # ---------------- same tunnel, different surveys ----------------
    passes = []
    order = list(range(len(y)))                      # load_test_set already sorts by line, then window
    for line in dict.fromkeys(lines):
        idx = [i for i in order if lines[i] == line]
        for a, b in runs([y[i] == 1 for i in idx]):
            passes.append(idx[(a + b) // 2])
    chosen = [passes[k] for k in np.linspace(0, len(passes) - 1, min(12, len(passes))).round().astype(int)]
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
    for k, a in enumerate(ax.ravel()):
        a.axis("off")
        if k < len(chosen):
            i = chosen[k]
            a.imshow(test[i], cmap="gray", vmin=0, vmax=1, extent=EXTENT, aspect="auto")
            a.set_title(f"{lines[i]}", fontsize=8)
    fig.suptitle("The same tunnel in different surveys (middle window of each pass)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_same_tunnel.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'figure_same_tunnel.png'}")
    print("\nStep 7 finished.")


if __name__ == "__main__":
    main()
