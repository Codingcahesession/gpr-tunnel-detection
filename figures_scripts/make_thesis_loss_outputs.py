"""
make_thesis_loss_outputs.py
Reads gpr_model_checkpoint.pth and produces:
  1. A clean thesis-quality loss curve (epoch-based, zoomed y-axis)
  2. A CSV table of train/val loss at fixed intervals (epoch 1, 5, 10, ..., 50)
  3. A markdown copy of the same table for easy pasting
"""

from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "gpr_model_checkpoint.pth"
OUT_DIR = SCRIPT_DIR / "Results" / "Training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load just the loss history (no need to instantiate the model).
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
train_losses = ckpt["train_epoch_losses"]
val_losses   = ckpt["val_epoch_losses"]

assert len(train_losses) == len(val_losses), "train/val length mismatch"
n_epochs = len(train_losses)
epochs = list(range(1, n_epochs + 1))
print(f"Loaded {n_epochs} epochs of loss history.")

# -------------------------------------------------------------------------
# 1. CLEAN THESIS LOSS CURVE
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Training loss",   linewidth=2, marker="o", markersize=3)
plt.plot(epochs, val_losses,   label="Validation loss", linewidth=2, marker="s", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE reconstruction loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
best_epoch = val_losses.index(min(val_losses)) + 1
plt.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.6)
plt.text(best_epoch + 0.5, max(train_losses) * 0.9,
        f"Best val (epoch {best_epoch})", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curve_thesis.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'loss_curve_thesis.png'}")

# Same figure but with a log-y axis, in case you want both options.
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Training loss",   linewidth=2, marker="o", markersize=3)
plt.plot(epochs, val_losses,   label="Validation loss", linewidth=2, marker="s", markersize=3)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE reconstruction loss (log scale)")
plt.title("Training and Validation Loss (log scale)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUT_DIR / "loss_curve_thesis_log.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'loss_curve_thesis_log.png'}")

# -------------------------------------------------------------------------
# 2. INTERVAL TABLE (you asked: epochs 20-50 at appropriate intervals)
#    I'm also including 1, 5, 10, 15 so the table shows the early descent.
# -------------------------------------------------------------------------
table_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Keep only epochs you actually trained
table_epochs = [e for e in table_epochs if e <= n_epochs]

best_val = min(val_losses)
best_epoch = val_losses.index(best_val) + 1  # 1-indexed

rows = []
for e in table_epochs:
    rows.append({
        "epoch": e,
        "train_loss": train_losses[e - 1],
        "val_loss":   val_losses[e - 1],
    })

# Always include the best-validation epoch if it isn't already in the list
if best_epoch not in [r["epoch"] for r in rows]:
    rows.append({
        "epoch": best_epoch,
        "train_loss": train_losses[best_epoch - 1],
        "val_loss":   val_losses[best_epoch - 1],
    })
    rows.sort(key=lambda r: r["epoch"])

# CSV
csv_path = OUT_DIR / "loss_table.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for r in rows:
        writer.writerow([r["epoch"], f"{r['train_loss']:.6f}", f"{r['val_loss']:.6f}"])
print(f"Saved: {csv_path}")

# Markdown (paste-able into Word / LaTeX after a small reformat)
md_path = OUT_DIR / "loss_table.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("| Epoch | Training loss | Validation loss |\n")
    f.write("|------:|--------------:|----------------:|\n")
    for r in rows:
        mark = " *"  if r["epoch"] == best_epoch else ""
        f.write(f"| {r['epoch']} | {r['train_loss']:.6f} | {r['val_loss']:.6f}{mark} |\n")
    f.write(f"\n* best validation loss (epoch {best_epoch}, val = {best_val:.6f})\n")
print(f"Saved: {md_path}")

# Console preview
print("\n--- Loss table preview ---")
print(f"{'Epoch':>5} | {'Train':>10} | {'Val':>10}")
print("-" * 32)
for r in rows:
    mark = "  <-- best" if r["epoch"] == best_epoch else ""
    print(f"{r['epoch']:>5} | {r['train_loss']:>10.6f} | {r['val_loss']:>10.6f}{mark}")