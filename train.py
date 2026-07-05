r"""
train.py
GPR Autoencoder Training Script (paper-faithful).

Denoising bottleneck convolutional autoencoder trained on NORMAL-ONLY
GPR B-scan windows. Paper Section IV-B: denoising objective (sigma=0.05),
horizontal-flip augmentation only (physically valid; other transforms
were evaluated and rejected as non-physical), best-validation-loss
checkpointing.

Expected data layout (relative to this script, or override via the
constants below):
       data/Training_and_Validation/Site_A/            (normal-only training pool)
       data/Inference_with_separate_folders/normal/    (normal test images)
       data/Inference_with_separate_folders/tunnel/    (tunnel test images)
"""

from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# BLOCK 1: PATHS AND MAIN SETTINGS
# =============================================================================
# Your paths are given here. Use r"..." raw strings for Windows paths.
# This script collects images recursively from Site A, including all subfolders.

SCRIPT_DIR = Path(__file__).resolve().parent

NORMAL_DIRS = [
    Path("data") / "Training_and_Validation" / "Site_A",
]

TEST_DIRS = Path("data") / "Inference_with_separate_folders"

OUTPUT_DIR = SCRIPT_DIR / "Results" / "Training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SCRIPT_DIR / "gpr_final_model.pt"
BEST_MODEL_PATH = SCRIPT_DIR / "gpr_final_model.pt"   # add this line
CHECKPOINT_PATH = SCRIPT_DIR / "gpr_model_checkpoint.pth"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_validation_loss_curve.png"
RANDOM_RAW_PREVIEW_PATH = OUTPUT_DIR / "preview_random_raw_training_image.png"
TRAINING_SAMPLES_PREVIEW_PATH = OUTPUT_DIR / "preview_training_samples_fed_to_model.png"
COMPARISON_PLOT_PATH = OUTPUT_DIR / "reconstruction_comparison.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 50
SAVE_INTERVAL = 1
NOISE_STD = 0.05 

# Train/validation split from the normal Site A images.
TRAIN_RATIO = 0.80

# If True, repeats the training dataset so random augmentation creates more
# variations per epoch. If USE_AUGMENTATION=False, keep this as 1.
AUGMENT_FACTOR = 10

# Set this to True only if you want to continue from an existing checkpoint.
# Keeping it False avoids accidentally skipping training because an old checkpoint exists.
RESUME_TRAINING = True

# Preview controls. These run before training starts.
SHOW_PREVIEW_BEFORE_TRAINING = False
SAVE_PREVIEW_TO_DISK = True
NUM_TRAINING_SAMPLES_TO_VIEW = 12

# Windows VS Code works most reliably with NUM_WORKERS = 0.
NUM_WORKERS = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =============================================================================
# BLOCK 2: REQUIRED PREPROCESSING TRANSFORM
# =============================================================================
# The model requires every image to become a 1-channel 128x128 tensor.
# Do not remove this basic preprocessing unless you also redesign the model.

basic_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# This clean transform is used for validation, preview comparison, and evaluation.
inference_transform = basic_transform


# =============================================================================
# BLOCK 3: OPTIONAL AUGMENTATION TRANSFORM
# =============================================================================
# To skip augmentation, change USE_AUGMENTATION to False.
# The code will still resize images and convert them to tensors using basic_transform.

USE_AUGMENTATION = True

augmentation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

train_transform = augmentation_transform if USE_AUGMENTATION else basic_transform

# If augmentation is off, do not artificially multiply the dataset.
EFFECTIVE_AUGMENT_FACTOR = AUGMENT_FACTOR if USE_AUGMENTATION else 1


# =============================================================================
# BLOCK 4: REPRODUCIBILITY HELPERS
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# BLOCK 5: IMAGE COLLECTION AND TRAIN/VALIDATION SPLIT
# =============================================================================

def collect_images_from_folder(folder: Path) -> List[Path]:
    """
    Recursively collect all image files from a folder and all its subfolders.
    Example: Site A may contain many internal subdirectories; this function scans them all.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"WARNING: Folder does not exist: {folder}")
        return []

    image_files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_files)


def collect_normal_training_images(normal_dirs: Sequence[Path]) -> List[Path]:
    """
    Collect ALL training images recursively from the configured folder(s).

    Important:
    This version does NOT skip paths containing words like 'tunnel', 'hole',
    or anything else. If an image is inside Site A and has a valid image
    extension, it will be included.
    """
    files: List[Path] = []
    for folder in normal_dirs:
        found = collect_images_from_folder(folder)
        print(f"Collected {len(found)} images from: {folder}")
        files.extend(found)

    # Remove duplicates while preserving order.
    files = list(dict.fromkeys(files))

    if not files:
        checked = "\n".join(str(p) for p in normal_dirs)
        raise FileNotFoundError(
            "No training images found. Checked these folder(s):\n" + checked
        )

    print(f"Total training images collected after duplicate removal: {len(files)}")
    return files


def split_train_validation(files: Sequence[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """Split image paths into train and validation file lists."""
    files = list(files)
    if len(files) < 2:
        raise ValueError("At least 2 images are required to create a train/validation split.")

    rng = random.Random(seed)
    rng.shuffle(files)

    val_count = max(1, int(round(len(files) * (1.0 - train_ratio))))
    train_count = len(files) - val_count

    if train_count < 1:
        raise ValueError("Train split became empty. Add more images or increase TRAIN_RATIO.")

    train_files = files[:train_count]
    val_files = files[train_count:]
    return train_files, val_files


# =============================================================================
# BLOCK 6: DATASET CLASS
# =============================================================================

class GPRImageDataset(Dataset):
    """
    Dataset for GPR image paths.

    factor is used only for training augmentation. If factor=10, each original
    image appears 10 times per epoch, but random transforms make each pass look different.
    """
    def __init__(self, files: Sequence[Path], transform=None, factor: int = 1):
        self.files = list(files)
        self.transform = transform
        self.factor = max(1, int(factor))

        if not self.files:
            raise ValueError("Dataset received an empty file list.")

    def __len__(self) -> int:
        return len(self.files) * self.factor

    def __getitem__(self, idx: int):
        actual_idx = idx % len(self.files)
        img_path = self.files[actual_idx]

        image = Image.open(img_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)

        return image, str(img_path)


# =============================================================================
# BLOCK 7: PREVIEW IMAGES BEFORE TRAINING
# =============================================================================

def show_random_raw_training_image(train_files: Sequence[Path]) -> None:
    """Show/save one raw random image from the training files before transform."""
    img_path = random.choice(list(train_files))
    image = Image.open(img_path).convert("L")

    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap="gray")
    plt.title(f"Random RAW training image\n{img_path}")
    plt.axis("off")
    plt.tight_layout()

    if SAVE_PREVIEW_TO_DISK:
        plt.savefig(RANDOM_RAW_PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved raw preview image: {RANDOM_RAW_PREVIEW_PATH}")

    plt.show()
    plt.close()


def show_training_samples_fed_to_model(train_dataset: Dataset, num_samples: int = 12) -> None:
    """
    Show/save transformed samples exactly as the model receives them.
    If augmentation is on, these images may be flipped/rotated/jittered.
    """
    n = min(num_samples, len(train_dataset))
    indices = random.sample(range(len(train_dataset)), k=n)

    images = []
    names = []
    for idx in indices:
        img_tensor, path_str = train_dataset[idx]
        images.append(img_tensor.squeeze(0).numpy())
        names.append(Path(path_str).name)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, img in enumerate(images):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(names[i], fontsize=9)
        axes[i].axis("off")

    aug_status = "ON" if USE_AUGMENTATION else "OFF"
    fig.suptitle(
        f"Training samples as fed to model | Resize={IMG_SIZE}x{IMG_SIZE} | Augmentation={aug_status}",
        fontsize=14,
    )
    plt.tight_layout()

    if SAVE_PREVIEW_TO_DISK:
        plt.savefig(TRAINING_SAMPLES_PREVIEW_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved transformed training samples preview: {TRAINING_SAMPLES_PREVIEW_PATH}")

    plt.show()
    plt.close()


# =============================================================================
# BLOCK 8: MODEL ARCHITECTURE
# =============================================================================

class GPRBottleneckAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 1 x 128 x 128 -> 256 latent features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),      # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),     # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),    # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),   # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),                  # IMPORTANT: 256, not 32
            nn.ReLU(),
        )

        # Decoder: 256 latent features -> 1 x 128 x 128 reconstructed image
        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 8 * 8),                  # IMPORTANT: 256, not 32
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 1 x 128 x 128
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# =============================================================================
# BLOCK 9: CHECKPOINT, LOSS PLOT, VALIDATION
# =============================================================================

def load_checkpoint_if_requested(model, optimizer):
    start_epoch = 1
    step_losses: List[float] = []
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []

    if RESUME_TRAINING and CHECKPOINT_PATH.exists():
        print(f"Found checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        step_losses = checkpoint.get("step_losses", [])
        train_epoch_losses = checkpoint.get("train_epoch_losses", checkpoint.get("epoch_losses", []))
        val_epoch_losses = checkpoint.get("val_epoch_losses", [])
        print(f"Resuming from epoch {start_epoch}")
    else:
        if CHECKPOINT_PATH.exists() and not RESUME_TRAINING:
            print("Existing checkpoint found, but RESUME_TRAINING=False, so training will start fresh.")
        else:
            print("No checkpoint found. Starting fresh training.")

    return start_epoch, step_losses, train_epoch_losses, val_epoch_losses


def save_checkpoint(epoch, model, optimizer, step_losses, train_epoch_losses, val_epoch_losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_losses": step_losses,
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
    }, CHECKPOINT_PATH)
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")


def evaluate_validation_loss(model, val_loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            recon = model(imgs)
            loss = criterion(recon, imgs)
            total_loss += float(loss.item())
    return total_loss / max(1, len(val_loader))


def plot_training_validation_loss(step_losses, train_epoch_losses, val_epoch_losses):
    plt.figure(figsize=(11, 6))

    if step_losses:
        plt.plot(step_losses, label="Batch training loss", alpha=0.25)

    if train_epoch_losses:
        x_train = np.linspace(0, len(step_losses), len(train_epoch_losses))
        plt.plot(x_train, train_epoch_losses, label="Epoch training loss", linewidth=2)

    if val_epoch_losses:
        x_val = np.linspace(0, len(step_losses), len(val_epoch_losses))
        plt.plot(x_val, val_epoch_losses, label="Epoch validation loss", linewidth=2)

    plt.title("Training and Validation Loss")
    plt.xlabel("Global training steps")
    plt.ylabel("MSE reconstruction loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved loss curve: {LOSS_PLOT_PATH}")


# =============================================================================
# BLOCK 10: RECONSTRUCTION COMPARISON AFTER TRAINING
# =============================================================================

def save_reconstruction_comparison(model, train_files: Sequence[Path], val_files: Sequence[Path]) -> None:
    """Save a visual comparison of original vs reconstructed images."""
    model.eval()
    plot_items = []

    normal_pool = list(train_files) + list(val_files)
    normal_sample = random.sample(normal_pool, k=min(3, len(normal_pool)))

    tunnel_dir = TEST_DIRS / "tunnel"
    tunnel_files = collect_images_from_folder(tunnel_dir) if tunnel_dir.exists() else []
    tunnel_sample = random.sample(tunnel_files, k=min(3, len(tunnel_files))) if tunnel_files else []

    with torch.no_grad():
        for img_path in normal_sample:
            image = Image.open(img_path).convert("L")
            img = inference_transform(image).unsqueeze(0).to(DEVICE)
            recon = model(img)
            mse = torch.mean((img - recon) ** 2).item()
            plot_items.append(("NORMAL", img_path, mse, img.cpu(), recon.cpu()))

        for img_path in tunnel_sample:
            image = Image.open(img_path).convert("L")
            img = inference_transform(image).unsqueeze(0).to(DEVICE)
            recon = model(img)
            mse = torch.mean((img - recon) ** 2).item()
            plot_items.append(("TUNNEL", img_path, mse, img.cpu(), recon.cpu()))

    if not plot_items:
        print("No images available for reconstruction comparison.")
        return

    fig, axes = plt.subplots(len(plot_items), 2, figsize=(10, 3 * len(plot_items)))
    if len(plot_items) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (label, img_path, mse, orig, recon) in enumerate(plot_items):
        axes[i, 0].imshow(orig[0][0], cmap="gray")
        axes[i, 0].set_title(f"[{label}] {img_path.name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon[0][0], cmap="gray")
        axes[i, 1].set_title(f"Reconstruction | MSE: {mse:.6f}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved reconstruction comparison: {COMPARISON_PLOT_PATH}")


# =============================================================================
# BLOCK 11: MAIN TRAINING PIPELINE
# =============================================================================

def run_project():
    set_seed(SEED)

    print("=" * 70)
    print("GPR AUTOENCODER TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("Normal training folder(s):")
    for folder in NORMAL_DIRS:
        print(f"  - {folder}")
    print(f"Inference/test root: {TEST_DIRS}")
    print(f"Model will be saved to: {MODEL_PATH}")
    print(f"Output previews/reports will be saved to: {OUTPUT_DIR}")
    print(f"Augmentation enabled: {USE_AUGMENTATION}")
    print("=" * 70)

    all_normal_files = collect_normal_training_images(NORMAL_DIRS)
    train_files, val_files = split_train_validation(all_normal_files, TRAIN_RATIO, SEED)

    print("\nDataset split:")
    print(f"  Total normal images collected recursively: {len(all_normal_files)}")
    print(f"  Training original images: {len(train_files)}")
    print(f"  Validation original images: {len(val_files)}")
    print(f"  Training samples per epoch after factor: {len(train_files) * EFFECTIVE_AUGMENT_FACTOR}")

    train_ds = GPRImageDataset(train_files, transform=train_transform, factor=EFFECTIVE_AUGMENT_FACTOR)
    val_ds = GPRImageDataset(val_files, transform=inference_transform, factor=1)

    # Preview images before training starts.
    if SHOW_PREVIEW_BEFORE_TRAINING:
        print("\nShowing preview images before training starts...")
        show_random_raw_training_image(train_files)
        show_training_samples_fed_to_model(train_ds, NUM_TRAINING_SAMPLES_TO_VIEW)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = GPRBottleneckAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch, step_losses, train_epoch_losses, val_epoch_losses = load_checkpoint_if_requested(model, optimizer)

    best_val = min(val_epoch_losses) if val_epoch_losses else float("inf")

    print("\nStarting training...")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        
        

        for batch_idx, (imgs, _) in enumerate(train_loader):    
            imgs = imgs.to(DEVICE)
            noisy = (imgs + NOISE_STD * torch.randn_like(imgs)).clamp(0.0, 1.0)
            recon = model(noisy)
            loss = criterion(recon, imgs)   # loss vs CLEAN image

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            step_losses.append(loss_value)
            running_train_loss += loss_value

            if (batch_idx + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}], "
                    f"Step [{batch_idx + 1}/{len(train_loader)}], "
                    f"Train Loss: {loss_value:.6f}"
                )

        avg_train_loss = running_train_loss / max(1, len(train_loader))
        avg_val_loss = evaluate_validation_loss(model, val_loader, criterion)

        train_epoch_losses.append(avg_train_loss)
        val_epoch_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch}/{EPOCHS}] completed | "
            f"Avg Train Loss: {avg_train_loss:.6f} | "
            f"Avg Val Loss: {avg_val_loss:.6f}"
        )
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ** New best validation loss: {best_val:.6f} — saved {BEST_MODEL_PATH.name}")

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            save_checkpoint(epoch, model, optimizer, step_losses, train_epoch_losses, val_epoch_losses)
            plot_training_validation_loss(step_losses, train_epoch_losses, val_epoch_losses)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nFinal model weights saved: {MODEL_PATH}")

    plot_training_validation_loss(step_losses, train_epoch_losses, val_epoch_losses)
    save_reconstruction_comparison(model, train_files, val_files)

    print("\nTraining complete.")


if __name__ == "__main__":
    run_project()