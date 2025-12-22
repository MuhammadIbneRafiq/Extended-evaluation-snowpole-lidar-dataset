import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _resolve_dataset_root() -> Path:
    """Mirror the notebook's path resolution so the script works in VS Code and Colab."""
    env_root = os.environ.get("SNOWPOLE_DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    scripts_dir = Path(__file__).resolve().parent
    candidates = [
        scripts_dir.parent / "SnowPole_Detection_Dataset",
        scripts_dir.parent / "data" / "SnowPole_Detection_Dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fall back to first candidate and create it so downstream code can populate it
    fallback = candidates[0]
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


# Define paths dynamically
base_path = _resolve_dataset_root()
range_path = base_path / "range"
# this will contain continuous [0,1] float32 tensors as .npy
output_path = base_path / "range-normalized-continuous"

# Create output directories
for split in ['train', 'valid', 'test']:
    (output_path / split).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) Log normalization with R_max = 80 m
# ---------------------------------------------------------------------

R_MAX_METERS = 80.0  # effective LiDAR range (OS2-128 ~80m @ 10% reflectivity)

def range_log_norm_0_1(img, max_range_m=R_MAX_METERS):
    """
    Logarithmic normalization of range image to [0, 1]. Inspired by the LiDARGen paper 

    - Clip to [0, max_range_m]
    - Apply log1p so nearby points get higher resolution
    - Divide by log1p(max_range_m) so  max_range_m -> 1.0

    Assumes 'img' stores range in meters (or proportional to meters).
    """
    img_float = img.astype(np.float32)

    # clip to [0, max_range_m] so extreme values don't dominate
    img_clipped = np.clip(img_float, 0.0, max_range_m)

    # log-scale mapping to [0,1]
    norm_0_1 = np.log1p(img_clipped) / np.log1p(max_range_m)
    return norm_0_1  # float32 in [0,1]


def range_log_norm_vis_uint8(img, max_range_m=R_MAX_METERS):
    """
    Visualization-friendly version of range_log_norm_0_1:
    returns an 8-bit [0,255] image for plotting / OpenCV saving.
    """
    norm_0_1 = range_log_norm_0_1(img, max_range_m)
    return (norm_0_1 * 255.0).astype(np.uint8)

# ---------------------------------------------------------------------
# 2) Debug helpers (optional, for checking what training sees)
# ---------------------------------------------------------------------

def show_training_view(img_path):
    """
    Show exactly what the training tensor looks like:
    - range_log_norm_0_1 in [0,1] (no *255)
    - fixed vmin=0, vmax=1 so there is NO autoscaling
    - histogram over [0,1]
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    norm_0_1 = range_log_norm_0_1(img)   # this is what you'll feed to the network

    print(f"norm_0_1 stats for {img_path.name}:")
    print(f"  min:  {norm_0_1.min():.4f}")
    print(f"  max:  {norm_0_1.max():.4f}")
    print(f"  mean: {norm_0_1.mean():.4f}")
    print(f"  std:  {norm_0_1.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # LEFT: the training tensor visualized with fixed 0–1 mapping
    im = axes[0].imshow(norm_0_1, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Training tensor (0–1, no autoscale)")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Value in [0,1]")

    # RIGHT: histogram of the values in [0,1]
    axes[1].hist(norm_0_1.ravel(), bins=50, range=(0.0, 1.0))
    axes[1].set_title("Value distribution")
    axes[1].set_xlabel("Value in [0,1]")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_old_vs_new_side_by_side(img_paths, output_dir):
    """
    For the given image paths, plot Original vs
    New log 80m → [0,1] normalization (visualized as 0-255) side by side.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        new_norm_vis = range_log_norm_vis_uint8(img)  # uint8 [0,255] from log scaling

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(new_norm_vis, cmap='gray')
        axes[1].set_title('Log norm (80m → [0,1] → vis)')
        axes[1].axis('off')

        plt.tight_layout()
        out_name = output_dir / f"{img_path.stem}_log_norm_comparison.png"
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison for {img_path.name} → {out_name}")

# ---------------------------------------------------------------------
# 3) Quick stats + comparisons on a few images (optional)
# ---------------------------------------------------------------------

print("\n[1] Analyzing image statistics...")
sample_images = list(range_path.glob("train/*.png"))[:5]

print(f"\nFound {len(sample_images)} sample images for analysis\n")

for img_path in sample_images:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image: {img_path.name}, shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

# Optional: visualize a couple of training tensors
# show_training_view(sample_images[0])

print("\n[3] Comparing ORIGINAL vs NEW log 80m → [0,1] normalization on 4 images...")
side_by_side_dir = output_path / "log_norm_80m_comparisons"
# Uncomment if you want to generate these comparison PNGs:
# plot_old_vs_new_side_by_side(sample_images[:4], side_by_side_dir)

# ---------------------------------------------------------------------
# 4) Process *all* images and save continuous [0,1] tensors as .npy
# ---------------------------------------------------------------------

print("\n[4] Processing full dataset with log 80m → [0,1] normalization...")

for split in ['train', 'valid', 'test']:
    split_path = range_path / split
    output_split_path = output_path / split
    
    image_files = list(split_path.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"  No images found in {split} split")
        continue
    
    print(f"\n  Processing {split} split ({len(image_files)} images)...")
    
    for img_path in tqdm(image_files, desc=f"  {split}"):
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to continuous [0,1]
        norm_0_1 = range_log_norm_0_1(img)  # float32 in [0,1]

        # Save as .npy (continuous, no 0–255 quantization)
        out_npy = output_split_path / f"{img_path.stem}.npy"
        np.save(out_npy, norm_0_1.astype(np.float32))

print(f"\nContinuous [0,1] log-normalized range tensors saved to: {output_path}")