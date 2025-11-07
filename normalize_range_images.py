import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

# Define paths
base_path = Path(r"C:\Users\muham\OneDrive - TU Eindhoven\Extended-evaluation-snowpole-lidar-dataset\SnowPole_Detection_Dataset")
range_path = base_path / "range"
output_path = base_path / "range-normalized"

# Create output directories
for split in ['train', 'valid', 'test']:
    (output_path / split).mkdir(parents=True, exist_ok=True)

def analyze_image_statistics(image_path):
    """Analyze the statistics of a single image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    return {
        'min': np.min(img),
        'max': np.max(img),
        'mean': np.mean(img),
        'std': np.std(img),
        'median': np.median(img),
        'dtype': img.dtype,
        'shape': img.shape
    }

def min_max_normalization(img):
    """Min-Max normalization to [0, 255]"""
    img_float = img.astype(np.float32)
    img_min = np.min(img_float)
    img_max = np.max(img_float)
    if img_max - img_min > 0:
        normalized = ((img_float - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img, dtype=np.uint8)
    return normalized

def percentile_normalization(img, lower_percentile=2, upper_percentile=98):
    """Percentile-based normalization (robust to outliers)"""
    img_float = img.astype(np.float32)
    p_low = np.percentile(img_float, lower_percentile)
    p_high = np.percentile(img_float, upper_percentile)
    
    # Clip values to percentile range
    img_clipped = np.clip(img_float, p_low, p_high)
    
    if p_high - p_low > 0:
        normalized = ((img_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img, dtype=np.uint8)
    return normalized

def histogram_equalization(img):
    """Histogram equalization for better contrast"""
    if len(img.shape) == 3:
        # Convert to grayscale if needed
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Normalize to 8-bit first if needed
    if img_gray.dtype != np.uint8:
        img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)
    
    equalized = cv2.equalizeHist(img_gray)
    return equalized

def clahe_normalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Normalize to 8-bit first if needed
    if img_gray.dtype != np.uint8:
        img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    normalized = clahe.apply(img_gray)
    return normalized

def z_score_normalization(img):
    """Z-score normalization (standardization)"""
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    std = np.std(img_float)
    
    if std > 0:
        z_normalized = (img_float - mean) / std
        # Scale to [0, 255]
        z_min = np.min(z_normalized)
        z_max = np.max(z_normalized)
        if z_max - z_min > 0:
            normalized = ((z_normalized - z_min) / (z_max - z_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
    else:
        normalized = np.zeros_like(img, dtype=np.uint8)
    
    return normalized

def plot_comparison(original, normalized_dict, sample_name, output_dir):
    """Plot original vs normalized images with histograms"""
    n_methods = len(normalized_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].hist(original.ravel(), bins=256, range=(0, original.max()), color='blue', alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Frequency')
    
    # Normalized images
    for idx, (method_name, norm_img) in enumerate(normalized_dict.items(), 1):
        axes[0, idx].imshow(norm_img, cmap='gray')
        axes[0, idx].set_title(method_name)
        axes[0, idx].axis('off')
        
        axes[1, idx].hist(norm_img.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7)
        axes[1, idx].set_title(f'{method_name} Histogram')
        axes[1, idx].set_xlabel('Intensity')
        axes[1, idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{sample_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def calculate_metrics(img):
    """Calculate image quality metrics"""
    return {
        'contrast': np.std(img),
        'dynamic_range': np.max(img) - np.min(img),
        'entropy': -np.sum(np.histogram(img, bins=256, range=(0, 255))[0] / img.size * 
                          np.log2(np.histogram(img, bins=256, range=(0, 255))[0] / img.size + 1e-10))
    }

# Main processing
print("=" * 80)
print("RANGE IMAGE NORMALIZATION ANALYSIS")
print("=" * 80)

# Step 1: Analyze statistics of sample images
print("\n[1] Analyzing image statistics...")
sample_images = list(range_path.glob("train/*.png"))[:5]  # Analyze first 5 images

print(f"\nFound {len(sample_images)} sample images for analysis\n")

all_stats = []
for img_path in sample_images:
    stats = analyze_image_statistics(img_path)
    all_stats.append(stats)
    print(f"Image: {img_path.name}")
    print(f"  Shape: {stats['shape']}, Dtype: {stats['dtype']}")
    print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Median: {stats['median']:.2f}")
    print()

# Step 2: Test different normalization techniques on a sample image
print("\n[2] Testing normalization techniques on sample image...")
sample_img_path = sample_images[0]
original_img = cv2.imread(str(sample_img_path), cv2.IMREAD_UNCHANGED)

if len(original_img.shape) == 3:
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

print(f"\nSample image: {sample_img_path.name}")
print(f"Original - Min: {np.min(original_img)}, Max: {np.max(original_img)}, "
      f"Mean: {np.mean(original_img):.2f}, Std: {np.std(original_img):.2f}")

# Apply all normalization techniques
normalized_images = {
    'Min-Max': min_max_normalization(original_img),
    'Percentile (2-98)': percentile_normalization(original_img, 2, 98),
    'Histogram Eq': histogram_equalization(original_img),
    'CLAHE': clahe_normalization(original_img),
    'Z-Score': z_score_normalization(original_img)
}

print("\n" + "=" * 80)
print("NORMALIZATION RESULTS")
print("=" * 80)

metrics_comparison = {}
for method_name, norm_img in normalized_images.items():
    metrics = calculate_metrics(norm_img)
    metrics_comparison[method_name] = metrics
    print(f"\n{method_name}:")
    print(f"  Range: [{np.min(norm_img)}, {np.max(norm_img)}]")
    print(f"  Mean: {np.mean(norm_img):.2f}, Std: {np.std(norm_img):.2f}")
    print(f"  Contrast (std): {metrics['contrast']:.2f}")
    print(f"  Dynamic Range: {metrics['dynamic_range']}")
    print(f"  Entropy: {metrics['entropy']:.2f}")

# Step 3: Create comparison plots
print("\n[3] Creating comparison plots...")
comparison_dir = output_path / "comparison_plots"
comparison_dir.mkdir(exist_ok=True)

plot_comparison(original_img, normalized_images, sample_img_path.stem, comparison_dir)
print(f"Comparison plot saved to: {comparison_dir / f'{sample_img_path.stem}_comparison.png'}")

# Step 4: Recommendation
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find best method based on entropy (information content) and contrast
best_method = max(metrics_comparison.items(), 
                  key=lambda x: x[1]['entropy'] * 0.6 + x[1]['contrast'] * 0.4)

print(f"\nBased on the analysis:")
print(f"  - Highest entropy (information content): {max(metrics_comparison.items(), key=lambda x: x[1]['entropy'])[0]}")
print(f"  - Highest contrast: {max(metrics_comparison.items(), key=lambda x: x[1]['contrast'])[0]}")
print(f"  - Best overall (weighted): {best_method[0]}")

print(f"\n>>> RECOMMENDED METHOD: {best_method[0]}")
print(f"\nRationale:")
if 'CLAHE' in best_method[0]:
    print("  - CLAHE provides adaptive local contrast enhancement")
    print("  - Reduces noise amplification compared to global histogram equalization")
    print("  - Preserves local details while enhancing overall contrast")
elif 'Percentile' in best_method[0]:
    print("  - Percentile normalization is robust to outliers")
    print("  - Prevents extreme values from dominating the normalization")
    print("  - Good for data with varying intensity distributions")
elif 'Min-Max' in best_method[0]:
    print("  - Simple and effective for uniform intensity distributions")
    print("  - Preserves relative intensity relationships")
elif 'Histogram Eq' in best_method[0]:
    print("  - Maximizes contrast across the entire image")
    print("  - Good for images with poor contrast")
else:
    print("  - Standardizes intensity distribution")

# Step 5: Apply best method to all images
print(f"\n[4] Applying {best_method[0]} to all images...")

# Determine which normalization function to use
if 'CLAHE' in best_method[0]:
    normalize_func = clahe_normalization
elif 'Percentile' in best_method[0]:
    normalize_func = percentile_normalization
elif 'Histogram Eq' in best_method[0]:
    normalize_func = histogram_equalization
elif 'Z-Score' in best_method[0]:
    normalize_func = z_score_normalization
else:
    normalize_func = min_max_normalization

# Process all images
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
        
        # Normalize
        normalized = normalize_func(img)
        
        # Save
        output_file = output_split_path / img_path.name
        cv2.imwrite(str(output_file), normalized)

print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)
print(f"\nNormalized images saved to: {output_path}")
print(f"Comparison plots saved to: {comparison_dir}")
print(f"\nMethod used: {best_method[0]}")
print("\nAll images have been normalized to [0, 255] intensity range.")
