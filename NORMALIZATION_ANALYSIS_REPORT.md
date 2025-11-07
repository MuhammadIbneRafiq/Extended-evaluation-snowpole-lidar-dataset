# Range Image Normalization Analysis Report

## Overview
This report documents the analysis and normalization of LiDAR range images from the SnowPole Detection Dataset. Multiple normalization techniques were evaluated to determine the optimal method for converting 16-bit range data to 8-bit intensity values (0-255).

---

## Original Data Characteristics

### Image Statistics (Sample Analysis)
The original range images have the following characteristics:

| Image | Shape | Data Type | Min | Max | Mean | Std Dev | Median |
|-------|-------|-----------|-----|-----|------|---------|--------|
| image_1.png | 128×1024×3 | uint16 | 0 | 30,771 | 2,644.71 | 3,037.64 | 2,484 |
| image_10.png | 128×1024×3 | uint16 | 0 | 33,462 | 2,635.34 | 3,076.11 | 2,438.5 |
| image_100.png | 128×1024×3 | uint16 | 0 | 32,678 | 2,308.34 | 3,700.67 | 0 |
| image_1000.png | 128×1024×3 | uint16 | 0 | 64,587 | 2,063.68 | 3,861.96 | 0 |
| image_1001.png | 128×1024×3 | uint16 | 0 | 60,835 | 2,073.41 | 3,830.92 | 0 |

**Key Observations:**
- Images are 16-bit unsigned integers with values ranging from 0 to ~65,000
- Wide dynamic range with significant variation between images
- Low mean values relative to maximum possible (65,535)
- High standard deviation indicates diverse intensity distributions
- Some images have median of 0, suggesting many zero/invalid pixels

---

## Normalization Techniques Evaluated

### 1. **Min-Max Normalization**
**Formula:** `(x - min) / (max - min) × 255`

**Results:**
- Range: [0, 255]
- Mean: 21.63, Std: 24.99
- Contrast: 24.99
- Entropy: 4.35

**Characteristics:**
- Simple linear scaling
- Preserves relative intensity relationships
- Sensitive to outliers (extreme values dominate scaling)

---

### 2. **Percentile Normalization (2nd-98th percentile)**
**Formula:** Clip to [p2, p98], then scale to [0, 255]

**Results:**
- Range: [0, 255]
- Mean: 69.69, Std: 71.87
- Contrast: 71.87
- **Entropy: 5.19** ⭐ (Highest information content)

**Characteristics:**
- Robust to outliers
- Clips extreme 2% on each end
- Better utilization of intensity range
- Preserves most data while reducing noise impact

---

### 3. **Histogram Equalization**
**Formula:** Cumulative distribution function mapping

**Results:**
- Range: [0, 255]
- Mean: 75.39, Std: 84.94
- **Contrast: 84.94** ⭐ (Highest contrast)
- Entropy: 4.28

**Characteristics:**
- Maximizes contrast across entire image
- Redistributes intensities for uniform histogram
- Excellent for low-contrast images
- May amplify noise in uniform regions

---

### 4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
**Parameters:** clip_limit=2.0, tile_grid_size=(8×8)

**Results:**
- Range: [1, 255]
- Mean: 31.66, Std: 32.67
- Contrast: 32.67
- Entropy: 4.72

**Characteristics:**
- Adaptive local contrast enhancement
- Reduces noise amplification vs. global histogram equalization
- Preserves local details
- Prevents over-enhancement in uniform regions

---

### 5. **Z-Score Normalization (Standardization)**
**Formula:** `(x - mean) / std`, then scale to [0, 255]

**Results:**
- Range: [0, 255]
- Mean: 21.63, Std: 24.99
- Contrast: 24.99
- Entropy: 4.35

**Characteristics:**
- Centers data around mean
- Standardizes variance
- Similar results to Min-Max for this dataset

---

## Decision Criteria & Recommendation

### Evaluation Metrics
1. **Entropy** (60% weight): Measures information content - higher is better
2. **Contrast** (40% weight): Measures intensity variation - higher is better

### Comparison Summary

| Method | Entropy | Contrast | Weighted Score | Rank |
|--------|---------|----------|----------------|------|
| **Histogram Equalization** | 4.28 | **84.94** | **36.54** | **1st** ⭐ |
| Percentile (2-98) | **5.19** | 71.87 | 31.86 | 2nd |
| CLAHE | 4.72 | 32.67 | 13.90 | 3rd |
| Min-Max | 4.35 | 24.99 | 12.61 | 4th |
| Z-Score | 4.35 | 24.99 | 12.61 | 4th |

---

## ✅ SELECTED METHOD: Histogram Equalization

### Rationale
**Histogram Equalization** was selected as the optimal normalization technique for the following reasons:

1. **Highest Overall Score**: Best weighted combination of entropy and contrast
2. **Maximum Contrast**: Achieves 84.94 std dev, significantly higher than other methods
3. **Optimal for Low-Contrast Data**: The original range images have poor contrast (most values clustered in low range)
4. **Enhanced Feature Visibility**: Redistributes intensities to make subtle features more visible
5. **Standard Practice**: Widely used in computer vision for preprocessing

### Trade-offs Considered
- **Percentile normalization** had higher entropy (5.19 vs 4.28) but lower contrast
- **CLAHE** provides more localized enhancement but lower overall contrast
- For object detection tasks (snow poles), **contrast is critical** for edge detection and feature extraction

---

## Implementation Details

### Processing Pipeline
1. **Input**: 16-bit PNG range images (uint16)
2. **Conversion**: Convert to grayscale if multi-channel
3. **Normalization**: Apply histogram equalization using OpenCV
4. **Output**: 8-bit PNG images (uint8) with values in [0, 255]

### Dataset Statistics
- **Train split**: 1,367 images processed
- **Valid split**: 0 images (empty)
- **Test split**: 197 images processed
- **Total**: 1,564 images normalized

### Output Structure
```
SnowPole_Detection_Dataset/
├── range/                    # Original 16-bit images
│   ├── train/
│   ├── valid/
│   └── test/
└── range-normalized/         # Normalized 8-bit images
    ├── train/                # 1,367 images
    ├── valid/                # 0 images
    ├── test/                 # 197 images
    └── comparison_plots/     # Before/after visualizations
        └── image_1_comparison.png
```

---

## Visualization

A comparison plot has been generated showing:
- **Row 1**: Visual comparison of original vs. all 5 normalization methods
- **Row 2**: Histogram distributions for each method

**Location**: `SnowPole_Detection_Dataset/range-normalized/comparison_plots/image_1_comparison.png`

This visualization demonstrates:
- Original image has most pixels in low intensity range
- Histogram equalization spreads intensities across full [0, 255] range
- Enhanced contrast makes features more distinguishable

---

## Usage Recommendations

### For Training Object Detection Models
✅ **Use the normalized images** in `range-normalized/` folder
- Better convergence during training
- Improved feature extraction
- Consistent intensity range across dataset

### For Visualization
✅ **Use histogram equalized images**
- Enhanced visibility of snow poles
- Better contrast for human inspection
- Easier annotation verification

### For Analysis
⚠️ **Consider percentile normalization** if:
- Dataset has significant outliers
- Need to preserve relative intensity relationships
- Noise reduction is priority

---

## Reproducibility

### Script Location
`normalize_range_images.py`

### Dependencies
```python
- opencv-python (cv2)
- numpy
- matplotlib
- pathlib
- tqdm
```

### Re-run Command
```bash
python normalize_range_images.py
```

---

## Conclusion

**Histogram Equalization** provides the best normalization for this LiDAR range image dataset, achieving:
- ✅ Maximum contrast (84.94 std dev)
- ✅ Full utilization of [0, 255] intensity range
- ✅ Enhanced feature visibility for object detection
- ✅ Standard, well-established technique

All 1,564 images have been successfully normalized and saved to:
`SnowPole_Detection_Dataset/range-normalized/`

---

**Generated**: 2025-11-07  
**Author**: Automated Analysis Pipeline  
**Dataset**: SnowPole Detection Dataset - LiDAR Range Images
