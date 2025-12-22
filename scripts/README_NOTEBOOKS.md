# Dual-Branch YOLOv9t Notebooks

## 📁 Files Created

1. **`dual_net_comb4_range_signal_reflec_v9t_COLAB.ipynb`** - Google Colab version
2. **`dual_net_comb4_range_signal_reflec_v9t_DESKTOP.ipynb`** - Desktop/Local version

## 🚀 Quick Start

### For Google Colab

1. Upload `dual_net_comb4_range_signal_reflec_v9t_COLAB.ipynb` to Google Colab
2. Mount your Google Drive when prompted
3. Ensure your dataset is in `/content/drive/MyDrive/SnowPole_Detection_Dataset/`
4. Run cells sequentially

### For Desktop (Windows/Linux/Mac)

1. Open `dual_net_comb4_range_signal_reflec_v9t_DESKTOP.ipynb` in Jupyter Lab/Notebook
2. Update the `BASE_PATH` in cell 3 to point to your local dataset
3. Install dependencies: `pip install ultralytics torch torchvision opencv-python numpy matplotlib pyyaml tqdm`
4. Run cells sequentially

## 🔧 Key Differences

| Feature | Colab Version | Desktop Version |
|---------|---------------|-----------------|
| **Drive Mount** | ✅ Google Drive mounting | ❌ No drive mounting |
| **Paths** | `/content/drive/MyDrive/...` | `C:\Users\...` (customizable) |
| **GPU** | Free T4 GPU | Your local GPU/CPU |
| **Training** | Shell commands (`!yolo train ...`) | Python commands (`os.system(...)`) |
| **Batch Size** | 16 (more VRAM) | 8 (conservative) |
| **Epochs** | 400 | 100 (for testing) |

## 📊 Dataset Structure Required

Both notebooks expect this structure:
```
SnowPole_Detection_Dataset/
├── images/
│   ├── images/
│   │   ├── train/*.png  (3-channel comb4)
│   │   ├── val/*.png
│   │   └── test/*.png
├── range-normalized-continuous/
│   ├── train/*.npy  (float32 [0,1])
│   ├── val/*.npy
│   └── test/*.npy
├── labels/
│   ├── train/*.txt  (YOLO format)
│   ├── val/*.txt
│   └── test/*.txt
└── comb4-range-signal-reflec_and_range_80m/  (output)
```

## 🧪 Testing Workflow

### Desktop Version Test
```python
# 1. Test with sample images first (Cell 4)
test_dual_pipeline_sample()

# 2. If successful, create full dataset (Cell 5)
for split in ["train", "val", "test"]:
    make_dual_split(split)

# 3. Modify model and train (Cells 6-9)
```

### Colab Version Test
- Cell 11: Tests on 1-2 images with visualization
- Cell 12: Processes full dataset if test passes
- Cells 13-16: Model training and validation

## ⚙️ Customization

### Adjust Paths (Desktop)
```python
# In cell 3, update:
BASE_PATH = Path(r"YOUR_DATASET_PATH_HERE")
```

### Adjust Training Parameters
```python
# Desktop version (cell 8):
epochs=100  # Increase for production
batch=8     # Increase if GPU has more VRAM
device=0    # Change to 'cpu' if no GPU

# Colab version (cell 15):
epochs=400
batch=16
patience=50
```

## 🔍 Verification

Both notebooks include test cells that:
1. Load 1-2 sample images
2. Verify .npy range files load correctly
3. Check shape compatibility
4. Display statistics
5. Create test 4-channel outputs

## 📚 Paper Reference

Yang et al., "Towards Generalized Range-View LiDAR Segmentation in Adverse Weather" (2025)
- arXiv: [2506.08979](https://arxiv.org/abs/2506.08979)
- Implements dual-branch architecture:
  - **Geometric branch**: Log-normalized range (80m)
  - **Reflectance branch**: Comb4 (Near-IR, Signal, Reflectivity)

## ⚠️ Common Issues

1. **Missing .npy files**: Ensure you've run the range normalization script first
2. **CUDA out of memory**: Reduce batch size
3. **Path not found**: Update `BASE_PATH` in desktop version
4. **Import errors**: Install all dependencies listed above

## 📝 Notes

- Desktop version tested for syntax validity ✅
- All cells are Python 3.8+ compatible
- Uses log-normalized range: `log1p(clip(range, 0, 80)) / log1p(80)`
- Sensor: Ouster OS2-128 (80m @ 10% reflectivity)
