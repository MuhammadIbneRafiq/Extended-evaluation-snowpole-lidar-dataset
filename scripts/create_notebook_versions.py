"""
Create desktop and Colab versions of the dual-branch YOLOv9t notebook
"""

import json
import os
from pathlib import Path

# Base path
BASE_PATH = Path(r"C:\Users\muham\OneDrive - TU Eindhoven\Extended-evaluation-snowpole-lidar-dataset")
SCRIPTS_PATH = BASE_PATH / "scripts"

def create_desktop_notebook():
    """Create desktop version of the notebook"""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Dual-Branch YOLOv9t for Snow Pole Detection - DESKTOP VERSION\n",
            "\n",
            "## Overview\n",
            "This notebook implements a **dual-branch architecture** for LiDAR-based object detection.\n",
            "\n",
            "**Paper Reference**: Yang et al., \"Towards Generalized Range-View LiDAR Segmentation in Adverse Weather\" (2025) - [arXiv:2506.08979](https://arxiv.org/abs/2506.08979)"
        ]
    })
    
    # Install cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# INSTALL DEPENDENCIES (if needed)\n",
            "# Uncomment if you need to install packages:\n",
            "# !pip install ultralytics torch torchvision opencv-python numpy matplotlib pyyaml tqdm"
        ]
    })
    
    # Imports cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# IMPORTS\n",
            "import torch\n",
            "from ultralytics import YOLO\n",
            "from pathlib import Path\n",
            "import shutil\n",
            "import cv2\n",
            "import numpy as np\n",
            "import yaml\n",
            "import matplotlib.pyplot as plt\n",
            "from tqdm import tqdm\n",
            "import os\n",
            "\n",
            "print(f\"PyTorch version: {torch.__version__}\")\n",
            "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")"
        ]
    })
    
    # Paths configuration
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PATHS CONFIGURATION - DESKTOP VERSION\n",
            "# Update this to your local dataset path\n",
            "BASE_PATH = Path(r\"C:\\Users\\muham\\OneDrive - TU Eindhoven\\Extended-evaluation-snowpole-lidar-dataset\\SnowPole_Detection_Dataset\")\n",
            "\n",
            "# 3-channel comb images (reflectance branch)\n",
            "COMB_ROOT = BASE_PATH / \"images\"\n",
            "\n",
            "# Continuous 1-channel range (log-norm, 80m) saved as .npy\n",
            "RANGE_ROOT = BASE_PATH / \"range-normalized-continuous\"\n",
            "\n",
            "# New 4-channel dual-input dataset (B, G, R, Range)\n",
            "DUAL_ROOT = BASE_PATH / \"comb4-range-signal-reflec_and_range_80m\"\n",
            "DUAL_ROOT.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Labels directory\n",
            "LABELS_ROOT = BASE_PATH / \"labels\"\n",
            "\n",
            "print(\"Dataset paths:\")\n",
            "print(f\"BASE_PATH  : {BASE_PATH}\")\n",
            "print(f\"COMB_ROOT  : {COMB_ROOT}\")\n",
            "print(f\"RANGE_ROOT : {RANGE_ROOT}\")\n",
            "print(f\"DUAL_ROOT  : {DUAL_ROOT}\")\n",
            "print(f\"LABELS_ROOT: {LABELS_ROOT}\")\n",
            "\n",
            "# Verify paths exist\n",
            "for name, path in [(\"COMB\", COMB_ROOT), (\"RANGE\", RANGE_ROOT), (\"LABELS\", LABELS_ROOT)]:\n",
            "    print(f\"{'✅' if path.exists() else '⚠️'} {name}: {path.exists()}\")"
        ]
    })
    
    # Test pipeline function
    test_code = '''# TEST PIPELINE ON SAMPLE IMAGES
def test_dual_pipeline_sample():
    """Test the dual-branch creation on 1-2 sample images"""
    
    # Get 2 sample images from train
    comb_img_dir = COMB_ROOT / "images" / "train"
    range_npy_dir = RANGE_ROOT / "train"
    
    if not comb_img_dir.exists():
        print(f"⚠️ Comb directory not found: {comb_img_dir}")
        return
    
    sample_files = sorted(comb_img_dir.glob("*.png"))[:2]
    
    if len(sample_files) == 0:
        print("⚠️ No images found")
        return
    
    print(f"Testing on {len(sample_files)} images...")
    
    for comb_path in sample_files:
        stem = comb_path.stem
        print(f"Processing: {stem}")
        
        # Read comb
        comb = cv2.imread(str(comb_path), cv2.IMREAD_COLOR)
        print(f"  Comb shape: {comb.shape}")
        
        # Read range .npy
        range_npy_path = range_npy_dir / f"{stem}.npy"
        if not range_npy_path.exists():
            print(f"  Missing: {range_npy_path}")
            continue
        
        range_arr = np.load(range_npy_path)
        print(f"  Range shape: {range_arr.shape}, dtype: {range_arr.dtype}")
        print(f"  Range stats: min={range_arr.min():.3f}, max={range_arr.max():.3f}")
        
        # Convert to uint8
        range_img = (np.clip(range_arr, 0, 1) * 255).astype(np.uint8)
        
        # Stack to 4-channel
        rgba = np.dstack([comb, range_img])
        print(f"  RGBA shape: {rgba.shape}")
        print(f"  ✅ Success\\n")
    
    print("Test completed!")

# Run test
test_dual_pipeline_sample()'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": test_code.split('\n')
    })
    
    # Make dual dataset function
    make_dual_code = '''# CREATE DUAL-BRANCH DATASET
def make_dual_split(split: str):
    """Create 4-channel dual-branch images"""
    comb_img_dir = COMB_ROOT / "images" / split
    range_npy_dir = RANGE_ROOT / split
    dual_img_dir = DUAL_ROOT / "images" / split
    
    dual_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy labels
    src_lbl_dir = LABELS_ROOT / split
    dual_lbl_dir = DUAL_ROOT / "labels" / split
    dual_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if src_lbl_dir.exists():
        for lbl in src_lbl_dir.glob("*.txt"):
            shutil.copy2(lbl, dual_lbl_dir / lbl.name)
    
    if not comb_img_dir.exists():
        print(f"[{split}] Warning: {comb_img_dir} not found")
        return
    
    img_files = sorted(comb_img_dir.glob("*.png"))
    print(f"[{split}] Found {len(img_files)} images")
    
    for comb_path in tqdm(img_files, desc=f"{split}"):
        stem = comb_path.stem
        
        # Read comb (3ch BGR)
        comb = cv2.imread(str(comb_path), cv2.IMREAD_COLOR)
        if comb is None:
            continue
        
        # Read range .npy
        range_npy_path = range_npy_dir / f"{stem}.npy"
        if not range_npy_path.exists():
            continue
        
        range_arr = np.load(range_npy_path)
        if range_arr.ndim == 3:
            range_arr = np.squeeze(range_arr)
        
        # Convert to uint8
        range_img = (np.clip(range_arr, 0, 1) * 255).astype(np.uint8)
        
        # Resize if needed
        if range_img.shape != comb.shape[:2]:
            range_img = cv2.resize(range_img, (comb.shape[1], comb.shape[0]))
        
        # Stack to 4-channel
        rgba = np.dstack([comb, range_img])
        
        out_path = dual_img_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), rgba)
    
    print(f"[{split}] Done!")

# Process all splits
for split in ["train", "val", "test"]:
    make_dual_split(split)'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_dual_code.split('\n')
    })
    
    # Model modification
    model_code = '''# MODIFY YOLOV9T FOR 4-CHANNEL INPUT
# Download model if needed
if not Path("yolov9t.pt").exists():
    print("Downloading YOLOv9t...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt",
        "yolov9t.pt"
    )

model = YOLO("yolov9t.pt")
model.model.eval()

first_conv = model.model.model[0]
old_conv = first_conv.conv

# Create 4-channel conv
new_conv = torch.nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# Transfer weights
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:] = old_conv.weight[:, :1] * 0.1
    if old_conv.bias is not None:
        new_conv.bias = old_conv.bias

first_conv.conv = new_conv
model.model.model[0] = first_conv

# Save model
model.save("yolov9t_dual_4ch.pt")
print("✅ Created yolov9t_dual_4ch.pt")'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": model_code.split('\n')
    })
    
    # Create YAML
    yaml_code = '''# CREATE DATA.YAML
DUAL_DATA_YAML = DUAL_ROOT / "data.yaml"

cfg = {
    "path": str(DUAL_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["snow_pole"],
    "channels": 4  # 4-channel input
}

with open(DUAL_DATA_YAML, "w") as f:
    yaml.safe_dump(cfg, f)

print("✅ Created data.yaml")
print(DUAL_DATA_YAML.read_text())'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": yaml_code.split('\n')
    })
    
    # Training
    train_code = '''# TRAIN MODEL
# Update device based on your system (0 for GPU, 'cpu' for CPU)
device = 0 if torch.cuda.is_available() else 'cpu'

# Build command
cmd = f"""yolo train model=yolov9t_dual_4ch.pt data={DUAL_DATA_YAML} epochs=100 imgsz=1024 device={device} batch=8 patience=20 name=dual_v9t_desktop project=runs"""

print(f"Training command: {cmd}")
# Run training (uncomment to execute)
# os.system(cmd)'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": train_code.split('\n')
    })
    
    # Validation
    val_code = '''# VALIDATE MODEL
device = 0 if torch.cuda.is_available() else 'cpu'

# Build validation command
cmd = f"""yolo val model=runs/dual_v9t_desktop/weights/best.pt data={DUAL_DATA_YAML} split=test imgsz=1024 device={device} batch=8"""

print(f"Validation command: {cmd}")
# Run validation (uncomment to execute)
# os.system(cmd)'''
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": val_code.split('\n')
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Save desktop notebook
    desktop_path = SCRIPTS_PATH / "dual_net_comb4_range_signal_reflec_v9t_DESKTOP.ipynb"
    with open(desktop_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Created desktop notebook: {desktop_path}")

def test_desktop_notebook():
    """Test that desktop notebook cells are valid Python"""
    import ast
    
    desktop_path = SCRIPTS_PATH / "dual_net_comb4_range_signal_reflec_v9t_DESKTOP.ipynb"
    
    with open(desktop_path, 'r') as f:
        notebook = json.load(f)
    
    print("Testing desktop notebook cells...")
    errors = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = '\n'.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            # Skip shell commands
            if source.strip().startswith('!'):
                continue
            try:
                # Try to parse as Python
                ast.parse(source)
                print(f"  Cell {i}: ✅")
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e}")
                print(f"  Cell {i}: ❌ {e}")
    
    if errors:
        print(f"\n⚠️ Found {len(errors)} syntax errors")
    else:
        print("\n✅ All cells valid!")
    
    return len(errors) == 0

if __name__ == "__main__":
    create_desktop_notebook()
    test_desktop_notebook()
