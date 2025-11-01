# Dual Network Pipeline for SnowPole Detection

This repository implements a comprehensive dual network pipeline for single-stage object detection, inspired by YOLOv9t and YOLOv11 architectures, specifically designed for detecting snow poles in Nordic winter conditions.

## Project Structure

```
dual_network_experiments/
│
├── README.md                         # Main documentation
├── requirements.txt                  # Python dependencies
├── config.yaml                      # Main configuration file
│
├── core/                           # Core architecture implementations
│   ├── dual_network.py            # Main dual network architecture
│   ├── attention_modules.py       # Attention mechanisms (EMA, CBAM, etc.)
│   ├── fusion_layers.py           # Different fusion strategies
│   └── merkle_tree.py             # Merkle tree for inference caching
│
├── experiment_1_baseline/          # Baseline RGBA input experiment
│   ├── train.py                   # Training script
│   ├── config.yaml                # Experiment-specific config
│   ├── data_loader.py             # RGBA data loading
│   └── README.md                  # Experiment documentation
│
├── experiment_2_attention/         # Attention module ablation
│   ├── train_with_attention.py    # Training with attention
│   ├── train_without_attention.py # Training without attention
│   ├── compare_results.py         # Performance comparison
│   └── README.md
│
├── experiment_3_modality/          # Modality branch ablation
│   ├── rgb_only.py                # RGB-only training
│   ├── alpha_only.py              # Alpha channel only
│   ├── combined_rgba.py           # Combined RGBA
│   └── README.md
│
├── experiment_5_fusion/            # Fusion architecture test
│   ├── concatenation_fusion.py    # Simple concatenation
│   ├── addition_fusion.py         # Element-wise addition
│   ├── gated_fusion.py            # Learnable gated fusion
│   └── README.md
│
├── experiment_7_merkle/            # Merkle tree caching
│   ├── tile_based_inference.py    # Tile-based caching
│   ├── merkle_cache.py            # Merkle tree implementation
│   ├── benchmark_speed.py         # Speed benchmarking
│   └── README.md
│
├── utils/                          # Utility functions
│   ├── dataset.py                 # Dataset utilities
│   ├── metrics.py                 # Evaluation metrics
│   ├── visualization.py           # Result visualization
│   └── augmentation.py            # Data augmentation
│
└── results/                        # Experiment results
    └── .gitkeep
```

## Overview

This implementation provides:
1. **Dual Network Architecture**: A sophisticated dual-branch network inspired by YOLOv9t and YOLOv11
2. **Multi-modal Fusion**: Support for RGB + additional modality (Range/IR) fusion
3. **Attention Mechanisms**: Efficient Multi-scale Attention (EMA) and CBAM modules
4. **Multiple Fusion Strategies**: Concatenation, addition, and gated fusion
5. **Merkle Tree Caching**: Efficient inference caching for real-time deployment
6. **Comprehensive Ablation Studies**: Systematic evaluation of each component

## Key Features

### 1. Dual Network Architecture
- **Primary Branch**: Processes RGB channels with pretrained weights
- **Secondary Branch**: Processes additional modality (Range/IR/Signal)
- **Cross-branch Attention**: Information exchange between branches
- **Multi-scale Feature Fusion**: Combines features at different scales

### 2. Attention Modules
- **Efficient Multi-scale Attention (EMA)**: Lightweight attention for feature enhancement
- **CBAM**: Channel and spatial attention blocks
- **Cross-modal Attention**: Attention between different modalities

### 3. Fusion Strategies
- **Early Fusion**: Concatenate inputs at the beginning
- **Late Fusion**: Combine features at detection head
- **Hierarchical Fusion**: Progressive fusion at multiple scales

### 4. Merkle Tree Inference
- **Tile-based Processing**: Divide image into tiles
- **Hash-based Caching**: Cache unchanged regions
- **Dynamic Update**: Only process changed tiles

## Experiments

### Experiment 1: Baseline with RGBA Inputs
Establishes baseline performance using 4-channel input (RGB + additional modality).

### Experiment 2: Attention Module Ablation
Compares performance with and without attention modules to quantify their contribution.

### Experiment 3: Modality Branch Ablation
Tests RGB-only, additional modality only, and combined configurations.

### Experiment 5: Fusion Architecture Test
Evaluates different fusion strategies: concatenation, addition, and gated fusion.

### Experiment 7: Merkle Tree Caching
Implements efficient inference caching using Merkle trees for real-time performance.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dual_network_experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Baseline Model
```bash
cd experiment_1_baseline
python train.py --config config.yaml
```

### Running Attention Ablation
```bash
cd experiment_2_attention
python train_with_attention.py
python train_without_attention.py
python compare_results.py
```

### Testing Fusion Strategies
```bash
cd experiment_5_fusion
python concatenation_fusion.py
python addition_fusion.py
python gated_fusion.py
```

## Configuration

Main configuration parameters are in `config.yaml`:
- Model architecture settings
- Training hyperparameters
- Dataset paths
- Fusion strategies
- Attention module configurations

## Dataset Format

The pipeline expects data in YOLO format:
- Images: `.jpg` or `.png` files
- Labels: `.txt` files with format: `class_id x_center y_center width height`
- Single class: `pole` (class_id = 0)

## Performance Metrics

The pipeline evaluates:
- **Precision**: Ratio of true positive detections
- **Recall**: Detection completeness
- **mAP@50**: Mean Average Precision at IoU 0.5
- **mAP@50-95**: Mean Average Precision across IoU thresholds
- **Inference Time**: GPU and CPU latency

## Notes

- All models are designed for single-class detection (pole)
- Supports multiple LiDAR modalities (Signal, Reflectance, Near-IR, Range)
- Optimized for real-time inference on embedded systems
- Includes comprehensive logging and visualization tools

## References

Based on the paper: "Extended Evaluation of SnowPole Detection for Machine-Perceivable Infrastructure for Nordic Winter Conditions"

## License

MIT License
