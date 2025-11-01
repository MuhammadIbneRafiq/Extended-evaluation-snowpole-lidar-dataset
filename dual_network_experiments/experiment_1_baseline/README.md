# Experiment 1: Baseline with RGBA Inputs

This experiment establishes the baseline performance of our dual network architecture using 4-channel RGBA inputs, where RGB channels come from pseudo-color LiDAR combinations and the Alpha channel contains additional modality data (Range/Near-IR/Signal).

## Overview

The baseline experiment implements the following key components:
1. **4-Channel Input Processing**: RGB + Additional modality fusion
2. **Dual-Branch Architecture**: Separate processing paths for RGB and Alpha
3. **Cross-Branch Attention**: Information exchange between modalities
4. **YOLOv9t-style Detection Head**: Anchor-free object detection

## Architecture Details

### Input Configuration
- **RGB Channels**: Pseudo-color combinations from LiDAR modalities
- **Alpha Channel**: Additional modality (Range, Near-IR, or Signal)
- **Input Size**: 640x640 pixels
- **Preprocessing**: Normalization and augmentation

### Dual Network Structure
```
Input (RGBA) → Split → [RGB Branch] → Cross-Attention → Fusion → Detection
                   ↘   [Alpha Branch] ↗                    ↗
```

### Key Components
1. **RGB Branch**: 
   - Full YOLOv9t backbone with pretrained weights
   - Multiple C3 blocks with bottleneck layers
   - SPPF module for multi-scale features

2. **Alpha Branch**:
   - Lightweight architecture (1/4 parameters of RGB)
   - Bottleneck blocks for efficiency
   - Aligned to RGB feature dimensions

3. **Cross-Branch Attention**:
   - Applied at P3, P4, P5 scales
   - Channel and spatial attention mechanisms
   - Learnable temperature parameters

4. **Feature Fusion**:
   - Hierarchical fusion at multiple scales
   - PANet for feature aggregation
   - Multi-scale detection heads

## Training Configuration

### Hyperparameters
- **Epochs**: 300
- **Batch Size**: 16
- **Learning Rate**: 0.001 (Cosine schedule)
- **Optimizer**: AdamW
- **Weight Decay**: 0.0005
- **Warmup Epochs**: 3

### Loss Functions
- **Box Loss**: CIoU loss (weight: 7.5)
- **Classification Loss**: Focal loss (weight: 0.5)
- **DFL Loss**: Distribution focal loss (weight: 1.5)

### Data Augmentation
- Mosaic augmentation (p=1.0)
- Horizontal flip (p=0.5)
- HSV augmentation
- Scale variation (±50%)
- Translation (±10%)

## Dataset Setup

### Data Organization
```
SnowPole_Detection_Dataset/
├── Combination[1-6]/     # Pseudo-color images
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/               # YOLO format annotations
│   ├── train/
│   ├── valid/
│   └── test/
└── [modality]/          # Additional modalities
    ├── train/
    ├── valid/
    └── test/
```

### Data Split
- **Training**: 70% (1,368 images)
- **Validation**: 20% (391 images)
- **Testing**: 10% (195 images)

## Running the Experiment

### Training
```bash
python train.py --config config.yaml --modality combination3 --alpha_channel range
```

### Evaluation
```bash
python evaluate.py --checkpoint best.pt --test_data ../test
```

### Inference
```bash
python inference.py --model best.pt --image sample.jpg --conf 0.4
```

## Expected Results

Based on the paper's findings, we expect:
- **mAP@50**: 0.90-0.92
- **mAP@50-95**: 0.45-0.46
- **Precision**: 0.85-0.90
- **Recall**: 0.85-0.87
- **GPU Inference**: 4-6 ms
- **CPU Inference**: 17-19 ms

## Files in this Experiment

- `train.py`: Main training script
- `config.yaml`: Experiment-specific configuration
- `data_loader.py`: RGBA data loading and preprocessing
- `model.py`: Baseline dual network model
- `loss.py`: Loss function implementations
- `evaluate.py`: Evaluation script
- `inference.py`: Inference and visualization
- `utils.py`: Utility functions

## Key Differences from Standard YOLO

1. **4-Channel Input**: Processes RGBA instead of RGB
2. **Dual-Branch Processing**: Separate paths for different modalities
3. **Cross-Modal Attention**: Information exchange between branches
4. **Modality-Aware Fusion**: Adaptive weighting of modalities
5. **Optimized for Single Class**: Specialized for pole detection

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Metrics to Track
- Training/Validation Loss
- mAP@50 and mAP@50-95
- Precision and Recall curves
- Learning rate schedule
- Gradient norms

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Enable mixed precision training
   - Use gradient accumulation

2. **Poor Convergence**:
   - Check data augmentation settings
   - Verify learning rate schedule
   - Ensure proper weight initialization

3. **Low mAP**:
   - Verify label quality
   - Adjust confidence thresholds
   - Check anchor configurations

## Notes

- Pretrained RGB weights significantly improve convergence
- Alpha channel initialization is critical for performance
- Cross-branch attention adds ~10% to inference time but improves mAP by 2-3%
- Best results with Combination 3 or 4 as RGB input

## References

Based on experiments described in:
"Extended Evaluation of SnowPole Detection for Machine-Perceivable Infrastructure for Nordic Winter Conditions"
