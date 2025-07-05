# Extended Evaluation of SnowPole Detection for Machine-Perceivable Infrastructure for Nordic Winter Conditions: A Comparative Study of Object Detection Models

## Authors
- Durga Prasad Bavirisetti¹,²
- Muhammad Ibne Rafiq³
- Shaira Tabassum¹
- Gabriel Hanssen Kiss¹
- Frank Lindseth¹

## Affiliations
1. Department of Computer Science, Norwegian University of Science and Technology (NTNU), Trondheim, Norway
2. Department of Computer Science, University of Gävle, Gävle, Sweden
3. Department of Mathematics and Computer Science, Eindhoven University of Technology, Netherlands

**Contact:** durga.prasad.bavirisetti@hig.se

## Abstract
This study presents an extensive evaluation of YOLO object detection architectures for identifying snow poles in LiDAR-derived imagery under challenging Nordic conditions. Building on our prior SnowPole Detection dataset[1] and LiDAR-GNSS localization framework[2], we benchmark six YOLO models—YOLOv5s, YOLOv7-tiny, YOLOv8n, YOLOv9t, YOLOv10n, and YOLOv11n—across multiple input modalities. We assess single-channel modalities (Reflectance, Signal, Near-Infrared) and six pseudo-color combinations derived from these channels. Model performance is quantified using Precision, Recall, mAP@50, mAP@50-95, and GPU inference latency. To enable systematic comparison, we define a composite Rank Score combining accuracy and real-time performance.

Results show YOLOv9t achieves the highest detection accuracy, while YOLOv11n offers the best balance between accuracy and inference speed, making it suitable for embedded real-time applications. Pseudo-color combinations, especially those fusing Near-Infrared, Signal, and Reflectance, outperform single modalities and yield the highest Rank Scores. We recommend multimodal LiDAR configurations such as Combination 4 and Combination 5 to enhance detection robustness.

All datasets, code, and trained models are publicly available to support reproducibility via our [GitHub repository](https://github.com/MuhammadIbneRafiq/Extended-evaluation-snowpole-lidar-dataset) and the [Mendeley dataset archive](https://data.mendeley.com/datasets/tt6rbx7s3h/3).

## 📁 Project Structure
```
project-root/
├── data/                        # All datasets and label files
│   ├── main_images/             # Raw multispectral images
│   ├── labels/                  # YOLO-format labels
│   ├── Combinations_yolo_format/# Dataset combinations in YOLO format
│   ├── VOC_COMBINATION_12.v1i.voc/  # VOC format dataset
│   └── yolo-to-coco-json-3/     # Format conversion utilities
├── scripts/                     # Data prep, training, and evaluation scripts
│   ├── create_combinations.py   # Generate spectral combinations
│   ├── create_correct_combinations.py  # Validate combinations
│   ├── generate_real_yolo_comparison.py  # Performance comparison
│   ├── plot_permutations.py     # Visualization utilities
│   ├── inference_on_single_image.py  # Single image inference
│   └── thebiggermodupdated.py   # Main benchmarking script
├── models/                      # Model weights and checkpoints
│   └── trained_weights/         # All trained model weights
├── results/                     # All results, outputs, and figures
│   ├── permutation_results/     # Spectral permutation visualizations
│   ├── runs-yolo-v7/           # YOLOv7 training outputs
│   ├── *.png                   # Result figures and comparisons
│   └── *.json                  # Model performance metrics
├── yolov5/                      # YOLOv5 codebase
├── yolov7/                      # YOLOv7 codebase
├── ultralytics/                 # YOLOv8+ codebase
├── requirements_multispectral.txt  # Python dependencies
├── main_dataset.yaml           # Main dataset configuration
├── train_command.md            # Training command reference
└── README.md                   # This file
```

## 🔬 Reproducibility

### System Requirements
- **Python**: 3.8+
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: NVIDIA GTX 1060+ (optional but recommended)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements_multispectral.txt
```

### Data Preparation
1. Raw multispectral images are located in `data/main_images/`
2. YOLO-format labels are in `data/labels/`
3. Use scripts in `scripts/` for dataset preparation and permutation generation
4. Dataset configurations are defined in `main_dataset.yaml`

## 🚀 Training Commands

### YOLOv5 Training
```bash
cd yolov5
# Single GPU training
python train.py --img 640 --batch 16 --epochs 100 --data ../main_dataset.yaml --weights yolov5s.pt --project ../results/yolov5_runs --name experiment_name

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node 4 train.py --img 640 --batch 64 --epochs 100 --data ../main_dataset.yaml --weights yolov5s.pt --project ../results/yolov5_runs --name experiment_name --device 0,1,2,3
```

### YOLOv7 Training
```bash
cd yolov7
# Single GPU training
python train.py --img 640 --batch 16 --epochs 100 --data ../main_dataset.yaml --weights yolov7-tiny.pt --project ../results/yolov7_runs --name experiment_name

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --img 640 --batch 64 --epochs 100 --data ../main_dataset.yaml --weights yolov7-tiny.pt --project ../results/yolov7_runs --name experiment_name --device 0,1,2,3
```

### Ultralytics YOLO (YOLOv8+) Training
```bash
# YOLOv8 training
yolo detect train data=main_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 project=results/ultralytics_runs name=experiment_name

# YOLOv9 training
yolo detect train data=main_dataset.yaml model=yolov9c.pt epochs=100 imgsz=640 project=results/ultralytics_runs name=experiment_name

# YOLOv10 training
yolo detect train data=main_dataset.yaml model=yolov10n.pt epochs=100 imgsz=640 project=results/ultralytics_runs name=experiment_name

# YOLOv11 training
yolo detect train data=main_dataset.yaml model=yolov11n.pt epochs=100 imgsz=640 project=results/ultralytics_runs name=experiment_name
```

## 📊 Results & Evaluation

### Performance Metrics
- **Precision**: Object detection precision
- **Recall**: Object detection recall  
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **FPS**: Frames per second (inference speed)
- **Latency**: Single image inference time
- **Resource Usage**: Memory and CPU utilization

### Result Files
- Individual model results: `results/model_test_results_*.json`
- Comparison figures: `results/*.png`
- Permutation analysis: `results/permutation_results/`
- Training logs: `results/runs-yolo-v7/`

### Running Evaluation
```bash
# Run comprehensive model testing
python scripts/thebiggermodupdated.py

# Generate comparison plots
python scripts/plot_permutations.py

# Single image inference
python scripts/inference_on_single_image.py --image path/to/image.jpg
```

## 🔬 Research Paper Guidelines

### Key Contributions
1. **Multispectral Dataset**: Comprehensive multispectral object detection dataset
2. **Systematic Benchmarking**: Evaluation across multiple YOLO versions
3. **Spectral Analysis**: Performance analysis across different spectral combinations
4. **Reproducible Framework**: Complete pipeline for multispectral object detection research

### Experimental Setup
- **Models Tested**: YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- **Input Resolutions**: 640x640, 1024x1024, 1280x1280
- **Spectral Combinations**: Various RGB+NIR+LiDAR combinations
- **Evaluation Metrics**: Standard COCO metrics + inference speed

## 📚 Citation

If you use this framework in your research, please cite:
```bibtex
@misc{multispectral2024yolo,
  title={Multispectral YOLO Benchmarking: A Comprehensive Framework for Object Detection Across Spectral Combinations},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo-url}},
  note={Comprehensive benchmarking and reproducible research for multispectral object detection}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: See `train_command.md` for detailed training instructions
- **Scripts**: Check individual script files for usage examples

---

**⭐ Star this repository if it helps your research!**

## References
[1] D. P. Bavirisetti, G. H. Kiss, P. Arnesen, H. Seter, S. Tabassum, and F. Lindseth, "SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions," Data in Brief, vol. 59, p. 111403, 2025.

[2] D. P. Bavirisetti, G. H. Kiss, and F. Lindseth, "A Pole Detection and Geospatial Localization Framework using LiDAR-GNSS Data Fusion," in 2024 27th International Conference on Information Fusion (FUSION), 2024, pp. 1–8.