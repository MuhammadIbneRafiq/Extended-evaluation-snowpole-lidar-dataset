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

**Contact:** muhammadibnerafiq@gmail.com || durga.prasad.bavirisetti@hig.se

## Abstract
This study presents an extensive evaluation of YOLO object detection architectures for identifying snow poles in LiDAR-derived imagery under challenging Nordic conditions. Building on our prior SnowPole Detection dataset[1] and LiDAR-GNSS localization framework[2], we benchmark six YOLO models—YOLOv5s, YOLOv7-tiny, YOLOv8n, YOLOv9t, YOLOv10n, and YOLOv11n—across multiple input modalities. We assess single-channel modalities (Reflectance, Signal, Near-Infrared) and six pseudo-color combinations derived from these channels. Model performance is quantified using Precision, Recall, mAP@50, mAP@50-95, and GPU inference latency. To enable systematic comparison, we define a composite Rank Score combining accuracy and real-time performance.

Results show YOLOv9t achieves the highest detection accuracy, while YOLOv11n offers the best balance between accuracy and inference speed, making it suitable for embedded real-time applications. Pseudo-color combinations, especially those fusing Near-Infrared, Signal, and Reflectance, outperform single modalities and yield the highest Rank Scores. We recommend multimodal LiDAR configurations such as Combination 4 and Combination 5 to enhance detection robustness.

All datasets, code, and trained models are publicly available to support reproducibility via our [GitHub repository](https://github.com/MuhammadIbneRafiq/Extended-evaluation-snowpole-lidar-dataset) and the [Mendeley dataset archive](https://data.mendeley.com/datasets/tt6rbx7s3h/3).

## 📁 Project Structure
```
project-root/
├── data/                        # Dataset files currently on gitignore, get them from Mendeley with the link from the abstract
├── scripts/                     # Data preparation, training, and evaluation scripts
│   ├── create_combinations.py   # Generate spectral combinations
│   ├── create_correct_combinations.py  # Validate combinations
│   ├── generate_real_yolo_comparison.py  # Performance comparison
│   ├── plot_permutations.py     # Visualization utilities
│   ├── inference_on_single_image.py  # Single image inference
│   ├── thebiggermodupdated.py   # Main benchmarking script
│   └── visualize_ground_truth.py # Ground truth visualization
├── results/                     # Evaluation results and visualizations
│   ├── ground_truth_*.png       # Ground truth visualizations
│   └── permutation_results/     # Spectral permutation analysis
├── yolov5/                      # YOLOv5 codebase
├── yolov7/                      # YOLOv7 codebase
├── ultralytics/                 # YOLOv8+ codebase
├── main_dataset.yaml           # Main dataset configuration
├── train_command.md            # Training command reference
├── test_commands.md         # Inference command reference
├── requirements_multispectral.txt  # Python dependencies
└── README.md                   # This file
```

## Trained Weights and Inference images
- **Link to drive** https://drive.google.com/file/d/18pZshkm-Yu9zKrhGDE5CMTTY9XbmAlwF/view?usp=sharing

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
1. Raw multispectral images are located in `data/`
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

## 🔍 Inference Commands

### Single Modality Inference
For detailed inference commands for all modalities (Signal, Reflectance, Near-IR, Range, Combined Color) and all YOLO versions (v8-v11), see `predict_commands.md`.

### Example Inference Commands
```bash
# Signal modality with YOLOv11
yolo predict model="trained_weights/train_signal_11n/weights/best.pt" source="data/single_modality/SnowPole_Detection_Dataset/signal/train/image_1967.png" conf=0.4 name=signal_v11_image_1967

# Combination 4 with YOLOv10
yolo predict model="trained_weights/v10N__perm4/weights/best.pt" source="data/Combinations_yolo_format/Combination4/train/image_1967.png" conf=0.4 name=perm4_v10_image_1967
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
- Ground truth visualizations: `results/ground_truth_*.png`
- Permutation analysis: `results/permutation_results/`
- Inference results: `inference/` directory with organized model outputs
- Training logs: Available in individual model directories under `trained_weights/`

### Running Evaluation
```bash
# Run comprehensive model testing
python scripts/thebiggermodupdated.py

# Generate comparison plots
python scripts/plot_permutations.py

# Single image inference
python scripts/inference_on_single_image.py --image path/to/image.jpg

# Visualize ground truth
python scripts/visualize_ground_truth.py
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: 
  - See `train_command.md` for detailed training instructions
  - See `predict_commands.md` for comprehensive inference commands
- **Scripts**: Check individual script files for usage examples

---

**⭐ Star this repository if it helps your research!**

## References
[1] D. P. Bavirisetti, G. H. Kiss, P. Arnesen, H. Seter, S. Tabassum, and F. Lindseth, "SnowPole Detection: A comprehensive dataset for detection and localization using LiDAR imaging in Nordic winter conditions," Data in Brief, vol. 59, p. 111403, 2025.

[2] D. P. Bavirisetti, G. H. Kiss, and F. Lindseth, "A Pole Detection and Geospatial Localization Framework using LiDAR-GNSS Data Fusion," in 2024 27th International Conference on Information Fusion (FUSION), 2024, pp. 1–8.
