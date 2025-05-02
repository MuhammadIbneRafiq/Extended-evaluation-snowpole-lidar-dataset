# Computer Vision Dataset & YOLOv5 Guide

This repository contains tools and instructions for working with custom computer vision datasets and running YOLOv5 models on different modalities.

## Dataset Access

### Downloading the Dataset from Roboflow

The dataset is hosted on Roboflow and can be downloaded using the following links:

```
[https://app.roboflow.com/ds/fcLLRJGEAC?key=BiJrWJZ9xW]
```

You can download the dataset in different formats depending on your needs:
- COCO JSON (ppreferable for yolo models)
- VOC XML (for R-CNNs)

### Viewing and Exploring the Dataset

Once downloaded, you can explore the dataset using various tools:

#### Using Roboflow's Web Interface
1. Log in to your Roboflow account
2. Navigate to your project
3. Click on "Explore" to browse images and annotations

#### Using Local Visualization Tools
1. For YOLO format datasets:
   ```
   python tools/visualize_dataset.py --path /path/to/dataset --format yolo
   ```

2. For COCO format datasets:
   ```
   python tools/visualize_dataset.py --path /path/to/dataset --format coco
   ```

## Setting Up YOLOv5

### Installation

1. Clone the YOLOv5 repository:
   ```
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional packages for different modalities:
   ```
   pip install opencv-python tensorflow torch torchvision
   ```

### Running YOLOv5 on Different Modalities

YOLOv5 can be run on various input modalities including RGB images, thermal images, depth maps, and more.

#### RGB Images (Standard)

```
python train.py --img 640 --batch 16 --epochs 100 --data /path/to/data.yaml --weights yolov5s.pt
```

#### Thermal Images

```
python train.py --img 640 --batch 16 --epochs 100 --data /path/to/thermal_data.yaml --weights yolov5s.pt --modality thermal
```

#### Depth Maps

```
python train.py --img 640 --batch 16 --epochs 100 --data /path/to/depth_data.yaml --weights yolov5s.pt --modality depth
```

#### Multi-Modal Training

For training with multiple input modalities:

```
python train_multimodal.py --img 640 --batch 16 --epochs 100 --rgb-data /path/to/rgb_data.yaml --thermal-data /path/to/thermal_data.yaml --weights yolov5s.pt
```

### Inference with YOLOv5

To run inference on different modalities:

#### RGB Images

```
python detect.py --weights /path/to/best.pt --source /path/to/images --img 640
```

#### Thermal Images

```
python detect.py --weights /path/to/thermal_best.pt --source /path/to/thermal_images --img 640 --modality thermal
```

#### Multi-Modal Inference

```
python detect_multimodal.py --weights /path/to/multimodal_best.pt --rgb-source /path/to/rgb_images --thermal-source /path/to/thermal_images --img 640
```

## Performance Evaluation

To evaluate model performance on different modalities:

```
python val.py --weights /path/to/best.pt --data /path/to/data.yaml --task test
```

## Visualization and Analysis

For visualizing results and analyzing model performance:

```
python tools/visualize_results.py --weights /path/to/best.pt --data /path/to/val_images --save-dir /path/to/output
```

## Troubleshooting

If you encounter issues:

1. Check that your dataset format matches the expected format for YOLOv5
2. Ensure you have the correct paths in your data.yaml file
3. For multi-modal training, verify that your images across modalities are properly aligned
4. Check GPU memory usage if you encounter CUDA out of memory errors

## Contact

For issues or questions regarding the dataset or models, please open an issue in this repository. 