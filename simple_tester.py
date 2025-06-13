#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path
import argparse

# Hardcoded dataset paths
DATASET_PATHS = {
    'signal': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\signal',
    'reflec': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\reflec',
    'range': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\range',
    'nearir': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\nearir',
    'combined': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\combined_color',
    'perm1': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation1',
    'perm3': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation3',
    'perm4': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation4',
    'perm5': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation5',
    'perm6': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation6'
}

def create_temp_yaml(dataset_path, yaml_file):
    """Create temporary YAML file for dataset"""
    yaml_content = f"""# Temporary dataset YAML
path: {dataset_path}

train: train
val: valid
test: test

nc: 1
names:
  0: pole
"""
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)

def test_yolov7_model():
    """Test a YOLOv7 model directly"""
    model_path = r"C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\runs-yolo-v7\runs\train\yolov7-tiny-combined-colo\weights\best.pt"
    dataset_path = DATASET_PATHS['combined']
    yolov7_dir = r"C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\yolov7"
    
    # Create temporary YAML in the yolov7 directory
    temp_yaml = Path(yolov7_dir) / "temp_test_dataset.yaml"
    create_temp_yaml(dataset_path, temp_yaml)
    
    print(f"Testing YOLOv7 model: {model_path}")
    print(f"Dataset: {dataset_path}")
    
    cmd = [
        sys.executable, "test.py",
        "--data", str(temp_yaml),
        "--weights", model_path,
        "--img", "1024",
        "--batch", "16", 
        "--conf", "0.001",
        "--iou", "0.65",
        "--device", "cpu",
        "--name", "direct_test"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=yolov7_dir, capture_output=True, text=True, timeout=600)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Command timed out after 10 minutes")
    except Exception as e:
        print(f"Error running command: {e}")
    finally:
        # Clean up temp file
        Path(temp_yaml).unlink(missing_ok=True)

def test_yolov5_model():
    """Test a YOLOv5 model directly"""
    # Find a YOLOv5 model
    import glob
    yolov5_models = glob.glob(r"C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\runs\runs\train\*/weights\best.pt")
    
    if not yolov5_models:
        print("No YOLOv5 models found")
        return
        
    model_path = yolov5_models[0]
    dataset_path = DATASET_PATHS['perm1']  # Use permutation 1
    yolov5_dir = r"C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\yolov5"
    
    # Create temporary YAML in the yolov5 directory
    temp_yaml = Path(yolov5_dir) / "temp_test_dataset.yaml"
    create_temp_yaml(dataset_path, temp_yaml)
    
    print(f"Testing YOLOv5 model: {model_path}")
    print(f"Dataset: {dataset_path}")
    
    cmd = [
        sys.executable, "val.py",
        "--data", str(temp_yaml),
        "--weights", model_path,
        "--img", "1024",
        "--batch", "16",
        "--conf", "0.001", 
        "--iou", "0.65",
        "--device", "cpu",
        "--name", "direct_test"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=yolov5_dir, capture_output=True, text=True, timeout=600)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Command timed out after 10 minutes")
    except Exception as e:
        print(f"Error running command: {e}")
    finally:
        # Clean up temp file
        Path(temp_yaml).unlink(missing_ok=True)

def check_datasets():
    """Check if dataset paths exist"""
    print("Checking dataset paths:")
    for name, path in DATASET_PATHS.items():
        exists = Path(path).exists()
        print(f"  {name}: {'✓' if exists else '✗'} {path}")

def main():
    parser = argparse.ArgumentParser(description='Simple direct model tester')
    parser.add_argument('--check', action='store_true', help='Check dataset paths')
    parser.add_argument('--test-yolov7', action='store_true', help='Test YOLOv7 model')
    parser.add_argument('--test-yolov5', action='store_true', help='Test YOLOv5 model')
    
    args = parser.parse_args()
    
    if args.check:
        check_datasets()
    elif args.test_yolov7:
        test_yolov7_model()
    elif args.test_yolov5:
        test_yolov5_model()
    else:
        print("Usage: python simple_tester.py --check | --test-yolov7 | --test-yolov5")

if __name__ == "__main__":
    main() 