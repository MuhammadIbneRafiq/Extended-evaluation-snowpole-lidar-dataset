#!/usr/bin/env python3
"""
Simple runner script for comprehensive model testing
This script handles environment setup and runs tests on all trained YOLO models
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['pandas', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        except subprocess.CalledProcessError:
            print("Failed to install required packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def setup_yolo_repos():
    """Check if YOLO repositories are available and provide guidance"""
    repos_needed = {
        'yolov5': 'https://github.com/ultralytics/yolov5.git',
        'yolov7': 'https://github.com/WongKinYiu/yolov7.git',
        'ultralytics': 'pip install ultralytics'
    }
    
    print("Checking for YOLO repositories...")
    
    # Check for YOLOv5
    yolov5_paths = ['yolov5', 'models/yolov5', '../yolov5']
    yolov5_found = any(Path(p).exists() for p in yolov5_paths)
    
    # Check for YOLOv7
    yolov7_paths = ['yolov7', 'models/yolov7', '../yolov7']
    yolov7_found = any(Path(p).exists() for p in yolov7_paths)
    
    # Check for Ultralytics
    try:
        import ultralytics
        ultralytics_found = True
    except ImportError:
        ultralytics_found = False
    
    print(f"✓ YOLOv5: {'Found' if yolov5_found else 'Not found'}")
    print(f"✓ YOLOv7: {'Found' if yolov7_found else 'Not found'}")
    print(f"✓ Ultralytics: {'Found' if ultralytics_found else 'Not found'}")
    
    if not (yolov5_found or yolov7_found or ultralytics_found):
        print("\nWarning: No YOLO repositories found!")
        print("To use this script effectively, please clone the repositories:")
        print("  git clone https://github.com/ultralytics/yolov5.git")
        print("  git clone https://github.com/WongKinYiu/yolov7.git")
        print("  pip install ultralytics")
        return False
    
    return True

def find_trained_models():
    """Quick check for trained models"""
    import glob
    
    patterns = [
        'runs/runs/train/*/weights/best.pt',
        'runs (1)/runs/train/*/weights/best.pt',
        'runs/train/*/weights/best.pt'
    ]
    
    models_found = []
    for pattern in patterns:
        models_found.extend(glob.glob(pattern))
    
    print(f"Found {len(models_found)} trained model weights:")
    for model in models_found[:10]:  # Show first 10
        model_name = Path(model).parent.parent.name
        print(f"  - {model_name}")
    
    if len(models_found) > 10:
        print(f"  ... and {len(models_found) - 10} more")
    
    return len(models_found) > 0

def create_missing_yaml_files():
    """Create basic YAML files if they don't exist"""
    yaml_files = [
        'Permutation1.yaml', 'Permutation2.yaml', 'Permutation3.yaml',
        'Permutation4.yaml', 'Permutation5.yaml', 'Permutation6.yaml'
    ]
    
    created_files = []
    for yaml_file in yaml_files:
        if not Path(yaml_file).exists():
            # Create a basic YAML template
            yaml_content = f"""# {yaml_file} Dataset Configuration
path: ./datasets/  # dataset root dir
train: train/images  # train images
val: valid/images   # validation images
test: test/images   # test images (optional)

# Classes
nc: 1  # number of classes
names:
  0: object  # class names
"""
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)
            created_files.append(yaml_file)
    
    if created_files:
        print(f"Created basic YAML files: {', '.join(created_files)}")
        print("Note: You may need to edit these files with correct dataset paths and class names")

def main():
    print("🚀 Comprehensive YOLO Model Testing Script")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to set up dependencies")
        return 1
    
    # Check YOLO repositories
    print("\n2. Checking YOLO repositories...")
    repos_available = setup_yolo_repos()
    
    # Check for trained models
    print("\n3. Scanning for trained models...")
    if not find_trained_models():
        print("❌ No trained models found in runs folders!")
        print("Please ensure you have trained models with weights in:")
        print("  - runs/runs/train/*/weights/best.pt")
        print("  - runs (1)/runs/train/*/weights/best.pt")
        print("  - runs/train/*/weights/best.pt")
        return 1
    
    # Create missing YAML files
    print("\n4. Checking dataset configuration files...")
    create_missing_yaml_files()
    
    # Run the comprehensive testing script
    print("\n5. Starting comprehensive model testing...")
    print("=" * 50)
    
    try:
        # Import and run the main testing script
        from comprehensive_model_testing import ModelTester
        
        tester = ModelTester('.')
        tester.run_all_tests()
        tester.save_results('test_results')
        
        print("\n✅ Testing completed successfully!")
        print("Check the 'test_results' folder for detailed results.")
        
    except ImportError as e:
        print(f"❌ Error importing testing module: {e}")
        print("Make sure comprehensive_model_testing.py is in the same directory")
        return 1
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 