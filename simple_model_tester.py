import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime
import glob
import re

class SimpleModelTester:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Hardcoded dataset paths - using actual paths from your system
        self.dataset_paths = {
            # Single modality datasets (already in YOLO format)
            'signal': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\signal',
            'reflec': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\reflec',
            'range': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\range',
            'nearir': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\nearir',
            'combined': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\main_images\SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions\SnowPole_Detection_Dataset\combined_color',
            
            # Permutation datasets (Pascal VOC format but work with YOLO commands)
            'perm1': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation1',
            'perm3': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation3', 
            'perm4': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation4',
            'perm5': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation5',
            'perm6': r'C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\Permutation6'
        }
        
        # Try to find the YOLO repositories
        self.yolov5_path = self.find_yolo_repo('yolov5')
        self.yolov7_path = self.find_yolo_repo('yolov7')
        
    def find_yolo_repo(self, repo_name):
        possible_paths = [
            self.base_dir / repo_name,
            Path.cwd() / repo_name,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def create_dataset_yaml(self, dataset_key, dataset_path):
        """Create a temporary YAML file for the dataset"""
        yaml_content = f"""# Dataset YAML for {dataset_key}
path: {dataset_path}

# Train/val/test sets
train: train
val: valid  
test: test

# Classes
nc: 1
names:
  0: pole
"""
        
        yaml_file = self.base_dir / f"temp_{dataset_key}_dataset.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        return str(yaml_file)
    
    def get_dataset_for_model(self, model_name):
        """Get appropriate dataset path for model"""
        model_lower = model_name.lower()
        
        # Check for specific patterns in model name
        for key in self.dataset_paths.keys():
            if key in model_lower:
                return self.dataset_paths[key], key
        
        # Default fallback
        return self.dataset_paths['combined'], 'combined'
    
    def find_model_weights(self):
        """Find all trained model weights"""
        models = []
        
        # Search patterns for different YOLO versions
        runs_patterns = [
            'runs/runs/train/*/weights/best.pt',
            'runs-yolo-v7/runs/train/*/weights/best.pt',
            'runs/train/*/weights/best.pt',
        ]
        
        for pattern in runs_patterns:
            weight_files = glob.glob(str(self.base_dir / pattern))
            for weight_file in weight_files:
                weight_path = Path(weight_file)
                model_name = weight_path.parent.parent.name
                
                # Skip problematic models
                if model_name in ['exp4']:
                    continue
                
                model_type = self._detect_model_type(weight_path)
                dataset_path, dataset_key = self.get_dataset_for_model(model_name)
                
                models.append({
                    'name': model_name,
                    'type': model_type,
                    'weights': str(weight_path),
                    'dataset_path': dataset_path,
                    'dataset_key': dataset_key
                })
        
        return models
    
    def _detect_model_type(self, model_path):
        """Detect YOLO model type based on path"""
        model_path_str = str(model_path).lower()
        
        if 'yolov5' in model_path_str or 'runs/runs/train' in model_path_str:
            return 'yolov5'
        elif 'yolov7' in model_path_str or 'runs-yolo-v7' in model_path_str:
            return 'yolov7'
        else:
            return 'ultralytics'
    
    def test_yolov5_model(self, model_info):
        """Test YOLOv5 model"""
        if not self.yolov5_path:
            return None
            
        # Create temporary dataset YAML
        yaml_file = self.create_dataset_yaml(model_info['dataset_key'], model_info['dataset_path'])
        
        cmd = [
            sys.executable, 'val.py',
            '--data', yaml_file,
            '--weights', model_info['weights'],
            '--img', '1024',
            '--batch', '16',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',
            '--name', f"test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, cwd=self.yolov5_path, capture_output=True, text=True, timeout=600)
        
        # Clean up temp file
        Path(yaml_file).unlink(missing_ok=True)
        
        if result.returncode == 0:
            return self.parse_yolo_results(result.stdout, model_info)
        else:
            print(f"YOLOv5 test failed for {model_info['name']}: {result.stderr}")
            return None
    
    def test_yolov7_model(self, model_info):
        """Test YOLOv7 model"""
        if not self.yolov7_path:
            return None
            
        # Create temporary dataset YAML
        yaml_file = self.create_dataset_yaml(model_info['dataset_key'], model_info['dataset_path'])
        
        cmd = [
            sys.executable, 'test.py',
            '--data', yaml_file,
            '--weights', model_info['weights'],
            '--img', '1024',
            '--batch', '16',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',
            '--name', f"test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, cwd=self.yolov7_path, capture_output=True, text=True, timeout=600)
        
        # Clean up temp file
        Path(yaml_file).unlink(missing_ok=True)
        
        if result.returncode == 0:
            return self.parse_yolo_results(result.stdout, model_info)
        else:
            print(f"YOLOv7 test failed for {model_info['name']}: {result.stderr}")
            return None
    
    def parse_yolo_results(self, output, model_info):
        """Parse YOLO test results from output"""
        # Pattern to match results line
        pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, output)
        
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            map50 = float(match.group(3))
            map50_95 = float(match.group(4))
        else:
            # Fallback patterns
            precision = self.extract_metric(output, r'(?:P|Precision)[:=\s]*([\d.]+)')
            recall = self.extract_metric(output, r'(?:R|Recall)[:=\s]*([\d.]+)')
            map50 = self.extract_metric(output, r'mAP@?\.?5[:\s]*([\d.]+)')
            map50_95 = self.extract_metric(output, r'mAP@?\.?5[:\-\.]*95[:\s]*([\d.]+)')
        
        return {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'dataset_used': model_info['dataset_key'],
            'weights_path': model_info['weights'],
            'dataset_path': model_info['dataset_path'],
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map50_95': map50_95,
            'test_date': datetime.now().isoformat()
        }
    
    def extract_metric(self, text, pattern):
        """Extract metric value using regex pattern"""
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0
    
    def test_single_model_by_name(self, model_name):
        """Test a single model by name"""
        models = self.find_model_weights()
        
        target_model = None
        for model in models:
            if model_name.lower() in model['name'].lower():
                target_model = model
                break
        
        if not target_model:
            print(f"Model '{model_name}' not found!")
            available = [m['name'] for m in models]
            print(f"Available models: {available}")
            return
        
        print(f"Testing: {target_model['name']} ({target_model['type']})")
        print(f"Dataset: {target_model['dataset_key']} -> {target_model['dataset_path']}")
        print(f"Weights: {target_model['weights']}")
        
        if target_model['type'] == 'yolov5':
            result = self.test_yolov5_model(target_model)
        elif target_model['type'] == 'yolov7':
            result = self.test_yolov7_model(target_model)
        else:
            print("Unknown model type")
            return
        
        if result:
            if result['map50'] == 0.0 and result['precision'] == 0.0:
                print(f"⚠ Test completed but metrics are 0 - check dataset structure")
            else:
                print(f"✓ Test successful!")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  mAP@0.5: {result['map50']:.4f}")
                print(f"  mAP@0.5:0.95: {result['map50_95']:.4f}")
        else:
            print(f"✗ Test failed")
    
    def run_all_tests(self):
        """Run tests on all found models"""
        models = self.find_model_weights()
        print(f"Found {len(models)} trained models to test")
        
        for model_info in models:
            print(f"\nTesting: {model_info['name']} ({model_info['type']}) with {model_info['dataset_key']}")
            
            if model_info['type'] == 'yolov5':
                result = self.test_yolov5_model(model_info)
            elif model_info['type'] == 'yolov7':
                result = self.test_yolov7_model(model_info)
            else:
                continue
            
            if result:
                self.results.append(result)
                print(f"✓ P={result['precision']:.3f}, R={result['recall']:.3f}, mAP50={result['map50']:.3f}, mAP50:95={result['map50_95']:.3f}")
            else:
                print(f"✗ Test failed")
    
    def save_results(self, output_dir='test_results'):
        """Save all test results"""
        if not self.results:
            print("No results to save!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = output_path / f'simple_model_test_results_{self.timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = output_path / f'simple_model_test_results_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {csv_file}")
        
        # Print summary
        print(f"\n{'Model Name':<25} {'Dataset':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50:95':<10}")
        print("-" * 95)
        
        for result in self.results:
            print(f"{result['model_name']:<25} {result['dataset_used']:<12} "
                  f"{result['precision']:<10.4f} {result['recall']:<10.4f} "
                  f"{result['map50']:<10.4f} {result['map50_95']:<10.4f}")
    
    def list_models(self):
        """List all available models and datasets"""
        models = self.find_model_weights()
        
        print(f"YOLO Repository Paths:")
        print(f"  YOLOv5: {self.yolov5_path}")
        print(f"  YOLOv7: {self.yolov7_path}")
        print()
        
        print(f"Available Datasets:")
        for key, path in self.dataset_paths.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {key}: {exists} {path}")
        print()
        
        if not models:
            print("No trained models found!")
            return
        
        print(f"Found {len(models)} trained models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['type']})")
            print(f"   → Will use dataset: {model['dataset_key']}")
            print(f"   → Weights: {model['weights']}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Simple YOLO model tester with hardcoded paths')
    parser.add_argument('--list', action='store_true', help='List available models and datasets')
    parser.add_argument('--test-single', help='Test a single model by name')
    parser.add_argument('--test-all', action='store_true', help='Test all models')
    
    args = parser.parse_args()
    
    tester = SimpleModelTester()
    
    if args.list:
        tester.list_models()
    elif args.test_single:
        tester.test_single_model_by_name(args.test_single)
    elif args.test_all:
        tester.run_all_tests()
        tester.save_results()
    else:
        tester.list_models()

if __name__ == "__main__":
    main() 