import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime
import glob
import re

class ModelTester:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Mapping of permutations/models to their YAML files
        self.yaml_mapping = {
            'perm1': 'Permutation1.yaml',
            'perm2': 'Permutation2.yaml', 
            'perm3': 'Permutation3.yaml',
            'perm4': 'Permutation4.yaml',
            'perm5': 'Permutation5.yaml',
            'perm6': 'Permutation6.yaml',
            'signal': 'signal_dataset.yaml',
            'reflec': 'reflec_dataset.yaml',
            'range': 'range_dataset.yaml',
            'nearir': 'nearir_dataset.yaml',
            'combined': 'combined_dataset.yaml'
        }
        
        # Try to find the YOLO repositories
        self.yolov5_path = self.find_yolo_repo('yolov5')
        self.yolov7_path = self.find_yolo_repo('yolov7')
        self.ultralytics_path = self.find_yolo_repo('ultralytics')
        
    def find_yolo_repo(self, repo_name):
        possible_paths = [
            self.base_dir / repo_name,
            self.base_dir / 'models' / repo_name,
            Path.cwd() / repo_name,
            Path.cwd().parent / repo_name
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def get_yaml_file(self, model_name):
        """Get appropriate YAML file for model"""
        model_lower = model_name.lower()
        
        # Check for specific permutation patterns
        for key, yaml_file in self.yaml_mapping.items():
            if key in model_lower:
                yaml_path = self.base_dir / yaml_file
                if yaml_path.exists():
                    return str(yaml_path)
        
        # Default fallback - check if permutation number is in name
        perm_match = re.search(r'perm(\d+)', model_lower)
        if perm_match:
            perm_num = perm_match.group(1)
            yaml_file = f'Permutation{perm_num}.yaml'
            yaml_path = self.base_dir / yaml_file
            if yaml_path.exists():
                return str(yaml_path)
        
        # Use dataset.yaml as default
        return 'dataset.yaml'
    
    def find_model_weights(self):
        """Find all trained model weights in runs folders"""
        models = []
        
        # Search patterns for different YOLO versions
        runs_patterns = [
            'runs/runs/train/*/weights/best.pt',
            'runs-yolo-v7/runs/train/*/weights/best.pt',
            'runs/train/*/weights/best.pt',
            'yolov5/runs/train/*/weights/best.pt',
            'yolov7/runs/train/*/weights/best.pt',
            'ultralytics/runs/train/*/weights/best.pt'
        ]
        
        for pattern in runs_patterns:
            weight_files = glob.glob(str(self.base_dir / pattern))
            for weight_file in weight_files:
                weight_path = Path(weight_file)
                model_name = weight_path.parent.parent.name
                
                if model_name == 'exp4':
                    continue
                
                model_type = self._detect_model_type(weight_path)
                
                models.append({
                    'name': model_name,
                    'type': model_type,
                    'weights': str(weight_path),
                    'yaml': self.get_yaml_file(model_name)
                })
        
        return models
    
    def _detect_model_type(self, model_path):
        """Detect YOLO model type based on path and folder name"""
        model_path_str = str(model_path).lower()
        
        if 'yolov5' in model_path_str or any(name in model_path_str for name in ['exp', 'exp2', 'exp3', 'exp5', 'exp6']):
            return 'yolov5'
        elif 'yolov7' in model_path_str or 'v7-tiny' in model_path_str:
            return 'yolov7'
        else:
        return 'ultralytics'
    
    def test_yolov5_model(self, model_info):
        if not self.yolov5_path or not self.yolov5_path.exists():
            return None
            
        cmd = [
            sys.executable, 'val.py',
            '--data', str(Path.cwd() / 'yolov5' / 'dataset.yaml'),
            '--weights', str(Path.cwd() / model_info['weights']),
            '--img', '1024',
            '--batch', '32',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',
            '--name', f"test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, cwd=self.yolov5_path, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            return self.parse_yolo_results(result.stdout, model_info)
        else:
            print(f"YOLOv5 test failed for {model_info['name']}: {result.stderr}")
        return None
    
    def test_yolov7_model(self, model_info):
        """Test YOLOv7 model"""
        if not self.yolov7_path or not self.yolov7_path.exists():
            return None
            
            cmd = [
                sys.executable, 'test.py',
            '--data', str(Path.cwd() / 'yolov7' / 'dataset.yaml'),
                '--weights', str(Path.cwd() / model_info['weights']),
                '--img', '1024',
            '--batch', '32',
                '--conf', '0.001',
                '--iou', '0.65',
                '--device', 'cpu',
                '--name', f"test_{model_info['name']}"
            ]
            
        result = subprocess.run(cmd, cwd=self.yolov7_path, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
            return self.parse_yolo_results(result.stdout, model_info)
            else:
            print(f"YOLOv7 test failed for {model_info['name']}: {result.stderr}")
            return None
    
    def test_ultralytics_model(self, model_info):
        """Test Ultralytics YOLO model"""
            cmd = [
                'yolo', 'val',
                f"model={model_info['weights']}",
                f"data={model_info['yaml']}",
            'imgsz=1024',
            'batch=8',
                'conf=0.001',
                'iou=0.65',
            'device=cpu',
                f"name=test_{model_info['name']}_{self.timestamp}"
            ]
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
            return self.parse_yolo_results(result.stdout, model_info)
            else:
            print(f"Ultralytics CLI test failed for {model_info['name']}: {result.stderr}")
            return self.test_ultralytics_python_api(model_info)
    
    def test_ultralytics_python_api(self, model_info):
        """Test using Ultralytics Python API as fallback"""
            from ultralytics import YOLO
            
            model = YOLO(model_info['weights'])
        metrics = model.val(data=model_info['yaml'], imgsz=1024, batch=8, conf=0.001, iou=0.65, device=0)
            
            return {
                'model_name': model_info['name'],
                'model_type': model_info['type'],
                'weights_path': model_info['weights'],
                'yaml_path': model_info['yaml'],
                'precision': getattr(metrics.box, 'mp', 0),
                'recall': getattr(metrics.box, 'mr', 0),
                'map50': getattr(metrics.box, 'map50', 0),
                'map50_95': getattr(metrics.box, 'map', 0),
                'test_date': datetime.now().isoformat()
            }
            
    def parse_yolo_results(self, output, model_info):
        """Parse YOLO test results from output"""
        # Pattern to match the "all" results line: "all        197        395      0.889      0.829      0.915      0.451"
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
            'weights_path': model_info['weights'],
            'yaml_path': model_info['yaml'],
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map50_95': map50_95,
            'test_date': datetime.now().isoformat()
        }
    
    def extract_metric(self, text, pattern):
        """Extract metric value using regex pattern"""
        match = re.search(pattern, text)
        if match:
                return float(match.group(1))
        return 0.0
    
    def run_all_tests(self):
        """Run tests on all found models"""
        models = self.find_model_weights()
        
        
        print(f"Found {len(models)} trained models to test")
        
        for model_info in models:
            print(f"Testing: {model_info['name']} ({model_info['type']})")
            
            if model_info['type'] == 'yolov5':
                result = self.test_yolov5_model(model_info)
            elif model_info['type'] == 'yolov7':
                result = self.test_yolov7_model(model_info)
            elif model_info['type'] == 'ultralytics':
                result = self.test_ultralytics_model(model_info)
            else:
                continue
            
            if result:
                self.results.append(result)
                print(f"✓ {model_info['name']}: P={result['precision']:.3f}, R={result['recall']:.3f}, mAP50={result['map50']:.3f}, mAP50:95={result['map50_95']:.3f}")
            else:
                print(f"✗ {model_info['name']}: Test failed")
    
    def save_results(self, output_dir='test_results'):
        """Save all test results to files"""
        if not self.results:
            print("No results to save!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = output_path / f'model_test_results_{self.timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = output_path / f'model_test_results_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {csv_file}")
        
        # Print summary
        print(f"\n{'Model Name':<25} {'Type':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50:95':<10}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result['model_name']:<25} {result['model_type']:<12} "
                  f"{result['precision']:<10.4f} {result['recall']:<10.4f} "
                  f"{result['map50']:<10.4f} {result['map50_95']:<10.4f}")

    def list_models_only(self):
        """Just list the models found without testing"""
        models = self.find_model_weights()
        
        print(f"YOLO Repository Paths:")
        print(f"  YOLOv5: {self.yolov5_path}")
        print(f"  YOLOv7: {self.yolov7_path}")
        print(f"  Ultralytics: {self.ultralytics_path}")
        print()
        
        if not models:
            print("No trained models found!")
            return []
        
        print(f"Found {len(models)} trained models:")
        
        for i, model_info in enumerate(models, 1):
            yaml_path = Path(model_info['yaml'])
            yaml_exists = yaml_path.exists() if model_info['yaml'] != 'dataset.yaml' else f"checking dataset.yaml"
            print(f"{i}. {model_info['name']} ({model_info['type']})")
            print(f"   Weights: {model_info['weights']}")
            print(f"   YAML: {model_info['yaml']} - {yaml_exists}")
            print()
        
        return models

    def test_single_model(self, model_name):
        """Test a single model by name"""
        models = self.find_model_weights()
        
        target_model = None
        for model in models:
            if model_name.lower() in model['name'].lower():
                target_model = model
                break
        
        if not target_model:
            print(f"Model '{model_name}' not found!")
            return
        
        print(f"Testing single model: {target_model['name']} ({target_model['type']})")
        print(f"Weights: {target_model['weights']}")
        print(f"YAML: {target_model['yaml']}")
        
        if target_model['type'] == 'yolov5':
            result = self.test_yolov5_model(target_model)
        elif target_model['type'] == 'yolov7':
            result = self.test_yolov7_model(target_model)
        elif target_model['type'] == 'ultralytics':
            result = self.test_ultralytics_model(target_model)
        else:
            print("Unknown model type")
            return
        
        if result:
            if result['map50'] == 0.0 and result['precision'] == 0.0:
                print(f"⚠ Test completed but no labels found (0 metrics)")
                print(f"  This usually means the dataset has images but no corresponding .txt label files")
                print(f"  Check that each image has a corresponding .txt file with YOLO format annotations")
            else:
                print(f"✓ Test successful!")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  mAP@0.5: {result['map50']:.4f}")
                print(f"  mAP@0.5:0.95: {result['map50_95']:.4f}")
        else:
            print(f"✗ Test failed")

def main():
    parser = argparse.ArgumentParser(description='Test all trained YOLO models')
    parser.add_argument('--base-dir', default='.', help='Base directory to search for models and YAML files')
    parser.add_argument('--output-dir', default='test_results', help='Directory to save test results')
    parser.add_argument('--debug', action='store_true', help='Only list models without testing')
    parser.add_argument('--test-single', help='Test a single model by name')
    
    args = parser.parse_args()
    
    tester = ModelTester(args.base_dir)
    
    if args.debug:
        tester.list_models_only()
    elif args.test_single:
        tester.test_single_model(args.test_single)
    else:
    tester.run_all_tests()
    tester.save_results(args.output_dir)

if __name__ == "__main__":
    main() 