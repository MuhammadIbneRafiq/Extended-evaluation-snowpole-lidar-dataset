import sys
import subprocess
import json
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import glob
import re

class ModelTester:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.results = []
        self.main_yaml = 'dataset.yaml'
        self.yolov5_path = Path.cwd() / 'yolov5'
        self.yolov7_path = Path.cwd() / 'yolov7'
        self.ultralytics_path = Path.cwd() / 'ultralytics'
        
    def find_model_weights(self):
        models = []
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
                    'yaml': self.main_yaml
                })
        
        return models
    
    def _detect_model_type(self, model_path):
        model_path_str = str(model_path).lower()
        
        if 'yolov5' in model_path_str or any(name in model_path_str for name in ['exp', 'exp2', 'exp3', 'exp5', 'exp6']):
            return 'yolov5'
        elif 'yolov7' in model_path_str or 'v7-tiny' in model_path_str:
            return 'yolov7'
        else:
            return 'ultralytics'
    
    def test_yolov5_model(self, model_info):
        # Convert weights path to relative path from yolov5 directory
        weights_path = Path(model_info['weights'])
        if weights_path.is_absolute():
            # Make relative to current working directory
            try:
                relative_weights = weights_path.relative_to(Path.cwd())
            except ValueError:
                relative_weights = weights_path
        else:
            relative_weights = weights_path
            
        print(f"Relative weights path: {relative_weights}")
            
        cmd = [
            sys.executable, 'val.py',
            '--data', 'dataset.yaml',
            '--weights', str(relative_weights),
            '--img', '1024',
            '--batch', '32',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', '0',
            '--name', f"test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, cwd=self.yolov5_path, capture_output=True, text=True, timeout=600)
        return self.parse_yolo_results(result.stdout, model_info)
    
    def test_yolov7_model(self, model_info):
        """Test YOLOv7 model"""
        if not self.yolov7_path or not self.yolov7_path.exists():
            return None
        
        # Convert weights path to relative path from yolov7 directory
        weights_path = Path(model_info['weights'])
        if weights_path.is_absolute():
            # Make relative to current working directory
            try:
                relative_weights = weights_path.relative_to(Path.cwd())
            except ValueError:
                relative_weights = weights_path
        else:
            relative_weights = weights_path
                        
        cmd = [
            sys.executable, 'test.py',
            '--data', 'dataset.yaml',
            '--weights', str(relative_weights),
            '--img', '1024',
            '--batch', '32',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', '0',
            '--name', f"test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, cwd=self.yolov7_path, capture_output=True, text=True, timeout=600)
        return self.parse_yolo_results(result.stdout, model_info)
    
    def test_ultralytics_model(self, model_info):
        """Test Ultralytics YOLO model"""
        cmd = [
            'yolo', 'val',
            f"model={model_info['weights']}",
            f"data=dataset.yaml",
            'imgsz=1024',
            'batch=8',
            'conf=0.001',
            'iou=0.65',
            'device=0',
            f"name=test_{model_info['name']}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return self.parse_yolo_results(result.stdout, model_info)

    
    def test_ultralytics_python_api(self, model_info):        
        model = YOLO(model_info['weights'])
        metrics = model.val(data='dataset.yaml', imgsz=1024, batch=8, conf=0.001, iou=0.65, device=0)
        
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
        models = self.find_model_weights()
        for model_info in models:           
            if model_info['type'] == 'yolov5':
                result = self.test_yolov5_model(model_info)
            elif model_info['type'] == 'yolov7':
                result = self.test_yolov7_model(model_info)
            else:
                result = self.test_ultralytics_model(model_info)

            self.results.append(result)
            print(f"✓ {model_info['name']}: P={result['precision']:.3f}, R={result['recall']:.3f}, mAP50={result['map50']:.3f}, mAP50:95={result['map50_95']:.3f}")

    
    def save_results(self, output_dir='test_results'):        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f'model_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'Model Name':<25} {'Type':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50:95':<10}")       

tester = ModelTester()
tester.run_all_tests()
tester.save_results('')
