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
        self.yolov5_path = Path.cwd() / 'yolov5'
        self.yolov7_path = Path.cwd() / 'yolov7'
        self.ultralytics_path = Path.cwd() / 'ultralytics'
        
    def find_model_weights(self):
        models = []
        runs_patterns = [
            'runs/runs/train/*/weights/best.pt',  # This one works for YOLOv5
            'runs-yolo-v7/runs/train/*/weights/best.pt',  # This one works for YOLOv7
        ]
        
        for pattern in runs_patterns:
            weight_files = glob.glob(str(self.base_dir / pattern))
            for weight_file in weight_files:
                weight_path = Path(weight_file)
                model_name = weight_path.parent.parent.name
                
                if model_name == 'exp4':
                    continue
                
                model_type = self._detect_model_type(weight_path)
                dataset_file = self._get_dataset_for_model(model_name, model_type)
                
                models.append({
                    'name': model_name,
                    'type': model_type,
                    'weights': str(weight_path),
                    'yaml': dataset_file
                })
        
        return models
    
    def _get_dataset_for_model(self, model_name, model_type):
        """Match each model with its corresponding dataset file"""
        model_name_lower = model_name.lower()
        
        if model_type == 'yolov5':
            # For yolov5 models, check what permutation they were trained on
            if 'perm1' in model_name_lower:
                return 'Permutation1.yaml'
            elif 'perm2' in model_name_lower:
                return 'Permutation2.yaml'
            elif 'perm3' in model_name_lower:
                return 'Permutation3.yaml'
            elif 'perm4' in model_name_lower:
                return 'Permutation4.yaml'
            elif 'perm5' in model_name_lower:
                return 'Permutation5.yaml'
            elif 'perm6' in model_name_lower:
                return 'Permutation6.yaml'
            else:
                return 'dataset.yaml'  # fallback
                
        elif model_type == 'yolov7':
            # For yolov7 models, check what permutation they were trained on
            if 'perm1' in model_name_lower:
                return 'Permutation1.yaml'
            elif 'perm2' in model_name_lower:
                return 'Permutation2.yaml'
            elif 'perm3' in model_name_lower:
                return 'Permutation3.yaml'
            elif 'perm4' in model_name_lower:
                return 'Permutation4.yaml'
            elif 'perm5' in model_name_lower:
                return 'Permutation5.yaml'
            elif 'perm6' in model_name_lower:
                return 'Permutation6.yaml'
            else:
                return 'dataset.yaml'  # fallback
        else:
            return 'dataset.yaml'  # fallback for ultralytics models
    
    def _detect_model_type(self, model_path):
        model_path_str = str(model_path).lower()
        
        if 'runs/runs/train' in model_path_str or 'yolov5' in model_path_str:
            return 'yolov5'
        elif 'runs-yolo-v7' in model_path_str or 'yolov7' in model_path_str:
            return 'yolov7'
        else:
            return 'ultralytics'
    
    def test_yolov5_model(self, model_info):
        # For YOLOv5, the weights path needs to be relative to the main directory, not yolov5 directory
        weights_path = Path(model_info['weights'])
        if weights_path.is_absolute():
            # Make relative to yolov5 directory by going up one level
            try:
                # The path is like: /main_dir/runs/runs/train/model/weights/best.pt
                # From yolov5 directory, it should be: ../runs/runs/train/model/weights/best.pt
                relative_weights = Path('..') / weights_path.relative_to(Path.cwd())
            except ValueError:
                relative_weights = weights_path
        else:
            relative_weights = Path('..') / weights_path
            
        cmd = [
            sys.executable, 'val.py',
            '--data', model_info['yaml'],  # Use the specific dataset for this model
            '--weights', str(relative_weights),
            '--img', '1024',
            '--batch', '4',  # Smaller batch for testing
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',  # Use CPU for now to avoid GPU issues
            '--name', f"test_{model_info['name']}"
        ]
        
        print(f"Testing YOLOv5 model: {model_info['name']} with dataset: {model_info['yaml']}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run with real-time output visible in terminal
        process = subprocess.Popen(cmd, cwd=self.yolov5_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Show in terminal
                output_lines.append(output)  # Capture for parsing
        
        full_output = ''.join(output_lines)
        print("=" * 80)
        
        return self.parse_yolo_results(full_output, model_info)

    def test_yolov7_model(self, model_info):
        """Test YOLOv7 model"""
        if not self.yolov7_path or not self.yolov7_path.exists():
            return None
        
        # For YOLOv7, similar path handling
        weights_path = Path(model_info['weights'])
        if weights_path.is_absolute():
            try:
                relative_weights = Path('..') / weights_path.relative_to(Path.cwd())
            except ValueError:
                relative_weights = weights_path
        else:
            relative_weights = Path('..') / weights_path
            
        cmd = [
            sys.executable, 'test.py',
            '--data', model_info['yaml'],  # Use the specific dataset for this model
            '--weights', str(relative_weights),
            '--img', '1024',
            '--batch', '4',
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',
            '--name', f"test_{model_info['name']}"
        ]
        
        print(f"Testing YOLOv7 model: {model_info['name']} with dataset: {model_info['yaml']}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {self.yolov7_path}")
        print("=" * 80)
        
        # Run with real-time output visible in terminal
        process = subprocess.Popen(cmd, cwd=self.yolov7_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Show in terminal
                output_lines.append(output)  # Capture for parsing
        
        full_output = ''.join(output_lines)
        print("=" * 80)
        
        return self.parse_yolo_results(full_output, model_info)

    def test_ultralytics_model(self, model_info):
        """Test Ultralytics YOLO model"""
        cmd = [
            'yolo', 'val',
            f"model={model_info['weights']}",
            f"data={model_info['yaml']}",
            'imgsz=1024',
            'batch=4',
            'conf=0.001',
            'iou=0.65',
            'device=cpu',
            f"name=test_{model_info['name']}"
        ]
        
        print(f"Testing Ultralytics model: {model_info['name']} with dataset: {model_info['yaml']}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80)
        
        # Run with real-time output visible in terminal
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Show in terminal
                output_lines.append(output)  # Capture for parsing
        
        full_output = ''.join(output_lines)
        print("=" * 80)
        
        return self.parse_yolo_results(full_output, model_info)

    
    def test_ultralytics_python_api(self, model_info):        
        model = YOLO(model_info['weights'])
        metrics = model.val(data=model_info['yaml'], imgsz=1024, batch=4, conf=0.001, iou=0.65, device='cpu')
        
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

        # Extract inference speed (e.g., "181.4ms inference")
        inference_speed = self.extract_inference_speed(output)

        result = {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'weights_path': model_info['weights'],
            'yaml_path': model_info['yaml'],
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map50_95': map50_95,
            'inference_speed_ms': inference_speed,
            'test_date': datetime.now().isoformat()
        }
        
        # Save individual JSON file for this model
        self.save_individual_result(result)
        
        return result
    
    def extract_metric(self, text, pattern):
        """Extract metric value using regex pattern"""
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def extract_inference_speed(self, text):
        """Extract inference speed in milliseconds from output"""
        # Look for patterns like "181.4ms inference" or "Speed: 0.5ms preprocess, 3.7ms inference"
        patterns = [
            r'([\d.]+)ms inference',
            r'Speed:.*?([\d.]+)ms inference',
            r'inference.*?([\d.]+)ms'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def save_individual_result(self, result):
        """Save individual model result to JSON file"""
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{result['model_name']}_{result['model_type']}_results.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Saved individual results: {filepath}")
        print(f"📊 {result['model_name']}: P={result['precision']:.3f}, R={result['recall']:.3f}, mAP50={result['map50']:.3f}, mAP50:95={result['map50_95']:.3f}, Speed={result['inference_speed_ms']:.1f}ms")
    
    def run_all_tests(self):
        models = self.find_model_weights()
        print(f"Found {len(models)} models to test")
        
        for model_info in models:           
            if model_info['type'] == 'yolov5':
                result = self.test_yolov5_model(model_info)
            elif model_info['type'] == 'yolov7':
                result = self.test_yolov7_model(model_info)
            else:
                result = self.test_ultralytics_model(model_info)

            self.results.append(result)
            # Individual results are now printed in save_individual_result method

    
    def save_results(self, output_dir='test_results'):        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save combined results
        with open(output_path / f'model_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save a CSV for easy viewing
        import csv
        csv_file = output_path / f'model_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(csv_file, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        
        print(f"\n{'Model Name':<25} {'Type':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50:95':<10} {'Speed(ms)':<10}")
        print("=" * 102)
        for result in self.results:
            print(f"{result['model_name']:<25} {result['model_type']:<12} {result['precision']:<10.3f} {result['recall']:<10.3f} {result['map50']:<10.3f} {result['map50_95']:<10.3f} {result['inference_speed_ms']:<10.1f}")
        
        print(f"\n✅ All results saved to: {output_path}")
        print(f"📊 Combined JSON: model_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"📈 CSV file: model_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

tester = ModelTester()
tester.run_all_tests()
tester.save_results('')
