#!/usr/bin/env python3
"""
Comprehensive Model Testing Script for YOLOv5, YOLOv7, and Ultralytics Models
Tests all trained models from runs folders and saves detailed results.
"""

import os
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
            'signal': 'signal_dataset.yaml',  # You may need to create this
            'reflec': 'reflec_dataset.yaml',  # You may need to create this
            'range': 'range_dataset.yaml',    # You may need to create this
            'nearir': 'nearir_dataset.yaml', # You may need to create this
            'combined': 'combined_dataset.yaml' # You may need to create this
        }
        
        # Try to find the YOLO repositories
        self.yolov5_path = self.find_yolo_repo('yolov5')
        self.yolov7_path = self.find_yolo_repo('yolov7')
        self.ultralytics_path = self.find_yolo_repo('ultralytics')
        
    def find_yolo_repo(self, repo_name):
        """Find YOLO repository path"""
        # Common locations to look for YOLO repos
        possible_paths = [
            self.base_dir / repo_name,
            self.base_dir / 'models' / repo_name,
            Path.cwd() / repo_name,
            Path.cwd().parent / repo_name
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        print(f"Warning: {repo_name} repository not found. Please ensure it's cloned.")
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
        
        # If no specific YAML found, try to use a default one
        default_yamls = ['Permutation1.yaml', 'data.yaml', 'dataset.yaml']
        for yaml_file in default_yamls:
            yaml_path = self.base_dir / yaml_file
            if yaml_path.exists():
                print(f"Warning: Using default YAML {yaml_file} for {model_name}")
                return str(yaml_path)
        
        print(f"Error: No suitable YAML file found for {model_name}")
        return None
    
    def find_model_weights(self):
        """Find all trained model weights in runs folders"""
        models = []
        
        # Search in runs folders
        runs_patterns = [
            'runs/runs/train/*/weights/best.pt',
            'runs (1)/runs/train/*/weights/best.pt', 
            'runs/train/*/weights/best.pt',
            'models/*/runs/train/*/weights/best.pt'
        ]
        
        for pattern in runs_patterns:
            weight_files = glob.glob(str(self.base_dir / pattern))
            for weight_file in weight_files:
                weight_path = Path(weight_file)
                model_name = weight_path.parent.parent.name
                
                # Skip exp4 as requested
                if model_name == 'exp4':
                    continue
                
                # Determine model type based on path and name
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
        
        # Check for YOLOv5 models
        if 'yolov5' in model_path_str or any(name in model_path_str for name in ['exp', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']):
            return 'yolov5'
        
        # Check for YOLOv7 models  
        if 'yolov7' in model_path_str or 'v7-tiny' in model_path_str or 'finetuned_v7' in model_path_str:
            return 'yolov7'
            
        # Default to ultralytics for newer models
        return 'ultralytics'
    
    def test_yolov5_model(self, model_info):
        """Test YOLOv5 model"""
        if not self.yolov5_path.exists():
            print(f"Skipping {model_info['name']} - YOLOv5 repository not found")
            return None
            
        try:
            print(f"Testing YOLOv5 model: {model_info['name']}")
            
            # Use dataset.yaml (should be in yolov5 directory)
            dataset_yaml = self.yolov5_path / 'dataset.yaml'
            
            cmd = [
                sys.executable, 'val.py',
                '--data', 'dataset.yaml',
                '--weights', str(Path.cwd() / model_info['weights']),
                '--img', '1024',
                '--batch', '8',
                '--conf', '0.001',
                '--iou', '0.65',
                '--device', 'cpu',
                '--save-txt',
                '--save-conf',
                '--name', f"test_{model_info['name']}"
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.yolov5_path,
                capture_output=True, 
                text=True, 
                timeout=3600
            )
            
            if result.returncode == 0:
                return self.parse_yolov5_results(result.stdout, model_info)
            else:
                print(f"Error testing {model_info['name']}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
            return None
    
    def test_yolov7_model(self, model_info):
        """Test YOLOv7 model"""
        if not self.yolov7_path.exists():
            print(f"Skipping {model_info['name']} - YOLOv7 repository not found")
            return None
            
        try:
            print(f"Testing YOLOv7 model: {model_info['name']}")
            
            # Use dataset.yaml (should be in yolov7 directory)
            dataset_yaml = self.yolov7_path / 'dataset.yaml'
            
            cmd = [
                sys.executable, 'test.py',
                '--data', 'dataset.yaml',
                '--weights', str(Path.cwd() / model_info['weights']),
                '--img', '1024',
                '--batch', '8',
                '--conf', '0.001',
                '--iou', '0.65',
                '--device', 'cpu',
                '--save-txt',
                '--save-conf',
                '--name', f"test_{model_info['name']}"
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.yolov7_path,
                capture_output=True, 
                text=True, 
                timeout=3600
            )
            
            if result.returncode == 0:
                return self.parse_yolov7_results(result.stdout, model_info)
            else:
                print(f"Error testing {model_info['name']}: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
            return None
    
    def test_ultralytics_model(self, model_info):
        """Test Ultralytics YOLO model"""
        if not model_info['yaml']:
            print(f"Skipping {model_info['name']} - No YAML file found")
            return None
        
        try:
            # Try using ultralytics CLI
            cmd = [
                'yolo', 'val',
                f"model={model_info['weights']}",
                f"data={model_info['yaml']}",
                'imgsz=640',
                'batch=16',
                'conf=0.001',
                'iou=0.65',
                'save_txt=True',
                'save_conf=True',
                'save_json=True',
                f"name=test_{model_info['name']}_{self.timestamp}"
            ]
            
            print(f"Testing Ultralytics model: {model_info['name']}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                return self.parse_ultralytics_results(result.stdout, model_info)
            else:
                print(f"Error testing {model_info['name']}: {result.stderr}")
                # Try alternative method with Python API
                return self.test_ultralytics_python_api(model_info)
                
        except subprocess.TimeoutExpired:
            print(f"Timeout testing {model_info['name']}")
            return None
        except Exception as e:
            print(f"Exception testing {model_info['name']}: {e}")
            return self.test_ultralytics_python_api(model_info)
    
    def test_ultralytics_python_api(self, model_info):
        """Test using Ultralytics Python API as fallback"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_info['weights'])
            metrics = model.val(data=model_info['yaml'], imgsz=640, batch=16, conf=0.001, iou=0.65)
            
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
            
        except ImportError:
            print("Ultralytics package not available for Python API testing")
            return None
        except Exception as e:
            print(f"Error using Ultralytics Python API for {model_info['name']}: {e}")
            return None
    
    def parse_yolov5_results(self, output, model_info):
        """Parse YOLOv5 test results from output"""
        # Debug: Print the last part of output to see format
        print(f"DEBUG - Last 500 chars of output:\n{output[-500:]}")
        
        # Look for the results line with format: "all        197        395      0.889      0.829      0.915      0.451"
        import re
        
        # Pattern to match the "all" results line
        pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, output)
        
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            map50 = float(match.group(3))
            map50_95 = float(match.group(4))
            print(f"DEBUG - Parsed values: P={precision}, R={recall}, mAP50={map50}, mAP50-95={map50_95}")
        else:
            print("DEBUG - 'all' pattern not found, trying fallback patterns")
            # Fallback patterns for different output formats
            precision = self.extract_metric(output, r'(?:P|Precision)[:=\s]*([\d.]+)')
            recall = self.extract_metric(output, r'(?:R|Recall)[:=\s]*([\d.]+)')
            map50 = self.extract_metric(output, r'mAP@?\.?5[:\s]*([\d.]+)')
            map50_95 = self.extract_metric(output, r'mAP@?\.?5[:\-\.]*95[:\s]*([\d.]+)')
            print(f"DEBUG - Fallback values: P={precision}, R={recall}, mAP50={map50}, mAP50-95={map50_95}")
        
        return {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'weights_path': model_info['weights'],
            'yaml_path': 'dataset.yaml',
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map50_95': map50_95,
            'test_date': datetime.now().isoformat()
        }
    
    def parse_yolov7_results(self, output, model_info):
        """Parse YOLOv7 test results from output"""
        # Look for the results line with format: "all        197        395      0.889      0.829      0.915      0.451"
        import re
        
        # Pattern to match the "all" results line
        pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, output)
        
        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            map50 = float(match.group(3))
            map50_95 = float(match.group(4))
        else:
            # Fallback patterns for different output formats
            precision = self.extract_metric(output, r'(?:P|Precision)[:=\s]*([\d.]+)')
            recall = self.extract_metric(output, r'(?:R|Recall)[:=\s]*([\d.]+)')
            map50 = self.extract_metric(output, r'mAP@?\.?5[:\s]*([\d.]+)')
            map50_95 = self.extract_metric(output, r'mAP@?\.?5[:\-\.]*95[:\s]*([\d.]+)')
        
        return {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'weights_path': model_info['weights'],
            'yaml_path': 'dataset.yaml',
            'precision': precision,
            'recall': recall,
            'map50': map50,
            'map50_95': map50_95,
            'test_date': datetime.now().isoformat()
        }
    
    def parse_ultralytics_results(self, output, model_info):
        """Parse Ultralytics test results from output"""
        # Look for metrics in output
        precision = self.extract_metric(output, r'precision:\s*([\d.]+)')
        recall = self.extract_metric(output, r'recall:\s*([\d.]+)')
        map50 = self.extract_metric(output, r'mAP50:\s*([\d.]+)')
        map50_95 = self.extract_metric(output, r'mAP50-95:\s*([\d.]+)')
        
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
        import re
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        return 0.0
    
    def run_all_tests(self):
        """Run tests on all found models"""
        models = self.find_model_weights()
        
        if not models:
            print("No trained models found in runs folders!")
            return
        
        print(f"Found {len(models)} trained models to test")
        
        for model_info in models:
            print(f"\n{'='*60}")
            print(f"Testing: {model_info['name']} ({model_info['type']})")
            print(f"Weights: {model_info['weights']}")
            print(f"YAML: {model_info['yaml']}")
            print(f"{'='*60}")
            
            if model_info['type'] == 'yolov5':
                result = self.test_yolov5_model(model_info)
            elif model_info['type'] == 'yolov7':
                result = self.test_yolov7_model(model_info)
            elif model_info['type'] == 'ultralytics':
                result = self.test_ultralytics_model(model_info)
            else:
                print(f"Unknown model type: {model_info['type']}")
                continue
            
            if result:
                self.results.append(result)
                print(f"✓ Test completed for {model_info['name']}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  mAP@0.5: {result['map50']:.4f}")
                print(f"  mAP@0.5:0.95: {result['map50_95']:.4f}")
            else:
                print(f"✗ Test failed for {model_info['name']}")
    
    def save_results(self, output_dir='test_results'):
        """Save all test results to files"""
        if not self.results:
            print("No results to save!")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = output_path / f'model_test_results_{self.timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Results saved to: {csv_file}")
        
        # Save as JSON
        json_file = output_path / f'model_test_results_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {json_file}")
        
        # Save summary
        summary_file = output_path / f'model_test_summary_{self.timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("Model Testing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Model: {result['model_name']} ({result['model_type']})\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n") 
                f.write(f"mAP@0.5: {result['map50']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {result['map50_95']:.4f}\n")
                f.write(f"Test Date: {result['test_date']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Summary saved to: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("TESTING SUMMARY")
        print("="*60)
        print(f"{'Model Name':<25} {'Type':<12} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50:95':<10}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result['model_name']:<25} {result['model_type']:<12} "
                  f"{result['precision']:<10.4f} {result['recall']:<10.4f} "
                  f"{result['map50']:<10.4f} {result['map50_95']:<10.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test all trained YOLO models')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory to search for models and YAML files')
    parser.add_argument('--output-dir', default='test_results',
                        help='Directory to save test results')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.base_dir)
    
    # Run all tests
    tester.run_all_tests()
    
    # Save results
    tester.save_results(args.output_dir)

if __name__ == "__main__":
    main() 