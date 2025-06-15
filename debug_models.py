import sys
import glob
from pathlib import Path

print('=== CHECKING MODEL DISCOVERY ===')
patterns = [
    'runs/runs/train/*/weights/best.pt',
    'runs-yolo-v7/runs/train/*/weights/best.pt', 
    'runs/train/*/weights/best.pt',
    'yolov5/runs/train/*/weights/best.pt',
    'yolov7/runs/train/*/weights/best.pt',
    'ultralytics/runs/train/*/weights/best.pt'
]

found_models = []
for pattern in patterns:
    weight_files = glob.glob(pattern)
    if weight_files:
        print(f'Pattern: {pattern}')
        for f in weight_files[:3]:  # Show first 3
            print(f'  Found: {f}')
            found_models.append(f)
        print()

print(f'Total models found: {len(found_models)}')

if found_models:
    print('\n=== TESTING FIRST MODEL ===')
    test_model = found_models[0]
    print(f'Testing model: {test_model}')
    
    # Check if it's a YOLOv5 model and test manually
    if 'yolov5' in test_model.lower() or any(name in test_model.lower() for name in ['exp', 'perm1', 'perm2']):
        print('Detected as YOLOv5 model, testing manually...')
        
        import subprocess
        yolov5_path = Path.cwd() / 'yolov5'
        weights_path = Path(test_model)
        
        if weights_path.is_absolute():
            try:
                relative_weights = weights_path.relative_to(Path.cwd())
            except ValueError:
                relative_weights = weights_path
        else:
            relative_weights = weights_path
            
        cmd = [
            sys.executable, 'val.py',
            '--data', 'dataset.yaml',
            '--weights', str(relative_weights),
            '--img', '1024',
            '--batch', '4',  # Smaller batch for testing
            '--conf', '0.001',
            '--iou', '0.65',
            '--device', 'cpu',  # Use CPU for testing
            '--name', 'debug_test'
        ]
        
        print(f'Running command: {" ".join(cmd)}')
        print(f'From directory: {yolov5_path}')
        
        try:
            result = subprocess.run(cmd, cwd=yolov5_path, capture_output=True, text=True, timeout=120)
            print(f'Return code: {result.returncode}')
            print(f'STDOUT:\n{result.stdout}')
            if result.stderr:
                print(f'STDERR:\n{result.stderr}')
        except Exception as e:
            print(f'Error running command: {e}')

print('\n=== CHECKING DATASET FILES ===')
dataset_files = ['dataset.yaml', 'main_dataset.yaml']
for dataset_file in dataset_files:
    if Path(dataset_file).exists():
        print(f'{dataset_file}: EXISTS')
    else:
        print(f'{dataset_file}: NOT FOUND') 