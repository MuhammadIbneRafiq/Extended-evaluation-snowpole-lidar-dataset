# Comprehensive YOLO Model Testing Suite

This testing suite automatically evaluates all your trained YOLO models (YOLOv5, YOLOv7, and Ultralytics) and provides detailed performance metrics including precision, recall, mAP@0.5, and mAP@0.5:0.95.

## 🚀 Features

- **Multi-Framework Support**: Tests YOLOv5, YOLOv7, and Ultralytics models
- **Automatic Discovery**: Finds all trained models in your runs folders
- **Smart YAML Mapping**: Automatically matches models to their dataset configurations
- **Comprehensive Metrics**: Extracts precision, recall, mAP@0.5, and mAP@0.5:0.95
- **Multiple Output Formats**: Saves results as CSV, JSON, and summary text files
- **Error Handling**: Robust error handling and timeout protection
- **Progress Reporting**: Clear progress indicators and detailed logging

## 📁 File Structure

```
├── comprehensive_model_testing.py  # Main testing script
├── run_comprehensive_tests.py      # Simple runner script
├── README_model_testing.md         # This file
└── test_results/                   # Output folder (created automatically)
    ├── model_test_results_YYYYMMDD_HHMMSS.csv
    ├── model_test_results_YYYYMMDD_HHMMSS.json
    └── model_test_summary_YYYYMMDD_HHMMSS.txt
```

## 🔧 Prerequisites

### Required YOLO Repositories

The script automatically detects YOLO repositories in these locations:
- `./yolov5/` - YOLOv5 repository
- `./yolov7/` - YOLOv7 repository
- Ultralytics package (installed via pip)

To set up the repositories:

```bash
# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5.git

# Clone YOLOv7
git clone https://github.com/WongKinYiu/yolov7.git

# Install Ultralytics
pip install ultralytics
```

### Required Python Packages

```bash
pip install pandas ultralytics
```

## 📊 Expected Folder Structure

Your trained models should be in one of these structures:

```
runs/
├── runs/train/                    # YOLOv5 models
│   ├── yolov5s_perm1/weights/best.pt
│   ├── yolov5s_perm2/weights/best.pt
│   └── ...
└── (1)/runs/train/               # YOLOv7 models
    ├── yolov7-tiny-perm1/weights/best.pt
    ├── yolov7-tiny-signal/weights/best.pt
    └── ...
```

## 🗂️ Dataset Configuration Files

The script expects YAML configuration files for your datasets:

```
Permutation1.yaml
Permutation2.yaml
Permutation3.yaml
Permutation4.yaml
Permutation5.yaml
Permutation6.yaml
```

### Example YAML Structure

```yaml
# Permutation1.yaml
path: /path/to/your/dataset  # dataset root dir
train: train/images          # train images
val: valid/images           # validation images
test: test/images           # test images

# Classes
nc: 1  # number of classes
names:
  0: object  # class names
```

## 🚀 Usage

### Quick Start (Recommended)

Simply run the automated script:

```bash
python run_comprehensive_tests.py
```

This script will:
1. Check dependencies and install missing packages
2. Verify YOLO repositories are available
3. Scan for trained models
4. Create basic YAML files if missing
5. Run comprehensive testing
6. Save results to the `test_results/` folder

### Advanced Usage

For more control, use the main testing script directly:

```bash
python comprehensive_model_testing.py --base-dir . --output-dir test_results
```

#### Command Line Options

- `--base-dir`: Base directory to search for models and YAML files (default: current directory)
- `--output-dir`: Directory to save test results (default: 'test_results')

## 📈 Output Files

### CSV Results (`model_test_results_YYYYMMDD_HHMMSS.csv`)

| model_name | model_type | weights_path | yaml_path | precision | recall | map50 | map50_95 | test_date |
|------------|------------|--------------|-----------|-----------|--------|--------|----------|-----------|
| yolov5s_perm1 | yolov5 | runs/.../best.pt | Permutation1.yaml | 0.8924 | 0.8756 | 0.9234 | 0.7891 | 2024-01-15T10:30:25 |

### JSON Results (`model_test_results_YYYYMMDD_HHMMSS.json`)

```json
[
  {
    "model_name": "yolov5s_perm1",
    "model_type": "yolov5", 
    "weights_path": "runs/runs/train/yolov5s_perm1/weights/best.pt",
    "yaml_path": "Permutation1.yaml",
    "precision": 0.8924,
    "recall": 0.8756,
    "map50": 0.9234,
    "map50_95": 0.7891,
    "test_date": "2024-01-15T10:30:25.123456"
  }
]
```

### Summary Report (`model_test_summary_YYYYMMDD_HHMMSS.txt`)

```
Model Testing Summary
==================================================

Model: yolov5s_perm1 (yolov5)
Precision: 0.8924
Recall: 0.8756
mAP@0.5: 0.9234
mAP@0.5:0.95: 0.7891
Test Date: 2024-01-15T10:30:25.123456
------------------------------
```

## 🔍 Understanding the Metrics

- **Precision**: Proportion of positive identifications that are actually correct
- **Recall**: Proportion of actual positives that are correctly identified
- **mAP@0.5**: Mean Average Precision at IoU threshold of 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95

## 🛠️ Troubleshooting

### Common Issues

1. **No models found**
   - Ensure your trained models are in the expected directory structure
   - Check that `best.pt` files exist in the weights folders

2. **YAML files missing**
   - The script creates basic YAML templates automatically
   - Edit them with correct dataset paths and class names

3. **YOLO repositories not found**
   - Clone the repositories to the expected locations
   - Or modify the `find_yolo_repo()` function to look in custom locations

4. **Permission errors**
   - Ensure you have write permissions in the current directory
   - Try running from a different location

5. **Memory errors**
   - Reduce batch size in the testing commands (currently set to 16)
   - Test models individually instead of all at once

### Debug Mode

To see detailed output during testing, you can modify the scripts to show stdout:

```python
# In comprehensive_model_testing.py, change:
result = subprocess.run(cmd, cwd=self.yolov5_path, capture_output=True, text=True, timeout=3600)
# To:
result = subprocess.run(cmd, cwd=self.yolov5_path, text=True, timeout=3600)
```

## 🎯 Customization

### Adding New Model Types

To add support for additional YOLO variants:

1. Add detection logic in `find_model_weights()`
2. Create a new test method (e.g., `test_yolov8_model()`)
3. Add parsing logic for the output format
4. Update the main testing loop in `run_all_tests()`

### Custom YAML Mapping

Modify the `yaml_mapping` dictionary in the `ModelTester` class:

```python
self.yaml_mapping = {
    'your_model_name': 'your_dataset.yaml',
    'custom_perm': 'custom_config.yaml',
    # ... add more mappings
}
```

### Custom Metrics Extraction

Modify the regex patterns in the parsing methods to extract additional metrics:

```python
def parse_custom_results(self, output, model_info):
    # Add custom metric extraction
    custom_metric = self.extract_metric(output, r'CustomMetric:\s*([\d.]+)')
    # ... rest of the method
```

## 📝 Example Complete Workflow

1. **Setup repositories**:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   git clone https://github.com/WongKinYiu/yolov7.git
   pip install ultralytics pandas
   ```

2. **Place your trained models** in the runs folders

3. **Create or verify YAML files** for your datasets

4. **Run the testing suite**:
   ```bash
   python run_comprehensive_tests.py
   ```

5. **Review results** in the `test_results/` folder

6. **Analyze performance** across different permutations and color modalities

## 📊 Performance Analysis

After running the tests, you can analyze the results to:

- Compare performance across different color modalities
- Identify the best performing permutation combinations
- Understand which model architecture works best for your specific data
- Track performance improvements across training iterations

The CSV output makes it easy to import into Excel, Python pandas, or other analysis tools for deeper investigation.

## 🤝 Contributing

Feel free to modify and extend these scripts for your specific needs. The modular design makes it easy to add new features or support for additional YOLO variants.

## 📄 License

This testing suite is provided as-is for research and development purposes. Make sure to comply with the licenses of the individual YOLO repositories you're using. 