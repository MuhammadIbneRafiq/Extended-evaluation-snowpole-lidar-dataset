# 🔧 Kernel Crash Fix Guide

## Quick Fix Steps

### 1. Restart Kernel
- **Jupyter Notebook**: Kernel → Restart & Clear Output
- **VS Code**: Click the restart button or Ctrl+Shift+P → "Jupyter: Restart Kernel"

### 2. Run Cells in Order
After restarting, run cells in this exact order:
1. Cell 1: Title (markdown - skip)
2. Cell 2: Imports
3. Cell 3: Paths configuration
4. Cell 4: **Simple test** (NEW - run this first!)
5. Cell 5: Full test (optional - only if Cell 4 works)

### 3. If Crash Persists

#### Option A: Use Standalone Script
```bash
python test_pipeline_standalone.py
```
This runs the same test outside Jupyter, avoiding notebook-specific issues.

#### Option B: Check Dependencies
```bash
pip install --upgrade opencv-python numpy matplotlib
pip install --upgrade notebook ipykernel
```

#### Option C: Clear Jupyter Cache
```bash
jupyter cache clear
jupyter kernelspec list
jupyter kernelspec remove <kernel_name>
jupyter kernelspec install --user --name=<kernel_name>
```

## What Was Fixed

1. **Matplotlib Backend Issue**
   - Changed to non-interactive 'Agg' backend
   - Moved imports outside function

2. **Memory Issues**
   - Added error handling
   - Save outputs instead of displaying

3. **Import Order**
   - Fixed import sequence
   - Added fallback imports

## Test Without Notebook

If the notebook keeps crashing, use the standalone script:

```python
# test_pipeline_standalone.py already created
python test_pipeline_standalone.py
```

This will:
- Test the pipeline
- Show results in console
- Save test images to disk
- Option to process full dataset

## Common Causes

1. **Matplotlib Display**: Trying to show plots in certain environments
2. **Memory**: Large images in memory
3. **OpenCV**: Version conflicts with notebook
4. **Path Issues**: Special characters in paths

## Emergency Workaround

If nothing works, create the dataset directly:

```python
import os
os.system("python test_pipeline_standalone.py")
```

Then continue with model training using the created dataset.
