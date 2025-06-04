# Enhanced Image Preprocessing for PaliGemma

## Problem Statement

Your PaliGemma model requires 896x896 square images, but your dataset contains 1024x128 rectangular images. Direct resizing would distort the images and invalidate the bounding box annotations from Roboflow.

## Solution Overview

This enhanced preprocessing system:

1. **Preserves aspect ratio** when resizing 1024x128 images
2. **Adds appropriate padding** to create 896x896 square images  
3. **Automatically scales coordinates** in PaliGemma location tokens
4. **Validates transformations** to ensure correctness

## Files Provided

### 1. `enhanced_preprocessing.py` 
Core preprocessing functions:
- `preprocess_image_with_padding()` - Main preprocessing function
- `scale_coordinates_to_square()` - Coordinate transformation  
- `validate_coordinate_scaling()` - Validation function
- Enhanced data iterators for training and validation

### 2. `integration_guide.md`
Step-by-step guide for integrating into your existing `thebiggermod.ipynb` notebook.

### 3. `example_usage.py`
Comprehensive examples demonstrating all functionality.

## Quick Start

### 1. Basic Usage

```python
from enhanced_preprocessing import preprocess_image_with_padding, scale_coordinates_to_square

# Process your 1024x128 image
image = Image.open("your_1024x128_image.jpg")
processed_image, transform_params = preprocess_image_with_padding(image, size=896)

# Scale your coordinates
original_coords = "<loc0512><loc0064><loc0768><loc0096> 0"
scaled_coords = scale_coordinates_to_square(original_coords, transform_params)

print(f"Original: {original_coords}")
print(f"Scaled:   {scaled_coords}")
```

### 2. Integration with Your Notebook

Replace your existing data iterators with enhanced versions:

```python
def train_data_iterator():
    dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    for example in dataset.as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))
        
        # Enhanced preprocessing
        processed_image, transform_params = preprocess_image_with_padding(image)
        
        prefix = example["prefix"].decode().lower()
        suffix = example["suffix"].decode().lower()
        
        # Scale coordinates
        scaled_suffix = scale_coordinates_to_square(suffix, transform_params)
        
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, scaled_suffix, SEQLEN)
        label, _, _, _ = preprocess_tokens(scaled_suffix, seqlen=SEQLEN)

        yield {
            "image": np.asarray(processed_image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_loss": np.asarray(mask_loss),
        }
```

## How It Works

### Image Transformation
For a 1024x128 image → 896x896:

1. **Calculate scale factor**: `min(896/1024, 896/128) = 0.875`
2. **Resize maintaining aspect ratio**: 1024x128 → 896x112
3. **Add padding**: Center the 896x112 image in 896x896 canvas
4. **Result**: 896x896 image with gray padding above and below

### Coordinate Transformation
For each bounding box `<loc_x1><loc_y1><loc_x2><loc_y2>`:

1. **Convert to original image space**: `(x1/1024)*1024, (y1/1024)*128, etc.`
2. **Apply scaling**: Multiply by scale factor (0.875)
3. **Add padding offsets**: Add x_offset=0, y_offset=392
4. **Convert back to PaliGemma space**: `(new_x/896)*1024, (new_y/896)*1024`
5. **Clamp to valid range**: Ensure coordinates are in [0, 1023]

## Example Transformations

```
Original 1024x128 image coordinates:
<loc0512><loc0064><loc0768><loc0096> 0

Transformed 896x896 image coordinates:  
<loc0512><loc0456><loc0768><loc0460> 0
```

- **X coordinates** (512, 768): Unchanged (no horizontal padding)
- **Y coordinates** (64, 96): Shifted to (456, 460) due to vertical padding

## Testing Results

The system has been tested with:
- ✅ 1024x128 input images
- ✅ Multiple bounding boxes per image
- ✅ Coordinate validation and verification
- ✅ Integration with existing PaliGemma workflow

Example test output:
```
Transform Parameters:
  scale: 0.875
  x_offset: 0
  y_offset: 392
  original_width: 1024
  original_height: 128
  new_width: 896
  new_height: 112

Validation: ✓ All coordinate transformations correct
```

## Benefits

1. **Maintains image quality**: No distortion from aspect ratio changes
2. **Preserves annotations**: Bounding boxes remain accurate after transformation  
3. **Easy integration**: Drop-in replacement for existing preprocessing
4. **Validated results**: Built-in validation ensures correctness
5. **Performance**: Minimal computational overhead

## Next Steps

1. **Test with your data**: Run `python example_usage.py` to see the system in action
2. **Integrate**: Follow the `integration_guide.md` to update your notebook
3. **Validate**: Use the validation functions to verify your coordinate transformations
4. **Train**: Your model should now work correctly with 1024x128 images!

## Support

If you encounter any issues:
1. Check the coordinate format matches PaliGemma expectations
2. Verify the transformation parameters make sense for your image dimensions
3. Use the validation functions to debug coordinate scaling
4. Ensure `enhanced_preprocessing.py` is in your Python path

The enhanced preprocessing system is ready for production use with your PaliGemma fine-tuning workflow! 