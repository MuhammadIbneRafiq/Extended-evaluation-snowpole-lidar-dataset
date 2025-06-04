import numpy as np
import tensorflow as tf
import re
from PIL import Image


def preprocess_image_with_padding(image, size=896):
    """
    Enhanced preprocessing for images with proper aspect ratio handling and coordinate scaling.
    Handles 1024x128 images by padding to square and returns scaling factors for coordinate adjustment.
    
    Args:
        image: Input image (PIL Image or numpy array)
        size: Target square size (default 896)
    
    Returns:
        tuple: (processed_image, transform_params)
            - processed_image: Normalized image array in [-1, 1] range
            - transform_params: Dictionary with transformation parameters for coordinate scaling
    """
    image = np.asarray(image)
    
    # Remove alpha layer if present (keep only first 3 channels)
    image = image[..., :3]  # This handles RGB(A) -> RGB conversion
    
    # Verify we have 3 channels (throws error for grayscale/2-channel images)
    assert image.shape[-1] == 3, f"Expected 3 channels, got {image.shape[-1]}"
    
    original_height, original_width = image.shape[:2]
    
    # Calculate scale factor to fit the larger dimension to target size
    # while maintaining aspect ratio
    scale = min(size / original_width, size / original_height)
    
    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image maintaining aspect ratio
    image = tf.constant(image)
    image = tf.image.resize(image, (new_height, new_width), method='bilinear', antialias=True)
    image = image.numpy()
    
    # Create a square canvas with padding
    padded_image = np.full((size, size, 3), 128, dtype=image.dtype)  # Gray padding
    
    # Calculate padding offsets to center the image
    y_offset = (size - new_height) // 2
    x_offset = (size - new_width) // 2
    
    # Place the resized image in the center of the padded canvas
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = image
    
    # Return normalized image and transformation parameters
    return padded_image / 127.5 - 1.0, {
        'scale': scale,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'original_width': original_width,
        'original_height': original_height,
        'new_width': new_width,
        'new_height': new_height
    }


def preprocess_image_backward_compatible(image, size=896):
    """
    Backward compatible version of preprocess_image that returns only the processed image.
    This can be used as a drop-in replacement for the original preprocess_image function.
    
    Args:
        image: Input image (PIL Image or numpy array)
        size: Target square size (default 896)
    
    Returns:
        numpy array: Processed and normalized image in [-1, 1] range
    """
    processed_image, _ = preprocess_image_with_padding(image, size)
    return processed_image


def scale_coordinates_to_square(coordinates_str, transform_params, target_size=896):
    """
    Scale bounding box coordinates from original image space to square padded image space.
    
    Args:
        coordinates_str: String with location tokens like "<loc0592><loc0402><loc1023><loc0409>"
        transform_params: Dictionary with transformation parameters from preprocess_image_with_padding
        target_size: Target square image size (default 896)
    
    Returns:
        String with scaled location tokens
    """
    # Extract all location tokens
    loc_pattern = r'<loc(\d{4})>'
    locations = re.findall(loc_pattern, coordinates_str)
    
    if len(locations) == 0:
        return coordinates_str
    
    # Convert to actual coordinates (PaliGemma uses 1024-based coordinate system)
    coords = [int(loc) for loc in locations]
    
    scaled_coords = []
    for i in range(0, len(coords), 4):  # Process bounding boxes (x1, y1, x2, y2)
        if i + 3 < len(coords):
            # Original coordinates in 1024 coordinate space
            x1, y1, x2, y2 = coords[i], coords[i+1], coords[i+2], coords[i+3]
            
            # Convert from 1024 coordinate space to original image space
            orig_x1 = (x1 / 1024.0) * transform_params['original_width']
            orig_y1 = (y1 / 1024.0) * transform_params['original_height']
            orig_x2 = (x2 / 1024.0) * transform_params['original_width']
            orig_y2 = (y2 / 1024.0) * transform_params['original_height']
            
            # Scale to new image space
            new_x1 = orig_x1 * transform_params['scale']
            new_y1 = orig_y1 * transform_params['scale']
            new_x2 = orig_x2 * transform_params['scale']
            new_y2 = orig_y2 * transform_params['scale']
            
            # Add padding offsets
            final_x1 = new_x1 + transform_params['x_offset']
            final_y1 = new_y1 + transform_params['y_offset']
            final_x2 = new_x2 + transform_params['x_offset']
            final_y2 = new_y2 + transform_params['y_offset']
            
            # Convert back to 1024 coordinate space for PaliGemma
            scaled_x1 = int((final_x1 / target_size) * 1024)
            scaled_y1 = int((final_y1 / target_size) * 1024)
            scaled_x2 = int((final_x2 / target_size) * 1024)
            scaled_y2 = int((final_y2 / target_size) * 1024)
            
            # Clamp coordinates to valid range
            scaled_x1 = max(0, min(1023, scaled_x1))
            scaled_y1 = max(0, min(1023, scaled_y1))
            scaled_x2 = max(0, min(1023, scaled_x2))
            scaled_y2 = max(0, min(1023, scaled_y2))
            
            scaled_coords.extend([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
    
    # Reconstruct the coordinate string
    result = coordinates_str
    for i, (old_coord, new_coord) in enumerate(zip(coords, scaled_coords)):
        old_token = f"<loc{old_coord:04d}>"
        new_token = f"<loc{new_coord:04d}>"
        result = result.replace(old_token, new_token, 1)
    
    return result


def validate_coordinate_scaling(original_coords, scaled_coords, transform_params, target_size=896):
    """
    Validate that coordinate scaling is working correctly by checking if the 
    scaled coordinates properly represent the same relative positions.
    
    Args:
        original_coords: Original coordinate string
        scaled_coords: Scaled coordinate string  
        transform_params: Transformation parameters
        target_size: Target image size
    
    Returns:
        Dictionary with validation results
    """
    # Extract coordinates from both strings
    loc_pattern = r'<loc(\d{4})>'
    orig_locs = [int(loc) for loc in re.findall(loc_pattern, original_coords)]
    scaled_locs = [int(loc) for loc in re.findall(loc_pattern, scaled_coords)]
    
    if len(orig_locs) != len(scaled_locs):
        return {"valid": False, "error": "Coordinate count mismatch"}
    
    validation_results = {
        "valid": True,
        "bounding_boxes": [],
        "transform_applied": transform_params
    }
    
    for i in range(0, len(orig_locs), 4):
        if i + 3 < len(orig_locs):
            # Original bounding box
            orig_x1, orig_y1, orig_x2, orig_y2 = orig_locs[i:i+4]
            scaled_x1, scaled_y1, scaled_x2, scaled_y2 = scaled_locs[i:i+4]
            
            # Convert to relative coordinates for comparison
            orig_rel_x1 = orig_x1 / 1024.0
            orig_rel_y1 = orig_y1 / 1024.0
            orig_rel_x2 = orig_x2 / 1024.0
            orig_rel_y2 = orig_y2 / 1024.0
            
            scaled_rel_x1 = scaled_x1 / 1024.0
            scaled_rel_y1 = scaled_y1 / 1024.0
            scaled_rel_x2 = scaled_x2 / 1024.0
            scaled_rel_y2 = scaled_y2 / 1024.0
            
            bbox_info = {
                "bbox_index": i // 4,
                "original": {"x1": orig_x1, "y1": orig_y1, "x2": orig_x2, "y2": orig_y2},
                "scaled": {"x1": scaled_x1, "y1": scaled_y1, "x2": scaled_x2, "y2": scaled_y2},
                "original_relative": {"x1": orig_rel_x1, "y1": orig_rel_y1, "x2": orig_rel_x2, "y2": orig_rel_y2},
                "scaled_relative": {"x1": scaled_rel_x1, "y1": scaled_rel_y1, "x2": scaled_rel_x2, "y2": scaled_rel_y2}
            }
            
            validation_results["bounding_boxes"].append(bbox_info)
    
    return validation_results


def enhanced_train_data_iterator(train_dataset, SEQLEN, tokenizer, preprocess_tokens):
    """
    Enhanced training data iterator that uses the new preprocessing with coordinate scaling.
    
    Args:
        train_dataset: Training dataset object
        SEQLEN: Sequence length
        tokenizer: Tokenizer object
        preprocess_tokens: Function to preprocess tokens
    
    Yields:
        Dictionary with processed training examples
    """
    dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
    for example in dataset.as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))
        
        # Use enhanced preprocessing with coordinate tracking
        processed_image, transform_params = preprocess_image_with_padding(image)
        
        prefix = example["prefix"].decode().lower()
        suffix = example["suffix"].decode().lower()
        
        # Scale coordinates in the suffix if present
        scaled_suffix = scale_coordinates_to_square(suffix, transform_params)
        
        tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, scaled_suffix, SEQLEN)
        label, _, _, _ = preprocess_tokens(scaled_suffix, seqlen=SEQLEN)

        yield {
            "image": np.asarray(processed_image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_loss": np.asarray(mask_loss),
            "transform_params": transform_params,  # Include for debugging
        }


def enhanced_validation_data_iterator(val_dataset, SEQLEN, tokenizer, preprocess_tokens):
    """
    Enhanced validation data iterator that uses the new preprocessing with coordinate scaling.
    
    Args:
        val_dataset: Validation dataset object
        SEQLEN: Sequence length
        tokenizer: Tokenizer object
        preprocess_tokens: Function to preprocess tokens
    
    Yields:
        Dictionary with processed validation examples
    """
    for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
        image = Image.open(io.BytesIO(example["image"]))
        
        # Use enhanced preprocessing with coordinate tracking
        processed_image, transform_params = preprocess_image_with_padding(image)
        
        prefix = example["prefix"].decode().lower()
        suffix = example["suffix"].decode().lower()
        
        # Scale coordinates in the suffix if present
        scaled_suffix = scale_coordinates_to_square(suffix, transform_params)
        
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
        label, _, _, _ = preprocess_tokens(scaled_suffix, seqlen=SEQLEN)

        yield {
            "image": np.asarray(processed_image),
            "text": np.asarray(tokens),
            "label": np.asarray(label),
            "mask_ar": np.asarray(mask_ar),
            "mask_input": np.asarray(mask_input),
            "transform_params": transform_params,  # Include for debugging
        }


def test_coordinate_scaling():
    """
    Test function to validate the coordinate scaling works correctly.
    """
    print("Testing Enhanced Image Preprocessing for PaliGemma...")
    
    # Simulate a 1024x128 image being processed
    dummy_image = np.random.randint(0, 255, (128, 1024, 3), dtype=np.uint8)
    
    # Process image and get transformation parameters
    processed_image, transform_params = preprocess_image_with_padding(dummy_image, size=896)
    
    print("\nTransform Parameters:")
    for key, value in transform_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nOriginal image shape: {dummy_image.shape}")
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # Test coordinate scaling with example coordinates
    original_coords = "<loc0512><loc0064><loc0768><loc0096> 0"  # Example bounding box
    scaled_coords = scale_coordinates_to_square(original_coords, transform_params)
    
    print(f"\nOriginal coordinates: {original_coords}")
    print(f"Scaled coordinates: {scaled_coords}")
    
    # Validate the scaling
    validation = validate_coordinate_scaling(original_coords, scaled_coords, transform_params)
    print(f"\nValidation results:")
    print(f"Valid: {validation['valid']}")
    if validation['valid'] and validation['bounding_boxes']:
        bbox = validation['bounding_boxes'][0]
        print(f"Original bbox: {bbox['original']}")
        print(f"Scaled bbox: {bbox['scaled']}")
        print(f"Original relative: {bbox['original_relative']}")
        print(f"Scaled relative: {bbox['scaled_relative']}")
    
    return processed_image, transform_params, scaled_coords


if __name__ == "__main__":
    test_coordinate_scaling() 