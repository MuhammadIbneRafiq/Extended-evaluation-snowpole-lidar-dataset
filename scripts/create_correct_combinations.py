#!/usr/bin/env python3
"""
Script to create correct combinations of RGB with different LiDAR modalities
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

def main():
    print("Creating CORRECT combination datasets...")
    
    # Define the source directory
    source_dir = "main_images/SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions/SnowPole_Detection_Dataset"
    
    # Define combinations: combination_number -> modality_name
    # Based on available modalities: nearir, range, reflec, signal
    combinations = {
        1: 'nearir',    # Combination 1: RGB + Near-IR
        3: 'range',     # Combination 3: RGB + Range
        4: 'reflec',    # Combination 4: RGB + Reflectivity  
        5: 'signal',    # Combination 5: RGB + Signal
        6: 'combined_all'  # Combination 6: RGB + all modalities combined
    }
    
    # Create each combination
    for combo_num, modality in combinations.items():
        create_combination_dataset(combo_num, modality, source_dir, "RGB_Combinations")
    
    print("All CORRECT combinations created successfully!")
    
    # Print summary
    print("\nSummary of created combinations:")
    for combo_num, modality in combinations.items():
        print(f"Combination {combo_num}: RGB + {modality}")

def create_combination_dataset(combination_num, modality_name, source_dir, output_dir):
    """Create a combination dataset by merging RGB with another modality"""
    
    print(f"\nCreating Combination {combination_num} - RGB + {modality_name}")
    
    # Define paths
    rgb_path = os.path.join(source_dir, "combined_color")
    labels_path = os.path.join(source_dir, "labels")
    
    # Create output directory
    output_path = os.path.join(output_dir, f"Combination{combination_num}")
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} set...")
        if modality_name == 'combined_all':
            process_split_all_modalities(rgb_path, source_dir, labels_path, output_path, split, combination_num)
        else:
            modality_path = os.path.join(source_dir, modality_name)
            process_split(rgb_path, modality_path, labels_path, output_path, split, combination_num, modality_name)
    
    # Create simple README
    create_simple_readme(output_path, combination_num, modality_name)
    print(f"Combination {combination_num} completed!")

def process_split(rgb_path, modality_path, labels_path, output_path, split, combination_num, modality_name):
    """Process a specific split (train/valid/test) for single modality"""
    
    rgb_split_path = os.path.join(rgb_path, split)
    modality_split_path = os.path.join(modality_path, split)
    labels_split_path = os.path.join(labels_path, split)
    output_split_path = os.path.join(output_path, split)
    
    if not os.path.exists(rgb_split_path) or not os.path.exists(modality_split_path):
        print(f"Warning: Missing source directories for {split}")
        return
    
    # Get list of RGB images
    rgb_images = [f for f in os.listdir(rgb_split_path) if f.endswith('.png')]
    processed_count = 0
    
    for rgb_img in rgb_images:
        try:
            # Load RGB image
            rgb_img_path = os.path.join(rgb_split_path, rgb_img)
            rgb_image = cv2.imread(rgb_img_path)
            
            if rgb_image is None:
                continue
            
            # Load corresponding modality image
            modality_img_path = os.path.join(modality_split_path, rgb_img)
            
            if os.path.exists(modality_img_path):
                modality_image = cv2.imread(modality_img_path, cv2.IMREAD_GRAYSCALE)
                
                if modality_image is None:
                    continue
                
                # Convert modality to 3-channel
                modality_3ch = cv2.cvtColor(modality_image, cv2.COLOR_GRAY2BGR)
                
                # Combine images horizontally
                combined_image = np.hstack((rgb_image, modality_3ch))
                
                # Keep original filename, just change extension
                output_filename = rgb_img.replace('.png', '.jpg')
                output_img_path = os.path.join(output_split_path, output_filename)
                
                # Save combined image
                cv2.imwrite(output_img_path, combined_image)
                
                # Copy label file with same basename
                label_file = rgb_img.replace('.png', '.txt')
                label_src_path = os.path.join(labels_split_path, label_file)
                
                if os.path.exists(label_src_path):
                    label_output_path = os.path.join(output_split_path, label_file)
                    shutil.copy2(label_src_path, label_output_path)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {rgb_img}: {str(e)}")
            continue
    
    print(f"Processed {processed_count} images for {split}")

def process_split_all_modalities(rgb_path, source_dir, labels_path, output_path, split, combination_num):
    """Process a split for combination 6 (all modalities)"""
    
    rgb_split_path = os.path.join(rgb_path, split)
    labels_split_path = os.path.join(labels_path, split)
    output_split_path = os.path.join(output_path, split)
    
    # Define all modality paths
    modalities = ['nearir', 'range', 'reflec', 'signal']
    modality_paths = {mod: os.path.join(source_dir, mod, split) for mod in modalities}
    
    # Check if all paths exist
    if not os.path.exists(rgb_split_path) or not all(os.path.exists(path) for path in modality_paths.values()):
        print(f"Warning: Missing source directories for {split}")
        return
    
    # Get list of RGB images
    rgb_images = [f for f in os.listdir(rgb_split_path) if f.endswith('.png')]
    processed_count = 0
    
    for rgb_img in rgb_images:
        try:
            # Load RGB image
            rgb_img_path = os.path.join(rgb_split_path, rgb_img)
            rgb_image = cv2.imread(rgb_img_path)
            
            if rgb_image is None:
                continue
            
            # Load all modality images
            modality_images = []
            for mod in modalities:
                mod_img_path = os.path.join(modality_paths[mod], rgb_img)
                if os.path.exists(mod_img_path):
                    mod_image = cv2.imread(mod_img_path, cv2.IMREAD_GRAYSCALE)
                    if mod_image is not None:
                        mod_3ch = cv2.cvtColor(mod_image, cv2.COLOR_GRAY2BGR)
                        modality_images.append(mod_3ch)
            
            if len(modality_images) == 4:  # All modalities loaded successfully
                # Combine RGB with all modalities horizontally
                all_images = [rgb_image] + modality_images
                combined_image = np.hstack(all_images)
                
                # Keep original filename, just change extension
                output_filename = rgb_img.replace('.png', '.jpg')
                output_img_path = os.path.join(output_split_path, output_filename)
                
                # Save combined image
                cv2.imwrite(output_img_path, combined_image)
                
                # Copy label file with same basename
                label_file = rgb_img.replace('.png', '.txt')
                label_src_path = os.path.join(labels_split_path, label_file)
                
                if os.path.exists(label_src_path):
                    label_output_path = os.path.join(output_split_path, label_file)
                    shutil.copy2(label_src_path, label_output_path)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {rgb_img}: {str(e)}")
            continue
    
    print(f"Processed {processed_count} images for {split}")

def create_simple_readme(output_path, combination_num, modality_name):
    """Create a simple README file"""
    
    if modality_name == 'combined_all':
        modality_description = "All modalities (Near-IR + Range + Reflectivity + Signal)"
        combination_description = "RGB + Near-IR + Range + Reflectivity + Signal horizontally concatenated"
    else:
        modality_descriptions = {
            'nearir': 'Near-infrared',
            'range': 'Range/distance data',
            'reflec': 'Reflectivity data',
            'signal': 'Signal intensity'
        }
        modality_description = modality_descriptions.get(modality_name, modality_name)
        combination_description = f"RGB and {modality_name} horizontally concatenated"
    
    readme_content = f"""# Combination {combination_num} - RGB + {modality_name}

This dataset combines RGB (combined_color) with {modality_name} modality from the original SnowPole Detection Dataset.

Dataset structure:
- train/: Training images and labels
- valid/: Validation images and labels  
- test/: Test images and labels

Image format: {combination_description}
Label format: YOLO format (.txt files)
Target class: Pole

Modalities:
- RGB: Combined color images (Near-IR + Signal + Reflectivity as RGB)
- {modality_name}: {modality_description}
"""
    
    with open(os.path.join(output_path, 'README.txt'), 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()