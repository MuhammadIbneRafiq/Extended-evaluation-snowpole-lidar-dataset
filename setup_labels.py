#!/usr/bin/env python3
"""
Script to copy centralized labels to modality-specific directories for YOLO training/testing.
The dataset has centralized labels but YOLO expects labels alongside images.
"""

import shutil
from pathlib import Path
import os

def setup_labels_for_yolo():
    # Base paths
    base_path = Path("main_images/SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions/SnowPole_Detection_Dataset")
    labels_source = base_path / "labels"
    
    # Available modalities
    modalities = ['combined_color', 'nearir', 'range', 'reflec', 'signal']
    splits = ['train', 'valid', 'test']
    
    print("Setting up labels for YOLO dataset structure...")
    
    for modality in modalities:
        modality_path = base_path / modality
        
        if not modality_path.exists():
            print(f"Skipping {modality} (directory not found)")
            continue
            
        print(f"\nProcessing {modality}...")
        
        for split in splits:
            # Source labels directory
            source_labels_dir = labels_source / split
            
            # Target labels directory (alongside images)
            target_labels_dir = modality_path / split / "labels" 
            images_dir = modality_path / split
            
            if not source_labels_dir.exists():
                print(f"  Warning: {source_labels_dir} not found")
                continue
                
            if not images_dir.exists():
                print(f"  Warning: {images_dir} not found")
                continue
            
            # Create labels directory
            target_labels_dir.mkdir(exist_ok=True)
            
            # Copy label files
            label_files = list(source_labels_dir.glob("*.txt"))
            copied_count = 0
            
            for label_file in label_files:
                target_file = target_labels_dir / label_file.name
                
                # Check if corresponding image exists
                image_name = label_file.stem
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_found = False
                
                for ext in image_extensions:
                    image_file = images_dir / f"{image_name}{ext}"
                    if image_file.exists():
                        image_found = True
                        break
                
                if image_found:
                    shutil.copy2(label_file, target_file)
                    copied_count += 1
            
            print(f"  {split}: copied {copied_count} label files to {target_labels_dir}")
    
    print("\nLabel setup complete!")

if __name__ == "__main__":
    setup_labels_for_yolo() 