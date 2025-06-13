import json
import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path
import cv2
import yaml
from typing import Dict, List, Tuple
import argparse

class AnnotationConverter:
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.converted_datasets = {}
        
    def convert_coco_to_yolo(self, coco_json_path: str, images_dir: str, output_dir: str, class_names: List[str]) -> str:
        """Convert COCO format annotations to YOLO format"""
        output_path = Path(output_dir)
        labels_dir = output_path / "labels"
        images_output_dir = output_path / "images"
        
        # Create directories
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        converted_count = 0
        
        # Convert each image and its annotations
        for image_id, image_info in images.items():
            # Copy image file
            src_image_path = Path(images_dir) / image_info['file_name']
            if src_image_path.exists():
                dst_image_path = images_output_dir / image_info['file_name']
                shutil.copy2(src_image_path, dst_image_path)
                
                # Create YOLO annotation file
                label_file = labels_dir / (Path(image_info['file_name']).stem + '.txt')
                
                with open(label_file, 'w') as f:
                    if image_id in annotations_by_image:
                        for ann in annotations_by_image[image_id]:
                            # Convert COCO bbox to YOLO format
                            x, y, w, h = ann['bbox']
                            img_w, img_h = image_info['width'], image_info['height']
                            
                            # YOLO format: class_id center_x center_y width height (normalized)
                            center_x = (x + w/2) / img_w
                            center_y = (y + h/2) / img_h
                            norm_w = w / img_w
                            norm_h = h / img_h
                            
                            class_id = ann.get('category_id', 0) - 1  # COCO is 1-indexed, YOLO is 0-indexed
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
                converted_count += 1
        
        print(f"Converted {converted_count} COCO images to YOLO format")
        return str(output_path)
    
    def convert_voc_to_yolo(self, voc_dir: str, output_dir: str, class_names: List[str]) -> str:
        """Convert Pascal VOC format annotations to YOLO format"""
        voc_path = Path(voc_dir)
        output_path = Path(output_dir)
        
        labels_dir = output_path / "labels"
        images_output_dir = output_path / "images"
        
        # Create directories
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for XML files in the VOC directory
        xml_files = list(voc_path.rglob("*.xml"))
        converted_count = 0
        
        for xml_file in xml_files:
            try:
                # Parse XML file
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image info
                filename = root.find('filename').text
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # Find corresponding image file
                possible_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_found = False
                
                for ext in possible_image_extensions:
                    image_name = Path(filename).stem + ext
                    src_image_paths = list(voc_path.rglob(image_name))
                    
                    if src_image_paths:
                        src_image_path = src_image_paths[0]
                        dst_image_path = images_output_dir / image_name
                        shutil.copy2(src_image_path, dst_image_path)
                        image_found = True
                        break
                
                if not image_found:
                    print(f"Warning: Image file not found for {filename}")
                    continue
                
                # Create YOLO annotation file
                label_file = labels_dir / (Path(filename).stem + '.txt')
                
                with open(label_file, 'w') as f:
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        class_id = 0  # Default to class 0 for pole detection
                        
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                        
                        # Convert to YOLO format
                        center_x = (xmin + xmax) / (2 * img_width)
                        center_y = (ymin + ymax) / (2 * img_height)
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
        
        print(f"Converted {converted_count} VOC images to YOLO format")
        return str(output_path)
    
    def create_yolo_dataset_yaml(self, train_path: str, val_path: str, test_path: str, class_names: List[str], output_file: str):
        """Create a YAML file for YOLO dataset configuration"""
        yaml_content = {
            'path': str(Path(output_file).parent.absolute()),
            'train': str(Path(train_path).relative_to(Path(output_file).parent)),
            'val': str(Path(val_path).relative_to(Path(output_file).parent)),
            'test': str(Path(test_path).relative_to(Path(output_file).parent)),
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created YOLO dataset YAML: {output_file}")
    
    def prepare_single_modality_datasets(self):
        """Prepare single modality datasets (COCO format) for YOLO"""
        base_dataset_path = self.base_dir / "main_images" / "SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions" / "SnowPole_Detection_Dataset"
        
        modalities = ['signal', 'reflec', 'range', 'nearir', 'combined_color']
        class_names = ['pole']
        
        for modality in modalities:
            print(f"\n=== Converting {modality} dataset ===")
            
            # Create output directory
            output_dir = self.base_dir / f"converted_datasets" / f"{modality}_yolo"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert train, val, test splits
            for split in ['train', 'valid', 'test']:
                split_name = 'val' if split == 'valid' else split
                
                # Check for COCO annotation files
                coco_json = base_dataset_path / "labels" / split / f"annotations.json"
                images_dir = base_dataset_path / modality / split
                output_split_dir = output_dir / split_name
                
                if coco_json.exists() and images_dir.exists():
                    self.convert_coco_to_yolo(str(coco_json), str(images_dir), str(output_split_dir), class_names)
                else:
                    print(f"Missing COCO annotations or images for {modality}/{split}")
            
            # Create YAML file
            yaml_file = output_dir / "dataset.yaml"
            self.create_yolo_dataset_yaml(
                str(output_dir / "train" / "images"),
                str(output_dir / "val" / "images"), 
                str(output_dir / "test" / "images"),
                class_names,
                str(yaml_file)
            )
            
            self.converted_datasets[modality] = str(yaml_file)
    
    def prepare_permutation_datasets(self):
        """Prepare permutation datasets (Pascal VOC format) for YOLO"""
        permutations = ['Permutation1', 'Permutation3', 'Permutation4', 'Permutation5', 'Permutation6']
        class_names = ['pole']
        
        for perm in permutations:
            print(f"\n=== Converting {perm} dataset ===")
            
            perm_dir = self.base_dir / perm
            if not perm_dir.exists():
                print(f"Permutation directory {perm} not found")
                continue
            
            # Create output directory
            output_dir = self.base_dir / f"converted_datasets" / f"{perm.lower()}_yolo"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert train, val, test splits
            for split in ['train', 'valid', 'test']:
                split_name = 'val' if split == 'valid' else split
                split_dir = perm_dir / split
                output_split_dir = output_dir / split_name
                
                if split_dir.exists():
                    self.convert_voc_to_yolo(str(split_dir), str(output_split_dir), class_names)
                else:
                    print(f"Missing {split} directory for {perm}")
            
            # Create YAML file
            yaml_file = output_dir / "dataset.yaml"
            self.create_yolo_dataset_yaml(
                str(output_dir / "train" / "images"),
                str(output_dir / "val" / "images"),
                str(output_dir / "test" / "images"), 
                class_names,
                str(yaml_file)
            )
            
            self.converted_datasets[perm.lower()] = str(yaml_file)
    
    def convert_all_datasets(self):
        """Convert all datasets to YOLO format"""
        print("Starting dataset conversion...")
        
        # Convert single modality datasets
        try:
            self.prepare_single_modality_datasets()
        except Exception as e:
            print(f"Error converting single modality datasets: {e}")
        
        # Convert permutation datasets  
        try:
            self.prepare_permutation_datasets()
        except Exception as e:
            print(f"Error converting permutation datasets: {e}")
        
        print(f"\n=== Conversion Summary ===")
        for dataset_name, yaml_path in self.converted_datasets.items():
            print(f"{dataset_name}: {yaml_path}")
        
        return self.converted_datasets

def main():
    parser = argparse.ArgumentParser(description='Convert annotations to YOLO format')
    parser.add_argument('--base-dir', default='.', help='Base directory')
    parser.add_argument('--convert-only', action='store_true', help='Only convert, do not test')
    
    args = parser.parse_args()
    
    converter = AnnotationConverter(args.base_dir)
    converted_datasets = converter.convert_all_datasets()
    
    if not args.convert_only:
        # Import and run model testing
        from comprehensive_model_testing import ModelTester
        
        # Update model tester to use converted datasets
        tester = ModelTester(args.base_dir)
        
        # Override yaml mapping with converted datasets
        tester.yaml_mapping.update(converted_datasets)
        
        print(f"\n=== Starting Model Testing ===")
        tester.run_all_tests()
        tester.save_results('test_results_with_converted_annotations')

if __name__ == "__main__":
    main() 