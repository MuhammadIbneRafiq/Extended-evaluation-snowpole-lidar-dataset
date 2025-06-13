#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import os
from pathlib import Path

def convert_xml_to_yolo(xml_file, output_dir):
    """Convert a single XML file to YOLO format"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            print(f"No size info in {xml_file}")
            return False
            
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Get filename without extension
        filename = root.find('filename').text
        output_file = output_dir / (Path(filename).stem + '.txt')
        
        # Convert all objects
        with open(output_file, 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                class_id = 0  # pole = class 0
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                center_x = (xmin + xmax) / (2 * img_width)
                center_y = (ymin + ymax) / (2 * img_height)
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"Error converting {xml_file}: {e}")
        return False

def convert_directory(xml_dir):
    """Convert all XML files in a directory"""
    xml_dir = Path(xml_dir)
    xml_files = list(xml_dir.glob("*.xml"))
    
    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    
    converted = 0
    for xml_file in xml_files:
        if convert_xml_to_yolo(xml_file, xml_dir):
            converted += 1
    
    print(f"Converted {converted}/{len(xml_files)} files")

if __name__ == "__main__":
    # Convert Permutation1/valid XML files to YOLO format
    convert_directory("Permutation1/valid")
    print("Conversion complete!") 