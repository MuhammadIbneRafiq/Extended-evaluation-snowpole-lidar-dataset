import cv2
import numpy as np
import os

def read_yolo_labels(label_path):
    """Read YOLO format labels and return list of [class, x, y, w, h]"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_id, x, y, w, h])
    return boxes

def draw_boxes(image, boxes):
    """Draw bounding boxes on image"""
    h, w = image.shape[:2]
    for box in boxes:
        class_id, x, y, width, height = box
        
        # Convert YOLO format to pixel coordinates
        x1 = int((x - width/2) * w)
        y1 = int((y - height/2) * h)
        x2 = int((x + width/2) * w)
        y2 = int((y + height/2) * h)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text 'pole' above the box
        text = 'pole'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Position text above the box
        text_x = x1
        text_y = y1 - 5  # 5 pixels above the box
        
        # Draw white background for text for better visibility
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (255, 255, 255), -1)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    
    return image

def process_modality(image_path, label_path, output_path):
    """Process a single modality/combination"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Read labels
    boxes = read_yolo_labels(label_path)
    
    # Draw boxes
    image_with_boxes = draw_boxes(image.copy(), boxes)
    
    # Save result
    cv2.imwrite(output_path, image_with_boxes)
    print(f"Saved ground truth visualization to: {output_path}")

def main():
    base_path = "C:/Users/x1 yoga/Documents/RA_5m_5L_6m_6L_7m_8m_9m_10m_11m"
    
    # Single modalities
    modalities = {
        'signal': f"{base_path}/data/single_modality/SnowPole_Detection_Dataset/signal/train/image_1967.png",
        'reflec': f"{base_path}/data/single_modality/SnowPole_Detection_Dataset/reflec/train/image_1967.png",
        'nearir': f"{base_path}/data/single_modality/SnowPole_Detection_Dataset/nearir/train/image_1967.png",
        'range': f"{base_path}/data/single_modality/SnowPole_Detection_Dataset/range/train/image_1967.png",
        'combined_color': f"{base_path}/data/single_modality/SnowPole_Detection_Dataset/combined_color/train/image_1967.png"
    }
    
    # Combinations
    combinations = {
        'combination1': f"{base_path}/data/Combinations_yolo_format/Combination1/train/image_1967.png",
        'combination3': f"{base_path}/data/Combinations_yolo_format/Combination3/train/image_1967.png",
        'combination4': f"{base_path}/data/Combinations_yolo_format/Combination4/train/image_1967.png",
        'combination5': f"{base_path}/data/Combinations_yolo_format/Combination5/train/image_1967.png",
        'combination6': f"{base_path}/data/Combinations_yolo_format/Combination6/train/image_1967.png"
    }
    
    label_path = f"{base_path}/data/labels/train/image_1967.txt"

    # Process single modalities
    for name, image_path in modalities.items():
        output_path = f"{base_path}/results/ground_truth_{name}_1967.png"
        process_modality(image_path, label_path, output_path)
    
    # Process combinations
    for name, image_path in combinations.items():
        output_path = f"{base_path}/results/ground_truth_{name}_1967.png"
        process_modality(image_path, label_path, output_path)

if __name__ == "__main__":
    main() 