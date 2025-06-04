import os
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow_hub as hub

# Configuration
IMG_HEIGHT = 640
IMG_WIDTH = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def parse_xml_to_dict(xml_path):
    """Parse Pascal VOC XML annotation to dictionary."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    result = {}
    result['filename'] = root.find('filename').text
    result['width'] = int(root.find('size/width').text)
    result['height'] = int(root.find('size/height').text)
    
    objects = []
    for obj in root.findall('object'):
        if obj.find('name').text == 'pole':
            obj_dict = {}
            obj_dict['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_dict['xmin'] = int(bbox.find('xmin').text)
            obj_dict['ymin'] = int(bbox.find('ymin').text)
            obj_dict['xmax'] = int(bbox.find('xmax').text)
            obj_dict['ymax'] = int(bbox.find('ymax').text)
            objects.append(obj_dict)
    
    result['objects'] = objects
    return result

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # box format: [ymin, xmin, ymax, xmax] (normalized)
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Calculate Average Precision for a single class."""
    if len(pred_boxes) == 0:
        return 0.0 if len(gt_boxes) > 0 else 1.0
    
    if len(gt_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Track which ground truth boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if prediction is a true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

def evaluate_model():
    """Evaluate the pre-trained Faster R-CNN model on test data."""
    print("Loading pre-trained Faster R-CNN ResNet50 model...")
    detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")
    print("Model loaded successfully!")
    
    # Get test images and annotations
    test_dir = 'Permutation1/test'
    image_files = glob.glob(os.path.join(test_dir, '*.jpg'))
    
    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    
    total_images = 0
    total_gt_boxes = 0
    total_predictions = 0
    
    print(f"Evaluating on {len(image_files)} test images...")
    
    for i, image_path in enumerate(image_files):
        xml_path = image_path.replace('.jpg', '.xml')
        
        if not os.path.exists(xml_path):
            continue
        
        # Load and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        original_shape = tf.shape(image)
        image_resized = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image_np = tf.cast(image_resized, tf.uint8).numpy()
        
        # Run inference
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = detector(input_tensor)
        
        # Extract predictions
        pred_boxes = detections['detection_boxes'][0].numpy()
        pred_scores = detections['detection_scores'][0].numpy()
        pred_classes = detections['detection_classes'][0].numpy().astype(int)
        
        # Filter predictions by confidence and class (assuming class 1 is closest to 'pole')
        valid_detections = (pred_scores >= CONFIDENCE_THRESHOLD)
        pred_boxes = pred_boxes[valid_detections]
        pred_scores = pred_scores[valid_detections]
        pred_classes = pred_classes[valid_detections]
        
        # Parse ground truth
        annotation = parse_xml_to_dict(xml_path)
        gt_boxes = []
        
        for obj in annotation['objects']:
            # Convert to normalized coordinates
            xmin = obj['xmin'] / annotation['width']
            ymin = obj['ymin'] / annotation['height']
            xmax = obj['xmax'] / annotation['width']
            ymax = obj['ymax'] / annotation['height']
            gt_boxes.append([ymin, xmin, ymax, xmax])
        
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        
        all_gt_boxes.extend(gt_boxes)
        all_pred_boxes.extend(pred_boxes)
        all_pred_scores.extend(pred_scores)
        
        total_images += 1
        total_gt_boxes += len(gt_boxes)
        total_predictions += len(pred_boxes)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    print(f"\nEvaluation Summary:")
    print(f"Total images: {total_images}")
    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total predictions (confidence > {CONFIDENCE_THRESHOLD}): {total_predictions}")
    print(f"Average GT boxes per image: {total_gt_boxes / total_images:.2f}")
    print(f"Average predictions per image: {total_predictions / total_images:.2f}")
    
    # Calculate mAP at different IoU thresholds
    if total_gt_boxes > 0 and total_predictions > 0:
        all_gt_boxes = np.array(all_gt_boxes)
        all_pred_boxes = np.array(all_pred_boxes)
        all_pred_scores = np.array(all_pred_scores)
        
        # Calculate mAP@0.5
        ap_50 = calculate_ap(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_threshold=0.5)
        
        # Calculate mAP@0.75
        ap_75 = calculate_ap(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_threshold=0.75)
        
        # Calculate mAP@0.5:0.95 (average over IoU thresholds from 0.5 to 0.95)
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []
        for iou_thresh in iou_thresholds:
            ap = calculate_ap(all_gt_boxes, all_pred_boxes, all_pred_scores, iou_threshold=iou_thresh)
            aps.append(ap)
        
        map_50_95 = np.mean(aps)
        
        print(f"\n=== OBJECT DETECTION METRICS ===")
        print(f"mAP@0.5:     {ap_50:.4f}")
        print(f"mAP@0.75:    {ap_75:.4f}")
        print(f"mAP@0.5:0.95: {map_50_95:.4f}")
        
        # Calculate precision and recall at IoU=0.5
        gt_boxes_flat = all_gt_boxes
        pred_boxes_flat = all_pred_boxes
        pred_scores_flat = all_pred_scores
        
        if len(pred_boxes_flat) > 0:
            # Sort by confidence
            sorted_indices = np.argsort(pred_scores_flat)[::-1]
            pred_boxes_sorted = pred_boxes_flat[sorted_indices]
            pred_scores_sorted = pred_scores_flat[sorted_indices]
            
            # Calculate precision and recall
            gt_matched = np.zeros(len(gt_boxes_flat), dtype=bool)
            tp = 0
            fp = 0
            
            for pred_box in pred_boxes_sorted:
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes_flat):
                    if gt_matched[j]:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= 0.5 and best_gt_idx != -1:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(gt_boxes_flat) if len(gt_boxes_flat) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            print(f"\nAdditional Metrics (IoU=0.5):")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1-Score:    {f1_score:.4f}")
            print(f"True Positives:  {tp}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {len(gt_boxes_flat) - tp}")
        
    else:
        print("No valid predictions or ground truth boxes found for evaluation.")
    
    print(f"\n=== MODEL INFORMATION ===")
    print(f"Model: Faster R-CNN ResNet50 (Pre-trained)")
    print(f"Input size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

if __name__ == "__main__":
    print("=== OBJECT DETECTION EVALUATION ===")
    print("Using pre-trained Faster R-CNN ResNet50 from TensorFlow Hub")
    print("Dataset: Pole Detection (Pascal VOC format)")
    print()
    
    evaluate_model() 