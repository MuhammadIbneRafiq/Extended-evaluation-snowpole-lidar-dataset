"""
Utility functions for training and evaluation
Includes logging, checkpointing, metrics computation, and visualization
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
from collections import defaultdict


def setup_logging(log_file: Path) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('SnowPoleDetection')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Validation loss
        
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict, save_path: Path,
                   config: Dict = None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        save_path: Path to save checkpoint
        config: Configuration dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    # Also save metrics as JSON for easy reading
    metrics_path = save_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        }, f, indent=2)


def load_checkpoint(model: nn.Module, checkpoint_path: Path,
                   optimizer: torch.optim.Optimizer = None,
                   strict: bool = True) -> Dict:
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state
        strict: Whether to strictly match state dict keys
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }


def compute_metrics(predictions: List[torch.Tensor], 
                   targets: List[Dict]) -> Dict[str, float]:
    """
    Compute detection metrics
    
    Args:
        predictions: List of prediction tensors
        targets: List of target dictionaries
    
    Returns:
        Dictionary of metrics
    """
    # Initialize metric accumulators
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    all_scores = []
    all_labels = []
    ious = []
    
    # IoU threshold for matching
    iou_threshold = 0.5
    
    for pred, target in zip(predictions, targets):
        if pred is None or len(pred) == 0:
            # No predictions
            fn += len(target['boxes'])
            continue
        
        if len(target['boxes']) == 0:
            # No ground truth
            fp += len(pred)
            continue
        
        # Match predictions to targets
        matched = match_predictions(pred, target['boxes'], iou_threshold)
        
        tp += matched['tp']
        fp += matched['fp']
        fn += matched['fn']
        ious.extend(matched['ious'])
    
    # Compute metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Compute mAP (simplified version)
    map50 = compute_map(predictions, targets, iou_threshold=0.5)
    map50_95 = compute_map_range(predictions, targets)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
        'map50_95': map50_95,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'mean_iou': np.mean(ious) if ious else 0.0
    }


def match_predictions(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                     iou_threshold: float = 0.5) -> Dict:
    """
    Match predictions to targets based on IoU
    
    Args:
        pred_boxes: Predicted boxes
        target_boxes: Target boxes
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with matching results
    """
    # Simplified matching - actual implementation would use Hungarian algorithm
    tp = 0
    fp = 0
    fn = 0
    ious = []
    
    matched_targets = set()
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_target = None
        
        for i, target_box in enumerate(target_boxes):
            if i in matched_targets:
                continue
            
            iou = compute_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_target = i
        
        if best_iou >= iou_threshold and best_target is not None:
            tp += 1
            matched_targets.add(best_target)
            ious.append(best_iou)
        else:
            fp += 1
    
    fn = len(target_boxes) - len(matched_targets)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'ious': ious
    }


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Compute IoU between two boxes
    
    Args:
        box1: First box [4]
        box2: Second box [4]
    
    Returns:
        IoU value
    """
    # Convert to numpy for easier computation
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def compute_map(predictions: List, targets: List, iou_threshold: float = 0.5) -> float:
    """
    Compute mean Average Precision at specific IoU threshold
    
    Args:
        predictions: List of predictions
        targets: List of targets
        iou_threshold: IoU threshold
    
    Returns:
        mAP value
    """
    # Simplified mAP computation
    # Actual implementation would compute AP for each class
    
    all_precisions = []
    all_recalls = []
    
    for pred, target in zip(predictions, targets):
        if pred is None or len(target['boxes']) == 0:
            continue
        
        matched = match_predictions(pred, target['boxes'], iou_threshold)
        
        if matched['tp'] > 0:
            precision = matched['tp'] / (matched['tp'] + matched['fp'] + 1e-6)
            recall = matched['tp'] / (matched['tp'] + matched['fn'] + 1e-6)
            all_precisions.append(precision)
            all_recalls.append(recall)
    
    if not all_precisions:
        return 0.0
    
    # Compute AP (simplified - actual would use interpolation)
    ap = np.mean(all_precisions)
    
    return ap


def compute_map_range(predictions: List, targets: List, 
                     iou_range: Tuple[float, float, float] = (0.5, 0.95, 0.05)) -> float:
    """
    Compute mAP across IoU range
    
    Args:
        predictions: List of predictions
        targets: List of targets
        iou_range: (start, stop, step) for IoU thresholds
    
    Returns:
        mAP@50:95 value
    """
    iou_thresholds = np.arange(iou_range[0], iou_range[1] + iou_range[2], iou_range[2])
    maps = []
    
    for iou_thresh in iou_thresholds:
        map_val = compute_map(predictions, targets, iou_thresh)
        maps.append(map_val)
    
    return np.mean(maps)


def visualize_predictions(image: np.ndarray, predictions: List[Dict],
                         targets: List[Dict] = None, 
                         confidence_threshold: float = 0.25,
                         save_path: Optional[Path] = None):
    """
    Visualize detection predictions on image
    
    Args:
        image: Input image (RGB or RGBA)
        predictions: List of prediction dictionaries
        targets: Optional list of target dictionaries
        confidence_threshold: Confidence threshold for displaying predictions
        save_path: Optional path to save visualization
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image (handle RGBA)
    if image.shape[-1] == 4:
        # Show RGB channels only
        ax.imshow(image[..., :3])
    else:
        ax.imshow(image)
    
    # Draw predictions
    for pred in predictions:
        if 'confidence' in pred and pred['confidence'] < confidence_threshold:
            continue
        
        box = pred['box']
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"Pole: {pred.get('confidence', 1.0):.2f}"
        ax.text(x1, y1-5, label, color='red', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw ground truth if provided
    if targets:
        for target in targets:
            box = target['box']
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='green', facecolor='none',
                                linestyle='--')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x2, y1-5, "GT", color='green', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis('off')
    ax.set_title('Snow Pole Detection Results')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Predictions'),
        Patch(facecolor='none', edgecolor='green', linestyle='--', label='Ground Truth')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_training_curves(log_dir: Path, save_path: Optional[Path] = None):
    """
    Plot training curves from logs
    
    Args:
        log_dir: Directory containing training logs
        save_path: Optional path to save plot
    """
    # Read metrics from JSON files
    metrics_files = sorted(log_dir.glob('*.json'))
    
    epochs = []
    train_losses = []
    val_losses = []
    maps50 = []
    maps50_95 = []
    
    for file in metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            epochs.append(data['epoch'])
            
            metrics = data['metrics']
            val_losses.append(metrics.get('loss', 0))
            maps50.append(metrics.get('map50', 0))
            maps50_95.append(metrics.get('map50_95', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, val_losses, 'b-', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mAP@50 plot
    axes[0, 1].plot(epochs, maps50, 'g-', label='mAP@50')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('mAP@50')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP@50-95 plot
    axes[1, 0].plot(epochs, maps50_95, 'r-', label='mAP@50-95')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('mAP@50-95')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics plot
    axes[1, 1].plot(epochs, maps50, 'g-', label='mAP@50')
    axes[1, 1].plot(epochs, maps50_95, 'r-', label='mAP@50-95')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Detection Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig


def create_confusion_matrix(predictions: List, targets: List,
                           num_classes: int = 1,
                           save_path: Optional[Path] = None):
    """
    Create confusion matrix for detection results
    
    Args:
        predictions: List of predictions
        targets: List of targets
        num_classes: Number of classes
        save_path: Optional path to save plot
    """
    # For single-class detection, create binary confusion matrix
    # (detected vs not detected)
    
    tp = fp = fn = tn = 0
    
    for pred, target in zip(predictions, targets):
        if pred is None:
            pred_boxes = []
        else:
            pred_boxes = pred
        
        target_boxes = target.get('boxes', [])
        
        # Match predictions to targets
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            matched = match_predictions(pred_boxes, target_boxes)
            tp += matched['tp']
            fp += matched['fp']
            fn += matched['fn']
        elif len(pred_boxes) > 0:
            fp += len(pred_boxes)
        elif len(target_boxes) > 0:
            fn += len(target_boxes)
        else:
            tn += 1
    
    # Create confusion matrix
    cm = np.array([[tp, fp], [fn, tn]])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Detected', 'Not Detected'],
               yticklabels=['True Pole', 'No Pole'])
    
    plt.title('Detection Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return cm


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    losses = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83]
    for i, loss in enumerate(losses):
        if early_stop(loss):
            print(f"Early stopping triggered at epoch {i}")
            break
    
    # Test metrics computation
    predictions = [torch.randn(5, 4) for _ in range(2)]  # Dummy predictions
    targets = [
        {'boxes': torch.randn(3, 4), 'labels': torch.zeros(3)},
        {'boxes': torch.randn(2, 4), 'labels': torch.zeros(2)}
    ]
    
    metrics = compute_metrics(predictions, targets)
    print(f"Sample metrics: {metrics}")
    
    print("Utility functions test completed!")
