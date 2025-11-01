"""
Data loader for RGBA inputs
Handles loading and preprocessing of 4-channel data (RGB + Additional modality)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import logging


class SnowPoleRGBADataset(Dataset):
    """
    Dataset for loading RGBA images for snow pole detection
    RGB channels: Pseudo-color combinations from LiDAR
    Alpha channel: Additional modality (Range/Near-IR/Signal)
    """
    
    def __init__(self, 
                 root_path: Path,
                 split: str = 'train',
                 rgb_modality: str = 'Combination3',
                 alpha_modality: str = 'range',
                 img_size: int = 640,
                 augment: bool = True,
                 cache_images: bool = False):
        """
        Initialize dataset
        
        Args:
            root_path: Path to SnowPole_Detection_Dataset
            split: Dataset split ('train', 'valid', 'test')
            rgb_modality: Modality for RGB channels
            alpha_modality: Modality for alpha channel
            img_size: Target image size
            augment: Whether to apply data augmentation
            cache_images: Whether to cache images in memory
        """
        self.root_path = Path(root_path)
        self.split = split
        self.rgb_modality = rgb_modality
        self.alpha_modality = alpha_modality
        self.img_size = img_size
        self.augment = augment
        self.cache_images = cache_images
        
        # Setup paths
        self.rgb_path = self.root_path / rgb_modality / split
        self.alpha_path = self.root_path / alpha_modality / split
        self.label_path = self.root_path / 'labels' / split
        
        # Verify paths exist
        self._verify_paths()
        
        # Get image list
        self.image_files = self._get_image_files()
        
        # Setup augmentation
        self.transform = self._setup_augmentation()
        
        # Cache for images
        self.image_cache = {} if cache_images else None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Dataset initialized: {len(self.image_files)} images")
        self.logger.info(f"RGB modality: {rgb_modality}, Alpha modality: {alpha_modality}")
    
    def _verify_paths(self):
        """Verify that all required paths exist"""
        if not self.rgb_path.exists():
            raise ValueError(f"RGB path does not exist: {self.rgb_path}")
        if not self.alpha_path.exists():
            raise ValueError(f"Alpha path does not exist: {self.alpha_path}")
        if not self.label_path.exists():
            raise ValueError(f"Label path does not exist: {self.label_path}")
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files"""
        # Get all image files from RGB directory
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(self.rgb_path.glob(f'*{ext}'))
        
        # Sort for consistency
        image_files = sorted([f.stem for f in image_files])
        
        # Verify corresponding files exist
        valid_files = []
        for img_name in image_files:
            # Check if alpha image exists
            alpha_file = None
            for ext in extensions:
                alpha_path = self.alpha_path / f"{img_name}{ext}"
                if alpha_path.exists():
                    alpha_file = alpha_path
                    break
            
            # Check if label exists
            label_file = self.label_path / f"{img_name}.txt"
            
            if alpha_file and label_file.exists():
                valid_files.append(img_name)
        
        return valid_files
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup data augmentation pipeline"""
        if self.augment and self.split == 'train':
            transform = A.Compose([
                # Geometric augmentations
                A.RandomResizedCrop(
                    height=self.img_size,
                    width=self.img_size,
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                
                # Color augmentations (applied to RGB only)
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.3),
                
                # Resize to target size
                A.Resize(self.img_size, self.img_size),
                
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.5],  # RGB + Alpha means
                    std=[0.229, 0.224, 0.225, 0.25],   # RGB + Alpha stds
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.5],
                    std=[0.229, 0.224, 0.225, 0.25],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transform
    
    def load_image(self, idx: int) -> np.ndarray:
        """
        Load RGBA image
        
        Args:
            idx: Image index
        
        Returns:
            RGBA image as numpy array [H, W, 4]
        """
        img_name = self.image_files[idx]
        
        # Check cache
        if self.image_cache is not None and img_name in self.image_cache:
            return self.image_cache[img_name]
        
        # Load RGB image
        rgb_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            path = self.rgb_path / f"{img_name}{ext}"
            if path.exists():
                rgb_file = path
                break
        
        if rgb_file is None:
            raise FileNotFoundError(f"RGB image not found: {img_name}")
        
        rgb_img = cv2.imread(str(rgb_file))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load alpha channel image
        alpha_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            path = self.alpha_path / f"{img_name}{ext}"
            if path.exists():
                alpha_file = path
                break
        
        if alpha_file is None:
            raise FileNotFoundError(f"Alpha image not found: {img_name}")
        
        alpha_img = cv2.imread(str(alpha_file), cv2.IMREAD_GRAYSCALE)
        
        # Ensure same size
        if rgb_img.shape[:2] != alpha_img.shape[:2]:
            alpha_img = cv2.resize(alpha_img, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # Combine into RGBA
        rgba_img = np.dstack([rgb_img, alpha_img])
        
        # Cache if enabled
        if self.image_cache is not None:
            self.image_cache[img_name] = rgba_img
        
        return rgba_img
    
    def load_labels(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load YOLO format labels
        
        Args:
            idx: Image index
        
        Returns:
            Tuple of (bboxes, class_labels)
            bboxes: [N, 4] in YOLO format (x_center, y_center, width, height)
            class_labels: [N] class indices
        """
        img_name = self.image_files[idx]
        label_file = self.label_path / f"{img_name}.txt"
        
        if not label_file.exists():
            # No objects in image
            return np.array([]), np.array([])
        
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return np.array([]), np.array([])
        
        bboxes = []
        class_labels = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
        
        return np.array(bboxes, dtype=np.float32), np.array(class_labels, dtype=np.int64)
    
    def apply_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation
        
        Args:
            idx: Current image index
        
        Returns:
            Tuple of (mosaic_image, mosaic_bboxes, mosaic_labels)
        """
        # Select 3 additional random images
        indices = [idx]
        indices.extend(random.sample(
            [i for i in range(len(self)) if i != idx], 3
        ))
        
        # Create mosaic image
        mosaic_size = self.img_size
        mosaic_img = np.zeros((mosaic_size * 2, mosaic_size * 2, 4), dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        for i, idx in enumerate(indices):
            # Load image and labels
            img = self.load_image(idx)
            bboxes, labels = self.load_labels(idx)
            
            # Resize to half size
            h, w = img.shape[:2]
            img = cv2.resize(img, (mosaic_size, mosaic_size))
            
            # Place in mosaic
            if i == 0:  # Top-left
                x1, y1, x2, y2 = 0, 0, mosaic_size, mosaic_size
            elif i == 1:  # Top-right
                x1, y1, x2, y2 = mosaic_size, 0, mosaic_size * 2, mosaic_size
            elif i == 2:  # Bottom-left
                x1, y1, x2, y2 = 0, mosaic_size, mosaic_size, mosaic_size * 2
            else:  # Bottom-right
                x1, y1, x2, y2 = mosaic_size, mosaic_size, mosaic_size * 2, mosaic_size * 2
            
            mosaic_img[y1:y2, x1:x2] = img
            
            # Adjust bboxes
            if len(bboxes) > 0:
                # Convert to pixel coordinates
                bboxes[:, 0] = bboxes[:, 0] * mosaic_size + x1
                bboxes[:, 1] = bboxes[:, 1] * mosaic_size + y1
                bboxes[:, 2] = bboxes[:, 2] * mosaic_size
                bboxes[:, 3] = bboxes[:, 3] * mosaic_size
                
                # Convert back to normalized coordinates
                bboxes[:, [0, 2]] /= (mosaic_size * 2)
                bboxes[:, [1, 3]] /= (mosaic_size * 2)
                
                mosaic_bboxes.append(bboxes)
                mosaic_labels.append(labels)
        
        # Combine all bboxes and labels
        if mosaic_bboxes:
            mosaic_bboxes = np.vstack(mosaic_bboxes)
            mosaic_labels = np.hstack(mosaic_labels)
        else:
            mosaic_bboxes = np.array([])
            mosaic_labels = np.array([])
        
        # Resize back to target size
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        return mosaic_img, mosaic_bboxes, mosaic_labels
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item from dataset
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (image, target)
            image: RGBA tensor [4, H, W]
            target: Dictionary with 'boxes', 'labels', 'image_id'
        """
        # Apply mosaic augmentation with probability
        if self.augment and self.split == 'train' and random.random() < 0.5:
            img, bboxes, class_labels = self.apply_mosaic(idx)
        else:
            # Load image and labels normally
            img = self.load_image(idx)
            bboxes, class_labels = self.load_labels(idx)
        
        # Apply transformations
        if len(bboxes) > 0:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            img = transformed['image']
            bboxes = np.array(transformed['bboxes'])
            class_labels = np.array(transformed['class_labels'])
        else:
            transformed = self.transform(image=img)
            img = transformed['image']
            bboxes = np.array([])
            class_labels = np.array([])
        
        # Prepare target dictionary
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        
        return img, target
    
    def collate_fn(self, batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Custom collate function for batching
        
        Args:
            batch: List of (image, target) tuples
        
        Returns:
            Tuple of batched images and list of targets
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        # Stack images
        images = torch.stack(images, 0)
        
        return images, targets


def create_data_loaders(config: Dict, 
                       rgb_modality: str,
                       alpha_modality: str) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        config: Configuration dictionary
        rgb_modality: RGB modality to use
        alpha_modality: Alpha modality to use
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SnowPoleRGBADataset(
        root_path=Path(config['dataset']['root_path']),
        split='train',
        rgb_modality=rgb_modality,
        alpha_modality=alpha_modality,
        img_size=config['model']['input_size'],
        augment=True,
        cache_images=False
    )
    
    val_dataset = SnowPoleRGBADataset(
        root_path=Path(config['dataset']['root_path']),
        split='valid',
        rgb_modality=rgb_modality,
        alpha_modality=alpha_modality,
        img_size=config['model']['input_size'],
        augment=False,
        cache_images=True
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['workers'],
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['workers'],
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loader
    import yaml
    
    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = SnowPoleRGBADataset(
        root_path=Path("../../SnowPole_Detection_Dataset"),
        split='train',
        rgb_modality='Combination3',
        alpha_modality='range',
        img_size=640,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target keys: {target.keys()}")
    print(f"Number of boxes: {len(target['boxes'])}")
    
    # Test data loader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    for i, (images, targets) in enumerate(loader):
        print(f"Batch {i}: Images shape: {images.shape}")
        print(f"Batch {i}: Number of targets: {len(targets)}")
        if i >= 2:
            break
    
    print("Data loader test completed!")
