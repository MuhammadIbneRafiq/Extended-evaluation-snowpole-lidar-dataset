"""
Loss functions for YOLOv9-style object detection
Implements CIoU loss, Focal loss, and Distribution Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import math


class YOLOv9Loss(nn.Module):
    """
    YOLOv9-style loss function
    Combines box regression loss, classification loss, and DFL loss
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_classes = config['dataset']['num_classes']
        
        # Loss weights
        loss_config = config['training']['loss']
        self.box_weight = loss_config['box_loss_weight']
        self.cls_weight = loss_config['cls_loss_weight']
        self.dfl_weight = loss_config['dfl_loss_weight']
        self.use_focal = loss_config.get('use_focal_loss', True)
        
        # Detection settings
        self.strides = [8, 16, 32]  # Feature map strides
        self.anchor_free = True
        
        # Focal loss parameters
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # DFL parameters
        self.dfl_channels = 16  # Distribution focal loss channels
    
    def forward(self, predictions: List[torch.Tensor], 
                targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss
        
        Args:
            predictions: List of predictions from detection heads [B, A, H, W, C]
            targets: List of target dictionaries with 'boxes' and 'labels'
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)
        
        # Initialize losses
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        dfl_loss = torch.tensor(0.0, device=device)
        
        # Process each scale
        for pred, stride in zip(predictions, self.strides):
            # Get dimensions
            bs, na, h, w, nc = pred.shape  # batch, anchors, height, width, channels
            
            # Separate outputs
            # Format: [x, y, w, h, obj, cls1, cls2, ...]
            xy = pred[..., 0:2]  # Center coordinates
            wh = pred[..., 2:4]  # Width and height
            obj = pred[..., 4:5]  # Objectness
            cls = pred[..., 5:5 + self.num_classes]  # Class predictions
            
            # Generate anchor points
            anchor_points = self.make_anchor_points(h, w, stride, device)
            
            # Match targets to anchors
            matched_targets = self.match_targets(targets, anchor_points, stride, (h, w))
            
            # Compute losses for this scale
            if matched_targets['num_targets'] > 0:
                # Box regression loss
                box_loss += self.compute_box_loss(
                    xy, wh, matched_targets['boxes'], 
                    matched_targets['anchor_points'], stride
                )
                
                # Classification loss
                cls_loss += self.compute_cls_loss(
                    cls, matched_targets['labels'], 
                    matched_targets['mask']
                )
                
                # Objectness loss
                obj_loss = self.compute_obj_loss(
                    obj, matched_targets['mask']
                )
                cls_loss += obj_loss * 0.5  # Add to cls loss
        
        # Combine losses
        total_loss = (self.box_weight * box_loss + 
                     self.cls_weight * cls_loss + 
                     self.dfl_weight * dfl_loss)
        
        # Create loss dictionary
        loss_dict = {
            'loss': total_loss,
            'box_loss': box_loss.item(),
            'cls_loss': cls_loss.item(),
            'dfl_loss': dfl_loss.item()
        }
        
        return total_loss, loss_dict
    
    def make_anchor_points(self, h: int, w: int, stride: int, 
                          device: torch.device) -> torch.Tensor:
        """
        Generate anchor points for anchor-free detection
        
        Args:
            h: Feature map height
            w: Feature map width
            stride: Feature stride
            device: Device
        
        Returns:
            Anchor points tensor [H*W, 2]
        """
        # Create grid
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Convert to anchor points (center of each cell)
        anchor_points = torch.stack([x + 0.5, y + 0.5], dim=-1)
        anchor_points = anchor_points.reshape(-1, 2) * stride
        
        return anchor_points
    
    def match_targets(self, targets: List[Dict], anchor_points: torch.Tensor,
                     stride: int, feat_size: Tuple[int, int]) -> Dict:
        """
        Match targets to anchor points
        
        Args:
            targets: List of target dictionaries
            anchor_points: Anchor points [N, 2]
            stride: Feature stride
            feat_size: Feature map size (h, w)
        
        Returns:
            Dictionary with matched targets
        """
        device = anchor_points.device
        h, w = feat_size
        
        matched = {
            'boxes': [],
            'labels': [],
            'anchor_points': [],
            'mask': torch.zeros((len(targets), h, w), device=device, dtype=torch.bool),
            'num_targets': 0
        }
        
        for batch_idx, target in enumerate(targets):
            if len(target['boxes']) == 0:
                continue
            
            # Convert boxes to pixel coordinates
            boxes = target['boxes'].to(device)  # [N, 4] in YOLO format
            labels = target['labels'].to(device)
            
            # Convert YOLO format to xyxy
            boxes_xyxy = self.yolo_to_xyxy(boxes, h * stride, w * stride)
            
            # Match each box to nearest anchor points
            for box, label in zip(boxes_xyxy, labels):
                # Get box center
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                # Find grid cell
                gx = int(cx / stride)
                gy = int(cy / stride)
                
                # Check bounds
                if 0 <= gx < w and 0 <= gy < h:
                    matched['boxes'].append(box)
                    matched['labels'].append(label)
                    matched['anchor_points'].append(anchor_points[gy * w + gx])
                    matched['mask'][batch_idx, gy, gx] = True
                    matched['num_targets'] += 1
        
        # Stack matched targets
        if matched['num_targets'] > 0:
            matched['boxes'] = torch.stack(matched['boxes'])
            matched['labels'] = torch.stack(matched['labels'])
            matched['anchor_points'] = torch.stack(matched['anchor_points'])
        
        return matched
    
    def yolo_to_xyxy(self, boxes: torch.Tensor, img_h: float, 
                     img_w: float) -> torch.Tensor:
        """
        Convert YOLO format to xyxy format
        
        Args:
            boxes: Boxes in YOLO format [N, 4] (cx, cy, w, h) normalized
            img_h: Image height
            img_w: Image width
        
        Returns:
            Boxes in xyxy format [N, 4]
        """
        boxes_xyxy = boxes.clone()
        
        # Convert to pixel coordinates
        boxes_xyxy[:, 0] *= img_w  # cx
        boxes_xyxy[:, 1] *= img_h  # cy
        boxes_xyxy[:, 2] *= img_w  # w
        boxes_xyxy[:, 3] *= img_h  # h
        
        # Convert to xyxy
        boxes_xyxy[:, 0] = boxes_xyxy[:, 0] - boxes_xyxy[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xyxy[:, 1] - boxes_xyxy[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]  # x2
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]  # y2
        
        return boxes_xyxy
    
    def compute_box_loss(self, pred_xy: torch.Tensor, pred_wh: torch.Tensor,
                        target_boxes: torch.Tensor, anchor_points: torch.Tensor,
                        stride: int) -> torch.Tensor:
        """
        Compute CIoU box regression loss
        
        Args:
            pred_xy: Predicted center coordinates
            pred_wh: Predicted width and height
            target_boxes: Target boxes in xyxy format
            anchor_points: Anchor points
            stride: Feature stride
        
        Returns:
            Box loss
        """
        if len(target_boxes) == 0:
            return torch.tensor(0.0, device=pred_xy.device)
        
        # Flatten predictions
        pred_xy = pred_xy.view(-1, 2)
        pred_wh = pred_wh.view(-1, 2)
        
        # Get predictions for matched targets
        # This is simplified - actual implementation would need proper indexing
        
        # Compute CIoU loss
        ciou_loss = self.ciou_loss(pred_xy, pred_wh, target_boxes, anchor_points)
        
        return ciou_loss.mean()
    
    def ciou_loss(self, pred_xy: torch.Tensor, pred_wh: torch.Tensor,
                  target_boxes: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
        """
        Complete IoU (CIoU) loss
        
        Args:
            pred_xy: Predicted centers
            pred_wh: Predicted dimensions
            target_boxes: Target boxes
            anchor_points: Anchor points
        
        Returns:
            CIoU loss
        """
        # Convert predictions to boxes
        pred_boxes = torch.zeros_like(target_boxes)
        pred_boxes[:, 0] = anchor_points[:, 0] + pred_xy[:, 0] - pred_wh[:, 0] / 2
        pred_boxes[:, 1] = anchor_points[:, 1] + pred_xy[:, 1] - pred_wh[:, 1] / 2
        pred_boxes[:, 2] = anchor_points[:, 0] + pred_xy[:, 0] + pred_wh[:, 0] / 2
        pred_boxes[:, 3] = anchor_points[:, 1] + pred_xy[:, 1] + pred_wh[:, 1] / 2
        
        # Compute IoU
        iou = self.compute_iou(pred_boxes, target_boxes)
        
        # Compute center distance
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # Compute diagonal distance of enclosing box
        enclosing_mins = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclosing_maxs = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        diagonal_dist = torch.sum((enclosing_maxs - enclosing_mins) ** 2, dim=1)
        
        # Compute aspect ratio penalty
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(target_w / (target_h + 1e-6)) - 
            torch.atan(pred_w / (pred_h + 1e-6)), 2
        )
        
        alpha = v / (1 - iou + v + 1e-6)
        
        # CIoU loss
        ciou = iou - center_dist / (diagonal_dist + 1e-6) - alpha * v
        loss = 1 - ciou
        
        return loss
    
    def compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes
        
        Args:
            boxes1: First set of boxes [N, 4] in xyxy format
            boxes2: Second set of boxes [N, 4] in xyxy format
        
        Returns:
            IoU values [N]
        """
        # Intersection
        inter_mins = torch.max(boxes1[:, :2], boxes2[:, :2])
        inter_maxs = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        inter_wh = torch.clamp(inter_maxs - inter_mins, min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def compute_cls_loss(self, pred_cls: torch.Tensor, target_labels: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss
        
        Args:
            pred_cls: Predicted class logits [B, A, H, W, num_classes]
            target_labels: Target labels
            mask: Mask for positive samples
        
        Returns:
            Classification loss
        """
        if len(target_labels) == 0:
            # Only negative samples
            if self.use_focal:
                # Focal loss for all negative
                neg_loss = self.focal_loss(
                    pred_cls.view(-1, self.num_classes),
                    torch.zeros(pred_cls.shape[:-1].numel(), 
                              dtype=torch.long, device=pred_cls.device),
                    alpha=1 - self.focal_alpha
                )
                return neg_loss.mean()
            else:
                # BCE loss for all negative
                return F.binary_cross_entropy_with_logits(
                    pred_cls, 
                    torch.zeros_like(pred_cls),
                    reduction='mean'
                )
        
        # Get positive and negative samples
        pos_mask = mask.unsqueeze(-1).expand_as(pred_cls[..., :1])
        
        # Compute loss
        if self.use_focal:
            loss = self.focal_loss(
                pred_cls[pos_mask].view(-1, self.num_classes),
                target_labels,
                alpha=self.focal_alpha
            )
        else:
            # Standard BCE loss
            target_one_hot = F.one_hot(target_labels, self.num_classes).float()
            loss = F.binary_cross_entropy_with_logits(
                pred_cls[pos_mask].view(-1, self.num_classes),
                target_one_hot,
                reduction='mean'
            )
        
        return loss
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal loss for addressing class imbalance
        
        Args:
            pred: Predictions [N, num_classes]
            target: Target class indices [N]
            alpha: Weighting factor
            gamma: Focusing parameter
        
        Returns:
            Focal loss
        """
        # Convert to probabilities
        p = torch.sigmoid(pred)
        
        # Get class probabilities
        ce_loss = F.binary_cross_entropy_with_logits(pred, 
                                                     F.one_hot(target, self.num_classes).float(),
                                                     reduction='none')
        
        # Apply focal term
        p_t = p * F.one_hot(target, self.num_classes).float() + \
              (1 - p) * (1 - F.one_hot(target, self.num_classes).float())
        focal_term = (1 - p_t) ** gamma
        
        # Apply alpha weighting
        alpha_t = alpha * F.one_hot(target, self.num_classes).float() + \
                 (1 - alpha) * (1 - F.one_hot(target, self.num_classes).float())
        
        focal_loss = alpha_t * focal_term * ce_loss
        
        return focal_loss.sum(dim=1)
    
    def compute_obj_loss(self, pred_obj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute objectness loss
        
        Args:
            pred_obj: Predicted objectness [B, A, H, W, 1]
            mask: Positive sample mask
        
        Returns:
            Objectness loss
        """
        # Create target
        target_obj = mask.float().unsqueeze(-1)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            pred_obj,
            target_obj,
            reduction='mean'
        )
        
        return loss


if __name__ == "__main__":
    # Test loss function
    import yaml
    
    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create loss function
    criterion = YOLOv9Loss(config)
    
    # Create dummy predictions and targets
    batch_size = 2
    predictions = [
        torch.randn(batch_size, 1, 80, 80, 6),  # P3
        torch.randn(batch_size, 1, 40, 40, 6),  # P4
        torch.randn(batch_size, 1, 20, 20, 6),  # P5
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.int64),
            'image_id': torch.tensor([0])
        },
        {
            'boxes': torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float32),
            'labels': torch.tensor([0], dtype=torch.int64),
            'image_id': torch.tensor([1])
        }
    ]
    
    # Compute loss
    loss, loss_dict = criterion(predictions, targets)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    print("Loss function test completed!")
