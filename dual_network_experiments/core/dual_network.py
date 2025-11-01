"""
Dual Network Architecture for SnowPole Detection
Inspired by YOLOv9t and YOLOv11 architectures
Implements a dual-branch network for RGB + Additional modality fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class ConvModule(nn.Module):
    """Basic convolution module with Conv2d + BatchNorm + Activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = None, groups: int = 1, 
                 activation: str = 'silu'):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """Bottleneck block inspired by YOLOv9"""
    
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, 
                 expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvModule(in_channels, hidden_channels, 1)
        self.conv2 = ConvModule(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.add:
            x = x + residual
        return x


class C3Block(nn.Module):
    """C3 block with multiple bottleneck layers"""
    
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int = 1, 
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvModule(in_channels, hidden_channels, 1)
        self.conv2 = ConvModule(in_channels, hidden_channels, 1)
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, 1)
        
        self.bottlenecks = nn.Sequential(
            *[BottleneckBlock(hidden_channels, hidden_channels, shortcut, 1.0) 
              for _ in range(n_blocks)]
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bottlenecks(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) module"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        
        self.conv1 = ConvModule(in_channels, hidden_channels, 1)
        self.conv2 = ConvModule(hidden_channels * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, 1, kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CrossBranchAttention(nn.Module):
    """Cross-branch attention for information exchange between RGB and Alpha branches"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * 2, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    
    def forward(self, rgb_feat: torch.Tensor, alpha_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-branch attention between RGB and Alpha features"""
        batch_size = rgb_feat.size(0)
        
        # Concatenate features for joint processing
        combined = torch.cat([rgb_feat, alpha_feat], dim=1)
        
        # Channel attention
        avg_out = self.avg_pool(combined).view(batch_size, -1)
        channel_att = self.fc(avg_out).view(batch_size, 2, self.channels, 1, 1)
        rgb_ch_att = channel_att[:, 0, :, :, :]
        alpha_ch_att = channel_att[:, 1, :, :, :]
        
        rgb_feat = rgb_feat * rgb_ch_att
        alpha_feat = alpha_feat * alpha_ch_att
        
        # Spatial attention for RGB
        rgb_avg = torch.mean(rgb_feat, dim=1, keepdim=True)
        rgb_max, _ = torch.max(rgb_feat, dim=1, keepdim=True)
        rgb_spatial = torch.cat([rgb_avg, rgb_max], dim=1)
        rgb_spatial_att = self.sigmoid_spatial(self.conv_spatial(rgb_spatial))
        rgb_feat = rgb_feat * rgb_spatial_att
        
        # Spatial attention for Alpha
        alpha_avg = torch.mean(alpha_feat, dim=1, keepdim=True)
        alpha_max, _ = torch.max(alpha_feat, dim=1, keepdim=True)
        alpha_spatial = torch.cat([alpha_avg, alpha_max], dim=1)
        alpha_spatial_att = self.sigmoid_spatial(self.conv_spatial(alpha_spatial))
        alpha_feat = alpha_feat * alpha_spatial_att
        
        return rgb_feat, alpha_feat


class DualBranchBackbone(nn.Module):
    """Dual-branch backbone for processing RGB and additional modality"""
    
    def __init__(self, rgb_channels: int = 3, alpha_channels: int = 1, 
                 base_channels: int = 64, depth_multiple: float = 0.33,
                 width_multiple: float = 0.25):
        super().__init__()
        
        # Calculate channel dimensions based on width multiple
        def ch(x):
            return max(8, int(x * width_multiple + 0.5) // 8 * 8)
        
        # RGB branch (main branch with more capacity)
        self.rgb_stem = ConvModule(rgb_channels, ch(base_channels), 6, 2, 2)
        self.rgb_stage1 = nn.Sequential(
            ConvModule(ch(base_channels), ch(base_channels * 2), 3, 2),
            C3Block(ch(base_channels * 2), ch(base_channels * 2), 
                   n_blocks=round(3 * depth_multiple))
        )
        self.rgb_stage2 = nn.Sequential(
            ConvModule(ch(base_channels * 2), ch(base_channels * 4), 3, 2),
            C3Block(ch(base_channels * 4), ch(base_channels * 4), 
                   n_blocks=round(6 * depth_multiple))
        )
        self.rgb_stage3 = nn.Sequential(
            ConvModule(ch(base_channels * 4), ch(base_channels * 8), 3, 2),
            C3Block(ch(base_channels * 8), ch(base_channels * 8), 
                   n_blocks=round(9 * depth_multiple))
        )
        self.rgb_stage4 = nn.Sequential(
            ConvModule(ch(base_channels * 8), ch(base_channels * 16), 3, 2),
            C3Block(ch(base_channels * 16), ch(base_channels * 16), 
                   n_blocks=round(3 * depth_multiple)),
            SPPF(ch(base_channels * 16), ch(base_channels * 16))
        )
        
        # Alpha branch (lightweight branch for additional modality)
        self.alpha_stem = ConvModule(alpha_channels, ch(base_channels // 2), 6, 2, 2)
        self.alpha_stage1 = nn.Sequential(
            ConvModule(ch(base_channels // 2), ch(base_channels), 3, 2),
            BottleneckBlock(ch(base_channels), ch(base_channels))
        )
        self.alpha_stage2 = nn.Sequential(
            ConvModule(ch(base_channels), ch(base_channels * 2), 3, 2),
            BottleneckBlock(ch(base_channels * 2), ch(base_channels * 2))
        )
        self.alpha_stage3 = nn.Sequential(
            ConvModule(ch(base_channels * 2), ch(base_channels * 4), 3, 2),
            BottleneckBlock(ch(base_channels * 4), ch(base_channels * 4))
        )
        self.alpha_stage4 = nn.Sequential(
            ConvModule(ch(base_channels * 4), ch(base_channels * 8), 3, 2),
            BottleneckBlock(ch(base_channels * 8), ch(base_channels * 8))
        )
        
        # Cross-branch attention modules
        self.cross_att2 = CrossBranchAttention(ch(base_channels * 4))
        self.cross_att3 = CrossBranchAttention(ch(base_channels * 8))
        self.cross_att4 = CrossBranchAttention(ch(base_channels * 16))
        
        # Feature alignment layers (to match dimensions)
        self.alpha_align2 = ConvModule(ch(base_channels * 2), ch(base_channels * 4), 1)
        self.alpha_align3 = ConvModule(ch(base_channels * 4), ch(base_channels * 8), 1)
        self.alpha_align4 = ConvModule(ch(base_channels * 8), ch(base_channels * 16), 1)
        
        self.ch = ch
    
    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-branch backbone
        
        Args:
            rgb: RGB input tensor [B, 3, H, W]
            alpha: Additional modality tensor [B, 1, H, W]
        
        Returns:
            Dictionary containing multi-scale features from both branches
        """
        # Process RGB branch
        rgb_s0 = self.rgb_stem(rgb)
        rgb_s1 = self.rgb_stage1(rgb_s0)
        rgb_s2 = self.rgb_stage2(rgb_s1)
        rgb_s3 = self.rgb_stage3(rgb_s2)
        rgb_s4 = self.rgb_stage4(rgb_s3)
        
        # Process Alpha branch
        alpha_s0 = self.alpha_stem(alpha)
        alpha_s1 = self.alpha_stage1(alpha_s0)
        alpha_s2 = self.alpha_stage2(alpha_s1)
        alpha_s3 = self.alpha_stage3(alpha_s2)
        alpha_s4 = self.alpha_stage4(alpha_s3)
        
        # Align alpha features to RGB dimensions
        alpha_s2_aligned = self.alpha_align2(alpha_s2)
        alpha_s3_aligned = self.alpha_align3(alpha_s3)
        alpha_s4_aligned = self.alpha_align4(alpha_s4)
        
        # Apply cross-branch attention
        rgb_s2, alpha_s2_aligned = self.cross_att2(rgb_s2, alpha_s2_aligned)
        rgb_s3, alpha_s3_aligned = self.cross_att3(rgb_s3, alpha_s3_aligned)
        rgb_s4, alpha_s4_aligned = self.cross_att4(rgb_s4, alpha_s4_aligned)
        
        return {
            'rgb_p3': rgb_s2,      # P3/8
            'rgb_p4': rgb_s3,      # P4/16
            'rgb_p5': rgb_s4,      # P5/32
            'alpha_p3': alpha_s2_aligned,
            'alpha_p4': alpha_s3_aligned,
            'alpha_p5': alpha_s4_aligned,
        }


class PANet(nn.Module):
    """Path Aggregation Network for feature pyramid"""
    
    def __init__(self, channels_list: List[int], num_repeats: List[int] = [3, 3, 3, 3]):
        super().__init__()
        
        # Bottom-up path augmentation
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P5 -> P4
        self.lat_p5 = ConvModule(channels_list[2], channels_list[1], 1)
        self.c3_p4 = C3Block(channels_list[1] * 2, channels_list[1], 
                            n_blocks=num_repeats[0], shortcut=False)
        
        # P4 -> P3
        self.lat_p4 = ConvModule(channels_list[1], channels_list[0], 1)
        self.c3_p3 = C3Block(channels_list[0] * 2, channels_list[0], 
                            n_blocks=num_repeats[1], shortcut=False)
        
        # Top-down path
        # P3 -> N3
        self.down_p3 = ConvModule(channels_list[0], channels_list[0], 3, 2)
        self.c3_n3 = C3Block(channels_list[0] + channels_list[1], channels_list[1], 
                            n_blocks=num_repeats[2], shortcut=False)
        
        # N3 -> N4
        self.down_n3 = ConvModule(channels_list[1], channels_list[1], 3, 2)
        self.c3_n4 = C3Block(channels_list[1] + channels_list[2], channels_list[2], 
                            n_blocks=num_repeats[3], shortcut=False)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Apply Path Aggregation Network to fused features"""
        
        # Fuse RGB and Alpha features
        p3 = features['rgb_p3'] + features['alpha_p3']
        p4 = features['rgb_p4'] + features['alpha_p4']
        p5 = features['rgb_p5'] + features['alpha_p5']
        
        # Bottom-up
        lat_p5 = self.lat_p5(p5)
        up_p5 = self.upsample(lat_p5)
        f_p4 = torch.cat([up_p5, p4], dim=1)
        f_p4 = self.c3_p4(f_p4)
        
        lat_p4 = self.lat_p4(f_p4)
        up_p4 = self.upsample(lat_p4)
        f_p3 = torch.cat([up_p4, p3], dim=1)
        f_p3 = self.c3_p3(f_p3)
        
        # Top-down
        down_p3 = self.down_p3(f_p3)
        f_n3 = torch.cat([down_p3, f_p4], dim=1)
        f_n3 = self.c3_n3(f_n3)
        
        down_n3 = self.down_n3(f_n3)
        f_n4 = torch.cat([down_n3, p5], dim=1)
        f_n4 = self.c3_n4(f_n4)
        
        return [f_p3, f_n3, f_n4]  # Multi-scale outputs


class DetectionHead(nn.Module):
    """YOLOv9-style detection head"""
    
    def __init__(self, num_classes: int, channels_list: List[int], 
                 num_anchors: int = 1, strides: List[int] = [8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_classes + 5  # cls + box(4) + obj(1)
        self.strides = strides
        
        # Detection heads for each scale
        self.heads = nn.ModuleList()
        for channels in channels_list:
            self.heads.append(
                nn.Sequential(
                    ConvModule(channels, channels, 3),
                    nn.Conv2d(channels, num_anchors * self.num_outputs, 1)
                )
            )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through detection heads"""
        outputs = []
        for i, (feat, head) in enumerate(zip(features, self.heads)):
            out = head(feat)
            bs, _, h, w = out.shape
            out = out.view(bs, self.num_anchors, self.num_outputs, h, w)
            out = out.permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(out)
        return outputs


class DualNetworkYOLO(nn.Module):
    """Complete Dual Network YOLO model"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.num_classes = config['dataset']['num_classes']
        self.input_size = config['model']['input_size']
        base_channels = 64
        
        # Width and depth multipliers for YOLOv9t
        width_multiple = 0.25  # For tiny model
        depth_multiple = 0.33
        
        # Initialize backbone
        self.backbone = DualBranchBackbone(
            rgb_channels=3,
            alpha_channels=1,
            base_channels=base_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple
        )
        
        # Calculate channel dimensions
        ch = self.backbone.ch
        channels_list = [
            ch(base_channels * 4),   # P3
            ch(base_channels * 8),   # P4
            ch(base_channels * 16),  # P5
        ]
        
        # Initialize neck (PANet)
        self.neck = PANet(channels_list, num_repeats=[3, 3, 3, 3])
        
        # Initialize detection head
        self.head = DetectionHead(
            num_classes=self.num_classes,
            channels_list=channels_list,
            num_anchors=1,  # Anchor-free
            strides=[8, 16, 32]
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [B, 4, H, W] where channels are [R, G, B, Alpha]
        
        Returns:
            List of detection outputs at different scales
        """
        # Split input into RGB and Alpha
        rgb = x[:, :3, :, :]
        alpha = x[:, 3:4, :, :]
        
        # Process through dual-branch backbone
        features = self.backbone(rgb, alpha)
        
        # Apply neck (PANet)
        neck_features = self.neck(features)
        
        # Get detection outputs
        outputs = self.head(neck_features)
        
        return outputs
    
    def predict(self, x: torch.Tensor, conf_thresh: float = 0.25, 
                iou_thresh: float = 0.45) -> List[torch.Tensor]:
        """
        Inference mode with NMS
        
        Args:
            x: Input tensor
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS
        
        Returns:
            List of detections [batch_size, num_detections, 7]
            Format: [x1, y1, x2, y2, confidence, class_score, class_id]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Post-processing will be handled by the detection pipeline
            # This is a placeholder for the actual NMS implementation
            predictions = []
            for output in outputs:
                # Process each scale output
                batch_size = output.size(0)
                for b in range(batch_size):
                    pred = output[b]  # [num_anchors, h, w, num_outputs]
                    # Apply confidence threshold and NMS here
                    # This would typically use torchvision.ops.nms
                    predictions.append(pred)
            
            return predictions


def build_dual_network(config: Dict) -> nn.Module:
    """Build dual network model from configuration"""
    model = DualNetworkYOLO(config)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test the model
    import yaml
    
    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = build_dual_network(config)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 4, 640, 640)
    outputs = model(dummy_input)
    
    print("Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Output shapes: {[out.shape for out in outputs]}")
