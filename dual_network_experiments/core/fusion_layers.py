"""
Fusion Layers for Multi-Modal Feature Integration
Implements various fusion strategies for combining RGB and additional modality features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion
    Concatenates features and optionally applies 1x1 convolution for channel reduction
    """
    
    def __init__(self, in_channels: List[int], out_channels: int, 
                 normalize: bool = True, use_conv: bool = True):
        super().__init__()
        self.normalize = normalize
        self.use_conv = use_conv
        total_channels = sum(in_channels)
        
        if use_conv:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.fusion_conv = nn.Identity()
            
        if normalize:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        else:
            self.norm = nn.Identity()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate and fuse features
        
        Args:
            features: List of feature tensors to concatenate
        
        Returns:
            Fused feature tensor
        """
        # Ensure all features have the same spatial dimensions
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Concatenate along channel dimension
        fused = torch.cat(aligned_features, dim=1)
        
        # Apply convolution if specified
        if self.use_conv:
            fused = self.fusion_conv(fused)
        
        # Apply normalization
        fused = self.norm(fused)
        
        return fused


class AdditionFusion(nn.Module):
    """
    Element-wise addition fusion
    Adds features with optional learnable weights
    """
    
    def __init__(self, num_inputs: int, channels: int, 
                 weighted: bool = True, normalize: bool = True):
        super().__init__()
        self.weighted = weighted
        self.normalize = normalize
        self.num_inputs = num_inputs
        
        if weighted:
            # Learnable weights for each input
            self.weights = nn.Parameter(torch.ones(num_inputs))
            self.weight_activation = nn.Softmax(dim=0)
        
        if normalize:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        else:
            self.norm = nn.Identity()
        
        # Channel alignment layers
        self.align_convs = nn.ModuleList()
        for _ in range(num_inputs):
            self.align_convs.append(nn.Conv2d(channels, channels, 1, bias=False))
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Add features with optional weighting
        
        Args:
            features: List of feature tensors to add
        
        Returns:
            Fused feature tensor
        """
        assert len(features) == self.num_inputs, \
            f"Expected {self.num_inputs} features, got {len(features)}"
        
        # Align channels if needed
        aligned_features = []
        for feat, align_conv in zip(features, self.align_convs):
            aligned_features.append(align_conv(feat))
        
        # Apply weights if specified
        if self.weighted:
            weights = self.weight_activation(self.weights)
            fused = sum(feat * weight.view(1, 1, 1, 1) 
                       for feat, weight in zip(aligned_features, weights))
        else:
            fused = sum(aligned_features)
        
        # Normalize
        fused = self.norm(fused)
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable gates
    Uses attention mechanism to determine feature importance
    """
    
    def __init__(self, channels: List[int], gate_type: str = 'sigmoid',
                 temperature: float = 1.0, use_attention: bool = True):
        super().__init__()
        self.gate_type = gate_type
        self.temperature = temperature
        self.use_attention = use_attention
        self.num_inputs = len(channels)
        
        # Gate computation network
        self.gate_conv = nn.ModuleList()
        for ch in channels:
            self.gate_conv.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 4, 1, 1)
                )
            )
        
        # Channel alignment
        max_channels = max(channels)
        self.align_convs = nn.ModuleList()
        for ch in channels:
            if ch != max_channels:
                self.align_convs.append(nn.Conv2d(ch, max_channels, 1))
            else:
                self.align_convs.append(nn.Identity())
        
        # Attention module if specified
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=max_channels,
                num_heads=8,
                batch_first=True
            )
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(max_channels, max_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(max_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply gated fusion to features
        
        Args:
            features: List of feature tensors
        
        Returns:
            Fused feature tensor
        """
        B = features[0].shape[0]
        
        # Compute gates for each feature
        gates = []
        for feat, gate_conv in zip(features, self.gate_conv):
            gate = gate_conv(feat)  # [B, 1, H, W]
            gates.append(gate)
        
        # Normalize gates
        gates = torch.cat(gates, dim=1)  # [B, num_inputs, H, W]
        
        if self.gate_type == 'sigmoid':
            gates = torch.sigmoid(gates / self.temperature)
        elif self.gate_type == 'softmax':
            gates = F.softmax(gates / self.temperature, dim=1)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")
        
        # Align channels
        aligned_features = []
        for feat, align_conv in zip(features, self.align_convs):
            aligned_features.append(align_conv(feat))
        
        # Apply gates
        gated_features = []
        for i, feat in enumerate(aligned_features):
            gate = gates[:, i:i+1, :, :]
            gated_features.append(feat * gate)
        
        # Sum gated features
        fused = sum(gated_features)
        
        # Apply attention if specified
        if self.use_attention:
            B, C, H, W = fused.shape
            fused_flat = fused.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
            attended, _ = self.attention(fused_flat, fused_flat, fused_flat)
            fused = attended.permute(0, 2, 1).view(B, C, H, W)
        
        # Output projection
        fused = self.out_conv(fused)
        
        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion at multiple scales
    Progressively fuses features from different levels
    """
    
    def __init__(self, channels_list: List[int], fusion_method: str = 'concatenation'):
        super().__init__()
        self.fusion_method = fusion_method
        self.num_scales = len(channels_list)
        
        # Create fusion modules for each scale
        self.fusion_modules = nn.ModuleList()
        
        for i, channels in enumerate(channels_list):
            if fusion_method == 'concatenation':
                fusion = ConcatenationFusion([channels, channels], channels)
            elif fusion_method == 'addition':
                fusion = AdditionFusion(2, channels)
            elif fusion_method == 'gated':
                fusion = GatedFusion([channels, channels])
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            self.fusion_modules.append(fusion)
        
        # Cross-scale connections
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample_modules = nn.ModuleList()
        
        for channels in channels_list[:-1]:
            self.downsample_modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True)
                )
            )
    
    def forward(self, rgb_features: Dict[str, torch.Tensor], 
                alpha_features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Hierarchically fuse multi-scale features
        
        Args:
            rgb_features: Dictionary of RGB features at different scales
            alpha_features: Dictionary of Alpha features at different scales
        
        Returns:
            List of fused features at different scales
        """
        # Extract features at each scale
        scales = ['p3', 'p4', 'p5']
        fused_features = []
        
        for i, scale in enumerate(scales):
            rgb_feat = rgb_features[f'rgb_{scale}']
            alpha_feat = alpha_features[f'alpha_{scale}']
            
            # Fuse at current scale
            fused = self.fusion_modules[i]([rgb_feat, alpha_feat])
            
            # Add cross-scale connection if not the first scale
            if i > 0:
                # Upsample previous scale and add
                prev_upsampled = self.upsample(fused_features[-1])
                
                # Ensure spatial dimensions match
                if prev_upsampled.shape[-2:] != fused.shape[-2:]:
                    prev_upsampled = F.interpolate(
                        prev_upsampled, 
                        size=fused.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                fused = fused + prev_upsampled
            
            fused_features.append(fused)
        
        return fused_features


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to select the best fusion strategy
    """
    
    def __init__(self, channels: int, num_strategies: int = 3):
        super().__init__()
        
        # Different fusion strategies
        self.concat_fusion = ConcatenationFusion([channels, channels], channels)
        self.add_fusion = AdditionFusion(2, channels)
        self.gate_fusion = GatedFusion([channels, channels])
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_strategies, 1)
        )
        
        self.num_strategies = num_strategies
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Adaptively fuse two features
        
        Args:
            feat1: First feature tensor
            feat2: Second feature tensor
        
        Returns:
            Fused feature tensor
        """
        # Compute fusion results for each strategy
        concat_result = self.concat_fusion([feat1, feat2])
        add_result = self.add_fusion([feat1, feat2])
        gate_result = self.gate_fusion([feat1, feat2])
        
        # Compute strategy weights
        combined = torch.cat([feat1, feat2], dim=1)
        weights = self.strategy_selector(combined)  # [B, num_strategies, 1, 1]
        weights = F.softmax(weights, dim=1)
        
        # Weight and combine results
        fused = (concat_result * weights[:, 0:1, :, :] +
                add_result * weights[:, 1:2, :, :] +
                gate_result * weights[:, 2:3, :, :])
        
        return fused


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for capturing second-order interactions
    """
    
    def __init__(self, channels1: int, channels2: int, output_channels: int,
                 pooling_size: int = 4):
        super().__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.output_channels = output_channels
        self.pooling_size = pooling_size
        
        # Dimension reduction
        reduction_dim = min(channels1, channels2, 256)
        self.reduce1 = nn.Conv2d(channels1, reduction_dim, 1)
        self.reduce2 = nn.Conv2d(channels2, reduction_dim, 1)
        
        # Bilinear pooling
        self.bilinear = nn.Bilinear(reduction_dim, reduction_dim, output_channels)
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Apply bilinear fusion
        
        Args:
            feat1: First feature tensor [B, C1, H, W]
            feat2: Second feature tensor [B, C2, H, W]
        
        Returns:
            Fused feature tensor [B, output_channels, H, W]
        """
        B, _, H, W = feat1.shape
        
        # Reduce dimensions
        feat1_reduced = self.reduce1(feat1)
        feat2_reduced = self.reduce2(feat2)
        
        # Pool spatially for bilinear operation
        if self.pooling_size > 1:
            feat1_pooled = F.adaptive_avg_pool2d(feat1_reduced, self.pooling_size)
            feat2_pooled = F.adaptive_avg_pool2d(feat2_reduced, self.pooling_size)
        else:
            feat1_pooled = feat1_reduced
            feat2_pooled = feat2_reduced
        
        # Reshape for bilinear
        feat1_flat = feat1_pooled.view(B, -1)
        feat2_flat = feat2_pooled.view(B, -1)
        
        # Apply bilinear fusion
        fused = self.bilinear(feat1_flat, feat2_flat)
        
        # Reshape back to spatial
        fused = fused.view(B, self.output_channels, 1, 1)
        fused = fused.expand(B, self.output_channels, H, W)
        
        # Output projection
        fused = self.out_conv(fused)
        
        return fused


def build_fusion_layer(fusion_type: str, **kwargs) -> nn.Module:
    """
    Factory function to build fusion layers
    
    Args:
        fusion_type: Type of fusion layer
        **kwargs: Additional arguments for specific fusion types
    
    Returns:
        Fusion layer module
    """
    fusion_layers = {
        'concatenation': ConcatenationFusion,
        'addition': AdditionFusion,
        'gated': GatedFusion,
        'hierarchical': HierarchicalFusion,
        'adaptive': AdaptiveFusion,
        'bilinear': BilinearFusion,
    }
    
    if fusion_type not in fusion_layers:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_layers[fusion_type](**kwargs)


if __name__ == "__main__":
    # Test fusion layers
    batch_size = 2
    channels = 256
    height, width = 40, 40
    
    # Create sample features
    feat1 = torch.randn(batch_size, channels, height, width)
    feat2 = torch.randn(batch_size, channels, height, width)
    
    # Test concatenation fusion
    concat_fusion = ConcatenationFusion([channels, channels], channels)
    fused = concat_fusion([feat1, feat2])
    print(f"Concatenation fusion output shape: {fused.shape}")
    
    # Test addition fusion
    add_fusion = AdditionFusion(2, channels)
    fused = add_fusion([feat1, feat2])
    print(f"Addition fusion output shape: {fused.shape}")
    
    # Test gated fusion
    gate_fusion = GatedFusion([channels, channels])
    fused = gate_fusion([feat1, feat2])
    print(f"Gated fusion output shape: {fused.shape}")
    
    # Test adaptive fusion
    adaptive_fusion = AdaptiveFusion(channels)
    fused = adaptive_fusion(feat1, feat2)
    print(f"Adaptive fusion output shape: {fused.shape}")
