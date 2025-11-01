"""
Attention Modules for SnowPole Detection
Implements various attention mechanisms including EMA, CBAM, SE, and Cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EfficientMultiScaleAttention(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA) Module
    Lightweight attention mechanism that processes features at multiple scales
    """
    
    def __init__(self, channels: int, factor: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.factor = factor
        self.head_dim = channels // num_heads
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(channels, channels // factor, 1)
        self.conv3x3 = nn.Conv2d(channels, channels // factor, 3, padding=1, groups=channels // factor)
        self.conv5x5 = nn.Conv2d(channels, channels // factor, 5, padding=2, groups=channels // factor)
        
        # Attention computation
        self.to_qkv = nn.Conv2d(channels // factor * 3, channels * 3, 1, bias=False)
        self.rescale = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient multi-scale attention"""
        B, C, H, W = x.shape
        
        # Multi-scale feature extraction
        feat1x1 = self.conv1x1(x)
        feat3x3 = self.conv3x3(x)
        feat5x5 = self.conv5x5(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([feat1x1, feat3x3, feat5x5], dim=1)
        
        # Generate Q, K, V
        qkv = self.to_qkv(multi_scale)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, head_dim, HW]
        
        # Scaled dot-product attention
        q = q.transpose(-2, -1)  # [B, num_heads, HW, head_dim]
        k = k.transpose(-2, -1)  # [B, num_heads, HW, head_dim]
        v = v.transpose(-2, -1)  # [B, num_heads, HW, head_dim]
        
        attn = (q @ k.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [B, num_heads, HW, head_dim]
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        
        # Output projection and residual connection
        out = self.proj(out)
        return self.norm(out + x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel attention and spatial attention
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM attention"""
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights"""
        B, C, _, _ = x.shape
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights"""
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    Channel-wise attention mechanism
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention"""
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Module
    Enables information exchange between different modalities (RGB and Additional)
    """
    
    def __init__(self, rgb_channels: int, alpha_channels: int, 
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # Projection layers for RGB
        self.rgb_q = nn.Conv2d(rgb_channels, hidden_dim, 1)
        self.rgb_k = nn.Conv2d(rgb_channels, hidden_dim, 1)
        self.rgb_v = nn.Conv2d(rgb_channels, hidden_dim, 1)
        
        # Projection layers for Alpha
        self.alpha_q = nn.Conv2d(alpha_channels, hidden_dim, 1)
        self.alpha_k = nn.Conv2d(alpha_channels, hidden_dim, 1)
        self.alpha_v = nn.Conv2d(alpha_channels, hidden_dim, 1)
        
        # Output projections
        self.rgb_out = nn.Conv2d(hidden_dim, rgb_channels, 1)
        self.alpha_out = nn.Conv2d(hidden_dim, alpha_channels, 1)
        
        # Layer normalization
        self.rgb_norm = nn.GroupNorm(num_groups=32, num_channels=rgb_channels)
        self.alpha_norm = nn.GroupNorm(num_groups=32, num_channels=alpha_channels)
        
        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention
        
        Args:
            rgb: RGB features [B, C_rgb, H, W]
            alpha: Alpha features [B, C_alpha, H, W]
        
        Returns:
            Tuple of attended RGB and Alpha features
        """
        B, C_rgb, H, W = rgb.shape
        _, C_alpha, _, _ = alpha.shape
        
        # Generate queries, keys, values for RGB
        rgb_q = self.rgb_q(rgb).view(B, self.num_heads, self.head_dim, H * W)
        rgb_k = self.rgb_k(rgb).view(B, self.num_heads, self.head_dim, H * W)
        rgb_v = self.rgb_v(rgb).view(B, self.num_heads, self.head_dim, H * W)
        
        # Generate queries, keys, values for Alpha
        alpha_q = self.alpha_q(alpha).view(B, self.num_heads, self.head_dim, H * W)
        alpha_k = self.alpha_k(alpha).view(B, self.num_heads, self.head_dim, H * W)
        alpha_v = self.alpha_v(alpha).view(B, self.num_heads, self.head_dim, H * W)
        
        # RGB attending to Alpha
        rgb_to_alpha_scores = torch.einsum('bhdn,bhdm->bhnm', rgb_q, alpha_k.transpose(-2, -1))
        rgb_to_alpha_scores = rgb_to_alpha_scores / (self.head_dim ** 0.5 * self.temperature)
        rgb_to_alpha_attn = F.softmax(rgb_to_alpha_scores, dim=-1)
        rgb_attended = torch.einsum('bhnm,bhdm->bhdn', rgb_to_alpha_attn, alpha_v)
        
        # Alpha attending to RGB
        alpha_to_rgb_scores = torch.einsum('bhdn,bhdm->bhnm', alpha_q, rgb_k.transpose(-2, -1))
        alpha_to_rgb_scores = alpha_to_rgb_scores / (self.head_dim ** 0.5 * self.temperature)
        alpha_to_rgb_attn = F.softmax(alpha_to_rgb_scores, dim=-1)
        alpha_attended = torch.einsum('bhnm,bhdm->bhdn', alpha_to_rgb_attn, rgb_v)
        
        # Reshape and project back
        rgb_attended = rgb_attended.view(B, self.hidden_dim, H, W)
        alpha_attended = alpha_attended.view(B, self.hidden_dim, H, W)
        
        rgb_out = self.rgb_out(rgb_attended)
        alpha_out = self.alpha_out(alpha_attended)
        
        # Residual connection with normalization
        rgb = self.rgb_norm(rgb + rgb_out)
        alpha = self.alpha_norm(alpha + alpha_out)
        
        return rgb, alpha


class AdaptiveAttentionGate(nn.Module):
    """
    Adaptive Attention Gate
    Dynamically weights the contribution of different modalities
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 2),
            nn.Sigmoid()
        )
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights for two features
        
        Args:
            feat1: First feature tensor
            feat2: Second feature tensor
        
        Returns:
            Weighted combination of features
        """
        B, C, H, W = feat1.shape
        
        # Global features
        global_feat1 = self.global_pool(feat1).view(B, C)
        global_feat2 = self.global_pool(feat2).view(B, C)
        
        # Compute attention weights
        combined = torch.cat([global_feat1, global_feat2], dim=1)
        weights = self.fc(combined)  # [B, 2]
        
        # Apply weights
        weight1 = weights[:, 0:1].view(B, 1, 1, 1)
        weight2 = weights[:, 1:2].view(B, 1, 1, 1)
        
        return feat1 * weight1 + feat2 * weight2


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention Module
    Processes features at multiple scales with attention
    """
    
    def __init__(self, channels_list: list, reduction: int = 16):
        super().__init__()
        self.attention_modules = nn.ModuleList()
        
        for channels in channels_list:
            self.attention_modules.append(
                nn.Sequential(
                    SEBlock(channels, reduction),
                    SpatialAttention()
                )
            )
    
    def forward(self, features: list) -> list:
        """Apply hierarchical attention to multi-scale features"""
        attended_features = []
        for feat, attn in zip(features, self.attention_modules):
            attended_features.append(attn(feat))
        return attended_features


def build_attention_module(attention_type: str, channels: int, **kwargs) -> nn.Module:
    """
    Factory function to build attention modules
    
    Args:
        attention_type: Type of attention ('ema', 'cbam', 'se', 'cross_modal')
        channels: Number of input channels
        **kwargs: Additional arguments for specific attention types
    
    Returns:
        Attention module
    """
    attention_modules = {
        'ema': EfficientMultiScaleAttention,
        'cbam': CBAM,
        'se': SEBlock,
        'cross_modal': CrossModalAttention,
    }
    
    if attention_type not in attention_modules:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    if attention_type == 'cross_modal':
        return attention_modules[attention_type](
            rgb_channels=channels,
            alpha_channels=kwargs.get('alpha_channels', channels),
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_heads=kwargs.get('num_heads', 8)
        )
    else:
        return attention_modules[attention_type](channels, **kwargs)


if __name__ == "__main__":
    # Test attention modules
    batch_size = 2
    channels = 256
    height, width = 40, 40
    
    # Test EMA
    ema = EfficientMultiScaleAttention(channels)
    x = torch.randn(batch_size, channels, height, width)
    out = ema(x)
    print(f"EMA output shape: {out.shape}")
    
    # Test CBAM
    cbam = CBAM(channels)
    out = cbam(x)
    print(f"CBAM output shape: {out.shape}")
    
    # Test Cross-modal Attention
    cross_attn = CrossModalAttention(channels, channels // 2)
    rgb = torch.randn(batch_size, channels, height, width)
    alpha = torch.randn(batch_size, channels // 2, height, width)
    rgb_out, alpha_out = cross_attn(rgb, alpha)
    print(f"Cross-modal RGB output shape: {rgb_out.shape}")
    print(f"Cross-modal Alpha output shape: {alpha_out.shape}")
