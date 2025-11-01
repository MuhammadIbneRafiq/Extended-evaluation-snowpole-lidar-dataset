"""
Training script with attention modules enabled
Part of Experiment 2: Attention Module Ablation
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_1_baseline.train import Trainer
from experiment_1_baseline.data_loader import create_data_loaders
from core.dual_network import DualNetworkYOLO
from core.attention_modules import (
    EfficientMultiScaleAttention,
    CBAM,
    CrossModalAttention,
    build_attention_module
)


class DualNetworkWithAttention(DualNetworkYOLO):
    """
    Extended dual network with configurable attention modules
    """
    
    def __init__(self, config: dict, attention_config: dict):
        super().__init__(config)
        self.attention_config = attention_config
        
        # Get channel dimensions
        ch = self.backbone.ch
        channels_p3 = ch(64 * 4)   # P3 channels
        channels_p4 = ch(64 * 8)   # P4 channels  
        channels_p5 = ch(64 * 16)  # P5 channels
        
        # Add attention modules based on configuration
        self.attention_modules = nn.ModuleDict()
        
        if attention_config['type'] == 'ema' or attention_config['type'] == 'all':
            # Add EMA attention after fusion
            self.attention_modules['ema_p3'] = EfficientMultiScaleAttention(channels_p3)
            self.attention_modules['ema_p4'] = EfficientMultiScaleAttention(channels_p4)
            self.attention_modules['ema_p5'] = EfficientMultiScaleAttention(channels_p5)
            
        if attention_config['type'] == 'cbam' or attention_config['type'] == 'all':
            # Add CBAM attention
            self.attention_modules['cbam_p3'] = CBAM(channels_p3)
            self.attention_modules['cbam_p4'] = CBAM(channels_p4)
            self.attention_modules['cbam_p5'] = CBAM(channels_p5)
            
        if attention_config['type'] == 'cross_modal' or attention_config['type'] == 'all':
            # Add cross-modal attention
            self.attention_modules['cross_p3'] = CrossModalAttention(
                channels_p3, channels_p3, hidden_dim=256
            )
            self.attention_modules['cross_p4'] = CrossModalAttention(
                channels_p4, channels_p4, hidden_dim=256
            )
            self.attention_modules['cross_p5'] = CrossModalAttention(
                channels_p5, channels_p5, hidden_dim=256
            )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass with attention modules
        """
        # Split input into RGB and Alpha
        rgb = x[:, :3, :, :]
        alpha = x[:, 3:4, :, :]
        
        # Process through dual-branch backbone
        features = self.backbone(rgb, alpha)
        
        # Apply attention modules if configured
        if self.attention_config['enabled']:
            features = self.apply_attention(features)
        
        # Apply neck (PANet)
        neck_features = self.neck(features)
        
        # Get detection outputs
        outputs = self.head(neck_features)
        
        return outputs
    
    def apply_attention(self, features: dict) -> dict:
        """
        Apply attention modules to features
        """
        # Apply EMA if enabled
        if 'ema_p3' in self.attention_modules:
            features['rgb_p3'] = self.attention_modules['ema_p3'](features['rgb_p3'])
            features['alpha_p3'] = self.attention_modules['ema_p3'](features['alpha_p3'])
        if 'ema_p4' in self.attention_modules:
            features['rgb_p4'] = self.attention_modules['ema_p4'](features['rgb_p4'])
            features['alpha_p4'] = self.attention_modules['ema_p4'](features['alpha_p4'])
        if 'ema_p5' in self.attention_modules:
            features['rgb_p5'] = self.attention_modules['ema_p5'](features['rgb_p5'])
            features['alpha_p5'] = self.attention_modules['ema_p5'](features['alpha_p5'])
        
        # Apply CBAM if enabled
        if 'cbam_p3' in self.attention_modules:
            features['rgb_p3'] = self.attention_modules['cbam_p3'](features['rgb_p3'])
            features['alpha_p3'] = self.attention_modules['cbam_p3'](features['alpha_p3'])
        if 'cbam_p4' in self.attention_modules:
            features['rgb_p4'] = self.attention_modules['cbam_p4'](features['rgb_p4'])
            features['alpha_p4'] = self.attention_modules['cbam_p4'](features['alpha_p4'])
        if 'cbam_p5' in self.attention_modules:
            features['rgb_p5'] = self.attention_modules['cbam_p5'](features['rgb_p5'])
            features['alpha_p5'] = self.attention_modules['cbam_p5'](features['alpha_p5'])
        
        # Apply cross-modal attention if enabled
        if 'cross_p3' in self.attention_modules:
            features['rgb_p3'], features['alpha_p3'] = self.attention_modules['cross_p3'](
                features['rgb_p3'], features['alpha_p3']
            )
        if 'cross_p4' in self.attention_modules:
            features['rgb_p4'], features['alpha_p4'] = self.attention_modules['cross_p4'](
                features['rgb_p4'], features['alpha_p4']
            )
        if 'cross_p5' in self.attention_modules:
            features['rgb_p5'], features['alpha_p5'] = self.attention_modules['cross_p5'](
                features['rgb_p5'], features['alpha_p5']
            )
        
        return features


class AttentionTrainer(Trainer):
    """
    Extended trainer for attention ablation experiment
    """
    
    def __init__(self, config: dict, args: argparse.Namespace):
        # Update config for attention experiment
        config['experiments']['current'] = 'exp2_attention'
        config['model']['attention']['enabled'] = True
        config['model']['attention']['type'] = args.attention_type
        
        super().__init__(config, args)
        
        # Log attention configuration
        self.logger.info(f"Attention configuration: {config['model']['attention']}")
    
    def build_model(self):
        """Build model with attention modules"""
        self.logger.info("Building dual network with attention modules...")
        
        # Create attention configuration
        attention_config = {
            'enabled': True,
            'type': self.args.attention_type,
            'reduction_ratio': self.config['model']['attention'].get('reduction_ratio', 16)
        }
        
        # Build model
        model = DualNetworkWithAttention(self.config, attention_config)
        
        # Move to device
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        attention_params = sum(
            p.numel() for n, p in model.named_parameters() 
            if 'attention' in n.lower()
        )
        
        self.logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        self.logger.info(f"Attention parameters: {attention_params / 1e6:.2f}M")
        self.logger.info(f"Attention overhead: {attention_params / total_params * 100:.1f}%")
        
        return model
    
    def save_attention_analysis(self, epoch: int):
        """
        Save attention weight analysis
        """
        analysis_dir = self.exp_dir / 'attention_analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # Extract attention weights
        attention_weights = {}
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                if hasattr(module, 'weight'):
                    attention_weights[name] = module.weight.detach().cpu().numpy()
        
        # Save weights
        import numpy as np
        np.savez(
            analysis_dir / f'attention_weights_epoch_{epoch}.npz',
            **attention_weights
        )
        
        # Log statistics
        for name, weights in attention_weights.items():
            self.logger.info(
                f"Attention {name}: "
                f"mean={weights.mean():.4f}, "
                f"std={weights.std():.4f}, "
                f"min={weights.min():.4f}, "
                f"max={weights.max():.4f}"
            )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train model with attention modules')
    parser.add_argument('--config', type=str, default='config_attention.yaml',
                       help='Path to configuration file')
    parser.add_argument('--attention_type', type=str, default='all',
                       choices=['ema', 'cbam', 'cross_modal', 'all'],
                       help='Type of attention to use')
    parser.add_argument('--modality', type=str, default='Combination3',
                       choices=['Combination1', 'Combination2', 'Combination3', 
                               'Combination4', 'Combination5', 'Combination6'],
                       help='RGB modality to use')
    parser.add_argument('--alpha_channel', type=str, default='range',
                       choices=['range', 'nearir', 'signal', 'reflec'],
                       help='Additional modality for alpha channel')
    parser.add_argument('--exp_name', type=str, default='with_attention',
                       help='Experiment name for logging')
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update experiment name
    config['experiments']['exp2_attention']['name'] = f"{args.exp_name}_{args.attention_type}"
    
    # Create trainer
    trainer = AttentionTrainer(config, args)
    
    # Start training
    trainer.train()
    
    # Save final attention analysis
    trainer.save_attention_analysis(trainer.epoch)
    
    print(f"Training completed with attention type: {args.attention_type}")


if __name__ == '__main__':
    main()
