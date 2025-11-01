"""
Training script for Experiment 1: Baseline with RGBA inputs
Implements training loop for dual network with 4-channel input
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
from tensorboardX import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dual_network import build_dual_network
from data_loader import SnowPoleRGBADataset, create_data_loaders
from loss import YOLOv9Loss
from utils import (
    setup_logging, 
    save_checkpoint, 
    load_checkpoint,
    compute_metrics,
    EarlyStopping
)


class Trainer:
    """Main trainer class for baseline RGBA experiment"""
    
    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.logger = setup_logging(self.log_dir / 'training.log')
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Log configuration
        self.logger.info(f"Training configuration:\n{yaml.dump(config)}")
        self.logger.info(f"Arguments: {args}")
        
        # Initialize model
        self.model = self.build_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Initialize training components
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.criterion = YOLOv9Loss(config)
        
        # Mixed precision training
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
        
        # Training state
        self.epoch = 0
        self.best_map = 0.0
        self.early_stopping = EarlyStopping(patience=30)
    
    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = Path(self.config['output']['results_dir']) / f"exp1_baseline_{timestamp}"
        
        self.exp_dir = base_dir
        self.weights_dir = base_dir / 'weights'
        self.log_dir = base_dir / 'logs'
        self.vis_dir = base_dir / 'visualizations'
        
        for dir_path in [self.exp_dir, self.weights_dir, self.log_dir, self.vis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def build_model(self):
        """Build and initialize the dual network model"""
        self.logger.info("Building dual network model...")
        
        # Build model
        model = build_dual_network(self.config)
        
        # Load pretrained weights for RGB branch if specified
        if self.config['model']['dual_network']['rgb_branch']['pretrained']:
            pretrained_path = self.config['model']['dual_network']['rgb_branch']['pretrained_weights']
            if os.path.exists(pretrained_path):
                self.logger.info(f"Loading pretrained weights from {pretrained_path}")
                self.load_pretrained_rgb(model, pretrained_path)
            else:
                self.logger.warning(f"Pretrained weights not found at {pretrained_path}")
        
        # Move to device
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        self.logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        return model
    
    def load_pretrained_rgb(self, model, pretrained_path):
        """Load pretrained weights for RGB branch"""
        try:
            # Load pretrained checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter and map to RGB branch
            rgb_state_dict = {}
            for k, v in state_dict.items():
                # Map to RGB branch keys
                if k.startswith('backbone'):
                    # Only load RGB-related backbone weights
                    if 'rgb' in k or not ('alpha' in k):
                        rgb_state_dict[k] = v
            
            # Load weights
            model.load_state_dict(rgb_state_dict, strict=False)
            
            # Freeze layers if specified
            freeze_layers = self.config['model']['dual_network']['rgb_branch'].get('freeze_layers', 0)
            if freeze_layers > 0:
                self.freeze_rgb_layers(model, freeze_layers)
                self.logger.info(f"Frozen first {freeze_layers} layers of RGB branch")
        
        except Exception as e:
            self.logger.error(f"Error loading pretrained weights: {e}")
    
    def freeze_rgb_layers(self, model, num_layers):
        """Freeze specified number of layers in RGB branch"""
        count = 0
        for name, param in model.named_parameters():
            if 'rgb' in name and count < num_layers:
                param.requires_grad = False
                count += 1
    
    def create_dataloaders(self):
        """Create training and validation data loaders"""
        self.logger.info("Creating data loaders...")
        
        # Get data configuration
        data_config = {
            'root_path': Path(self.config['dataset']['root_path']),
            'rgb_modality': self.args.modality,
            'alpha_modality': self.args.alpha_channel,
            'img_size': self.config['model']['input_size'],
            'augment': self.config['training']['augmentation']['enabled']
        }
        
        # Create datasets
        train_dataset = SnowPoleRGBADataset(
            data_config['root_path'],
            'train',
            data_config['rgb_modality'],
            data_config['alpha_modality'],
            img_size=data_config['img_size'],
            augment=True
        )
        
        val_dataset = SnowPoleRGBADataset(
            data_config['root_path'],
            'valid',
            data_config['rgb_modality'],
            data_config['alpha_modality'],
            img_size=data_config['img_size'],
            augment=False
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['workers'],
            pin_memory=True,
            collate_fn=train_dataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['workers'],
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )
        
        self.logger.info(f"Train dataset: {len(train_dataset)} images")
        self.logger.info(f"Val dataset: {len(val_dataset)} images")
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """Create optimizer"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['type']}")
        
        return scheduler
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_items = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss, loss_items = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_box_loss += loss_items['box_loss']
            total_cls_loss += loss_items['cls_loss']
            total_dfl_loss += loss_items['dfl_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items["box_loss"]:.4f}',
                'cls': f'{loss_items["cls_loss"]:.4f}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/BoxLoss', loss_items['box_loss'], global_step)
                self.writer.add_scalar('Train/ClsLoss', loss_items['cls_loss'], global_step)
                self.writer.add_scalar('Train/DFLLoss', loss_items['dfl_loss'], global_step)
        
        # Compute epoch averages
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_box_loss = total_box_loss / n_batches
        avg_cls_loss = total_cls_loss / n_batches
        avg_dfl_loss = total_dfl_loss / n_batches
        
        return {
            'loss': avg_loss,
            'box_loss': avg_box_loss,
            'cls_loss': avg_cls_loss,
            'dfl_loss': avg_dfl_loss
        }
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss, _ = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss, _ = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                predictions = self.model.predict(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"mAP@50: {val_metrics.get('map50', 0):.4f}, "
                f"mAP@50-95: {val_metrics.get('map50_95', 0):.4f}"
            )
            
            # Save checkpoint
            if epoch % self.config['logging']['checkpoint']['save_period'] == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Save best model
            if val_metrics.get('map50', 0) > self.best_map:
                self.best_map = val_metrics['map50']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"New best model! mAP@50: {self.best_map:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info("Early stopping triggered")
                break
            
            # Log to tensorboard
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/mAP50', val_metrics.get('map50', 0), epoch)
            self.writer.add_scalar('Val/mAP50-95', val_metrics.get('map50_95', 0), epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            save_path = self.weights_dir / 'best.pt'
        else:
            save_path = self.weights_dir / f'epoch_{epoch}.pt'
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train baseline RGBA model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--modality', type=str, default='Combination3',
                       choices=['Combination1', 'Combination2', 'Combination3', 
                               'Combination4', 'Combination5', 'Combination6'],
                       help='RGB modality to use')
    parser.add_argument('--alpha_channel', type=str, default='range',
                       choices=['range', 'nearir', 'signal', 'reflec'],
                       help='Additional modality for alpha channel')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config, args)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
