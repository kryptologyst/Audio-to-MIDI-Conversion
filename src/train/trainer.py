"""Training script for Audio-to-MIDI conversion models."""

import warnings
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ..utils.device import get_device, set_seed, get_device_info
from ..models.baseline import BaselineAudioToMIDI, BaselineConfig
from ..models.deep_learning import create_model, DeepLearningConfig
from ..data.dataset import AudioMIDIDataset, SyntheticDataset, create_dataloader, DataConfig
from ..metrics.evaluation import AudioToMIDIMetrics, ModelEvaluator


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


class AudioToMIDITrainer:
    """
    Trainer for Audio-to-MIDI conversion models.
    
    This class handles the training loop, validation, checkpointing,
    and logging for Audio-to-MIDI models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: torch.device,
        logger: logging.Logger,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize metrics
        self.metrics = AudioToMIDIMetrics()
        self.evaluator = ModelEvaluator(self.metrics)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on model type."""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
            if self.model.config.model_type == 'transformer':
                return nn.CrossEntropyLoss(ignore_index=0)
            else:
                return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            audio_features = batch["audio_features"].to(self.device)
            midi_events = batch["midi_events"].to(self.device)
            midi_pitches = batch["midi_pitches"].to(self.device)
            midi_velocities = batch["midi_velocities"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward'):
                outputs = self.model(audio_features)
                
                # Compute loss
                loss = self._compute_loss(outputs, midi_events, midi_pitches, midi_velocities)
            else:
                # For baseline models, compute loss differently
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                audio_features = batch["audio_features"].to(self.device)
                midi_events = batch["midi_events"].to(self.device)
                midi_pitches = batch["midi_pitches"].to(self.device)
                midi_velocities = batch["midi_velocities"].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(audio_features)
                    
                    # Compute loss
                    loss = self._compute_loss(outputs, midi_events, midi_pitches, midi_velocities)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        midi_events: torch.Tensor,
        midi_pitches: torch.Tensor,
        midi_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from model outputs."""
        # Event loss
        event_loss = self.criterion(
            outputs["event_logits"].view(-1, outputs["event_logits"].size(-1)),
            midi_events.view(-1)
        )
        
        # Pitch loss
        pitch_loss = self.criterion(
            outputs["pitch_logits"].view(-1, outputs["pitch_logits"].size(-1)),
            midi_pitches.view(-1)
        )
        
        # Velocity loss
        velocity_loss = self.criterion(
            outputs["velocity_logits"].view(-1, outputs["velocity_logits"].size(-1)),
            midi_velocities.view(-1)
        )
        
        # Combined loss
        total_loss = event_loss + pitch_loss + velocity_loss
        
        return total_loss
    
    def train(self, num_epochs: int, save_dir: Path) -> None:
        """Train the model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        writer = SummaryWriter(save_dir / 'tensorboard')
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Tensorboard logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_dir / 'best_model.pth')
                self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pth')
        
        writer.close()
        self.logger.info("Training completed")
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Audio-to-MIDI model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    
    # Log device info
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create datasets
    data_config = DataConfig(**config.data)
    
    if config.get('use_synthetic', False):
        logger.info("Using synthetic dataset")
        train_dataset = SyntheticDataset(data_config, num_samples=1000)
        val_dataset = SyntheticDataset(data_config, num_samples=200)
    else:
        logger.info(f"Loading datasets from {args.data_dir}")
        train_dataset = AudioMIDIDataset(args.data_dir, data_config, split='train')
        val_dataset = AudioMIDIDataset(args.data_dir, data_config, split='val')
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.get('num_workers', 4),
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.get('num_workers', 4),
    )
    
    # Create model
    if config.model.type == 'baseline':
        logger.info("Creating baseline model")
        model_config = BaselineConfig(**config.model.baseline)
        model = BaselineAudioToMIDI(model_config)
    else:
        logger.info(f"Creating {config.model.type} model")
        model_config = DeepLearningConfig(**config.model.deep_learning)
        model = create_model(model_config)
    
    # Create trainer
    trainer = AudioToMIDITrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config.training,
        device=device,
        logger=logger,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # Train model
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_dir=output_dir,
    )


if __name__ == '__main__':
    main()
