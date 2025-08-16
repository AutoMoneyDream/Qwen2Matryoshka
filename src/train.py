"""
Core training pipeline for multimodal search intent alignment model.
Implements SOTA training techniques with Matryoshka Representation Learning.
"""

import os
import sys
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from tqdm import tqdm
import wandb

from .config import get_config
from .data_loader import MultimodalDataset, create_dataloader
from .model import (
    MultimodalEncoder, 
    matryoshka_in_batch_negative_loss,
    compute_accuracy,
    ContrastiveLossWithHardNegatives,
    MultiLevelContrastiveLoss
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Advanced trainer for multimodal search intent alignment with SOTA optimizations.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = torch.device(self.config.device)
        self.global_step = 0
        self.best_metric = 0.0
        
        # Setup distributed training if available
        self.is_distributed = self._setup_distributed()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize components
        self._setup_logging()
        self._set_seed()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_monitoring()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=self.config.mixed_precision and self.config.use_grad_scaler)
        
        # Gradient accumulation
        self.accumulation_steps = self.config.gradient_accumulation_steps
        
    def _setup_distributed(self) -> bool:
        """Setup distributed training if available."""
        if 'RANK' in os.environ:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            return True
        return False
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        if self.local_rank == 0:
            # Create checkpoint directory
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            
            # Setup tensorboard
            if self.config.use_tensorboard:
                self.tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.config.log_dir, f"run_{int(time.time())}")
                )
            
            # Setup wandb
            if self.config.use_wandb:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"{self.config.experiment_name}_{int(time.time())}",
                    config=self.config.__dict__
                )
    
    def _set_seed(self):
        """Set random seed for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Ensure deterministic behavior
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        # Create datasets
        train_dataset = MultimodalDataset(
            data_path=self.config.train_data_path,
            video_meta_path=self.config.video_meta_path,
            config=self.config,
            split='train'
        )
        
        if self.config.eval_data_path:
            eval_dataset = MultimodalDataset(
                data_path=self.config.eval_data_path,
                video_meta_path=self.config.video_meta_path,
                config=self.config,
                split='eval'
            )
        else:
            eval_dataset = None
        
        # Create data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            distributed=self.is_distributed
        )
        
        if eval_dataset:
            self.eval_loader = create_dataloader(
                eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                distributed=self.is_distributed
            )
        else:
            self.eval_loader = None
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    def _setup_model(self):
        """Setup model and move to device."""
        logger.info("Setting up model...")
        
        self.model = MultimodalEncoder(self.config)
        self.model.to(self.device)
        
        # Setup distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        
        # Setup loss functions
        self.mrl_loss_fn = lambda q, t: matryoshka_in_batch_negative_loss(
            q, t, 
            mrl_dims=self.config.mrl_dims,
            temperature=self.config.temperature,
            mrl_weights=self.config.mrl_weights
        )
        
        if self.config.use_hard_negatives:
            self.hard_negative_loss_fn = ContrastiveLossWithHardNegatives(
                temperature=self.config.temperature,
                margin=self.config.margin
            )
        
        if self.config.use_multi_level_loss:
            self.multi_level_loss_fn = MultiLevelContrastiveLoss(
                levels=self.config.mrl_dims,
                temperature=self.config.temperature,
                level_weights=self.config.mrl_weights
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        # Get model parameters
        if self.is_distributed:
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
        
        # Filter trainable parameters
        trainable_params = [p for p in model_params if p.requires_grad]
        
        # Setup optimizer
        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Setup scheduler
        if self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs * len(self.train_loader),
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'linear':
            total_steps = self.config.num_epochs * len(self.train_loader)
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {self.config.optimizer}")
        logger.info(f"Scheduler: {self.config.scheduler}")
    
    def _setup_monitoring(self):
        """Setup monitoring and evaluation metrics."""
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Progress bar
        if self.local_rank == 0:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                leave=False
            )
        else:
            pbar = self.train_loader
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device
                def move_to_device(inputs):
                    if isinstance(inputs, dict):
                        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    elif isinstance(inputs, torch.Tensor):
                        return inputs.to(self.device)
                    else:
                        return inputs
                
                query_inputs = move_to_device(batch['query_inputs'])
                target_inputs = move_to_device(batch['target_inputs'])
                
                # Forward pass with mixed precision
                with autocast(self.device.type, enabled=self.config.mixed_precision):
                    query_embeddings, target_embeddings = self.model(query_inputs, target_inputs)
                    
                    # Compute loss
                    if self.config.use_multi_level_loss:
                        loss_dict = self.multi_level_loss_fn(query_embeddings, target_embeddings)
                        loss = loss_dict['total_loss']
                    else:
                        loss = self.mrl_loss_fn(query_embeddings, target_embeddings)
                    
                    # Add hard negative mining loss if enabled
                    if self.config.use_hard_negatives:
                        hard_loss = self.hard_negative_loss_fn(
                            query_embeddings, 
                            target_embeddings,
                            hard_negative_ratio=self.config.hard_negative_ratio
                        )
                        loss = loss + self.config.hard_negative_weight * hard_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.global_step += 1
                
                # Compute metrics
                with torch.no_grad():
                    accuracy = compute_accuracy(query_embeddings, target_embeddings)
                    total_loss += loss.item() * self.accumulation_steps
                    total_accuracy += accuracy
                    num_batches += 1
                
                # Update progress bar
                if self.local_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                        'acc': f"{accuracy:.3f}",
                        'lr': f"{current_lr:.2e}"
                    })
                
                # Log to tensorboard/wandb
                if self.local_rank == 0 and self.global_step % self.config.log_interval == 0:
                    self._log_metrics({
                        'train/loss': loss.item() * self.accumulation_steps,
                        'train/accuracy': accuracy,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.global_step
                    })
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        if self.local_rank == 0:
            pbar = tqdm(self.eval_loader, desc="Evaluating", leave=False)
        else:
            pbar = self.eval_loader
        
        for batch in pbar:
            try:
                # Move batch to device
                def move_to_device(inputs):
                    if isinstance(inputs, dict):
                        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    elif isinstance(inputs, torch.Tensor):
                        return inputs.to(self.device)
                    else:
                        return inputs
                
                query_inputs = move_to_device(batch['query_inputs'])
                target_inputs = move_to_device(batch['target_inputs'])
                
                # Forward pass
                with autocast(self.device.type, enabled=self.config.mixed_precision):
                    query_embeddings, target_embeddings = self.model(query_inputs, target_inputs)
                    loss = self.mrl_loss_fn(query_embeddings, target_embeddings)
                
                # Compute metrics
                accuracy = compute_accuracy(query_embeddings, target_embeddings)
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                if self.local_rank == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{accuracy:.3f}"
                    })
                
            except Exception as e:
                logger.error(f"Error in evaluation step: {e}")
                continue
        
        # Calculate metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb."""
        if hasattr(self, 'tb_writer'):
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.global_step)
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
        
        # Get model state dict
        if self.is_distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")
        logger.info(f"Total steps: {self.config.num_epochs * len(self.train_loader)}")
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            start_epoch = self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            if self.local_rank == 0:
                logger.info(f"\n=== Epoch {epoch+1}/{self.config.num_epochs} ===")
            
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate(epoch)
            
            # Log epoch metrics
            if self.local_rank == 0:
                # Update metrics history
                self.metrics['train_loss'].append(train_metrics['loss'])
                self.metrics['train_accuracy'].append(train_metrics['accuracy'])
                self.metrics['learning_rate'].append(train_metrics['learning_rate'])
                
                if eval_metrics:
                    self.metrics['eval_loss'].append(eval_metrics['loss'])
                    self.metrics['eval_accuracy'].append(eval_metrics['accuracy'])
                
                # Log to monitoring
                log_dict = {
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'train/epoch_lr': train_metrics['learning_rate']
                }
                
                if eval_metrics:
                    log_dict.update({
                        'eval/epoch_loss': eval_metrics['loss'],
                        'eval/epoch_accuracy': eval_metrics['accuracy']
                    })
                
                self._log_metrics(log_dict)
                
                # Print summary
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.3f}")
                if eval_metrics:
                    logger.info(f"Eval Loss: {eval_metrics['loss']:.4f}, "
                              f"Eval Acc: {eval_metrics['accuracy']:.3f}")
                
                # Check if best model
                current_metric = eval_metrics.get('accuracy', train_metrics['accuracy'])
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    logger.info(f"New best metric: {self.best_metric:.3f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0:
                    all_metrics = {**train_metrics}
                    if eval_metrics:
                        all_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                    self.save_checkpoint(epoch, all_metrics, is_best)
        
        if self.local_rank == 0:
            logger.info("Training completed!")
            
            # Save final model
            final_metrics = {
                'final_train_loss': self.metrics['train_loss'][-1],
                'final_train_accuracy': self.metrics['train_accuracy'][-1],
                'best_metric': self.best_metric
            }
            if self.metrics['eval_loss']:
                final_metrics.update({
                    'final_eval_loss': self.metrics['eval_loss'][-1],
                    'final_eval_accuracy': self.metrics['eval_accuracy'][-1]
                })
            
            self.save_checkpoint(self.config.num_epochs - 1, final_metrics, False)
            
            # Close monitoring
            if hasattr(self, 'tb_writer'):
                self.tb_writer.close()
            if self.config.use_wandb:
                wandb.finish()


def main():
    """Main training function."""
    config = get_config()
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()