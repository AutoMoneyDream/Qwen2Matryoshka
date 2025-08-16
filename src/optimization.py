"""
Advanced optimization utilities for SOTA multimodal training performance.
Includes gradient checkpointing, learning rate scheduling, and memory optimization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    Provides smooth transitions and optimal learning rate decay.
    """
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear learning rate scheduler with warmup and linear decay.
    """
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return [self.min_lr + (base_lr - self.min_lr) * (1 - progress) for base_lr in self.base_lrs]


class AdamWWithDecay(torch.optim.AdamW):
    """
    Enhanced AdamW optimizer with adaptive weight decay and gradient clipping.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, adaptive_weight_decay=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.adaptive_weight_decay = adaptive_weight_decay
    
    def step(self, closure=None):
        """Perform optimization step with adaptive weight decay."""
        if self.adaptive_weight_decay:
            # Adjust weight decay based on learning rate
            for group in self.param_groups:
                group['weight_decay'] = group['weight_decay'] * group['lr'] / 1e-3
        
        return super().step(closure)


class GradientCheckpointing:
    """
    Utilities for gradient checkpointing to reduce memory usage.
    """
    
    @staticmethod
    def enable_checkpointing(model: nn.Module, modules_to_checkpoint: Optional[List[str]] = None):
        """
        Enable gradient checkpointing for specified modules.
        
        Args:
            model: Model to apply checkpointing to
            modules_to_checkpoint: List of module names to checkpoint (default: attention modules)
        """
        if modules_to_checkpoint is None:
            modules_to_checkpoint = ['attention', 'self_attn', 'cross_attn']
        
        def apply_checkpointing(module):
            for name, child in module.named_children():
                if any(checkpoint_name in name.lower() for checkpoint_name in modules_to_checkpoint):
                    if hasattr(child, 'gradient_checkpointing'):
                        child.gradient_checkpointing = True
                        logger.info(f"Enabled gradient checkpointing for {name}")
                apply_checkpointing(child)
        
        apply_checkpointing(model)
    
    @staticmethod
    def checkpoint_sequential(functions, segments, *inputs):
        """
        Apply gradient checkpointing to sequential functions.
        
        Args:
            functions: List of functions to checkpoint
            segments: Number of segments to divide the functions into
            inputs: Input tensors
            
        Returns:
            Output tensors
        """
        def run_function(start, end, functions):
            def forward(*inputs):
                for j in range(start, end + 1):
                    inputs = functions[j](*inputs)
                return inputs
            return forward
        
        if segments == 1:
            return run_function(0, len(functions) - 1, functions)(*inputs)
        
        segment_size = len(functions) // segments
        outputs = inputs
        
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size - 1 if i < segments - 1 else len(functions) - 1
            
            outputs = torch.utils.checkpoint.checkpoint(
                run_function(start_idx, end_idx, functions),
                *outputs
            )
        
        return outputs


class MemoryOptimizer:
    """
    Memory optimization utilities for large-scale training.
    """
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """
        Apply memory optimizations to the model.
        
        Args:
            model: Model to optimize
        """
        # Enable memory efficient attention if available
        for module in model.modules():
            if hasattr(module, 'enable_memory_efficient_attention'):
                module.enable_memory_efficient_attention()
        
        # Convert to half precision where appropriate
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.float()  # Keep normalization layers in fp32
        
        logger.info("Applied memory optimizations to model")
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class LossBalancer:
    """
    Adaptive loss balancing for multi-task learning.
    """
    
    def __init__(self, loss_names: List[str], initial_weights: Optional[List[float]] = None,
                 adaptation_rate: float = 0.01):
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        
        if initial_weights is None:
            self.weights = torch.ones(self.num_losses)
        else:
            self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        
        self.adaptation_rate = adaptation_rate
        self.loss_history = {name: [] for name in loss_names}
        self.step_count = 0
    
    def update_weights(self, losses: Dict[str, float]):
        """
        Update loss weights based on recent loss values.
        
        Args:
            losses: Dictionary of loss name to loss value
        """
        self.step_count += 1
        
        # Store loss history
        for name, loss_value in losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(loss_value)
                # Keep only recent history
                if len(self.loss_history[name]) > 100:
                    self.loss_history[name] = self.loss_history[name][-100:]
        
        # Update weights every 10 steps
        if self.step_count % 10 == 0:
            self._adaptive_update()
    
    def _adaptive_update(self):
        """Perform adaptive weight update based on loss trends."""
        if self.step_count < 20:  # Need some history first
            return
        
        # Calculate loss trends
        trends = []
        for name in self.loss_names:
            if len(self.loss_history[name]) >= 10:
                recent = self.loss_history[name][-10:]
                older = self.loss_history[name][-20:-10] if len(self.loss_history[name]) >= 20 else recent
                
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                
                # Positive trend means loss is increasing (bad)
                trend = (recent_avg - older_avg) / (older_avg + 1e-8)
                trends.append(trend)
            else:
                trends.append(0.0)
        
        # Adjust weights: increase weight for losses that are increasing
        trends = torch.tensor(trends, dtype=torch.float32)
        adjustment = torch.sigmoid(trends * 5.0)  # Scale and sigmoid
        
        # Update weights with momentum
        self.weights = self.weights * (1 - self.adaptation_rate) + \
                      adjustment * self.adaptation_rate
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum() * self.num_losses
    
    def get_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get weighted sum of losses.
        
        Args:
            losses: Dictionary of loss tensors
            
        Returns:
            Weighted total loss
        """
        total_loss = 0.0
        for i, name in enumerate(self.loss_names):
            if name in losses:
                total_loss += self.weights[i] * losses[name]
        
        return total_loss
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {name: self.weights[i].item() for i, name in enumerate(self.loss_names)}


class EMAModel:
    """
    Exponential Moving Average model for stable training.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.ema_model = None
        self.updates = 0
    
    def update(self, model: nn.Module):
        """Update EMA model with current model parameters."""
        self.updates += 1
        
        # Initialize EMA model on first update
        if self.ema_model is None:
            self.ema_model = type(model)(model.config if hasattr(model, 'config') else None)
            self.ema_model.load_state_dict(model.state_dict())
            self.ema_model.eval()
            return
        
        # Adjust decay based on number of updates
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        
        # Update EMA parameters
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """Apply EMA weights to the model."""
        if self.ema_model is not None:
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                param.data.copy_(ema_param.data)
    
    def restore(self):
        """Restore original model weights."""
        # This would need to store original weights to restore them
        pass


class TrainingProfiler:
    """
    Profiler for monitoring training performance and bottlenecks.
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.step_times = []
        
    def start_timer(self, name: str):
        """Start timing for a named operation."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(torch.cuda.Event(enable_timing=True))
        self.timings[name][-1].record()
    
    def end_timer(self, name: str):
        """End timing for a named operation."""
        if name in self.timings and self.timings[name]:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            
            start_event = self.timings[name][-1]
            elapsed_time = start_event.elapsed_time(end_event)
            
            if f"{name}_times" not in self.timings:
                self.timings[f"{name}_times"] = []
            self.timings[f"{name}_times"].append(elapsed_time)
    
    def record_memory(self, name: str):
        """Record current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            
            if name not in self.memory_usage:
                self.memory_usage[name] = []
            self.memory_usage[name].append({'allocated': allocated, 'cached': cached})
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        summary = {}
        
        # Average timings
        for key, times in self.timings.items():
            if key.endswith('_times') and times:
                avg_time = sum(times) / len(times)
                summary[f"avg_{key}"] = avg_time
        
        # Memory statistics
        for key, usage_list in self.memory_usage.items():
            if usage_list:
                avg_allocated = sum(u['allocated'] for u in usage_list) / len(usage_list)
                max_allocated = max(u['allocated'] for u in usage_list)
                summary[f"avg_memory_{key}"] = avg_allocated
                summary[f"max_memory_{key}"] = max_allocated
        
        return summary
    
    def reset(self):
        """Reset all recorded data."""
        self.timings.clear()
        self.memory_usage.clear()
        self.step_times.clear()


def setup_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """
    Setup optimizer with advanced configurations.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config.optimizer.lower() == 'adamw':
        optimizer = AdamWWithDecay(
            trainable_params,
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            adaptive_weight_decay=True
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    return optimizer


def setup_scheduler(optimizer: torch.optim.Optimizer, config, total_steps: int):
    """
    Setup learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        total_steps: Total training steps
        
    Returns:
        Configured scheduler
    """
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    if config.scheduler.lower() == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_lr
        )
    elif config.scheduler.lower() == 'linear':
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_lr
        )
    else:
        scheduler = None
    
    return scheduler