#!/usr/bin/env python3
"""
Training runner script with advanced optimizations for SOTA performance.
Includes distributed training, gradient checkpointing, and monitoring setup.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config, update_config
from src.train import Trainer
from src.data_loader import create_sample_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    destroy_process_group()


def run_training(rank: int, world_size: int, config):
    """Run training on a single GPU."""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    try:
        # Update config for current process
        config.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        
        # Initialize trainer
        trainer = Trainer(config)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        if world_size > 1:
            cleanup_distributed()


def create_data_if_missing(config):
    """Create sample data if training data doesn't exist."""
    if not os.path.exists(config.train_data_path):
        logger.info("Training data not found. Creating sample data...")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(config.train_data_path), exist_ok=True)
        
        # Create sample data
        create_sample_data(
            config.train_data_path,
            config.video_meta_path,
            num_samples=1000  # Create more samples for better training
        )
        
        logger.info("Sample data created successfully!")
    else:
        logger.info(f"Using existing training data: {config.train_data_path}")


def optimize_for_hardware(config):
    """Optimize configuration based on available hardware."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        logger.info(f"Detected {gpu_count} GPU(s) with {gpu_memory:.1f}GB memory each")
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 8:
            config.batch_size = 4
            config.eval_batch_size = 8
            config.gradient_accumulation_steps = 4
            logger.info("Reduced batch size for limited GPU memory")
        elif gpu_memory < 16:
            config.batch_size = 8
            config.eval_batch_size = 16
            config.gradient_accumulation_steps = 2
        else:
            config.batch_size = 16
            config.eval_batch_size = 32
            config.gradient_accumulation_steps = 1
            
        # Enable optimizations for better performance
        config.mixed_precision = True
        config.pin_memory = True
        
        # Adjust number of workers based on CPU cores
        config.num_workers = min(4, os.cpu_count())
        
    else:
        logger.warning("No CUDA GPUs detected. Training will be slow on CPU.")
        config.batch_size = 2
        config.eval_batch_size = 4
        config.mixed_precision = False
        config.pin_memory = False
        config.num_workers = 2


def setup_monitoring(config):
    """Setup monitoring and logging directories."""
    # Create necessary directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Setup wandb if requested
    if config.use_wandb:
        try:
            import wandb
            logger.info("W&B monitoring enabled")
        except ImportError:
            logger.warning("W&B not installed. Disabling wandb monitoring.")
            config.use_wandb = False
    
    # Setup tensorboard
    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            logger.info("TensorBoard monitoring enabled")
        except ImportError:
            logger.warning("TensorBoard not available. Disabling tensorboard monitoring.")
            config.use_tensorboard = False


def validate_config(config):
    """Validate configuration parameters."""
    assert config.learning_rate > 0, "Learning rate must be positive"
    assert config.batch_size > 0, "Batch size must be positive"
    assert config.num_epochs > 0, "Number of epochs must be positive"
    assert config.temperature > 0, "Temperature must be positive"
    assert len(config.mrl_dims) > 0, "MRL dimensions list cannot be empty"
    
    # Validate paths
    if not os.path.exists(os.path.dirname(config.train_data_path)):
        os.makedirs(os.path.dirname(config.train_data_path), exist_ok=True)
    
    logger.info("Configuration validation passed")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Multimodal Search Intent Alignment Model")
    
    # Training arguments
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--video_meta_path", type=str, help="Path to video metadata")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    
    # Optimization arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone model")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Monitoring arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Distributed training
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs for distributed training")
    
    # Other options
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample data for testing")
    parser.add_argument("--validate_only", action="store_true", help="Only validate config without training")
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config()
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Set resume checkpoint if provided
    if args.resume:
        config.resume_from_checkpoint = args.resume
    
    # Create sample data if requested
    if args.create_sample_data:
        create_data_if_missing(config)
        return
    
    # Optimize configuration for available hardware
    optimize_for_hardware(config)
    
    # Setup monitoring
    setup_monitoring(config)
    
    # Validate configuration
    validate_config(config)
    
    # Print configuration summary
    logger.info("=== Training Configuration ===")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"MRL dimensions: {config.mrl_dims}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"LoRA: {config.use_lora}")
    logger.info(f"Freeze backbone: {config.freeze_backbone}")
    logger.info("=" * 30)
    
    if args.validate_only:
        logger.info("Configuration validation complete. Exiting.")
        return
    
    # Create training data if it doesn't exist
    create_data_if_missing(config)
    
    # Run training
    world_size = args.world_size if args.distributed else 1
    
    if world_size > 1:
        logger.info(f"Starting distributed training on {world_size} GPUs")
        mp.spawn(
            run_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("Starting single GPU training")
        run_training(0, 1, config)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()