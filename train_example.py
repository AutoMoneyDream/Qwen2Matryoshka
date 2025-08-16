#!/usr/bin/env python3
"""
Example training script demonstrating SOTA multimodal training with Qwen2.5-VL.
This script showcases the complete training pipeline with optimizations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config, update_config
from src.train import Trainer
from src.data_loader import create_sample_data
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run example training with optimized settings."""
    
    # Get configuration
    config = get_config()
    
    # Update configuration for demonstration
    config.batch_size = 4  # Small batch size for demo
    config.num_epochs = 3  # Short training for demo
    config.learning_rate = 1e-4
    config.mixed_precision = True
    config.use_multi_level_loss = True
    config.use_tensorboard = True
    config.use_wandb = False  # Disable wandb for demo
    config.save_interval = 1  # Save every epoch
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create sample data for demonstration
    logger.info("Creating sample training data...")
    create_sample_data(
        config.train_data_path,
        config.video_meta_path,
        num_samples=100  # Small dataset for quick demo
    )
    
    logger.info("=== Starting SOTA Multimodal Training Demo ===")
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"MRL dimensions: {config.mrl_dims}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info("=" * 50)
    
    try:
        # Initialize trainer
        trainer = Trainer(config)
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("This is expected if Qwen2.5-VL model dependencies are not installed.")
        logger.info("\nTo run this training, install required dependencies:")
        logger.info("pip install torch transformers Pillow tqdm tensorboard")
        logger.info("pip install peft wandb einops  # Optional for advanced features")
        
        # Show what would happen
        logger.info("\n=== Training Process Overview ===")
        logger.info("1. Load Qwen2.5-VL model with LoRA/full fine-tuning")
        logger.info("2. Setup Matryoshka Representation Learning loss")
        logger.info("3. Train with in-batch negative sampling")
        logger.info("4. Use mixed precision and gradient accumulation")
        logger.info("5. Monitor with TensorBoard/W&B")
        logger.info("6. Save checkpoints with best model selection")
        logger.info("7. Apply SOTA optimizations (EMA, adaptive scheduling)")


if __name__ == "__main__":
    main()