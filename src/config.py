"""
Configuration file for multimodal search intent alignment model training.
Contains all hyperparameters and model configurations.
"""

import torch
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration class containing all hyperparameters."""
    
    # Model configuration
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # Path to pretrained Qwen2.5-VL model
    max_text_length: int = 1024  # Maximum sequence length for text inputs
    embedding_dim: int = 2560  # Default embedding dimension for Qwen2.5-VL-3B
    vision_model_max_patches: int = 256  # Maximum number of image patches
    max_image_size: int = 448  # Maximum image size for processing
    
    # Data paths
    data_path: str = "data/train_data.jsonl"
    video_meta_path: str = "data/video_meta.jsonl"
    checkpoint_dir: str = "checkpoints"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip_norm: float = 1.0
    
    # Matryoshka Representation Learning parameters
    mrl_dims: List[int] = None  # Will be set to [128, 256, 512, 896] if None
    mrl_loss_weight: float = 1.0
    temperature: float = 0.07  # Temperature for contrastive loss
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision
    gradient_accumulation_steps: int = 1
    
    # Logging and saving
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Data processing
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Model specific for Qwen2.5-VL
    freeze_backbone: bool = False  # Whether to freeze the backbone model
    use_lora: bool = False  # Whether to use LoRA fine-tuning
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Qwen2.5-VL specific parameters
    trust_remote_code: bool = True  # Required for Qwen models
    torch_dtype: str = "bfloat16"  # Recommended dtype for Qwen2.5-VL
    attn_implementation: str = "flash_attention_2"  # Use flash attention for efficiency
    
    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.mrl_dims is None:
            # Updated MRL dimensions for Qwen2.5-VL
            self.mrl_dims = [256, 512, 1024, 2048, self.embedding_dim]
        
        # Ensure checkpoint directory exists
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)


# Global configuration instance
config = TrainingConfig()


def update_config(**kwargs):
    """Update configuration parameters."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")


def get_config():
    """Get the current configuration."""
    return config


# For backwards compatibility and easy access
MODEL_PATH = config.model_path
DATA_PATH = config.data_path
VIDEO_META_PATH = config.video_meta_path
CHECKPOINT_DIR = config.checkpoint_dir
DEVICE = config.device
BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
NUM_EPOCHS = config.num_epochs
MRL_DIMS = config.mrl_dims