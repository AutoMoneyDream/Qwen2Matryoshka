"""
Multimodal Search Intent Alignment Model Training Package

This package contains modules for training a multimodal model based on Qwen2-VL
for search query-target semantic alignment using Matryoshka Representation Learning.
"""

__version__ = "1.0.0"
__author__ = "Vlm4rec Project"

from .config import config, get_config, update_config

__all__ = [
    "config",
    "get_config", 
    "update_config"
]