"""
Multimodal encoder model based on Qwen2.5-VL with Matryoshka Representation Learning.
Implements SOTA techniques for multimodal search intent alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
    AutoModelForCausalLM
)
from typing import Dict, List, Optional, Tuple, Union
import math
import logging
from .config import get_config

logger = logging.getLogger(__name__)


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder using Qwen2.5-VL for feature extraction.
    Supports both text and image inputs with L2 normalized embeddings.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or get_config()
        
        # Load pretrained Qwen2.5-VL model and processor
        logger.info(f"Loading Qwen2.5-VL model from {self.config.model_path}")
        
        # Load model with optimized settings and fallback mechanism
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation=self.config.attn_implementation,
            )
            logger.info(f"Model loaded successfully with {self.config.attn_implementation} attention")
        except Exception as e:
            logger.warning(f"Failed to load model with {self.config.attn_implementation} attention: {e}")
            logger.info("Falling back to eager attention implementation")
            
            # Fallback to eager attention
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation="eager",
            )
            self.config.attn_implementation = "eager"
            logger.info("Model loaded successfully with eager attention (fallback)")
        
        # Load processor
        self.processor = Qwen2VLProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        self.config.embedding_dim = self.embedding_dim
        
        # Projection layer for embedding normalization and dimensionality
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim)
        )
        
        # Initialize projection weights
        self._init_projection_weights()
        
        # Freeze backbone if specified
        if self.config.freeze_backbone:
            self._freeze_backbone()
        
        # Setup LoRA if specified
        if self.config.use_lora:
            self._setup_lora()
    
    def _init_projection_weights(self):
        """Initialize projection layer weights using Xavier initialization."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _freeze_backbone(self):
        """Freeze the backbone model parameters."""
        logger.info("Freezing backbone model parameters")
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _setup_lora(self):
        """Setup LoRA fine-tuning for efficient training."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configuration applied successfully")
        except ImportError:
            logger.warning("PEFT library not installed. LoRA will be disabled.")
            self.config.use_lora = False
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings with attention mask consideration.
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_embeddings: [batch_size, hidden_size]
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Apply mask and sum
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from Qwen2.5-VL model.
        
        Args:
            inputs: Dictionary containing input_ids, attention_mask, and optionally pixel_values
            
        Returns:
            features: [batch_size, hidden_size]
        """
        # Forward pass through the model
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs = self.model.model(**inputs, output_hidden_states=True)
        
        # Get last hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        attention_mask = inputs.get('attention_mask')
        
        if attention_mask is not None:
            # Use mean pooling with attention mask
            pooled_features = self._mean_pooling(last_hidden_states, attention_mask)
        else:
            # Simple mean pooling across sequence dimension
            pooled_features = last_hidden_states.mean(dim=1)
        
        return pooled_features
    
    def forward(self, 
                query_inputs: Dict[str, torch.Tensor], 
                target_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both query and target inputs.
        
        Args:
            query_inputs: Processed query inputs (text and/or images)
            target_inputs: Processed target inputs (text and/or images)
            
        Returns:
            Tuple of (query_embeddings, target_embeddings), both L2 normalized
        """
        # Extract features
        query_features = self._extract_features(query_inputs)
        target_features = self._extract_features(target_inputs)
        
        # Project to final embedding space
        query_embeddings = self.projection(query_features)
        target_embeddings = self.projection(target_features)
        
        # L2 normalization for cosine similarity
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)
        
        return query_embeddings, target_embeddings
    
    def encode_single(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single input (query or target).
        
        Args:
            inputs: Processed inputs
            
        Returns:
            embeddings: L2 normalized embeddings
        """
        features = self._extract_features(inputs)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=-1)


def matryoshka_in_batch_negative_loss(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    mrl_dims: List[int],
    temperature: float = 0.07,
    mrl_weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Compute Matryoshka Representation Learning loss with in-batch negative sampling.
    
    This function implements the core MRL loss by computing contrastive loss at multiple
    representation dimensions, encouraging the model to learn nested representations.
    
    Args:
        query_embeddings: [batch_size, embedding_dim] - Query embeddings
        target_embeddings: [batch_size, embedding_dim] - Target embeddings  
        mrl_dims: List of dimensions to compute loss at (e.g., [256, 512, 1024, 2048])
        temperature: Temperature scaling for contrastive loss
        mrl_weights: Optional weights for each MRL dimension
        
    Returns:
        total_loss: Weighted sum of losses across all MRL dimensions
    """
    batch_size = query_embeddings.size(0)
    device = query_embeddings.device
    total_loss = 0.0
    
    # Default equal weights if not provided
    if mrl_weights is None:
        mrl_weights = [1.0] * len(mrl_dims)
    
    # Ensure mrl_dims are sorted and valid
    max_dim = min(query_embeddings.size(-1), target_embeddings.size(-1))
    valid_dims = [d for d in sorted(mrl_dims) if d <= max_dim]
    
    if not valid_dims:
        raise ValueError(f"No valid MRL dimensions. Max available: {max_dim}, requested: {mrl_dims}")
    
    # Create labels for contrastive loss (diagonal elements are positive pairs)
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    
    # Compute loss for each MRL dimension
    for i, dim in enumerate(valid_dims):
        # Truncate embeddings to current dimension
        query_emb_truncated = query_embeddings[:, :dim]  # [batch_size, dim]
        target_emb_truncated = target_embeddings[:, :dim]  # [batch_size, dim]
        
        # Re-normalize after truncation (important for cosine similarity)
        query_emb_truncated = F.normalize(query_emb_truncated, p=2, dim=-1)
        target_emb_truncated = F.normalize(target_emb_truncated, p=2, dim=-1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        # Each (i,j) entry is cosine similarity between query_i and target_j
        similarity_matrix = torch.matmul(query_emb_truncated, target_emb_truncated.T)
        
        # Scale by temperature
        logits = similarity_matrix / temperature
        
        # Compute contrastive loss
        # For query-to-target direction
        loss_q2t = F.cross_entropy(logits, labels)
        
        # For target-to-query direction (symmetric loss)
        loss_t2q = F.cross_entropy(logits.T, labels)
        
        # Average bidirectional loss
        contrastive_loss = (loss_q2t + loss_t2q) / 2
        
        # Add weighted loss
        weight = mrl_weights[i] if i < len(mrl_weights) else 1.0
        total_loss += weight * contrastive_loss
    
    # Average across all dimensions
    return total_loss / len(valid_dims)


def compute_accuracy(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Compute retrieval accuracy (top-k accuracy).
    
    Args:
        query_embeddings: [batch_size, embedding_dim]
        target_embeddings: [batch_size, embedding_dim]
        top_k: Number of top predictions to consider
        
    Returns:
        accuracy: Top-k accuracy as a float
    """
    batch_size = query_embeddings.size(0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(query_embeddings, target_embeddings.T)
    
    # Get top-k predictions
    _, top_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
    
    # Check if correct target is in top-k for each query
    correct_labels = torch.arange(batch_size, device=query_embeddings.device).unsqueeze(1)
    correct_predictions = (top_indices == correct_labels).any(dim=1)
    
    return correct_predictions.float().mean().item()


class ContrastiveLossWithHardNegatives(nn.Module):
    """
    Advanced contrastive loss with hard negative mining for improved training.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, 
                query_embeddings: torch.Tensor, 
                target_embeddings: torch.Tensor,
                hard_negative_ratio: float = 0.3) -> torch.Tensor:
        """
        Compute contrastive loss with hard negative mining.
        
        Args:
            query_embeddings: [batch_size, embedding_dim]
            target_embeddings: [batch_size, embedding_dim]
            hard_negative_ratio: Ratio of hardest negatives to include
            
        Returns:
            loss: Computed contrastive loss
        """
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_embeddings, target_embeddings.T) / self.temperature
        
        # Create mask for positive pairs (diagonal)
        positive_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        
        # Get negative similarities (off-diagonal elements)
        negative_similarities = similarity_matrix.masked_fill(positive_mask, float('-inf'))
        
        # Hard negative mining: select top similarities as hard negatives
        num_hard_negatives = max(1, int((batch_size - 1) * hard_negative_ratio))
        hard_negatives, _ = torch.topk(negative_similarities, k=num_hard_negatives, dim=1)
        
        # Positive similarities (diagonal elements)
        positive_similarities = similarity_matrix.diag().unsqueeze(1)
        
        # Combine positive and hard negative similarities
        combined_similarities = torch.cat([positive_similarities, hard_negatives], dim=1)
        
        # Labels: first column (index 0) contains positive pairs
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(combined_similarities, labels)
        
        return loss


class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level contrastive loss for hierarchical representation learning.
    """
    
    def __init__(self, 
                 levels: List[int] = [256, 512, 1024], 
                 temperature: float = 0.07,
                 level_weights: Optional[List[float]] = None):
        super().__init__()
        self.levels = levels
        self.temperature = temperature
        self.level_weights = level_weights or [1.0] * len(levels)
        
    def forward(self, 
                query_embeddings: torch.Tensor, 
                target_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-level contrastive loss.
        
        Args:
            query_embeddings: [batch_size, embedding_dim]
            target_embeddings: [batch_size, embedding_dim]
            
        Returns:
            Dictionary containing total loss and per-level losses
        """
        total_loss = 0.0
        level_losses = {}
        
        for i, level in enumerate(self.levels):
            if level > query_embeddings.size(-1):
                continue
                
            # Truncate to current level
            query_level = F.normalize(query_embeddings[:, :level], p=2, dim=-1)
            target_level = F.normalize(target_embeddings[:, :level], p=2, dim=-1)
            
            # Compute contrastive loss for this level
            level_loss = matryoshka_in_batch_negative_loss(
                query_level, target_level, [level], self.temperature
            )
            
            level_losses[f'level_{level}'] = level_loss
            total_loss += self.level_weights[i] * level_loss
        
        return {
            'total_loss': total_loss,
            **level_losses
        }