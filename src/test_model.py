"""
Test script for validating the multimodal encoder and loss functions.
"""

import torch
import torch.nn.functional as F
from model import (
    MultimodalEncoder, 
    matryoshka_in_batch_negative_loss,
    compute_accuracy,
    ContrastiveLossWithHardNegatives,
    MultiLevelContrastiveLoss
)
from config import get_config
import numpy as np
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mock_inputs(batch_size: int = 4, seq_len: int = 128):
    """Create mock inputs for testing."""
    config = get_config()
    
    # Mock text inputs
    text_inputs = {
        'input_ids': torch.randint(1, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    # Mock image inputs (if needed)
    if hasattr(config, 'vision_model_max_patches'):
        # Create mock pixel values for vision model
        pixel_values = torch.randn(batch_size, 3, 448, 448)  # Standard image size
        text_inputs['pixel_values'] = pixel_values
    
    return text_inputs


def test_model_loading():
    """Test if the model loads correctly."""
    print("Testing model loading...")
    try:
        config = get_config()
        # Use a smaller model for testing if available
        config.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        model = MultimodalEncoder(config)
        print(f"✓ Model loaded successfully")
        print(f"  - Embedding dimension: {model.embedding_dim}")
        print(f"  - Device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with mock data."""
    print("\nTesting forward pass...")
    try:
        batch_size = 2
        query_inputs = create_mock_inputs(batch_size)
        target_inputs = create_mock_inputs(batch_size)
        
        # Move to same device as model
        device = next(model.parameters()).device
        for key in query_inputs:
            query_inputs[key] = query_inputs[key].to(device)
            target_inputs[key] = target_inputs[key].to(device)
        
        with torch.no_grad():
            query_emb, target_emb = model(query_inputs, target_inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Query embeddings shape: {query_emb.shape}")
        print(f"  - Target embeddings shape: {target_emb.shape}")
        print(f"  - Query embeddings norm: {query_emb.norm(dim=-1).mean():.4f}")
        print(f"  - Target embeddings norm: {target_emb.norm(dim=-1).mean():.4f}")
        
        return query_emb, target_emb
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return None, None


def test_matryoshka_loss(query_emb, target_emb):
    """Test Matryoshka loss computation."""
    print("\nTesting Matryoshka loss...")
    try:
        config = get_config()
        mrl_dims = [256, 512, 1024]  # Test with smaller dimensions
        
        loss = matryoshka_in_batch_negative_loss(
            query_emb, target_emb, mrl_dims, temperature=0.07
        )
        
        print(f"✓ Matryoshka loss computed successfully")
        print(f"  - Loss value: {loss.item():.4f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        # Test with different batch sizes
        for batch_size in [2, 4, 8]:
            test_query = torch.randn(batch_size, query_emb.size(-1))
            test_target = torch.randn(batch_size, target_emb.size(-1))
            test_query = F.normalize(test_query, p=2, dim=-1)
            test_target = F.normalize(test_target, p=2, dim=-1)
            
            test_loss = matryoshka_in_batch_negative_loss(
                test_query, test_target, mrl_dims[:2]  # Use smaller dims for testing
            )
            print(f"  - Batch size {batch_size}: loss = {test_loss.item():.4f}")
        
        return loss
    except Exception as e:
        print(f"✗ Matryoshka loss failed: {e}")
        return None


def test_accuracy_computation(query_emb, target_emb):
    """Test accuracy computation."""
    print("\nTesting accuracy computation...")
    try:
        acc_1 = compute_accuracy(query_emb, target_emb, top_k=1)
        acc_3 = compute_accuracy(query_emb, target_emb, top_k=3)
        
        print(f"✓ Accuracy computation successful")
        print(f"  - Top-1 accuracy: {acc_1:.4f}")
        print(f"  - Top-3 accuracy: {acc_3:.4f}")
        
        return acc_1, acc_3
    except Exception as e:
        print(f"✗ Accuracy computation failed: {e}")
        return None, None


def test_advanced_losses(query_emb, target_emb):
    """Test advanced loss functions."""
    print("\nTesting advanced loss functions...")
    
    try:
        # Test ContrastiveLossWithHardNegatives
        hard_neg_loss = ContrastiveLossWithHardNegatives(temperature=0.07)
        loss_hard = hard_neg_loss(query_emb, target_emb)
        print(f"✓ Hard negatives loss: {loss_hard.item():.4f}")
        
        # Test MultiLevelContrastiveLoss
        multi_level_loss = MultiLevelContrastiveLoss(
            levels=[256, 512], temperature=0.07
        )
        loss_dict = multi_level_loss(query_emb, target_emb)
        print(f"✓ Multi-level loss: {loss_dict['total_loss'].item():.4f}")
        for key, value in loss_dict.items():
            if key != 'total_loss':
                print(f"  - {key}: {value.item():.4f}")
        
    except Exception as e:
        print(f"✗ Advanced losses failed: {e}")


def test_gradient_flow(model, query_emb, target_emb):
    """Test gradient flow through the model."""
    print("\nTesting gradient flow...")
    try:
        # Enable gradients
        model.train()
        
        # Create loss and compute gradients
        config = get_config()
        loss = matryoshka_in_batch_negative_loss(
            query_emb, target_emb, config.mrl_dims[:2], temperature=0.07
        )
        
        loss.backward()
        
        # Check if gradients exist
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if len(grad_norms) <= 3:  # Print first few
                    print(f"  - {name}: grad norm = {grad_norm:.6f}")
        
        print(f"✓ Gradient flow successful")
        print(f"  - Parameters with gradients: {len(grad_norms)}")
        print(f"  - Average gradient norm: {np.mean(grad_norms):.6f}")
        
    except Exception as e:
        print(f"✗ Gradient flow failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MULTIMODAL ENCODER MODEL VALIDATION")
    print("=" * 60)
    
    # Test 1: Model loading
    model = test_model_loading()
    if model is None:
        print("\n❌ Model loading failed. Cannot proceed with other tests.")
        return
    
    # Test 2: Forward pass
    query_emb, target_emb = test_forward_pass(model)
    if query_emb is None:
        print("\n❌ Forward pass failed. Cannot proceed with other tests.")
        return
    
    # Test 3: Loss computation
    loss = test_matryoshka_loss(query_emb, target_emb)
    
    # Test 4: Accuracy computation
    test_accuracy_computation(query_emb, target_emb)
    
    # Test 5: Advanced losses
    test_advanced_losses(query_emb, target_emb)
    
    # Test 6: Gradient flow
    test_gradient_flow(model, query_emb, target_emb)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()