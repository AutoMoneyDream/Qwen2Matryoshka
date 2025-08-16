# Qwen2Matryoshka: Multimodal Search Alignment with Matryoshka Embeddings

This project provides a state-of-the-art training framework for building a powerful multimodal search alignment model. It leverages the **Qwen2.5-VL** large vision-language model and implements **Matryoshka Representation Learning (MRL)** to produce highly efficient, multi-granularity embeddings.

The resulting model can understand and encode both text and visual data (videos/images) into a shared semantic space, making it ideal for tasks like cross-modal retrieval, search intent alignment, and recommendation systems. The Matryoshka embeddings allow for flexible, adaptive performance at inference time, where shorter embedding vectors can be used for faster, lower-cost operations without significant accuracy loss.

## ✨ Core Features

- **Powerful Backbone**: Built on the state-of-the-art **Qwen2.5-VL-3B-Instruct** model for robust multimodal understanding.
- **Matryoshka Representation Learning (MRL)**: Trains nested embeddings at multiple dimensions (e.g., 256, 512, 1024, 2048) simultaneously from a single model.
- **In-Batch Contrastive Loss**: Employs an efficient in-batch negative sampling strategy for contrastive learning.
- **Advanced Optimizations**:
    - **Flash Attention 2**: Integrated for faster and more memory-efficient attention computation.
    - **Mixed Precision Training**: Supports Automatic Mixed Precision (AMP) with `bfloat16` for significant training speedup.
    - **Distributed Training**: Full support for Distributed Data Parallel (DDP) to scale training across multiple GPUs.
    - **Parameter-Efficient Fine-Tuning (PEFT)**: Optional support for **LoRA**.
- **Flexible & Configurable**: A centralized configuration file (`src/config.py`) allows for easy management of all hyperparameters.
- **Monitoring**: Integrated support for **TensorBoard** and **Weights & Biases (W&B)**.

## 📁 File Structure

```
Qwen2Matryoshka/
├───data/
│   ├───train_data.jsonl      # Training pairs (query-target)
│   └───video_meta.jsonl      # Metadata for video/image content
├───checkpoints/              # Directory for saving model checkpoints
├───logs/                     # Directory for TensorBoard logs
├───src/
│   ├───config.py             # Centralized configuration for all parameters
│   ├───data_loader.py        # Multimodal dataset and dataloader
│   ├───model.py              # Qwen2.5-VL model wrapper and MRL loss
│   ├───train.py              # Core training and evaluation pipeline
│   └───optimization.py       # Optimization utilities
├───requirements.txt          # Python dependencies
└───train_runner.py           # Main script to launch advanced training
└───train_example.py          # A simple script to test the pipeline
```

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd Qwen2Matryoshka
    ```

2.  **Install dependencies**:
    It is highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `flash-attn` is commented out in `requirements.txt`. For maximum performance on compatible hardware (NVIDIA Ampere/Hopper GPUs), install it manually.*

## 🚀 How to Train

All configurations can be adjusted in `src/config.py`.

### 1. Simple Example (Recommended for a first run)

This script runs a minimal training loop on a tiny, auto-generated dataset to verify that the environment and pipeline are working correctly.

```bash
python train_example.py
```

### 2. Advanced Training

This is the main script for running a full training session.

**Basic Training on a Single GPU:**
```bash
python train_runner.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --mixed_precision \
    --use_tensorboard \
    --experiment_name "qwen2.5-mrl-run-1"
```

**Distributed Training on Multiple GPUs:**
To run on 4 GPUs, for example:
```bash
python train_runner.py \
    --distributed \
    --world_size 4 \
    --batch_size 32
```

## 📈 Monitoring

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir logs
```
Navigate to `http://localhost:6006` in your browser to view real-time graphs.

## 🔧 Troubleshooting

### 1. Hugging Face HTTP 429 Error

If you encounter an `HTTP Error 429 Too Many Requests` while downloading the model, it means you have been rate-limited by Hugging Face Hub. This typically happens with frequent, unauthenticated requests.

**Solution**: Authenticate with a Hugging Face access token.

1.  **Get a Token**: Create a new access token on the [Hugging Face website](https://huggingface.co/settings/tokens).
2.  **Login via CLI**: Run the following command in your terminal and paste your token when prompted. This will store your token locally for future use.
    ```bash
    huggingface-cli login
    ```
    After logging in, re-run your training script.

### 2. BFloat16 Mixed Precision Issues

When using `bfloat16` for mixed-precision training, you might encounter compatibility errors with `GradScaler`, especially on certain hardware like AMD GPUs.

**Solution**: The training script automatically handles this. The `src/config.py` file detects if `torch_dtype` is set to `"bfloat16"` and, if so, disables `GradScaler` (`use_grad_scaler = False`). Mixed precision is still used via `torch.amp.autocast`, but without gradient scaling, which is generally safe for `bfloat16` due to its larger numerical range compared to `float16`.

```python
# From src/config.py
if self.torch_dtype == "bfloat16" and self.use_grad_scaler:
    logger.warning("BFloat16 detected. Disabling GradScaler for compatibility.")
    self.use_grad_scaler = False
```
This ensures training stability without manual intervention.
