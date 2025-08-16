# Qwen2Matryoshka: Multimodal Search Alignment with Matryoshka Embeddings

This project provides a state-of-the-art training framework for building a powerful multimodal search alignment model. It leverages the **Qwen2.5-VL** large vision-language model and implements **Matryoshka Representation Learning (MRL)** to produce highly efficient, multi-granularity embeddings.

The resulting model can understand and encode both text and visual data (videos/images) into a shared semantic space, making it ideal for tasks like cross-modal retrieval, search intent alignment, and recommendation systems. The Matryoshka embeddings allow for flexible, adaptive performance at inference time, where shorter embedding vectors can be used for faster, lower-cost operations without significant accuracy loss.

## ✨ Core Features

- **Powerful Backbone**: Built on the state-of-the-art **Qwen2.5-VL-3B-Instruct** model for robust multimodal understanding.
- **Matryoshka Representation Learning (MRL)**: Trains nested embeddings at multiple dimensions (e.g., 256, 512, 1024, 2048) simultaneously from a single model, enabling adaptive inference performance.
- **In-Batch Contrastive Loss**: Employs an efficient in-batch negative sampling strategy for contrastive learning, with a symmetric query-to-target and target-to-query loss.
- **Advanced Optimizations**:
    - **Flash Attention 2**: Integrated for faster and more memory-efficient attention computation.
    - **Mixed Precision Training**: Supports Automatic Mixed Precision (AMP) with `bfloat16` for significant training speedup and reduced memory footprint.
    - **Distributed Training**: Full support for Distributed Data Parallel (DDP) to scale training across multiple GPUs.
    - **Parameter-Efficient Fine-Tuning (PEFT)**: Optional support for **LoRA** to fine-tune the model with a fraction of the memory and computational cost.
- **Flexible & Configurable**: A centralized configuration file (`src/config.py`) allows for easy management of all hyperparameters, paths, and training settings.
- **Monitoring**: Integrated support for **TensorBoard** and **Weights & Biases (W&B)** for real-time monitoring of training progress, losses, and other metrics.

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
│   ├───data_loader.py        # Multimodal dataset and dataloader implementation
│   ├───model.py              # Qwen2.5-VL model wrapper and MRL loss function
│   ├───train.py              # Core training and evaluation pipeline
│   └───optimization.py       # (Implicit) SOTA optimization utilities
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

## 📊 Data Preparation

The model expects two main data files in the `data/` directory:

1.  `train_data.jsonl`: Contains the query-target pairs for training. Each line is a JSON object.
    ```json
    {"query": "a user's search query", "target": "a relevant text document"}
    {"query": "find videos of cats playing piano", "target": "video_id_123"}
    {"query": "video_id_456", "target": "video_id_789"}
    ```

2.  `video_meta.jsonl`: Contains metadata for video/image content. The `data_loader` uses this file to look up video IDs and retrieve their associated text (title, description, OCR, etc.) and a representative image frame.
    ```json
    {"video_id": "video_id_123", "title": "Cat Playing Piano", "caption": "My cat plays a beautiful melody.", "images": ["path/to/frame.jpg"], "ocr": "text found in video", "asr": "audio transcript"}
    ```

If these files are not found, the training script will automatically generate a small sample dataset for demonstration purposes.

## 🚀 How to Train

The project provides two main scripts for training. All configurations can be adjusted in `src/config.py`.

### 1. Simple Example (Recommended for a first run)

This script runs a minimal training loop on a tiny, auto-generated dataset to verify that the environment and pipeline are working correctly.

```bash
python train_example.py
```

### 2. Advanced Training

This is the main script for running a full training session with all advanced features.

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

**Parameter-Efficient Fine-Tuning (PEFT) with LoRA:**
This is ideal for training on hardware with limited memory.
```bash
python train_runner.py \
    --use_lora \
    --freeze_backbone \
    --batch_size 8
```

## 📈 Monitoring

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir logs
```

Navigate to `http://localhost:6006` in your browser to view real-time graphs of the training loss, accuracy, learning rate, and other metrics. If you enable Weights & Biases in the config, logs will be automatically synced to your W&B account.