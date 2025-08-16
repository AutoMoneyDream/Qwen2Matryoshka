# 多模态搜索意图对齐模型训练任务指导说明书

## 1. 任务概述

本项目旨在训练一个多模态模型（基于 Qwen2.5-VL 系列），用于理解和对齐搜索查询（query）与目标（target）的语义。输入数据中的 query 和 target 既可以是文本，也可以是视频。我们将采用 Matryoshka Representation Learning (MRL) 范式进行训练，并结合批处理（in-batch）负采样策略来优化表征学习，最终实现高效、多粒度的向量表征。

## 2. 代码框架结构

为了保持项目简洁性，建议采用以下扁平化的代码结构。

```
/Vlm4rec
|-- data/                  # (建议) 存放训练/验证数据
|   |-- train_data.jsonl
|   `-- video_meta.jsonl
|-- checkpoints/           # (建议) 存放训练好的模型权重
|-- src/
|   |-- __init__.py
|   |-- data_loader.py     # 负责数据加载、预处理和批处理构建
|   |-- model.py           # 定义模型结构、封装 Qwen2.5-VL 和 Matryoshka 损失
|   |-- train.py           # 核心训练脚本，负责启动和管理训练流程
|   `-- config.py          # 存放所有超参数和配置信息
|-- requirements.txt       # 项目依赖
`-- task.md                # 任务指导说明书 (本文件)
```

---

## 3. 子任务拆解与执行步骤

请按照以下步骤，逐一完成各个模块的开发。

### **Step 1: 环境设置与配置**

1.  **任务描述**: 创建并配置项目运行所需的环境。
2.  **执行指引**:
    *   在 `requirements.txt` 文件中，列出所有必要的 Python 库，例如：
        *   `torch`
        *   `transformers`
        *   `datasets`
        *   `Pillow` (用于图像处理)
        *   `tqdm`
    *   在 `src/config.py` 文件中，定义一个配置类或字典，用于统一管理所有可调参数，例如：
        *   模型路径 (`model_path`)
        *   数据路径 (`data_path`, `video_meta_path`)
        *   训练参数 (`learning_rate`, `batch_size`, `num_epochs`, `weight_decay`)
        *   Matryoshka Loss 相关参数 (`mrl_dims`, `mrl_loss_weight`)
        *   其他 (`device`, `max_text_length`)
3.  **验证标准**:
    *   能够通过 `pip install -r requirements.txt` 成功安装所有依赖。
    *   `src/config.py` 中包含所有必要的配置项，并可在其他脚本中被成功导入和访问。

### **Step 2: 数据加载与预处理 (`src/data_loader.py`)**

1.  **任务描述**: 实现一个能够处理文本和视频两种模态、并将其转换为模型输入格式的 `Dataset` 和 `DataLoader`。
2.  **执行指引**:
    *   **创建 `MultimodalDataset(torch.utils.data.Dataset)` 类**:
        *   在 `__init__` 方法中，加载 query-target 对数据和视频元数据，并将视频元数据存储在一个以 `video_id` 为键的字典中，以便快速查找。
        *   实现 `__getitem__(self, idx)` 方法。此方法是核心，需要处理四种情况：
            1.  `query`: text, `target`: text
            2.  `query`: text, `target`: video_id
            3.  `query`: video_id, `target`: text
            4.  `query`: video_id, `target`: video_id
        *   对于 video_id，需要根据 id 查找到对应的视频元数据，**将所有文本字段（`caption`, `title`, `text`, `ocr`, `asr`）拼接成一个长文本**，并加载对应的图片（`images` 列表中的第一张或随机一张作为代表）。
        *   对于 text，直接使用文本内容。
        *   最终，`__getitem__` 应返回一个包含 query 和 target 信息的字典，例如 `{'query_text': str, 'query_image': PIL.Image, 'target_text': str, 'target_image': PIL.Image}`。如果某个部分不存在（如 query 是文本），则对应的 image 位可以为 `None`。
    *   **创建 `collate_fn` 函数**:
        *   该函数接收一个批次的数据列表（来自 `__getitem__` 的输出）。
        *   使用 Qwen2.5-VL 的 `processor` 或 `tokenizer` 对批次内的所有文本进行编码和填充（padding）。
        *   对批次内的所有图像进行预处理和堆叠（stacking）。
        *   返回一个模型可以直接接受的批处理字典，例如 `{'query_input': {...}, 'target_input': {...}}`，其中 `...` 部分包含 `input_ids`, `attention_mask`, `pixel_values` 等。
3.  **验证标准**:
    *   能够成功创建一个 `DataLoader` 实例，并能从中迭代出第一个批次的数据。
    *   打印出批次数据的 `shape`，确认文本和图像的维度符合预期（例如，文本已填充到相同长度，图像已转换为 `torch.Tensor`）。

### **Step 3: 模型与损失函数定义 (`src/model.py`)**

1.  **任务描述**: 封装 Qwen2.5-VL 模型用于特征提取，并实现 Matryoshka 对比损失函数。
2.  **执行指引**:
    *   **创建 `MultimodalEncoder(torch.nn.Module)` 类**:
        *   在 `__init__` 方法中，加载预训练的 Qwen2.5-VL 模型和对应的 processor。
        *   实现 `forward(self, text_inputs, image_inputs)` 方法，该方法接收经过 `collate_fn` 处理后的文本和图像张量。
        *   在 `forward` 方法内部，调用 Qwen2.5-VL 模型，**提取最后一层隐藏层状态的 `[CLS]` token 或对所有 token 的输出进行平均池化（mean pooling）**，作为最终的 embedding 输出。确保 embedding 在返回前被归一化（L2 normalization）。
    *   **实现 `matryoshka_in_batch_negative_loss(query_embeddings, target_embeddings, mrl_dims)` 函数**:
        *   该函数接收一批 query 和 target 的 embedding。
        *   **核心逻辑**:
            1.  计算 query 和 target embedding 之间的相似度矩阵（通过矩阵乘法 `query @ target.T`）。由于 embedding 已归一化，这等价于余弦相似度。
            2.  对于一个大小为 `N` 的批次，相似度矩阵的对角线元素是正样本对（`query_i` vs `target_i`），其余 `N-1` 个元素是批内负样本。
            3.  构造一个标准的交叉熵损失（cross-entropy loss），其 logits 为相似度矩阵，标签为 `torch.arange(N)`。
            4.  **Matryoshka 嵌套**: 在一个循环中，对 `mrl_dims` 中定义的每个维度 `d`，截取 embedding 的前 `d` 维 (`embeddings[:, :d]`)，然后重复上述 1-3 步计算损失。
            5.  将所有维度的损失加权求和（或简单求和）作为最终的总损失返回。
3.  **验证标准**:
    *   能够成功加载 Qwen2.5-VL 模型。
    *   给定一个模拟的批次数据，`MultimodalEncoder` 的 `forward` 方法能正确返回 embedding，其维度符合预期。
    *   给定两组模拟的 embedding，损失函数能计算出一个标量（scalar）损失值，且该值随着正样本对相似度的增加而减小。

### **Step 4: 核心训练流程 (`src/train.py`)**

1.  **任务描述**: 编写主训练脚本，整合数据加载、模型、损失函数和优化器，以执行完整的训练循环。
2.  **执行指引**:
    *   **初始化**:
        *   加载 `src/config.py` 中的配置。
        *   设置随机种子和计算设备（CPU/GPU）。
        *   实例化 `MultimodalDataset` 和 `DataLoader`。
        *   实例化 `MultimodalEncoder` 模型并将其移动到指定设备。
        *   实例化优化器（如 `AdamW`）和学习率调度器（可选）。
    *   **训练循环**:
        *   外层循环遍历 `num_epochs`。
        *   内层循环使用 `tqdm` 遍历 `DataLoader` 中的每个批次。
        *   **循环体内部**:
            1.  将数据批次移动到指定设备。
            2.  调用模型分别计算 query 和 target 的 embedding。
            3.  调用 `matryoshka_in_batch_negative_loss` 函数计算损失。
            4.  执行标准的反向传播三步曲：`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`。
            5.  记录并打印训练指标（如 loss）。
    *   **模型保存**: 在每个 epoch 结束或达到指定步数时，保存模型权重（checkpoint）到 `checkpoints/` 目录。
3.  **验证标准**:
    *   训练脚本可以顺利启动并运行，没有出现 shape 不匹配或设备不一致的错误。
    *   在训练过程中，打印出的 loss 值有明显的下降趋势。
    *   训练结束后，在 `checkpoints/` 目录下能找到保存的模型文件。
