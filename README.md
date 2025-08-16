# HRPROJ - 视觉语言模型训练

这是一个使用Qwen-VL（一种强大的视觉语言模型）对自定义数据集进行微调的项目。该项目利用PEFT（参数高效微调）和LoRA（低秩适应）等技术，以高效地训练模型。

## 安装

1.  克隆代码库：
    ```bash
    git clone <repository-url>
    cd HRPROJ
    ```

2.  安装所需的依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 数据

该项目使用两个主要的数据文件：

*   `data/video_meta.jsonl`: 包含视频元数据。
*   `data/train_data.jsonl`: 包含训练数据，每个条目都是一个JSON对象，其中包含图像或视频帧的引用以及相关的文本描述。

请确保这些文件存在于`data`目录中，然后再开始训练。

## 配置

训练配置在`src/config.py`文件中定义。您可以修改此文件以更改超参数、模型路径、数据路径和其他设置。

关键配置选项：

*   `model_path`: 预训练Qwen-VL模型的路径。
*   `data_path`: 训练数据的路径。
*   `meta_path`: 元数据文件的路径。
*   `output_dir`: 保存检查点和日志的目录。
*   `batch_size`: 训练的批量大小。
*   `learning_rate`: 优化器的学习率。
*   `epochs`: 训练的轮数。

## 用法

要开始训练，请运行`train_runner.py`脚本：

```bash
python train_runner.py
```

该脚本将初始化配置、数据加载器、模型和训练器，并开始训练过程。

## 项目结构

```
HRPROJ/
├───.gitignore
├───implement_plan.md
├───MODEL_STEP3_README.md
├───README.md
├───README_TRAINING.md
├───requirements.txt
├───train_example.py
├───train_runner.py
├───.claude/
│   └───settings.local.json
├───.git/...
├───checkpoints/
├───data/
│   ├───train_data.jsonl
│   └───video_meta.jsonl
├───logs/
│   └───run_1755325661/
│       └───events.out.tfevents.1755325661.yudeMac-mini.16052.0
└───src/
    ├───__init__.py
    ├───config.py
    ├───data_loader.py
    ├───model.py
    ├───optimization.py
    ├───test_model.py
    ├───train.py
    └───__pycache__/
```

## 监控

该项目支持使用TensorBoard和Weights & Biases（WandB）进行训练监控。

*   **TensorBoard**: 日志保存在`logs`目录中。要启动TensorBoard，请运行：
    ```bash
    tensorboard --logdir logs
    ```

*   **Weights & Biases**: 如果在`src/config.py`中启用了WandB，训练运行将自动记录到您的WandB帐户中。

## 贡献

欢迎贡献！如果您想为此项目做贡献，请fork代码库并提交拉取请求。

## 许可证

该项目根据MIT许可证授权。有关更多信息，请参阅`LICENSE`文件。
