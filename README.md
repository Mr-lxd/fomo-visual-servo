# fomo-visual-servo

面向水下目标视觉伺服的轻量化 PyTorch FOMO 项目。训练与开发目标是 Windows，最终部署目标是 Raspberry Pi 5 上的 ONNX Runtime CPU。

当前仓库已包含 YAML 配置加载、YOLOv5 数据读取与 stride-8 heatmap 标签生成、MobileNetV2-lite FOMO 模型，以及 CPU/CUDA 训练和验证流程。预测和正式 ONNX 导出脚本仍未实现。

## 环境边界

- **训练环境**：Python 3.10、PyTorch、OpenCV 与开发测试工具。训练代码必须支持 CPU；CUDA wheel 不写入 `environment.yml` 或 `pyproject.toml`，而是由激活后的项目环境通过明确的官方 pip 命令安装一次。
- **导出环境**：`onnx` 仅用于 Windows 训练机将 PyTorch 模型序列化为 ONNX；Raspberry Pi 运行时不需要它。
- **部署环境**：Python 3.10、ONNX Runtime 与 OpenCV。它不要求安装 PyTorch，面向 Raspberry Pi 5 的 CPU 推理。
- **组合环境**：只有需要在同一台机器同时验证 PyTorch、ONNX 导出和 ONNX Runtime 时，才安装相关 optional dependency groups。

`pyproject.toml` 中的依赖组为：

- `.[training]`：Windows 训练所需的项目依赖（不包含 torch）。
- `.[dev]`：pytest 开发测试工具（不包含 torch）。
- `.[export]`：固定尺寸 ONNX 导出和 ONNX 图校验所需的 `onnx`（不包含 torch 或 ONNX Runtime）。
- `.[deployment]`：Raspberry Pi 部署运行时。
- `.[training,export,deployment]`：同机端到端验证所需的项目依赖（不包含 torch）。

## Windows：创建并激活独立训练环境

以下命令应在仓库根目录的 Anaconda Prompt 中运行。它们只会创建名为 `fomo-servo-train` 的新环境，不会修改 base 或现有 YOLO 环境。

```powershell
conda env create -f environment.yml
conda activate fomo-servo-train
python --version
```

`environment.yml` 只创建 Python 3.10 基础环境并提供 pip；它**不安装 torch**。请先确认上一步输出为 Python 3.10。

### RTX 4060：安装唯一的 PyTorch CUDA wheel

本机 NVIDIA 驱动 `531.61` 的 `nvidia-smi` 输出支持 CUDA 12.1。因此使用 PyTorch 官方 CUDA 12.1 wheel 索引；项目当前只需要 `torch`，不额外安装尚未使用的 `torchvision` 或 `torchaudio`：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

该命令是此训练环境中唯一安装 torch 的步骤。之后安装项目依赖时，`.[training]` 和 `requirements-dev.txt` 都不包含 torch，不会覆盖或再次安装它。PyTorch 官方历史版本页列出了 `2.5.1` 的 CUDA 12.1 wheel；如将来升级 NVIDIA 驱动，应重新使用官方安装选择器评估新的 wheel，而不要在当前环境中混装 CUDA 构建。

如只需要 CPU 训练，可在新的 `fomo-servo-train` 环境中改用下列**替代命令**，不要与 CUDA 命令同时执行：

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### 安装项目训练与开发依赖

```powershell
python -m pip install -e ".[training]"
python -m pip install -r requirements-dev.txt
python scripts/check_env.py --profile training
```

上述两条项目依赖命令不包含 torch。若环境已经创建且 torch wheel 已按上节安装，可仅重新运行这两条命令更新项目依赖；不要重新执行另一种 torch 安装命令。

如需在同一训练环境中执行固定尺寸 ONNX 导出、ONNX 图校验和 ONNX Runtime CPU 数值一致性测试，再额外安装导出与部署依赖：

```powershell
python -m pip install -e ".[export,deployment]"
python scripts/check_env.py --profile all
```

`onnx` 是 PyTorch ONNX 导出器生成并校验 ONNX 图所需的依赖；`onnxruntime` 只在 CPU 运行时推理与 PyTorch/ONNX 输出一致性测试中需要。两者均不改变已安装的 PyTorch CUDA wheel。

## Raspberry Pi 5：独立部署环境

在 Raspberry Pi 的项目副本中使用 Python 3.10 创建虚拟环境；不要把 Windows 的训练环境复制到 Pi：

```bash
python3.10 -m venv .venv-fomo-servo-deploy
source .venv-fomo-servo-deploy/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deployment]"
python scripts/check_env.py --profile deployment
```

部署 profile 会检查 Python、OpenCV、ONNX Runtime 和当前设备；即使 PyTorch 未安装，也不会因此将部署 profile 判为失败。

## 环境检查

`scripts/check_env.py` 只检查**当前已激活**的 Python 环境，不创建 conda 环境、不安装依赖，也不修改系统环境。

```powershell
python scripts/check_env.py --profile training
python scripts/check_env.py --profile deployment
python scripts/check_env.py --profile all
```

它会报告 Python 版本、torch 版本、CUDA 是否可用、OpenCV、ONNX Runtime 与当前设备。`all` profile 仅适用于已经同时安装训练和部署依赖的环境。

官方安装参考：[PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/) 与 [PyTorch Previous Versions](https://docs.pytorch.org/get-started/previous-versions/)。

项目开发约束、模块边界、数据格式、测试要求和 ONNX 接口见 [AGENTS.md](AGENTS.md) 与 [docs/architecture_decision.md](docs/architecture_decision.md)。

## 训练运行时与设备选择

训练运行时配置位于 YAML 的小写 `training` 节。为兼容已有配置，也可使用 `TRAIN` 及其全大写字段；同一个 YAML 中不能同时写两种节名。

```yaml
training:
  device: auto       # auto | cpu | cuda | cuda:N
  amp: true          # 仅在 CUDA 上启用 float16 autocast
  num_workers: 4     # DataLoader 并行加载进程数，0 表示主进程加载
  pin_memory: true   # 仅 CUDA DataLoader 使用 pinned memory
```

`device: auto` 在 CUDA 可用时选择 GPU，否则选择 CPU。若显式请求 `cuda` 而当前 PyTorch 不可用 CUDA，程序会报出明确错误；不会静默回退。AMP 和 pin memory 在 CPU 上会被禁用，并由命令输出明确说明。

`scripts/train.py` 会执行完整的训练与验证流程，并支持设备与 checkpoint 覆盖：

```powershell
python scripts/train.py --config configs/aquarium_creature_192.yaml --device cuda
python scripts/train.py --config configs/aquarium_creature_192.yaml --device cpu
python scripts/train.py --config configs/aquarium_creature_192.yaml --resume outputs/aquarium_creature/last.pt
```

未提供 `--device` 或 `--resume` 时，分别使用 YAML 的 `training.device`、`training.resume`。训练内部采用以下可复用接口：

```python
runtime = create_training_runtime(config.training, device_override=args.device)
model = prepare_model(model, runtime)  # model.to(runtime.device)
images, targets = move_training_batch(images, targets, runtime)

with autocast_context(runtime):
    logits = model(images)
```

训练 YAML 同时控制数据类映射、loss、AdamW、scheduler、early stopping 与输出目录：

```yaml
dataset:
  class_mode: merge_single     # merge_single | preserve
  merged_class_name: creature

loss:
  name: focal_cross_entropy    # weighted_cross_entropy | focal_cross_entropy
  gamma: 2.0
  class_weights: [1.0, 3.0]    # background + N foreground classes

training:
  batch_size: 16
  epochs: 50
  seed: 42
  output_dir: outputs/aquarium_creature
  resume: null
  early_stopping_patience: 10  # 0 表示关闭
  early_stopping_min_delta: 0.0
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
  scheduler:
    name: step_lr              # none | step_lr
    step_size: 10
    gamma: 0.5
```

每个 epoch 会记录 `train_loss`、`val_loss`、前景网格级 micro `precision`、`recall` 和 `F1` 到 `<output_dir>/history.csv`。始终保存 `<output_dir>/last.pt`，仅当 validation F1 严格提升时保存 `<output_dir>/best_val_f1.pt`。checkpoint 含模型、优化器、scheduler、AMP scaler、随机状态和 early stopping 状态，因此可从 `last.pt` 继续训练。

## Roboflow YOLO 数据集

数据 loader 同时支持本项目布局 `images/train`、`labels/train` 与 Roboflow 布局 `train/images`、`train/labels`。逻辑 validation split 为 `val` 时，会自动识别 Roboflow 的物理目录 `valid/images` 与 `valid/labels`；不需要移动或复制原始数据。

已提供七类预训练配置：[configs/aquarium_pretrain_192.yaml](configs/aquarium_pretrain_192.yaml)。它保留 `fish`、`jellyfish`、`penguin`、`puffin`、`shark`、`starfish`、`stingray` 的原始 class id 顺序，并产生 background 加七类共八个输出通道。配置中的数据根目录通过环境变量 `FOMO_DATASET_ROOT` 指定，不包含任何机器绝对路径。

该配置使用 `dataset.collision_policy: keep_first`。当不同类别的目标中心量化到同一个 8 像素网格时，FOMO 单个网格只能表达一个类别，因此按标签文件顺序保留第一个类别，并在 heatmap 元数据中累计碰撞次数；默认策略仍为 `error`，适合发现数据或输入尺寸问题。

先检查标签、letterbox 和 centroid heatmap：

```powershell
$env:FOMO_DATASET_ROOT = 'C:/path/to/aquarium_pretrain'

python scripts/visualize_yolo_heatmap.py `
  --dataset-root $env:FOMO_DATASET_ROOT `
  --split train `
  --index 0 `
  --input-size 192 `
  --stride 8 `
  --class-mode preserve `
  --output outputs/aquarium_pretrain_label_check.jpg
```

确认可视化正确后开始训练：

```powershell
python scripts/train.py --config configs/aquarium_pretrain_192.yaml --device cuda
```
