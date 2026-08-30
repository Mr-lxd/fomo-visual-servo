# fomo-visual-servo

面向水下目标视觉伺服的轻量化 PyTorch FOMO 项目。训练与导出在 Windows 上进行；正式 D2 MobileNetV2-FOMO 已在 Raspberry Pi 4 ARM64 / Python 3.13 上完成 ONNX Runtime CPU、静态图片、预录视频、USB UVC 摄像头和 VNC 实时预览验证。

当前仓库包含 YAML 配置加载、YOLOv5 数据读取与 stride-8 heatmap 标签生成、MobileNetV2-lite FOMO 模型、CPU/CUDA 训练验证、固定尺寸 ONNX 导出、ONNX Runtime predictor，以及图片、视频和摄像头 CLI。正式 checkpoint、ONNX、数据集和部署运行产物由 `.gitignore` 排除，不随源码发布。

## 当前稳定基线与后续范围

当前 integration 定义为“已验证的 D2 baseline + Raspberry Pi 4 / ONNX Runtime / USB camera / VNC preview 稳定工程基线”，用于长期回退和后续开发；它不是最终论文代码或项目终点。

- **Training / research**：保留 D2 MobileNetV2-FOMO 的 augmentation、checkpoint selection、validation/locked evaluation、threshold protocol、model card 和实验 provenance。未经单独实验计划，不在稳定主线上混入对比模型或消融依赖。
- **Headless Pi deployment**：正式无人值守路径使用最小 ORT bundle 和 `opencv-python-headless`，不安装 PyTorch、torchvision、训练代码、数据集或 checkpoint。
- **VNC preview / debug**：独立 preview 环境仅用于实时查看 annotated frame、调整摄像头和验证资源释放，不改变推理语义，也不把桌面刷新率当作模型 FPS。
- **Future robot-control integration**：摄像头到 perception 的链路已经打通；target selection、visual servo/control、actuator/hardware 闭环，以及后续方法创新、对比/消融、实验室与真实环境测试、投稿冻结和最终开源仍属于未来阶段。

未进入该稳定基线的科研探索继续保留在 `experiment/*` 或专用 feature branch 中。“不进入 main”不等于删除实验历史。

合并该 baseline 后，机器人闭环建议从最新 `main` 创建 `feature/robot-control-integration`；新的模型对比、方法创新和消融原则上使用 `experiment/*`，不直接在稳定 `main` 上试验。

## 环境边界

- **训练环境**：Python 3.10、PyTorch、OpenCV 与开发测试工具。训练代码必须支持 CPU；CUDA wheel 不写入 `environment.yml` 或 `pyproject.toml`，而是由激活后的项目环境通过明确的官方 pip 命令安装一次。
- **导出环境**：`onnx` 仅用于 Windows 训练机将 PyTorch 模型序列化为 ONNX；Raspberry Pi 运行时不需要它。
- **部署环境**：Raspberry Pi 4 使用 Python 3.13、ONNX Runtime 与 OpenCV 的最小 bundle，不安装完整训练项目或 PyTorch。headless 与 GUI preview 使用互斥的 OpenCV profile。
- **组合环境**：只有需要在同一台机器同时验证 PyTorch、ONNX 导出和 ONNX Runtime 时，才安装相关 optional dependency groups。

`pyproject.toml` 中的依赖组为：

- `.[training]`：Windows 训练所需的项目依赖（不包含 torch）。
- `.[dev]`：pytest 开发测试工具（不包含 torch）。
- `.[export]`：固定尺寸 ONNX 导出和 ONNX 图校验所需的 `onnx`（不包含 torch 或 ONNX Runtime）。
- `.[deployment]`：Raspberry Pi 部署运行时。
- `.[tflite]`：仅用于审计 Edge Impulse 导出的 TFLite/LiteRT 模型；使用官方 LiteRT wheel，不安装完整 TensorFlow。
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

如需运行 Edge Impulse TFLite parity audit，再额外安装轻量解释器：

```powershell
python -m pip install -e ".[tflite]"
```

它只用于读取和执行导出的 `.tflite`；不替代 ONNX Runtime 的 Raspberry Pi 部署路径。

## Raspberry Pi 4：已验证的最小部署环境

Pi 端不安装受 `pyproject.toml` Python 3.10 边界约束的完整训练包，而是使用只包含 ORT inference 闭包的 bundle。当前实装版本为：

```text
Python 3.13.5
numpy==2.5.2
onnxruntime==1.29.0
opencv-python-headless==5.0.0.93  # headless profile
opencv-python==5.0.0.93           # preview profile，独立 venv
```

仓库根目录的 `requirements-pi4-headless.txt` 与 `requirements-pi4-preview.txt` 分别描述两个互斥 profile。正式 headless venv 不依赖桌面 GUI；preview venv 提供 Qt HighGUI，并且必须从 VNC 桌面中的 Terminal 自然继承 Wayland/XWayland 会话变量。源码和启动命令不写死 `DISPLAY`。

以下是当前 Pi 上已经实测的 bundle 与 venv 路径；`run.py` 本身从所在目录解析 bundle root，因此 bundle 可整体移动。

### Headless deployment

适用于 benchmark、无人值守运行以及后续 systemd 集成。`--display` 默认关闭：

```bash
cd /home/pi/fomo-ort-d2-epoch40
run_id="$(date +%Y%m%d-%H%M%S)"
/home/pi/venvs/fomo-ort-d2-epoch40/bin/python run.py predict_video \
  --onnx artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx \
  --onnx-report artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json \
  --source 0 \
  --max-frames 300 \
  --output-video "camera_runs/${run_id}/camera-overlay.mp4" \
  --output-csv "camera_runs/${run_id}/camera-telemetry.csv" \
  --output-jsonl "camera_runs/${run_id}/camera-telemetry.jsonl"
```

### VNC preview / calibration

适用于调整摄像头角度、距离和视野，以及人工查看 annotated detections。必须在 VNC 桌面的 Terminal 中运行；不要从普通 SSH shell 手工设置 `DISPLAY`：

```bash
cd /home/pi/fomo-ort-d2-epoch40
run_id="$(date +%Y%m%d-%H%M%S)"
/home/pi/venvs/fomo-ort-d2-epoch40-preview/bin/python run.py predict_video \
  --onnx artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx \
  --onnx-report artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json \
  --source 0 \
  --display \
  --output-video "camera_runs/${run_id}/camera-overlay.mp4" \
  --output-csv "camera_runs/${run_id}/camera-telemetry.csv" \
  --output-jsonl "camera_runs/${run_id}/camera-telemetry.jsonl"
```

参数含义：

- `--onnx`：正式固定尺寸模型。
- `--onnx-report`：sidecar/provenance contract；启动时校验模型 SHA、size、shape、opset 和正式 threshold。
- `--source`：视频路径或摄像头索引；`0` 对应已验证的 `/dev/video0`。
- `--display`：默认关闭的桌面/VNC annotated-frame preview；按 `q`、`Q` 或 Esc 正常停止。
- `--output-video`：保存带 overlay 的录像。
- `--output-csv`：保存便于表格分析的逐帧 telemetry。
- `--output-jsonl`：保存完整机器可读的逐帧结果。

上述正式命令不传 `--confidence-threshold`，因此使用 sidecar 锁定的 `0.40`。摄像头拍摄显示器时出现的 false positive 属于明显 domain shift，只能用于 pipeline/资源释放回归，不能作为模型精度评价。完整 provenance、跨平台 parity 和 Pi benchmark 见 [Raspberry Pi 4 deployment handoff](docs/handoffs/2026-08-28-raspberry-pi4-deployment-handoff.md)。

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

## 在线 augmentation suite

增强只在 train split 的每次样本读取阶段执行，不缓存增强图像。固定顺序为：

```text
horizontal flip → affine → color jitter → Gaussian blur → Gaussian noise
→ letterbox → normalize → FOMO heatmap
```

推荐使用 preset：

```yaml
augmentation:
  enabled: true
  preset: underwater_conservative
  overrides: {}
```

内置 preset 为 `none`、`photometric`、`underwater_conservative` 和 `custom`。`overrides` 只能覆盖已知的点号字段；未知 preset 或字段会明确报错。旧的逐项 `color_jitter`、`horizontal_flip` 等配置仍可加载，但会发出弃用警告。

Dataset 的 augmentation seed 由 `base_seed + epoch + sample_index` 的稳定 64-bit hash 派生，不依赖 worker id。训练在每个 epoch 开始前更新 Dataset epoch；同一 epoch/index 可复现，不同 epoch 通常产生不同增强。当前 DataLoader 明确使用 `persistent_workers=false`。validation/test 始终跳过全部增强。

完整 suite 的可视化命令示例：

```powershell
$env:FOMO_DATASET_ROOT = 'C:/path/to/aquarium_pretrain'
python scripts/visualize_augmentations.py `
  --config configs/experiments/augmentation_suite.yaml `
  --split train `
  --num-images 16 `
  --suite
```

运行结果写入 `outputs/experiments/augmentation_suite/visualization/`，包括不同 epoch、photometric、underwater 和 affine geometry 接触表以及 `augmentation_samples.json`。增强触发率、bbox clipping/drop、目标数量和 collision 统计写入 `history.csv` 与 `training_summary.json`。

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

每个 epoch 会记录 `train_loss`、`val_loss`、网格级指标和固定阈值下的 centroid 指标到 `<output_dir>/history.csv`。始终保存 `<output_dir>/last.pt`，并分别按固定 `evaluation.checkpoint_threshold` 保存 `best_grid_f1.pt` 与 `best_centroid_f1.pt`；`best_val_f1.pt` 是当前 `training.checkpoint_criterion` 对应文件的兼容别名。checkpoint 含模型、优化器、scheduler、AMP scaler、随机状态、选择指标和选择阈值，因此可从 `last.pt` 继续训练。

推理阈值与 checkpoint 选择阈值严格分离：`postprocess.inference_threshold` 供图片/视频推理默认使用，`evaluation.checkpoint_threshold` 只用于每个 epoch 的 legacy 固定阈值指标。当前正式 D2 候选为 seed42 的 epoch40 snapshot，validation threshold 为 `0.40`；它由 validation `centroid_pr_auc_macro` 选择 epoch，再在同一 validation split 上选择 threshold。Test 只读取锁定的 D2 epoch40 candidate 和 validation threshold，禁止 sweep、自动搜索或比较多个 checkpoint。历史 C1/C2 epoch58 结果只用于阶段比较，不是当前候选。详细协议见 [docs/threshold_protocol.md](docs/threshold_protocol.md)。旧配置中的 `postprocess.confidence_threshold` 仍可加载，但会发出弃用警告，并且只映射为推理阈值。

正式 Stage B 流程使用 `scripts/tune_validation_threshold.py` 生成 `threshold_tuning.json` 和 `locked_test_protocol.json`，再使用 `scripts/evaluate_locked_test.py` 输出一次 `final_test_metrics.json` 与 `final_test_metrics.csv`。独立 calibration split 仍可作为高级可选模式，但不是默认要求。

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

## 后处理、验证与推理

模型 checkpoint 只保存 logits；softmax、8 邻域连通域、质心坐标反变换和目标选择在模型外执行。图片推理示例：

```powershell
$env:FOMO_DATASET_ROOT = 'C:/path/to/aquarium_pretrain'
python scripts/predict_image.py `
  --config configs/aquarium_pretrain_192.yaml `
  --checkpoint outputs/aquarium_pretrain_7class_192/best_val_f1.pt `
  --image C:/path/to/image.jpg `
  --device cuda `
  --strategy highest_confidence `
  --output-image outputs/prediction.jpg `
  --output-json outputs/prediction.json
```

完整 validation 会使用与推理相同的后处理，并在配置的 validation split 上执行 confidence threshold sweep；训练期 checkpoint 仍只使用固定阈值：

```powershell
python scripts/evaluate.py `
  --config configs/aquarium_pretrain_192.yaml `
  --checkpoint outputs/aquarium_pretrain_7class_192/best_val_f1.pt `
  --device cuda `
  --output-json outputs/validation_report.json
```

视频推理使用单帧 latest buffer，旧帧会被丢弃以避免实时运行时积压：

```powershell
python scripts/predict_video.py `
  --source input.mp4 `
  --config configs/aquarium_pretrain_192.yaml `
  --checkpoint outputs/aquarium_pretrain_7class_192/best_val_f1.pt `
  --device cuda `
  --output-video outputs/prediction.mp4 `
  --output-csv outputs/prediction.csv `
  --output-jsonl outputs/prediction.jsonl
```

`normalized_x/y` 使用原图像素坐标归一化到 `[-1,1]`；左/上为负，右/下为正。视频中的 jitter、availability、loss rate 和 reacquisition 是运行稳定性统计，不是 MOT 身份跟踪指标。

## 当前正式候选与实验索引

- 当前模型候选：MobileNetV2 FOMO、alpha `0.35`、输入 `192×192 RGB`、输出 `[B,8,24,24]`、stride `8`、Edge Impulse pretrained backbone、seed `42`、epoch `40`、threshold `0.40`。
- D2 validation multi-seed Strict F1：`0.418215 ± 0.005413`（seed `42/123/2027`，sample std，validation-only）。
- D2 seed42 locked-test Strict F1：`0.451977`；EI legacy F1：`0.484959`。EI legacy 只用于 Edge Impulse parity，不是正式主指标。
- pretrained H5 来源：Edge Impulse transfer-learning CDN 的 [Keras MobileNetV2 artifact](https://cdn.edgeimpulse.com/transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5)，仅用于 backbone initialization；SHA-256 为 `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`。来源审计见 [D1.1 pretrained source audit](docs/experiments/stage_d1_1_architecture_parity.md)。H5 未提交，license/redistribution status requires confirmation，不从本仓库或 GitHub Release 重新分发。
- dataset、checkpoint、H5、TFLite、ONNX、ZIP 和 outputs 均不在 Git 仓库中；配置通过 `FOMO_DATASET_ROOT` 与 `FOMO_PRETRAINED_WEIGHTS` 指向本地文件。
- 评估协议、阶段结论和 provenance 索引见 [docs/experiments/README.md](docs/experiments/README.md)；候选模型限制见 [docs/model_card_d2_seed42.md](docs/model_card_d2_seed42.md)；发布前审计见 [docs/release_preflight_report.md](docs/release_preflight_report.md)。
