# AGENTS.md

## 项目状态与目标

本项目名为 `fomo-visual-servo`，Python 包名为 `fomo_servo`。仓库当前已经建立可长期回退的 **D2 MobileNetV2-FOMO 稳定工程基线**：完成模型选择、validation/locked evaluation、正式 ONNX 导出、PyTorch/ONNX Runtime parity，以及 Raspberry Pi 4 ARM64 上的静态图片、预录视频、USB UVC 摄像头和 VNC preview 验证。

该基线不是最终论文代码或项目终点。后续仍包括机器人硬件控制闭环、baseline 方法创新、模型对比与消融、实验室/真实环境测试、投稿版本冻结，以及最终代码与数据集开源。稳定 `main` 用于可复现基线；未成熟科研探索应优先位于 `experiment/*` 或明确的 feature branch，不得为了保留实验而污染正式 D2/Pi runtime 依赖。

Windows 训练/导出环境使用 Python 3.10 和 PyTorch；训练实现必须同时兼容 CPU 与 CUDA，但不得依赖 CUDA 专用算子。已验证部署环境为 Raspberry Pi 4 ARM64 / Python 3.13 / ONNX Runtime CPU，且最小运行 bundle 不依赖 torch、torchvision、训练代码、数据集或 checkpoint。Raspberry Pi 5 和其他硬件平台属于后续可验证目标，不能用推测结果替代实测。

## 设计原则

- 保持模块边界清晰、实现可替换、配置可复现、错误可见。
- 优先小型且常见的 Python 依赖；引入新依赖前，说明其必要性和替代方案。
- 不要静默吞掉异常。捕获异常时必须保留上下文，并以可操作的信息重新抛出或明确报告。
- 源码中不得写死数据集的绝对路径、用户目录或机器相关路径。路径只能来自 YAML 配置、命令行参数或受控环境变量。
- 所有可调参数必须位于 YAML 配置中；源码只负责读取、校验和使用配置。
- 任何新增模块都必须配套 pytest 测试；测试应与模块在同一任务中提交。
- 正式 D2 candidate 固定为 seed42 epoch40、validation threshold `0.40`；除非开启经过批准的新模型 milestone，不得重新训练、利用 test split 重选模型/epoch/threshold，或改变已冻结的 preprocessing/postprocessing 与 ONNX contract。

## 当前目录布局

采用 `src` layout。后续实现应维持如下职责边界，而非把功能堆入单一脚本：

```text
src/
  fomo_servo/
    config.py     # YAML 加载、schema 与配置校验
    datasets/     # YOLOv5 数据集、augmentation、letterbox、标签/热力图生成
    models/       # backbone、FOMO head、模型工厂
    losses/       # 分类与加权损失
    training/     # 训练循环、损失、优化器、checkpoint、调度器
    evaluation/   # validation、locked evaluation 与 parity 流程
    metrics/      # 分类、质心与序列指标
    postprocess/  # softmax、connected components 与 detections
    inference/    # 共享预处理、PyTorch/ORT predictor 与视频缓冲
    deployment/   # 固定尺寸 ONNX 导出、provenance 与校验
    geometry/     # 坐标、letterbox 与 stride 映射
tests/
  ...             # 与 src 模块对应的 pytest 测试
configs/
  ...             # 数据、模型、训练、评估、导出 YAML
```

模型、数据、训练、评估、推理、部署和几何变换必须保持分离。模块之间应使用明确的数据结构和张量约定交互，而不是依赖隐式全局状态。

## 稳定 D2 基线合同

### 数据与类别

- 输入数据集采用 YOLOv5 目标检测格式。
- 首先支持单类 `creature`；类别表必须来自 YAML，后续可扩展至多类。
- 数据集根目录、train/val split、类别名称、输入尺寸及增强参数均由 YAML 提供。
- 图像预处理必须使用 letterbox；禁止使用中心裁剪作为默认或替代路径。
- letterbox 的缩放比例、上下左右 padding、原图尺寸和目标输入尺寸必须被保留，以支持坐标反变换。

### 模型与张量约定

- 模型支持 96、160、192、224 等正方形输入尺寸；具体尺寸由 YAML 配置，且应验证能被输出 stride 整除。
- 模型输出为 stride=8 的分类热力图，空间形状为 `[B, 1 + N, H/8, W/8]`。
  - `B`：batch size。
  - `N`：前景类别数。
  - 通道 `0`：background。
  - 通道 `1..N`：各前景类别。
- 除非某接口明确声明概率，模型前向输出应为 logits；softmax 由损失或推理后处理显式执行。
- 每个新增函数、类和模块的 docstring 或类型注释必须写清输入与输出张量 shape、dtype、坐标系及单位。

### 训练、评估与推理

- 训练、评估、推理必须使用相同的 class mapping、letterbox 语义和 stride 定义。
- 后处理必须输出 `centroid`、`class_id` 和 `confidence`；`centroid` 必须明确是在 letterbox 输入坐标还是原图坐标中，并提供到另一坐标系的转换。
- 所有从热力图到目标坐标的变换必须可逆或明确说明不可逆部分（例如网格量化）。
- 训练与评估路径都必须在 CPU 可运行；CUDA 只能作为可选加速路径。

## 坐标与几何测试要求

- 为每种输入尺寸测试 letterbox 的正向变换：原图坐标 → letterbox 坐标。
- 为每种输入尺寸测试反向变换：letterbox 坐标 → 原图坐标。
- 覆盖横向、纵向、正方形、奇数尺寸和存在 padding 的图像。
- 测试 round-trip 误差，并显式设置可接受容差；对 stride=8 网格映射的量化误差单独断言。
- 标签生成、热力图网格中心、后处理质心与原图坐标之间必须有端到端测试。

## ONNX 与 Raspberry Pi 部署约束

- 网络优先使用 ONNX Runtime CPU 友好的常见算子：`Conv2d`、`DepthwiseConv2d`、`BatchNorm2d`、`ReLU6`、普通/自适应 pooling 和 `1x1 Conv`。
- 禁止自定义 C++/CUDA op，避免仅在 CUDA 下可用的实现与未验证的导出路径。
- 每个可部署模型必须支持固定输入尺寸的 ONNX 导出；导出的输入 shape、输出 shape、opset 和配置来源必须被记录。
- 必须使用相同的固定样本比较 PyTorch 与 ONNX Runtime 输出，并以数值容差断言一致性；至少验证 logits/概率热力图，必要时也验证后处理结果。
- Raspberry Pi 部署优化应优先考虑 batch=1、固定输入尺寸、CPU 延迟、内存占用和稳定性，而不是依赖 GPU 特性；任何新硬件平台的性能结论都必须来自该平台实测。
- Headless 与 VNC preview 必须保持可选依赖边界；正式 headless runtime 不得无条件依赖桌面 GUI、PyTorch、torchvision、训练模块或科研对比模型。

## 测试与质量门槛

- 使用 pytest。每个新增模块至少有一个对应测试文件；新增几何、数据、模型、导出模块时，必须覆盖其关键正常路径与失败路径。
- 测试不得依赖本机绝对路径、真实大规模数据集、网络下载或 CUDA 可用性。
- 对配置错误、缺失文件、类别不匹配、非法输入尺寸、异常标签和 ONNX 输出不一致，应明确失败并给出可诊断的错误信息。
- 在开始实现任何模型或训练代码前，先建立张量 shape、坐标系和异常行为的测试。

## 每次任务的交付报告

每次完成任务后，最终回复必须包含以下四项：

1. 修改文件；
2. 运行命令；
3. 测试结果；
4. 未解决问题。

若任务只修改文档、尚未存在可运行测试，也必须明确说明“未运行测试”的原因，而不能省略该项。

