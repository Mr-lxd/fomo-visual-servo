# Stage E：bhoke/FOMO 上游主干审计

审计日期：2026-07-18
上游仓库：[bhoke/FOMO](https://github.com/bhoke/FOMO)（审计 ref：`main`）
用途：为本项目的 PyTorch Stage E 主干筛选提供只读证据；不复制 Keras 源码或权重。

## 上游范围与许可证

上游 README 将项目描述为 Edge Impulse FOMO 的从零实现，并列出 MobileNetV2、MobileNetV3 与 MobileViT；训练入口还注册了 `squeezenet`。仓库许可证为 MIT（Copyright 2022 EmbeddedML）。Stage E 仅借鉴其“选择真实 stride-8 特征后接轻量 FOMO head”的设计意图；本项目不会复制其 Keras 源码，也不会将 Keras 权重转换为 PyTorch 权重。

## 模型选择、训练与输入语义

`train.py` 通过 YAML 的 `MODEL.BACKBONE` 选择 `mobilenetv2`、`squeezenet`、`mobilenetv3` 或 `mobilevit`，然后以 Keras `weighted_dice_loss`、`OneHotIoU` 和 Keras callbacks 训练。该实现是 TensorFlow/Keras（NHWC），而本项目是 PyTorch（NCHW）、外部 letterbox/`float32(image)/255` 预处理、cross-entropy logits 训练和独立后处理，二者不可直接混用。

上游 Squeeze 模型在网络内部添加 `Rescaling(1/255)`；其余候选模型没有统一的网络内归一化层。Stage E 的模型不加入任何 rescaling，继续复用本项目数据集与推理路径的归一化，避免重复缩放。

## MobileNetV3-Small

上游的 `MobileFOMOv3` 使用 Keras `MobileNetV3Small(include_top=False)`，优先取名为 `expanded_conv_3_expand` 的 1/8 层，否则按空间尺寸动态选择；随后是 `1x1 Conv(32, ReLU)` 与 `1x1 Conv(num_classes, softmax)`。Keras 代码仅在 alpha 为 0.75 或 1.0 时尝试 ImageNet 权重，其他 alpha 会打印警告并退回随机初始化。

Stage E 使用 torchvision 0.20.1 的 `mobilenet_v3_small` 定义、`MobileNet_V3_Small_Weights.IMAGENET1K_V1` 官方权重格式，以及真实的 `features.2` 输出：`[B,24,24,24]`（输入为 `[B,3,192,192]`）。该截断点由 stem stride 2、第一倒残差 stride 2、第二倒残差 stride 2 组成，总 stride 为 8；不会使用 stride-16/32 特征插值。原生 Hardswish、ReLU 和 SE 保留。所有模型输出仍是 logits，softmax 只在项目既有损失/后处理内显式执行。

## SqueezeNet

上游 `SqueezeFOMO` 不是 torchvision SqueezeNet 1.1：它使用 `7x7/96` stem、网络内 rescale，并在 `fire4` 后 max-pool 接单层预测卷积。它不提供 ImageNet 加载或严格权重覆盖验证，因此不能作为本阶段的实现来源。

Stage E 使用 torchvision 0.20.1 的明确 `squeezenet1_1`。选择 `features.6`（Fire4）为截断输出：stem stride 2、第一次 max-pool stride 2、第二次 max-pool stride 2 后保持空间尺寸。标准无 padding 的 192 输入在该处会得到 `23x23`（首个 valid `3x3/2` 和两次 ceil-mode pooling 的累积边界效应）；为满足本项目固定 24x24 网格，encoder 仅在输入右/下显式补 1 像素，再使用原生 Fire4 前缀，得到真实 `[B,256,24,24]` stride-8 特征。没有插值、没有改变 Fire 权重或模块。Fire 模块自身不改变分辨率；后续 Fire5 及 pooling 不进入 FOMO encoder。

## 统一 FOMO head 与载入策略

两个 Stage E 主干均使用 D2 同样的 head policy：`Conv2d(feature_channels, 32, 1)`、`ReLU`、`Conv2d(32, 8, 1)`，训练前向返回 `[B,8,24,24]` float32 logits。只有第一层的输入通道随主干变化；head 宽度、activation、类别顺序和 stride 均不变。

官方预训练权重以 YAML `pretrained_source` 指向的本地 `.pth` 文件和 SHA-256 为唯一输入。加载器先验证文件哈希，再以严格键集合验证完整 torchvision state dict，随后仅严格复制所选 feature 前缀。它记录 torchvision 版本、权重枚举、官方 URL、源文件 SHA-256、完整/截断加载键数及 missing/unexpected keys；任何版本、哈希或键集不匹配都会报错。不会静默下载、不会 partial load，也不会把生成的权重写入仓库。

## MobileViT：仅记录，不进入 Stage E

上游 MobileViT-XXS 在 stride-8 后使用 reshape/transpose、LayerNorm、MultiHeadAttention、Dense、Dropout 和 Swish。它能产生 stride-8 特征，但 attention 的 token 重排、ONNX 图稳定性、ONNX Runtime CPU 内存访问和 Raspberry Pi 5 batch=1 延迟均比卷积候选风险更高；上游也未提供可核验的官方 ImageNet 加载路径。

建议仅在以下全部条件达成后单列 Stage E2：两个卷积候选均完成静态 ONNX 导出与 ORT FP32 parity；CPU benchmark 管线稳定；明确选定可再分发/可验证的 MobileViT 预训练来源；并为固定 192 输入完成 ORT 与 Pi 5 实测预算评估。

## 部署风险结论

| 主干 | Stage E 决策 | ONNX/Pi 风险 | 主要控制措施 |
| --- | --- | --- | --- |
| MobileNetV3-Small | 实现与筛选 | 中等：SE/Hardswish 图转换需实测 | 固定 192、opset 17、ORT parity gate |
| SqueezeNet 1.1 | 实现与筛选 | 低到中等：pooling 早期损失定位信息 | Fire4 stride-8 cut、统一 head、benchmark |
| MobileViT | 暂缓 | 高：attention 与 token reshape | Stage E2 前不训练、不导出 |
