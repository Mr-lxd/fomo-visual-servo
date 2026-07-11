# 第一版架构决策

## 状态与范围

本文是第一版 PyTorch FOMO 的设计决策记录，不包含任何 Python 实现。它以 `AGENTS.md` 的约束为准，并综合 `docs/reference_analysis.md` 的两份参考实现分析。

目标是先获得一个在 Windows CPU/CUDA 上可训练、可固定尺寸导出、可由 Raspberry Pi 5 上 ONNX Runtime CPU 执行的稳定基线。第一版优先接口一致性、标签正确性和部署可验证性，不以最大模型精度为首要目标。

## 已考虑的方案

| 方案 | 优点 | 主要风险 | 结论 |
|---|---|---|---|
| A. `/8` MobileNetV2-lite encoder + 1×1 FOMO head | 使用深度可分离卷积，输出小，ONNX/Pi CPU 友好，和目标接口直接对齐 | 没有 decoder，极小目标可能需要更高输入尺寸 | **采用** |
| B. deebuls 风格 `/2` U-Net decoder | skip connection 有利于细粒度定位 | 输出 stride 不符合要求；decoder 激活、concat 和上采样增加 Pi 成本；参考代码不完整 | 不采用 |
| C. MobileViT `/8` backbone | 可能获得更强全局建模能力 | attention 与 reshape 增加导出、内存和 CPU 风险 | 后续实验项，不作为第一版 |

## 明确决定

| 决策项 | 第一版决定 |
|---|---|
| Backbone | 自定义、可配置宽度的 **MobileNetV2-lite encoder**；使用 Conv2d、DepthwiseConv2d、BatchNorm2d、ReLU6 和 inverted residual block，在第一次产生 `/8` 特征后停止下采样 |
| Head | `1×1 Conv(F→32) → ReLU6 → 1×1 Conv(32→1+N)`；不使用 decoder、不使用 attention、不在模型内 softmax |
| 默认输入尺寸 | `160×160` RGB；模型配置允许 96、160、192、224 等正方形且能被 8 整除的尺寸；每个 ONNX 文件固定一种尺寸，第一份导出为 160 |
| 输出 stride | `8` |
| 输出通道 | `1+N`。第一版单类 `creature`：`N=1`，故输出 2 通道（0=background，1=creature） |
| 标签生成 | YOLOv5 bbox 经 letterbox 后取 centroid，落入唯一的 `/8` 网格单元；其他网格单元均为 background |
| Loss | 对 logits 和 class-index 标签使用 **加权多类 focal cross-entropy**；`gamma`、类别权重、ignore/冲突策略均由 YAML 控制 |
| 后处理 | `softmax → argmax/置信度阈值 → 每类 8 连通域 → 概率加权网格质心 → stride 映射 → letterbox 反变换`，输出 centroid、class_id、confidence |
| 训练接口 | `images [B,3,S,S]`、`target [B,S/8,S/8]`、`int64`；`model(images) → logits [B,1+N,S/8,S/8]` |
| 部署接口 | 仅导出模型 logits：固定 `float32 [1,3,160,160] → float32 [1,2,20,20]` ONNX；预处理和后处理在 ONNX 外执行 |

## 模型与张量合同

### 输入与特征

输入图像在数据层完成 RGB 转换和 letterbox，进入模型前仅为一次 `uint8 → float32 / 255` 归一化。模型不再包含 Rescaling 或 softmax。

```text
images:        float32 [B, 3, S, S]      # S ∈ {96,160,192,224,...}; S % 8 == 0
encoder_feat:  float32 [B, F, S/8, S/8]  # F 由 MobileNetV2-lite 宽度配置决定
logits:        float32 [B, 1+N, S/8, S/8]
```

默认单类尺寸为：

```text
[B,3,160,160] → [B,F,20,20] → [B,2,20,20]
```

backbone 的输入/输出通道、宽度倍率、BatchNorm 参数和 head 宽度都是 YAML 配置项。第一版只使用仓库规定的常见 ONNX 友好算子，不使用自定义 C++/CUDA op、Transformer attention 或动态 reshape。

### background 与前景类别

`target` 存储类别索引而不是 one-hot：

```text
0              background
1              creature（单类阶段）
1 + class_id   多类阶段的前景类别，class_id 为 YAML 类别表的 0-based 索引
```

`N=1` 和 `N>1` 使用同一 softmax、loss、后处理和 ONNX 输出接口。禁止为单类分出 sigmoid/BCE 模型分支；这样多类扩展只改变 YAML 类别表、最后一层输出通道数和 class weights。

## 标签与坐标决策

### Letterbox 合同

对原图 `(W0,H0)` 和目标正方形尺寸 `S`：

```text
r = min(S / W0, S / H0)
W1 = round(W0 * r), H1 = round(H0 * r)
p_left, p_top = letterbox 左、上 padding
x_lb = r * x_original + p_left
y_lb = r * y_original + p_top
```

数据模块必须将 `r`、四边 padding、原图/输入尺寸与每个 bbox 的变换结果作为元数据返回。中心裁剪不是允许的替代方案。

### Centroid grid 标签

对 YOLOv5 原图归一化 bbox 的 `(class_id,x_center,y_center,width,height)`：

```text
x0 = x_center * W0
y0 = y_center * H0
gx = clamp(floor(x_lb / 8), 0, S/8 - 1)
gy = clamp(floor(y_lb / 8), 0, S/8 - 1)
target[gy,gx] = 1 + class_id
```

选择单格 centroid 而不是填满 bbox 的原因是：视觉伺服消费的是目标中心，且 bbox 面积不应通过大面积正样本改变类别监督的含义。网格量化造成的最大轴向误差应在测试中单独界定，不得被当作 letterbox 反变换误差。

如果两个 bbox 中心落在同一 grid cell：

- 同一前景类：标签写入相同值，但数据验证必须累积并报告 collision 数量。
- 不同前景类：该样本不可表达，数据验证必须以清晰错误终止，而不能覆盖标签或静默丢弃。

## Loss 决策

第一版损失为从 `logits [B,C,G,G]` 和 `target [B,G,G]` 计算的加权多类 focal cross-entropy：

```text
p_t = softmax(logits)[target]
loss = - class_weight[target] * (1 - p_t)^gamma * log(p_t)
```

`gamma` 与每类权重均由 YAML 提供；默认建议 `gamma=2`，但不在源码写死。background 权重也必须配置化，以应对 centroid 标签造成的大量 background 网格。

第一版不组合 Dice loss：class-index focal cross-entropy 的输入/输出语义更直接，且有利于在单类与多类之间保持相同接口。若后续实验显示前景召回不足，可在保留同一 logits/target 合同的前提下新增可配置的 Dice 辅助项，并配套消融与导出回归测试。

## 后处理决策

给定单张 logits `[1,1+N,G,G]`：

1. 在类别维执行 softmax，得到 `probabilities [1,1+N,G,G]`。
2. 对每个网格取 `class_id = argmax(probabilities)` 与 `confidence = max(probabilities)`。
3. 保留 `class_id != 0` 且 `confidence >= threshold[class_id]` 的网格；阈值来自 YAML。
4. 对每个类别执行 8 连通域聚合。每个连通区域对应一个检测候选。
5. 用该类别概率作为权重，在网格中心 `(gx+0.5,gy+0.5)` 上计算质心；再乘以 stride 8 得到 letterbox 输入坐标。
6. 使用该样本的 letterbox 元数据反变换为原图坐标，裁剪到原图边界。
7. 返回稳定的数据结构：`centroid=(x,y)`、`class_id`、`confidence=max(component_probabilities)`。

后处理不属于 ONNX 图。第一版实现应使用 CPU 友好、无大依赖的 8 连通域逻辑；禁止因为部署方便而把阈值、连通域或坐标变换硬编码进模型 forward。

## 训练、评估与部署接口

| 环节 | 输入 | 输出 | 不变量 |
|---|---|---|---|
| Dataset | 图像路径、YOLO 标签、YAML 配置 | image、class-index target、letterbox metadata | ID 配对、绝对路径不写入源码、label collision 可见 |
| Train step | `[B,3,S,S]` 与 `[B,G,G]` | scalar loss、`[B,C,G,G]` logits | CPU 可运行；CUDA 为可选加速 |
| Evaluate | logits、target、metadata | 前景 IoU/定位/检出指标 | 使用与训练相同的 class mapping、stride、letterbox |
| Inference | 原图与 YAML 配置 | `centroid,class_id,confidence` | 对原图坐标输出；保留 letterbox 调试元数据 |
| Export | eval 模式模型与固定 dummy input | ONNX logits 图 | 输入/输出 shape、opset、配置来源被记录 |
| ONNX verification | 相同预处理后的固定输入 | PyTorch/ONNX logits 比较 | 使用数值容差断言；必要时也比较后处理检测结果 |

## 交付前的设计验收

在开始任何 Python 实现前，后续任务必须先为以下事项建立 pytest：

- 对 96、160、192、224 的模型 shape：输出必须是 `[B,1+N,S/8,S/8]`。
- letterbox 正向与反向变换，包括横图、竖图、正方形、奇数尺寸与有 padding 图像。
- YOLO bbox centroid 到网格标签的映射、边界 clamp 与冲突检测。
- 单类与多类 target、loss、后处理的通道/索引一致性。
- 固定 160 输入时 PyTorch 与 ONNX Runtime logits 的数值一致性。

这些测试是后续模型、数据、训练、推理和导出模块的前置条件；本任务仅记录决策，未创建任何实现或测试代码。
