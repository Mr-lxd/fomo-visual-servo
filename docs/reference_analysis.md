# FOMO 参考实现分析

## 范围与证据

本文件仅分析以下两份本地参考资料，不实现模型或训练代码：

- **bhoke**：`references/bhoke_fomo_notes.md`，其中记录了 Keras/TensorFlow 项目的源码位置与迁移风险。
- **deebuls**：`references/deebuls_fomo.ipynb`，一个未包含训练流程的 PyTorch 网络草图。

引用 `bhoke:Lx` 指本仓库笔记的行号；引用 `deebuls:Cell x` 指 notebook 单元格。对 notebook 的 shape 结论按其卷积参数推导，且以 Cell 8 的输入 `[1,3,512,384]` 为例。

## 结论摘要

两份实现共享“将目标定位转化为低分辨率密集分类”的 FOMO 思路，但接口差异很大：

| 维度 | bhoke 参考 | deebuls 参考 | 对本项目的含义 |
|---|---|---|---|
| 主干结构 | 截断的 MobileNetV2/V3、MobileViT 或 SqueezeNet，加 1×1 FOMO head | 四层下采样、三层上采样且带 skip connection 的浅层 U-Net | 采用 bhoke 的轻量 `/8` head 思路；不采用完整 decoder |
| 张量布局 | Keras 默认 NHWC | PyTorch NCHW | 本项目统一 NCHW |
| 输出分辨率 | `/8` | `/2` | 采用 `/8`，符合仓库既定合同 |
| 类别表达 | `C` 个 softmax 通道；配置中 `C=2` 表示背景 + 一个前景类 | 单通道二值图；文档称 sigmoid，但代码没有 sigmoid | 采用 `1+N` logits，softmax 在 loss/推理中显式执行 |
| 标注来源 | bbox 填充为低分辨率区域掩码，或直接读取语义掩码 | 未实现训练与标签生成 | YOLO bbox 在 letterbox 后映射为质心网格单元 |
| 部署成熟度 | 有视频连通域后处理，但路径/尺寸硬编码 | 只有模型草图和粗略计时 | 固定 shape ONNX，仅导出 logits；预处理/后处理置于图外 |

来源：`bhoke:L62-L77,L93-L97,L101-L110,L145-L160`；`deebuls:Cell 4, Cell 6, Cell 8, Cell 11`。

## 1. 模型结构差异

### 1.1 bhoke：截断轻量 backbone + FOMO 分类头

bhoke 在多个 backbone 的 **1/8 分辨率特征图** 上添加共同的轻量 head。MobileNetV2、MobileNetV3 和 MobileViT 的 head 都是近似的 `1×1 Conv(32) → activation → 1×1 Conv(num_classes)`；MobileNetV2 使用 depthwise convolution 与 inverted residual block，MobileNetV3 使用 Keras 内置骨干，MobileViT 引入 attention 和 patch reshape。`SqueezeFOMO` 的输入预处理和输出激活与其他分支不一致。来源：`bhoke:L71-L78,L106-L109`。

其优点是：不做解码器、不恢复高分辨率，网络末端直接产生低分辨率类别热力图，因此激活内存和 CPU 后处理输入都较小。来源：`bhoke:L73,L93,L147-L151`。

### 1.2 deebuls：浅层 U-Net 风格 encoder-decoder

deebuls 的 `FOMO` 依次使用四个 `3×3, stride=2` 卷积：

```text
input → conv1(3→4, /2) → conv2(4→8, /4)
      → conv3(8→16, /8) → conv4(16→32, /16)
      → upsample + conv5 + concat(out3)
      → upsample + conv6 + concat(out2)
      → upsample + conv7 + concat(out1)
      → conv8(8→1)
```

来源：`deebuls:Cell 6:L10-L26,L32-L52`。

它保留了 `/8`、`/4`、`/2` 三个 encoder 特征作为 skip connection，随后用双线性上采样恢复到输入的一半尺寸。对于高度和宽度均可被 16 整除的输入，输出空间尺寸为 `H/2 × W/2`。在 Cell 8 的 `[1,3,512,384]` 输入上，逐层 shape 为：

```text
[1,3,512,384]
→ [1,4,256,192]   # conv1
→ [1,8,128,96]    # conv2
→ [1,16,64,48]    # conv3
→ [1,32,32,24]    # conv4
→ [1,1,256,192]   # 三次 ×2 上采样后的 conv8 输出
```

来源：`deebuls:Cell 6:L11-L26,L35-L52`；输入示例：`deebuls:Cell 8:L1-L3`。

该网络没有激活函数、BatchNorm 或显式输出激活；因此连续卷积和双线性插值构成的网络缺少通常用于视觉特征提取的非线性。Cell 4 的文字描述提到 sigmoid，而 Cell 6 的实际 `forward` 只返回 `conv8` 输出，两者不一致。来源：`deebuls:Cell 4`；`deebuls:Cell 6:L10-L26,L50-L52`。

## 2. 输入、输出与下采样倍率

| 项目 | bhoke 参考 | deebuls 参考 | 本项目第一版接口 |
|---|---|---|---|
| 输入布局 | `[B,H,W,3]`、float、loader 中 `/255` | `[B,3,H,W]` | `[B,3,S,S]`，`float32`，范围 `[0,1]` |
| 默认/示例输入 | 配置示例含 `224×224` 与 `360×640` | Cell 8：`[1,3,512,384]` | 默认 `S=160`；模型接受 96/160/192/224 等 `8` 的倍数；每个 ONNX 产物固定一种输入尺寸 |
| 输出布局 | `[B,H/8,W/8,C]` | `[B,1,H/2,W/2]`（尺寸可整除时） | `[B,1+N,S/8,S/8]` logits |
| 主输出 stride | 8 | 2 | 8 |
| 目标表示 | one-hot 分类掩码 | 未定义 | class-index 网格标签，背景为 0 |

bhoke 的 MFF loader 分配 NHWC 图像和掩码，其中掩码尺寸为输入尺寸除以 8；其模型 head 与标签空间尺度一致。来源：`bhoke:L64-L68,L73,L93,L115-L117`。deebuls 代码采用 PyTorch NCHW，但输出只恢复到半分辨率，不满足本仓库的 stride=8 约束。来源：`deebuls:Cell 6:L11-L26,L35-L52`。

## 3. background 与类别表达

### bhoke

bhoke 的配置以 `NUM_CLASSES=2` 与两项类别权重为例，训练 IoU 仅计算索引 `1..C-1`，可推断通道 `0` 被当作 background，剩余通道表示前景类别。MobileNetV2/V3/MobileViT head 使用 softmax 输出。来源：`bhoke:L56-L60,L82-L87,L93-L95,L117,L140-L142,L158`。

但 MFF loader 的 bbox 掩码是 `(H/8,W/8,C)`，`cv2.rectangle(..., 1, ...)` 会向所有通道填 1，因此不能正确表达互斥的 background/前景 one-hot 标签。来源：`bhoke:L101-L102`。

### deebuls

deebuls 仅输出一个通道。其文字说明是“bee / no bee”的 sigmoid 二分类，但实现中既没有 background 通道，也没有 sigmoid；因此该单通道输出到底是 logits、概率还是掩码没有被代码定义。来源：`deebuls:Cell 4`；`deebuls:Cell 6:L24-L26,L50-L52`。

### 采用的语义

本项目必须使用互斥分类语义：`C=1+N`，其中通道 0 是 background，通道 `1+foreground_class_id` 是前景类。模型返回 logits；训练使用多类损失，推理再 softmax。这个接口在单类和多类情况下保持不变，避免在二值 sigmoid 与多类 softmax 间切换。该决定符合仓库的既定张量合同，并纠正了两份参考实现的输出语义不一致问题。

## 4. bbox/centroid 到训练标签的转换

### 4.1 参考实现的实际做法

- bhoke 的 MFF loader 把 JSON bbox 的左上和右下坐标缩放到 `/8` 网格，再以矩形填充方式写入 mask；它不是 centroid 监督。来源：`bhoke:L64-L68,L101-L102`。
- bhoke 的 VIRAT loader 读取已有的 PNG 类别掩码，最近邻缩放到 `/8` 后 one-hot；它也不从 bbox 生成 centroid。来源：`bhoke:L64-L68`。
- deebuls notebook 未实现数据集、标签生成、训练或评估；末尾仍标注 “ToDo train on a dataset”。来源：`deebuls:Cell 11`。

### 4.2 本项目应采用的转换

YOLOv5 标签的每行是原图归一化 bbox：`[foreground_class_id, x_center, y_center, width, height]`。第一版只定位目标中心，因此训练标签按以下确定性流程生成：

1. 读取原图尺寸 `(W0,H0)`，将 bbox 中心还原到原图像素坐标 `(x0,y0)`。
2. 用与图像完全相同的 letterbox 比例 `r` 和 left/top padding `(p_left,p_top)` 得到输入坐标：`x_lb = r*x0 + p_left`，`y_lb = r*y0 + p_top`。
3. 对 stride `s=8` 映射到输出网格：`gx = floor(x_lb / s)`，`gy = floor(y_lb / s)`，再 clamp 到 `[0,G-1]`，其中 `G=S/8`。
4. 初始化标签 `target[B,G,G]` 为 0（background），然后写入 `target[gy,gx] = 1 + foreground_class_id`。

这与“bbox 区域全部填满”的 bhoke MFF 做法不同：bbox 大小不进入分类标签，FOMO 第一版只学习类别与中心位置。bbox 面积、宽高和 letterbox 元数据仍须保留用于诊断、评估和坐标反变换。

一个网格单元不能表示两个不同类别中心。实现时必须检测并报告这类冲突：不同类冲突是无效样本错误；同类冲突可写入相同标签，但必须计数并在数据验证报告中显式呈现，不能静默忽略。

## 5. 可复用的工程思想

1. **配置驱动。** bhoke 使用基础配置与实验 YAML 的组合，适合将输入尺寸、类别、loss、阈值、导出尺寸等参数移出源码。来源：`bhoke:L44-L60,L89-L90`。
2. **可替换 backbone 与稳定 head 合同。** backbone 工厂与统一 `/8` head 使模型替换不影响数据和后处理接口。来源：`bhoke:L71-L77,L91-L94,L140-L142`。
3. **低分辨率输出与图外后处理分离。** bhoke 在视频推理中把低分辨率类别图阈值化、做连通域并映射回原图；这个边界应保留，但移除硬编码的尺寸、阈值和路径。来源：`bhoke:L97,L110,L151,L155-L160`。
4. **训练与部署分离。** 不将数据加载、损失和 OpenCV 后处理塞进导出图；ONNX 只表达固定 shape 的模型前向。来源：`bhoke:L134,L155-L160`。
5. **skip connection 是可选的定位增强手段。** deebuls 展示了利用 encoder 特征恢复空间细节的思路；它可作为精度不足时的候选实验，而不是第一版默认架构。来源：`deebuls:Cell 6:L39-L50`。

## 6. 不应直接复制的部分

| 参考 | 不应复制的做法 | 原因与替代 |
|---|---|---|
| bhoke | MFF 多通道矩形填充 | 破坏 one-hot 语义；改为 class-index centroid grid 标签 |
| bhoke | NHWC、模型内/loader 内混杂归一化、部分 head 已 softmax | PyTorch/ONNX 统一 NCHW，模型只输出 logits，归一化只在预处理发生一次 |
| bhoke | 配置字段存在但训练入口未使用 | 新项目必须校验 YAML 并显式传递每个已声明训练参数 |
| bhoke | 独立排序图像和标注、硬编码推理路径/尺寸/阈值 | 用 ID 配对，路径/阈值/输入尺寸均由 YAML 或调用参数提供 |
| bhoke | MobileViT 的 attention 与静态 reshape 作为首选部署模型 | 增加 ONNX 和 CPU 风险；第一版优先 depthwise MobileNetV2 |
| deebuls | `/2` 输出 stride | 与本项目 `/8` 热力图合同不符，显著增加输出与后处理成本 |
| deebuls | 无非线性、无归一化、无训练/标签/评估 | 不是可训练基线；第一版使用标准 MobileNetV2 block 与独立训练模块 |
| deebuls | 单通道且未定义 sigmoid/logit 语义 | 无法自然扩展多类；改为 `1+N` logits + softmax |
| deebuls | 依赖上下采样与 concat 的隐式尺寸可配 | 非 16 倍数输入可能 skip shape 不匹配；第一版不使用 decoder |

来源：`bhoke:L101-L110,L113-L143,L155-L160`；`deebuls:Cell 4, Cell 6, Cell 11`。

## 7. ONNX 与 Raspberry Pi 5 兼容性

### bhoke 的启示

MobileNetV2/V3 的深度可分离卷积与轻量 1×1 head 是较好的 CPU 起点；MobileViT 的 attention 和 reshape 更适合作为后续实验项。部署时应固定输入 shape、batch=1，导出模型 logits，保持 softmax、阈值、连通域与坐标映射在 ONNX 图外。来源：`bhoke:L145-L160`。

### deebuls 的启示

其 `Conv2d`、双线性 `Upsample` 和 `concat` 都是常见图算子，但 `/2` 输出会扩大 decoder 激活与后处理图尺寸；双线性插值的导出还必须固定 `align_corners` 语义。由于它没有 ONNX 验证、训练或部署代码，不能作为直接部署基线。来源：`deebuls:Cell 6:L11-L26,L39-L50`；`deebuls:Cell 9`。

### 本项目部署合同

- 首个导出模型固定为 `float32 [1,3,160,160] → [1,2,20,20]` logits。
- 每个其他输入尺寸产生独立 ONNX 文件，不以动态高宽替代固定导出。
- ONNX Runtime CPU 与 PyTorch 必须对相同预处理输入比较 logits；数值容差、opset 和配置来源应记录在导出测试中。
- letterbox、softmax、连通域、置信度筛选和坐标反变换保留在宿主推理层，以便 Raspberry Pi 侧诊断并避免把非模型逻辑导入 ONNX 图。

## 8. 单类与多类的实现差异

| 项目 | 单类 `creature` | 多类扩展 |
|---|---|---|
| 类别数 | `N=1` | `N=K`，来自 YAML 类别表 |
| 输出 | `[B,2,G,G]` | `[B,1+K,G,G]` |
| 标签值 | `0=background`，`1=creature` | `0=background`，`1..K` 为各前景类别 |
| loss | 同一多类 focal cross-entropy | 同一 loss，仅扩展 class weights |
| 后处理 | foreground 仅通道 1 | 每个非背景 argmax 类别分别聚合连通域 |
| 冲突 | 同类同格可计数记录 | 异类同格必须报错；同类同格必须计数 |

因此不建议为单类单独实现 sigmoid/BCE 路径；从第一天起保持 `background + N classes` 的 softmax 合同，才不会在扩展多类时更换输出层、loss 和 ONNX 接口。

