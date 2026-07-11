# bhoke/FOMO 项目结构与迁移笔记

## 0. 说明与源码范围

本文分析的源码根目录是用户提供的 FOMO 参考实现目录（本公开仓库不记录其本机绝对路径）。

文中位置均使用 `文件路径:L起始-L结束` 标注；目录结构依据该目录下的实际文件树，代码结论依据对应源码行。该仓库是 Keras/TensorFlow 实现，训练入口直接从 `utils.losses` 导入损失函数，因此没有独立的顶层 `losses/` 目录【源码：`train.py:L4-L8`；`utils/losses.py:L1-L3`】。

## 1. 项目结构概览

```text
FOMO-main/
├── train.py                         # 训练入口、数据集/模型/损失/回调组装
├── predict.py                       # 单图推理示例
├── predict_video.py                 # 视频逐帧推理与连通域后处理
├── backbones/
│   ├── mobilenetv2.py               # 手写 MobileNetV2 + FOMO head
│   ├── mobilenetv3.py               # Keras MobileNetV3Small + FOMO head
│   ├── mobilevit.py                 # MobileViT-XXS 风格网络 + FOMO head
│   ├── squeezenet.py                # SqueezeNet 风格网络
│   └── __init__.py                  # 导出四个模型工厂
├── configs/
│   ├── default.py                   # YACS 默认配置
│   ├── mff/*.yaml                   # Mediterranean Fruit Fly 实验配置
│   └── VIRAT/*.yaml                 # VIRAT 实验配置
├── dataloaders/
│   ├── mff.py                       # JSON bounding box → 1/8 网格掩码
│   ├── virat.py                     # TensorFlow Dataset 数据管线
│   ├── virat_keras.py               # Keras PyDataset 版本
│   └── __init__.py                  # 当前仅导出 mff、virat
├── utils/
│   ├── losses.py                    # weighted xent/dice/focal loss
│   ├── callbacks.py                 # warmup + cosine 学习率函数
│   └── data_utils.py                # 数据集下载函数
├── demos/                           # GIF/视频演示文件
├── samples/                         # 示例视频
└── virat_mobilenetv2.keras          # 已保存的 Keras 权重/模型文件
```

依据：`README.md:L1-L30`、`backbones/__init__.py:L1-L4`、`dataloaders/__init__.py:L1-L2`、以及上述源码根目录的实际文件树。README 将该项目定义为基于 MobileNetV2 的 FOMO，并列出 MobileNetV2、MobileNetV3、MobileViT 三种支持模型【源码：`README.md:L1-L3,L24-L27`】。

## 2. 核心目录与文件职责

### 2.1 `train.py`

`train.py` 是实验编排入口：

- 解析 `--cfg`，将 YAML 合并进 YACS 配置【源码：`train.py:L16-L29`；`configs/default.py:L60-L64`】。
- 通过 `config.DATASET.NAME` 动态取得数据集类，并按 `get_dataset()` 是否存在兼容 Keras `PyDataset` 和 TensorFlow `tf.data.Dataset`【源码：`train.py:L57-L80`】。
- 通过 backbone 名称分派 `MobileFOMOv2`、`SqueezeFOMO`、`MobileFOMOv3`、`MobileFOMOViT`，并根据 `RESUME` 决定加载预训练权重还是保存文件【源码：`train.py:L32-L54`】。
- 使用 `weighted_dice_loss`，创建优化器，设置学习率，以忽略背景类的 `OneHotIoU` 作为指标【源码：`train.py:L82-L91`；`utils/losses.py:L31-L50`】。
- 注册最佳验证 IoU 检查点和 warmup/cosine 学习率调度器，然后调用 `model.fit`【源码：`train.py:L93-L120`；`utils/callbacks.py:L3-L13`】。

因此它更像一个“配置驱动的组装脚本”，而不是把模型、数据和训练循环分别封装成独立服务的训练框架【源码：`train.py:L57-L120`】。

### 2.2 `configs`

`configs/default.py` 定义输出目录、设备/worker、模型、损失、数据集、训练和测试配置节点【源码：`configs/default.py:L8-L58`】；各 YAML 只覆盖具体实验，例如数据集路径、类别数、输入尺寸、backbone、优化器、学习率、类别权重和保存路径【源码：`configs/mff/mff_mobilenetv2.yaml:L1-L30`；`configs/mff/mff_mobilenetv3.yaml:L1-L31`；`configs/VIRAT/virat_mobilenetv2.yaml:L1-L34`】。

其主要职责是让同一训练入口切换数据集、输入尺寸、backbone 和超参数【源码：`train.py:L19-L27,L57-L86`】。需要注意的是，配置中声明的 `MODEL.PRETRAINED`、`TRAIN.MOMENTUM`、`TRAIN.WD`、`TRAIN.NESTEROV` 等字段没有在 `train.py` 的优化器构造处实际传入，迁移时不能把“存在配置项”误认为“已经生效”【源码：`configs/default.py:L18-L21,L43-L47`；`train.py:L85-L87`】。

### 2.3 `dataloaders`

数据加载器负责读取图像和标注、缩放到训练尺寸、构造 1/8 分辨率的分类掩码，并输出图像与 one-hot 标签：

- `mff.py` 读取 `<split>_labels.json`，将 bounding box 映射到 `img_size // 8` 的网格【源码：`dataloaders/mff.py:L10-L25,L30-L53`】。
- `virat.py` 读取 `images/` 与 `annotations/`，用 TensorFlow 解码 JPEG/PNG，图像双线性缩放、掩码最近邻缩放，再 one-hot，并支持同步水平翻转【源码：`dataloaders/virat.py:L4-L17,L19-L47`】；`get_dataset()` 再负责 shuffle、map、batch、prefetch【源码：`dataloaders/virat.py:L49-L59`】。
- `virat_keras.py` 是 Keras `PyDataset` 版本，使用灰度掩码的类别索引再 `np.eye` one-hot【源码：`dataloaders/virat_keras.py:L13-L30,L36-L60`】。
- `dataloaders/__init__.py` 当前只把 `mff` 和 `virat` 暴露给动态工厂，未暴露 `virat_keras`【源码：`dataloaders/__init__.py:L1-L2`】。

### 2.4 `backbones`

四个模型都试图在 backbone 的 1/8 分辨率特征上接轻量 FOMO 输出头；MobileNetV2、MobileNetV3、MobileViT 的共同头部基本是 `1×1 Conv(32) + 1×1 Conv(num_classes)`【源码：`backbones/mobilenetv2.py:L128-L137`；`backbones/mobilenetv3.py:L38-L42`；`backbones/mobilevit.py:L101-L112`】。

- `mobilenetv2.py` 手写 inverted residual、depthwise convolution、残差连接，并支持 `alpha` 和 ImageNet 权重【源码：`backbones/mobilenetv2.py:L43-L111,L114-L158`】。
- `mobilenetv3.py` 复用 Keras `MobileNetV3Small`，尝试选取 1/8 分辨率层；找不到固定层名时按空间尺寸动态选择【源码：`backbones/mobilenetv3.py:L17-L36`】。
- `mobilevit.py` 组合卷积、倒残差、Multi-Head Attention、patch unfold/fold 和融合层【源码：`backbones/mobilevit.py:L3-L76`】，模型注释明确以 1/8 分辨率作为 FOMO head 输入【源码：`backbones/mobilevit.py:L78-L110`】。
- `squeezenet.py` 通过 fire module 降低参数量，经过三次主要下采样后输出预测特征【源码：`backbones/squeezenet.py:L5-L30,L33-L50`】。

### 2.5 `losses`

实际对应文件是 `utils/losses.py`，不是 `losses/` 目录【源码：`train.py:L7-L8`；`utils/losses.py:L1-L3`】：

- `weighted_xent` 对预测概率做裁剪、取 log，并按 one-hot 标签和类别权重加权【源码：`utils/losses.py:L5-L28`】。
- `weighted_dice_loss` 在 batch、高、宽三个维度上计算每个类别的 Dice，再按类别权重求加权平均【源码：`utils/losses.py:L31-L50`】。
- `weighted_focal_loss` 对 one-hot 概率做 focal 调制和类别加权【源码：`utils/losses.py:L53-L81`】。
- 当前训练入口只实际使用 `weighted_dice_loss`，另外两个函数只是可选实现【源码：`train.py:L7,L82-L89`】。

## 3. 可复用的工程设计

1. **配置驱动实验。** 默认配置与数据集/模型 YAML 分离，训练入口只接收 `--cfg`，适合批量复现实验【源码：`train.py:L16-L29`；`configs/default.py:L8-L64`】。
2. **轻量模型工厂。** `backbones/__init__.py` 统一导出模型，`get_model_by_name()` 将配置字符串映射到模型构造函数，便于替换 backbone【源码：`backbones/__init__.py:L1-L4`；`train.py:L32-L54`】。
3. **统一低分辨率输出接口。** 各 backbone 都围绕 1/8 特征图接分类 head，数据标签也按 `img_size // 8` 构造，模型和标注的空间尺度关系清晰【源码：`dataloaders/mff.py:L17-L22`；`dataloaders/virat.py:L14-L16`；`backbones/mobilenetv2.py:L134-L137`】。
4. **数据管线适配层。** 训练入口通过 `get_dataset()` 兼容两类 Keras/TensorFlow 数据源，保留了不同数据集实现的自由度【源码：`train.py:L57-L80`】。
5. **类别不平衡处理。** 类别权重从 YAML 传入加权 Dice，且 IoU 指标只监控非背景类，符合前景稀疏的边缘检测场景【源码：`configs/mff/mff_mobilenetv2.yaml:L27-L30`；`train.py:L83-L90`；`utils/losses.py:L42-L48`】。
6. **训练期间保存最佳模型并调度学习率。** 以验证 IoU 最大值保存模型，并使用 warmup 后 cosine annealing，适合作为迁移后的默认训练策略【源码：`train.py:L93-L110`；`utils/callbacks.py:L3-L13`】。
7. **后处理与模型输出解耦。** 视频示例先按类别取低分辨率概率图，再阈值化、连通域分析、将质心乘以 8 映射回图像坐标；这为 PyTorch/ONNX 迁移提供了明确的后处理边界【源码：`predict_video.py:L57-L69`】。

## 4. 不应直接照搬的问题

1. **MFF 掩码写法会污染类别通道。** `mask` 是 `(H,W,num_classes)`，但 `cv2.rectangle(..., 1, ...)` 对所有通道写入 1；对于 two-class one-hot 标签，这不是“只写前景通道”的语义，迁移时应明确写 `mask[..., class_id]`，并保持背景通道为 1/0 的一致定义【源码：`dataloaders/mff.py:L37-L50`】。
2. **MFF 的增强参数目前没有真正生效。** 构造函数保存了 `augment`，但增强代码被注释，`__getitem__` 直接返回原图和掩码；不能仅在 PyTorch 配置中保留同名字段就认为增强已迁移【源码：`dataloaders/mff.py:L22-L25,L51-L65`】。
3. **不同 loader 的尺寸约定不一致。** `tf.image.resize` 使用 `(height,width)`，OpenCV 的 `cv2.resize` 使用 `(width,height)`；MFF 直接传 `self.img_size`，VIRAT Keras 版本传 `self.img_size[::-1]`，非正方形输入下容易产生转置或标注错位【源码：`dataloaders/mff.py:L17,L41-L43`；`dataloaders/virat.py:L14,L28-L34`；`dataloaders/virat_keras.py:L19,L46-L53`】。
4. **VIRAT 图像和标注只分别排序，没有按文件名校验配对。** 文件新增、缺失或命名不完全一致时可能静默错配；PyTorch Dataset 应按 stem/ID 建立显式映射并检查数量【源码：`dataloaders/virat.py:L10-L13`；`dataloaders/virat_keras.py:L16-L25`】。
5. **`virat_keras.py` 直接导入会失败。** 它导入 `utils.data_utils.Augment`，但该文件实际只定义 `download_dataset`；同时它没有被 `dataloaders/__init__.py` 导出【源码：`dataloaders/virat_keras.py:L10-L13`；`utils/data_utils.py:L5-L15`；`dataloaders/__init__.py:L1-L2`】。
6. **SqueezeNet 分支的输入形状和其他 backbone 不一致。** 训练分派时传入二维 `IMAGE_SIZE`，而 `SqueezeFOMO` 直接把它作为 `Input(shape=...)`，没有补 RGB 通道；此外它在模型内做 `/255`，其他模型由 loader 做归一化，输出也没有 softmax【源码：`train.py:L39-L42`；`backbones/squeezenet.py:L33-L49`；`dataloaders/mff.py:L36-L42`】。
7. **训练配置并非全部生效。** `MODEL.PRETRAINED` 没有被 `get_model_by_name()` 使用；优化器只按字符串构造并设置学习率，YAML 中的 weight decay、momentum、Nesterov 没有传入【源码：`configs/mff/mff_mobilenetv2.yaml:L15-L30`；`configs/default.py:L18-L21,L43-L47`；`train.py:L32-L46,L85-L87`】。
8. **恢复训练逻辑过于粗糙。** `RESUME=True` 时把 `BEST_SAVE_PATH` 作为完整模型权重路径，但没有恢复优化器、epoch、调度器状态；因此 PyTorch 迁移时应使用包含 model/optimizer/scheduler/epoch 的 checkpoint 字典【源码：`train.py:L32-L34,L93-L110`】。
9. **MobileViT 的 patch reshape 依赖静态、可整除尺寸。** `num_patches` 和多次 `Reshape` 直接从静态空间尺寸计算，动态输入或不能被 `patch_size=2` 整除的尺寸会影响导出和运行【源码：`backbones/mobilevit.py:L47-L67`】。
10. **推理脚本包含硬编码和接口漂移。** `predict.py` 加载 `model.h5`，而默认训练配置保存 `best.keras`；图片路径也是机器相关的绝对路径【源码：`predict.py:L6-L10`；`configs/default.py:L49-L52`】。视频脚本还固定 640×360、阈值 0.9 和 8 倍坐标缩放，不能直接当成通用部署程序【源码：`predict_video.py:L9-L17,L42-L67`】。
11. **优化器调用的语义不能原样移植。** Keras 这里直接使用 `optimizers.get(config.TRAIN.OPTIMIZER)`，并在之后修改学习率；PyTorch 需要显式选择 Adam/SGD 并传入每个配置字段，否则训练结果不可比【源码：`train.py:L85-L90`；`configs/mff/mff_mobilenetv2.yaml:L19-L26`】。

## 5. 迁移为 PyTorch 项目的建议

### 5.1 先固定接口和张量语义

建议统一以下约定：输入为 `float32 [N,3,H,W]`、范围 `[0,1]`；输出为未归一化 logits `[N,C,H/8,W/8]`；标签优先使用类别索引 `[N,H/8,W/8]`，若继续使用 Dice，再在 loss 内转换 one-hot。这样可以避免 Keras 当前的 NHWC、概率输出、SqueezeNet 内部归一化混在一起【源依据：`dataloaders/mff.py:L36-L53`；`dataloaders/virat.py:L24-L47`；`backbones/mobilenetv2.py:L135-L137`；`backbones/squeezenet.py:L33-L49`】。

### 5.2 推荐的 PyTorch 目录

```text
fomo_torch/
├── configs/                 # dataclass/YAML 解析与校验
├── datasets/                # MFFDataset、VIRATDataset
├── models/
│   ├── backbones/           # MobileNetV2/V3、SqueezeNet、MobileViT
│   └── fomo.py              # 统一 1/8 head
├── losses/                 # weighted dice / CE / focal
├── engine/                  # train/eval/checkpoint/scheduler
├── export.py                # TorchScript/ONNX 导出
└── infer.py                 # 预处理、模型调用、连通域后处理
```

该拆分保留原项目“配置—数据—backbone—head—loss—后处理”的边界，同时把当前散落在 `train.py` 和推理脚本中的状态显式化【源依据：`train.py:L57-L120`；`utils/losses.py:L5-L81`；`predict_video.py:L52-L69`】。

### 5.3 迁移步骤

1. **配置层：** 将 YACS 节点映射为 dataclass 或 YAML 配置对象；明确区分 `DATASET.IMAGE_SIZE` 与 `TRAIN.IMAGE_SIZE`，并删除未使用字段或在代码中补齐其语义【源依据：`configs/default.py:L28-L52`；`dataloaders/mff.py:L17-L21`；`dataloaders/virat.py:L14-L17`】。
2. **数据层：** 实现 `torch.utils.data.Dataset`，在 `__getitem__` 中返回 CHW 图像和低分辨率 target；用 `(W,H)` 明确调用 OpenCV，用文件 stem 配对 VIRAT；修复 MFF 前景通道写入，并将增强和图像/标签几何变换绑定【源依据：`dataloaders/mff.py:L30-L53`；`dataloaders/virat.py:L19-L47`】。
3. **模型层：** 为每个 backbone 实现 `nn.Module`，提取与原实现相同的 1/8 特征；统一使用 `FOMOHead(Conv2d(in,32,1) → activation → Conv2d(32,C,1))`，训练时返回 logits，不在模型末尾固定 softmax【源依据：`backbones/mobilenetv2.py:L134-L137`；`backbones/mobilenetv3.py:L38-L42`；`backbones/mobilevit.py:L107-L112`】。
4. **损失层：** 将加权 Dice 改写为 NCHW 轴 `(0,2,3)`；如果使用 `CrossEntropyLoss`，传类别索引和 logits，不要先做概率裁剪；若保留 one-hot Dice，则统一在 loss 内部调用 softmax，并验证类别权重长度等于 `C`【源依据：`utils/losses.py:L31-L50`；`configs/mff/mff_mobilenetv2.yaml:L27-L30`】。
5. **训练层：** 显式构造 Adam/SGD，应用学习率、weight decay、momentum、Nesterov；实现 train/eval epoch、非背景 IoU、best checkpoint 和 warmup+cosine scheduler，并保存 epoch、优化器和 scheduler 状态【源依据：`train.py:L82-L120`；`configs/default.py:L41-L52`；`utils/callbacks.py:L3-L13`】。
6. **一致性验证：** 先固定一批输入，逐项对比 Keras 与 PyTorch 的预处理、输出尺寸、类别通道、低分辨率掩码和后处理质心；确认输出空间尺寸确实为输入的 1/8 后，再进行权重转换或重新训练【源依据：`dataloaders/mff.py:L21,L44-L50`；`dataloaders/virat.py:L15,L30-L36`；`predict_video.py:L57-L67`】。

## 6. Raspberry Pi 部署影响

- **有利因素：** MobileNetV2/MobileNetV3/SqueezeNet 使用深度可分离卷积、fire module 或轻量 head；输出只保留 1/8 网格，推理和后处理的数据量较小，适合优先评估 batch=1、固定输入尺寸的 CPU 部署【源码：`backbones/mobilenetv2.py:L75-L110`；`backbones/mobilenetv3.py:L17-L21`；`backbones/squeezenet.py:L5-L30`；`dataloaders/mff.py:L21-L22`】。
- **运行时成本：** 原始项目依赖 TensorFlow/Keras、OpenCV、NumPy，并在视频脚本中执行逐帧解码、连通域和窗口显示；在 Raspberry Pi 上应把训练依赖移除，只保留轻量推理运行时和必要的图像后处理【源码：`train.py:L4-L10`；`predict_video.py:L1-L5,L15-L17,L63-L71`】。
- **优先模型：** 首轮建议从 MobileNetV2 或 MobileNetV3 开始；MobileViT 含 Multi-Head Attention 及多次 patch reshape，虽然可能提升表达能力，但会增加 CPU/内存压力和导出风险【源码：`backbones/mobilevit.py:L26-L44,L53-L67`】。这属于基于算子结构的工程判断，最终延迟和精度必须在目标 Pi 型号上实测。
- **量化与输入：** 固定 `H×W`、batch=1、统一 `[0,1]` 预处理，并在量化前后检查小目标/稀疏前景的召回；类别权重只影响训练，不应出现在部署图中【源依据：`dataloaders/mff.py:L36-L42`；`utils/losses.py:L42-L48`】。
- **后处理不可丢失：** 仅导出网络不能得到最终目标点；部署端仍需对前景通道做 softmax/阈值、连通域分析和质心坐标映射，并保持原脚本中的输出步长约定【源码：`predict_video.py:L57-L69`】。

## 7. ONNX 部署影响

1. **导出边界应是“模型 logits”，不是完整训练对象。** PyTorch 版本应导出单一 `NCHW` 输入到 `[N,C,H/8,W/8]` logits 的前向图；softmax、阈值、连通域和坐标映射放在 ONNX 外部，便于替换后端并避免把 OpenCV 后处理塞进计算图【源依据：`train.py:L87-L91`；`predict_video.py:L57-L69`】。
2. **固定形状更稳妥。** MobileViT 的 `num_patches` 和 `Reshape` 依赖静态尺寸及可整除关系；在 ONNX 首版导出中固定输入尺寸，待验证后再尝试动态高宽【源码：`backbones/mobilevit.py:L47-L67`】。
3. **预处理必须写入部署合同。** MFF/VIRAT loader 将图像转 RGB、缩放并除以 255；SqueezeNet 又在模型内做一次 `Rescaling`，所以迁移后的 ONNX 前处理必须先统一，否则可能出现重复归一化或颜色通道错误【源码：`dataloaders/mff.py:L40-L42`；`dataloaders/virat.py:L24-L29`；`backbones/squeezenet.py:L35-L37`】。
4. **输出语义必须显式记录。** MobileNet 系列头部直接输出 softmax 概率，SqueezeNet 输出未激活预测；PyTorch/ONNX 若统一输出 logits，部署端要统一执行 softmax，并按类别索引选择前景通道【源码：`backbones/mobilenetv2.py:L135-L137`；`backbones/mobilenetv3.py:L39-L42`；`backbones/mobilevit.py:L109-L110`；`backbones/squeezenet.py:L47-L50`】。
5. **端到端验收指标应包含几何一致性。** 除了数值误差和分类指标，还要检查输出网格尺寸、阈值后的连通区域数量、质心位置与 Keras 结果是否一致；原视频后处理将质心乘以 8，因此任何下采样层变化都会直接改变部署坐标【源码：`predict_video.py:L57-L67`；`dataloaders/virat.py:L14-L16`】。
6. **Pi + ONNX 的落地路线。** 推荐先在桌面环境完成 PyTorch → ONNX 的固定形状数值对齐，再在 Raspberry Pi 上使用 CPU ONNX 后端测量单帧延迟、内存峰值和视频吞吐；若 MobileViT 导出或性能不稳定，先部署 MobileNetV2/V3，并保留同一 FOMO head 与后处理接口【源依据：`backbones/__init__.py:L1-L4`；`backbones/mobilevit.py:L47-L67`；`predict_video.py:L52-L71`】。

## 8. 最小迁移验收清单

- [ ] MFF 掩码的背景/前景通道经过单元测试，且 one-hot 约束成立【源码：`dataloaders/mff.py:L37-L50`】。
- [ ] 非正方形输入在 OpenCV、PyTorch 和 ONNX 三处的 `(H,W)/(W,H)` 约定一致【源码：`dataloaders/mff.py:L41-L43`；`dataloaders/virat.py:L28-L34`；`dataloaders/virat_keras.py:L48-L53`】。
- [ ] 每个 VIRAT 图像与标注通过 ID 配对，而不是仅靠两个排序列表【源码：`dataloaders/virat.py:L10-L13`】。
- [ ] 输入归一化只发生一次，输出统一为 logits 或统一为概率【源码：`dataloaders/mff.py:L40-L42`；`backbones/squeezenet.py:L35-L37`；`backbones/mobilenetv2.py:L136-L137`】。
- [ ] PyTorch checkpoint 可恢复模型、优化器、调度器和 epoch【源依据：`train.py:L93-L120`】。
- [ ] 固定输入尺寸下，PyTorch 与 ONNX 的输出网格为输入的 1/8，后处理质心与原实现对齐【源码：`dataloaders/mff.py:L21-L22`；`predict_video.py:L57-L67`】。
