# Stage E：Alternative FOMO Backbone Screening 设计

## 目标与不变量

Stage E 在不改变 D2 数据、损失、增强、优化、checkpoint-selection 和 evaluator 语义的前提下，新增 `mobilenet_v3_small_fomo` 与 `squeezenet1_1_fomo`。它们的共同输入为 RGB `float32 [B,3,192,192]`，共同输出为 raw logits `float32 [B,8,24,24]`，共同输出 stride 为 8。D2 的 `mobilenet_v2_lite`、`mobilenet_v2_fomo` 与既有 checkpoint 格式保持不变。

本阶段是工程候选比较，而不是纯 backbone 单变量因果消融：D2 使用 EI/Keras MobileNetV2 H5，而新候选使用各自官方 torchvision ImageNet 权重。新模型只允许 validation-only 筛选，绝不访问 test split。

## 小范围兼容接口

不重写既有两个模型。新增模块只实现符合现有模型约定的网络：公开 `backbone` 和 `head` 属性；backbone 提供 `output_channels`、`output_stride`、`cut_point`；model 提供 `input_size`、`num_classes`、`pretrained`、`initialization` 与 `pretrained_load_report`。现有 `build_fomo_model()` 仅增加显式分支，未知名称仍明确失败。

`ModelConfig` 保持旧字段兼容，并增加预训练来源说明字段：`pretrained_format`、`pretrained_torchvision_version`、`pretrained_weights_enum`、`pretrained_url`。仅 Stage E 预训练模型要求这些值；MobileNetV2 H5 配置不变。`describe_model()` 因此能把 model identity、cut point、参数量和预训练 provenance 写入 checkpoint metadata；既有 snapshots 的精确 metadata 比较继续拒绝跨模型恢复。

## 预训练与公平性

配置使用受控环境变量指向只读官方 `.pth`：`FOMO_MOBILENET_V3_SMALL_WEIGHTS` 和 `FOMO_SQUEEZENET1_1_WEIGHTS`。源文件哈希写在 YAML，路径不写入源码或 Git。加载器拒绝：不存在文件、哈希不匹配、非 mapping payload、torchvision 版本/enum/URL 不匹配、任何完整权重或截断前缀的 missing/unexpected key。classifier 权重不属于 FOMO encoder，必须在完整官方 state dict 验证之后被有意忽略；FOMO head 始终使用 PyTorch 初始化。

## 验证门

每个新模型必须通过：config/registry、CPU/CUDA（可用时）、192 输入 shape、loss/backward/AMP finite、random 与预训练、provenance 拒绝、checkpoint/resume、旧模型回归及参数量稳定测试。随后对 D2 和两个新模型导出静态 batch=1 ONNX（opset 17），以 ORT CPU 执行，并用 `rtol=1e-4`、`atol=1e-5` 比较 logits。任一新候选导出或 ORT parity 不通过，就不开始该候选训练。

CPU benchmark 固定输入 192、batch 1、warmup 和重复次数，分别记录默认线程与单线程的 PyTorch/ORT latency、参数量、state_dict 与 ONNX 大小。结果仅是开发机代理，不能代替 Raspberry Pi 5 实测。

## Validation-only 筛选规则

按顺序：MobileNetV3-Small seed 42，再 SqueezeNet1.1 seed 42。两者使用从 D2 复制的 60 epochs、AMP、optimizer、scheduler、loss、object weight、snapshots、PR-AUC epoch selection、validation strict threshold selection 和 parity-clean protocol。主指标为 validation Strict F1；进入后续阶段的性能门槛是比 D2 `0.422235` 高至少 0.01，效率门槛是 F1 不低于 D2 超过 0.01 且 latency/params/ONNX size 至少一项改善 20%/30%/30%。
