# FOMO Augmentation Framework Design

## Scope

建立仅包含 no-op 行为的 augmentation framework。它只扩展配置、dataset 处理边界和可视化输入输出接口；不实现 color jitter、flip、blur、noise 或 affine，也不改变 locked baseline 的任何训练、评价或推理参数。

## Data flow

```text
读取原图与原始 YOLO bbox
→ 仅 train split 调用 AugmentationPipeline
→ letterbox
→ 变换 bbox 与 centroid
→ stride=8 class-index / one-hot heatmap
```

`AugmentationPipeline.apply` 接受原始 RGB `uint8 [H,W,3]`、原图像素坐标 bbox 和由 seed/worker seed 派生的随机源，返回相同语义的 image/bbox。全局 `enabled=false` 时不消费随机数、不复制或修改输入语义，并返回与旧 dataset 路径逐元素一致的结果。

`is_train` 由 dataset split 与 YAML `dataset.train_split` 的匹配决定；任何 validation 或 test split 即使传入 augmentation config 也强制 no-op。

## Configuration

新增顶层 `augmentation`，包含全局开关及五个操作的 enabled/probability/参数字段。schema 完整解析所有字段并校验布尔值、概率 `[0,1]` 和非负强度。当前 locked config 所有 enabled=false、probability=0.0。若未来配置启用尚未实现的算法，pipeline 抛出明确的 `AugmentationNotImplementedError`。

## Determinism and visualization

pipeline 接收显式 NumPy RNG；现有 training seed 和 DataLoader worker seed 仍是随机性的唯一来源。本阶段 no-op 不调用 RNG，因此固定样本的输出和 RNG 状态均可验证。`scripts/visualize_augmentations.py` 读取 dataset sample 并展示原图、bbox、letterbox 和 heatmap 所需的四个接口面板；增强全部关闭时不生成新像素。

## Tests

- 完整 YAML augmentation schema 与非法值错误；
- train/validation/test split 的 no-op 限制；
- enabled=false 时 image、original/letterbox boxes、transform metadata、class-index heatmap、one-hot heatmap、collision count 与现有 dataset 逐元素一致；
- 相同 seed/worker seed 下 no-op 输出与 RNG 状态稳定；
- visualization CLI 使用微型 fixture 成功输出；
- 不执行正式训练，不触碰模型、loss、optimizer、scheduler、threshold 或 AMP。
