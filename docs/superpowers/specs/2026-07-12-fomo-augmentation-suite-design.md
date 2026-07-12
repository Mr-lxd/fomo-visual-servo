# FOMO Online Augmentation Suite Design

## Scope

阶段 A 只整理在线 augmentation framework，不运行正式训练，不修改模型、loss、优化器、scheduler、输入尺寸、stride、标签基本语义或评价协议。

增强只在 train split 的每次样本读取阶段执行。validation/test 直接跳过全部 augmentation。固定执行顺序为：

```text
原始 RGB image + bbox
→ horizontal flip
→ affine
→ color jitter
→ Gaussian blur
→ Gaussian noise
→ letterbox
→ normalize
→ FOMO heatmap
```

## RNG 语义

`YOLOv5FOMODataset` 保存 `current_epoch`，提供 `set_epoch(epoch: int)`。每个样本使用稳定的 64-bit seed：

```text
sample_seed = stable_hash(augmentation_base_seed, epoch, sample_index)
```

hash 使用明确的字节编码和 SHA-256/BLAKE2 派生值，不使用 Python 内置 `hash()`，也不包含 worker id。这样可以保证：

- 同一 epoch、同一 index、同一 base seed 的增强一致；
- 不同 epoch、同一 index 通常不同；
- `num_workers=0/2/4` 对同一 epoch/index 产生相同结果；
- 相同 seed 重新运行可复现；
- 不同 seed 产生不同序列。

训练循环在每个 epoch 开始前调用 `train_dataset.set_epoch(epoch)`。resume 从 checkpoint 的下一个 epoch 继续。当前明确使用 `persistent_workers=False`；如果以后启用 persistent workers，必须显式传播 epoch，禁止使用过期 epoch 状态。

## Preset 与兼容性

`src/fomo_servo/datasets/presets.py` 是内置 preset 的唯一来源，包含：

- `none`：全部 disabled；
- `photometric`：color jitter + Gaussian blur + Gaussian noise；
- `underwater_conservative`：photometric 加 horizontal flip 和 mild affine；
- `custom`：完全使用显式字段。

新配置支持：

```yaml
augmentation:
  enabled: true
  preset: underwater_conservative
  overrides: {}
```

旧的逐项字段继续解析，但当未使用 preset 时发出 deprecation warning。未知 preset、未知 override 路径和类型错误都明确抛出 `ConfigurationError`。`enabled=false` 在 preset 展开后仍强制整个 pipeline no-op。

解析后的最终 augmentation 参数通过序列化 helper 保存到 `config.yaml`/resolved metadata、checkpoint、`training_summary.json` 和 experiment metadata。aug00、aug01、aug02 的非 augmentation 字段以及现有逐项 augmentation 字段不发生静默漂移。

## 变换定义

### Horizontal flip

连续像素坐标使用：

```text
x_min' = W - x_max
x_max' = W - x_min
```

不使用 `W-1`。类别、目标顺序、y 坐标和 bbox 尺寸保持不变。

### Mild affine

使用 OpenCV 常见 2x3 仿射矩阵，围绕 `(W/2, H/2)` 组合 scale、rotation 和像素平移。平移范围为 `±translate_fraction * W/H`，border fill 使用 RGB 配置值。

bbox 四角先变换，再重建 axis-aligned bbox。变换后的 bbox 使用连续坐标并裁剪到 `[0,W] × [0,H]`。visibility 定义为：

```text
clipped_bbox_area / transformed_axis_aligned_bbox_area
```

visibility 小于 `min_visibility` 或裁剪后宽/高小于 1 像素的目标被丢弃。保留目标的 centroid 使用裁剪后 bbox 中心，并重新生成 heatmap 与 collision 统计。

### Color jitter

保持当前 RGB 语义和参数范围。颜色变换只作用于原始 RGB 图像，不改变 bbox 或 heatmap 几何。

### Gaussian blur

使用 OpenCV `GaussianBlur`。kernel 从配置的正奇数列表中由 sample RNG 选择，sigma 在 `[sigma_min, sigma_max]` 均匀采样。输出 shape/dtype 不变。

### Gaussian noise

在 uint8 像素尺度 `[0,255]` 上采样正态噪声，std 在 `[std_min,std_max]` 均匀采样，结果 round/clip 后转换回 uint8。bbox 和 heatmap 不变。

## Metadata 与统计

`FOMOSample.augmentation_metadata` 保留旧字段兼容性，并新增：

- epoch、sample_index、sample_seed；
- color_jitter_applied、horizontal_flip_applied、gaussian_blur_applied、gaussian_noise_applied、affine_applied；
- 采样的 color、blur、noise、affine 参数；
- clipped_bbox_count、dropped_bbox_count；
- pre/post object count 与 collision counts。

metadata 不进入模型 tensor。训练 epoch 聚合并写入 `history.csv` 与 `training_summary.json`，绝对路径不进入 metadata。

## 测试策略

测试使用现有合成 fixture，不依赖真实数据集。覆盖 RNG、resume、worker 一致性、preset 展开/override 错误、旧配置兼容、disabled no-op、blur/noise 数值范围、affine geometry/visibility/clipping、heatmap/collision 重建和配置漂移。

## 可视化

`scripts/visualize_augmentations.py` 在运行时选择至少 16 张真实 train 图片，生成 epoch 0/1/2、photometric、underwater 和 affine geometry 接触表，以及只包含相对路径的 JSON。产物位于 `outputs/experiments/augmentation_suite/visualization/`，不进入 Git。
