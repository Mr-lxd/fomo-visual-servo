# FOMO 训练与验证流程设计

## 范围

实现 YAML 驱动的 CPU/CUDA FOMO 训练、验证、checkpoint 和恢复流程。训练使用现有 YOLOv5 FOMO 数据集，输入为 RGB `float32 [B,3,S,S]`，标签为 class-index `int64 [B,S/8,S/8]`，模型返回 logits `float32 [B,1+N,S/8,S/8]`。

## YAML 合同

`dataset` 新增 `class_mode`（`merge_single` 或 `preserve`）和 `merged_class_name`。训练创建的数据集必须生成与 YAML `dataset.classes` 完全一致的类别表。

```yaml
loss:
  name: focal_cross_entropy       # weighted_cross_entropy | focal_cross_entropy
  gamma: 2.0                      # focal loss 的非负指数
  class_weights: [1.0, 4.0]       # 长度必须是 background + N classes

training:
  seed: 42
  batch_size: 16
  epochs: 50
  output_dir: outputs/aquarium_creature
  resume: null
  early_stopping_patience: 10     # 0 表示关闭
  early_stopping_min_delta: 0.0
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
  scheduler:
    name: step_lr                 # none | step_lr
    step_size: 10
    gamma: 0.5
```

`--device` 和 `--resume` CLI 参数分别覆盖 YAML `training.device`、`training.resume`。其他可调参数只来自 YAML。

## 数据、损失与指标

DataLoader collate 将 `FOMOSample` 列表转换为 `FOMOBatch(images, targets)`。加权 CE 使用 `torch.nn.functional.cross_entropy` 和长度 `1+N` 的 class weight。Focal CE 使用相同的类别权重，并计算 `-(1-p_t)^gamma * log(p_t)`；`gamma=0` 与加权 CE 一致。

验证指标为前景网格级 micro 统计：

```text
TP = predicted == target and target > 0
FP = predicted > 0 and predicted != target
FN = target > 0 and predicted != target
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2PR / (P + R)
```

该定义同时适用于单类和多类；背景不直接参与 F1。

## 训练与恢复

开始时设置 Python、NumPy、PyTorch CPU/CUDA 随机种子，并以受控 generator/worker seed 构建 DataLoader。每个 train step 使用已有运行时 API：`model.to(device)`、`images.to(device, non_blocking=True)`、`targets.to(device, non_blocking=True)` 与 CUDA AMP autocast。反向传播后、optimizer step 前，逐一检查所有存在的参数梯度为有限数；发现 NaN/Inf 立即抛出包含参数名的异常。

每个 epoch 运行 train 和 eval，写入 `history.csv`：`epoch,train_loss,val_loss,precision,recall,f1,learning_rate`。每轮总是原子式保存 `last.pt`；validation F1 严格提升时保存 `best_val_f1.pt`。checkpoint 含模型、优化器、scheduler、scaler、epoch、best F1、早停计数和随机状态。恢复时从 checkpoint 下一 epoch 开始，保留 history，恢复状态并继续早停判定。

## 早停

`val_f1 > best_val_f1 + min_delta` 才视为改善。`early_stopping_patience > 0` 时，连续未改善次数达到 patience 后停止；零明确关闭早停。最佳 checkpoint 永远按 F1 而非最后 epoch 保存。

## 测试与 smoke test

- CE/focal 数值、权重长度和非法配置。
- 单类/多类前景 micro 指标。
- FOMOSample collate shape/dtype。
- NaN/Inf 梯度拒绝。
- 合成 YOLO fixture 的 CPU 2 epoch 训练：生成 `last.pt`、`best_val_f1.pt`、两行 history，并验证 resume 可继续。
- CUDA 可用时仅增加设备/AMP smoke；所有主测试不依赖 CUDA。

## 自检

- 无自定义 C++/CUDA op 或绝对数据路径。
- checkpoint 与 CSV 的输出路径只能来自 YAML 或 CLI。
- 不将后处理、ONNX 或真实大规模训练纳入本任务。
