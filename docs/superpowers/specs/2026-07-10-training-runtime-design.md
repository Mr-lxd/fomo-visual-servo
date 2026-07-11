# 训练运行时与设备控制设计

## 目标

为后续 FOMO 训练循环提供可复现、可测试的设备、AMP 与 DataLoader 运行时设置，并让 `scripts/train.py` 支持 YAML 配置与 `--device` 覆盖。

## 配置合同

项目规范的主键保持为小写 `training`，并接受大写 `TRAIN` 作为兼容别名；两个键同时出现时必须报错，避免静默选取其一。

```yaml
training:
  device: auto        # auto | cpu | cuda | cuda:N
  amp: true           # 仅 CUDA 生效
  num_workers: 4      # 非负整数
  pin_memory: true    # 仅 CUDA DataLoader 生效
```

`device` 的 CLI 覆盖优先级最高：`python scripts/train.py --config <path> --device cpu`。未提供该参数时使用 YAML 中的值。

## 运行时行为

`create_training_runtime` 调用现有的 `resolve_device`：`auto` 优先 CUDA，否则 CPU；显式 CUDA 在不可用时抛出错误。它返回不可变的 `TrainingRuntime`：

- `device: torch.device`：调用者使用 `model.to(device)`。
- `amp_enabled: bool`：只有 `amp: true` 且有效设备为 CUDA 时为真；CPU 时显式记录为禁用。
- `data_loader_kwargs`：`num_workers` 保持配置值，`pin_memory` 仅在 CUDA 且请求为真时为真。
- `diagnostics`：解释 AMP 或 pin memory 因 CPU 而未启用，避免静默行为。

`move_training_batch(images, targets, runtime)` 对 `images float32 [B,3,S,S]` 与 `targets int64 [B,S/8,S/8]` 执行：

```python
images = images.to(runtime.device, non_blocking=True)
targets = targets.to(runtime.device, non_blocking=True)
```

AMP 上下文仅在 CUDA 可用时创建 `torch.autocast(device_type="cuda", dtype=torch.float16)`；CPU 返回无操作上下文。

## CLI 范围

当前没有 loss、optimizer、epoch 或 checkpoint 实现，因此 `scripts/train.py` 只做训练预检：加载配置、应用覆盖、构造模型并迁移到选择的设备，打印实际设置及限制后成功退出。它不加载数据，也不声称已经训练。后续真实训练循环必须复用本设计的运行时 API。

## 测试

- YAML 小写配置、大写别名与双键冲突。
- `auto`、显式 CPU、无效 device 与当前 CUDA 行为。
- CPU 的 batch/model 迁移；CUDA 可用时的设备迁移和 AMP。
- DataLoader 参数和 CPU 下诊断信息。
- CLI 的 `--device cpu` 覆盖与缺失配置错误；不依赖真实数据集。

## 自检

- 没有引入依赖、CUDA 专用算子或固定机器路径。
- AMP 和 pin memory 的有效状态由设备决定且可报告。
- 范围不扩展到未定义的 loss、optimizer、epoch 或 checkpoint 策略。
