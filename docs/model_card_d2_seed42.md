# Model Card: D2 seed42 FOMO candidate

## 模型身份

- 用途：轻量化水下目标 centroid detection / visual servo perception。
- 架构：MobileNetV2 风格 FOMO backbone，alpha `0.35`，输出 stride `8`。
- 输入：RGB `float32 [B,3,192,192]`，使用 letterbox，不使用中心裁剪。
- 输出：logits `float32 [B,8,24,24]`；通道 0 为 background，通道 1–7 为 fish、jellyfish、penguin、puffin、shark、starfish、stingray。
- 后处理：softmax、连通区域/质心解码，输出 centroid、class_id、confidence；正式指标使用 strict one-to-one centroid matching。

## 训练与选择

- 配置：[stage_d2_fomo_ei_w100_pretrained.yaml](../configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml)
- loss：`ei_weighted_xent_legacy`，background weight `1`，object weight `100`。
- optimizer/scheduler：AdamW、learning rate `0.001`、StepLR（step 20、gamma 0.5）。
- 输入尺寸：`192×192`；训练 60 epoch；CUDA AMP；seed `42`。
- checkpoint selection：validation `centroid_pr_auc_macro` 选择 epoch，再在 validation strict one-to-one evaluator 上选择 threshold。
- 当前 candidate：epoch `40`，validation threshold `0.40`。

## 数据与 provenance

- dataset 不在仓库中。
- dataset content hash：`0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`。
- pretrained H5 来源：外部 Edge Impulse/Keras MobileNetV2 pretrained artifact；SHA-256：`a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`。
- H5 只用于 backbone initialization，95 tensors loaded，missing/unexpected 为 `0/0`；head 和 classifier 使用 PyTorch 默认初始化。
- H5 未提交到 Git；license/redistribution status requires confirmation，不应从本仓库或 GitHub Release 重新分发。

## 指标

### Validation

D2 seed42 validation：Strict F1 `0.422235`，Macro F1 `0.382332`，PR-AUC macro `0.218902`，Count MAE `3.992126`。

D2 multi-seed validation（42/123/2027）：Strict F1 `0.418215 ± 0.005413`，sample standard deviation，validation-only。starfish 和 stingray 的 per-class F1 对 seed 更敏感，不能只引用 seed42。

### Locked test

seed42 使用 parity-clean-v1、固定 epoch40 和 validation threshold `0.40`，只评价一次：

- Strict one-to-one：P `0.500000`、R `0.412371`、F1 `0.451977`。
- EI legacy：P `0.554167`、R `0.431118`、F1 `0.484959`。

EI legacy 的 many-to-one 语义只为 Edge Impulse parity 保留，不是正式主指标。seed123/2027 没有运行 test。

## 使用限制与部署状态

- 模型表达的是每个 stride-8 cell 一个类别/质心；同 cell 多目标会发生量化或 collision 信息损失。
- test 结果只适用于记录的 cleaned view、checkpoint、threshold 和 evaluator protocol，不可用于重新调参。
- ONNX 固定输入导出路径已提供，但当前环境缺少 `onnx`/`onnxruntime` 时相关测试会 skip；安装方式见根 [README.md](../README.md)。
- Raspberry Pi 5 尚未完成实际 batch=1 CPU 延迟、内存、功耗和稳定性测试。
- Edge Impulse TFLite parity audit 与本地 D2 模型不是同一模型；其结果不能直接替换本 model card 的正式候选指标。
