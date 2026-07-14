# Stage C：EI-style object-weight controlled ablation

状态：实现已完成，正式 C1–C4 训练结果待按锁定顺序生成。

## 目的与边界

本阶段只研究 FOMO 输出热力图中 foreground 与 background 的 loss 权重。沿用
`configs/experiments/checkpoint_v2_lite_aug03_locked.yaml` 的数据、模型、增强、
优化器、scheduler、batch、seed、训练轮数和评估协议。C0 是已有 focal baseline，
不重新训练。禁止修改七类 per-class weights、`object_weight` 之外的训练配方、
阈值选择规则或 test 驱动的决策。

实现所在提交：`d10f07e22cb80b61592f46ca963c9de8a156b612`

## loss 语义审计

当前旧 baseline（`src/fomo_servo/losses/classification.py`）是多类 softmax CE：

```text
ce = -log_softmax(logits)[target]
focal = (1 - exp(-ce)) ** gamma
loss = sum(focal * ce * class_weight[target]) / sum(class_weight[target])
```

因此旧 `focal_cross_entropy` 会同时使用 focal modulation 与每个类别的
`class_weights`；旧 `weighted_cross_entropy` 只去掉 focal modulation。它们不是
Edge Impulse 的 object-vs-background 控制变量。

Edge Impulse 的公开 FOMO 文档说明默认 background 权重为 1、非 background
object weight 为 100，并将默认方法称为 `weighted_cross_entropy_with_logits`。
公开性能文档还明确把它与 `sigmoid_cross_entropy_with_logits` 和
`sparse_softmax_cross_entropy_with_logits` 作为不同重平衡方式。TensorFlow 的
`weighted_cross_entropy_with_logits` 是逐通道 one-vs-rest logistic loss，
`pos_weight` 只乘 positive label 项；这与多类 softmax CE 不同。

因此本项目实现两个明确模式：

| 配置 `loss.type` | 数学语义 | per-class weights | focal |
|---|---|---:|---:|
| `weighted_softmax_ce` | 每个 grid cell 的 sparse softmax CE；target=0 用 `background_weight`，target>0 用 `object_weight` | 禁用 | 禁用 |
| `ei_weighted_xent_legacy` | one-hot `binary_cross_entropy_with_logits`；background positive 用 `background_weight`，每个 foreground positive channel 用 `object_weight`，negative channel 项保持 1 | 禁用 | 禁用 |

旧 `focal_cross_entropy` 与旧 `weighted_cross_entropy` 保持向后兼容。所有模式
的 collision policy 仍由数据集配置决定，loss 不静默重写标签。

参考：

- [Edge Impulse FOMO object weighting](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo)
- [Edge Impulse public FOMO training example](https://studio.edgeimpulse.com/public/109997/latest/impulse/1/learning/keras-object-detection/25)
- [TensorFlow weighted cross entropy semantics](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits)

## 配置差异审计

四个 YAML 都继承同一份锁定协议的字段值。相对
`checkpoint_v2_lite_aug03_locked.yaml`，允许变化仅为实验命名、输出目录和
`loss` policy；数据 split/hash、MobileNetV2 lite、width 0.35、192 输入、stride 8、
`underwater_conservative`、AdamW、scheduler、batch、workers、seed、60 epochs、
AMP、snapshots、validation selection 和 threshold grid 均保持不变。

| 配置 | loss type | background weight | object weight | per-class weights |
|---|---|---:|---:|---|
| `loss_ce_object_w1.yaml` | `weighted_softmax_ce` | 1 | 1 | disabled |
| `loss_ei_object_w10.yaml` | `ei_weighted_xent_legacy` | 1 | 10 | disabled |
| `loss_ei_object_w30.yaml` | `ei_weighted_xent_legacy` | 1 | 30 | disabled |
| `loss_ei_object_w100.yaml` | `ei_weighted_xent_legacy` | 1 | 100 | disabled |

`loss.type` 是 `loss.name` 的 YAML 别名；若两者同时出现，必须相同。object-weight
模式禁止提供 `class_weights`，并要求 `gamma: 0.0`，防止不透明的权重叠加。

## 运行顺序

每个实验开始前必须确认：工作树干净、固定 Git 提交、数据 content hash 与基线
一致、输出目录不存在、配置 diff 只有上表允许字段。严格顺序：

1. C1 `weighted_softmax_ce`, object weight 1
2. C2 `ei_weighted_xent_legacy`, object weight 10
3. C3 `ei_weighted_xent_legacy`, object weight 30
4. C4 `ei_weighted_xent_legacy`, object weight 100

先用合成 fixture 做 CPU smoke：确认 CE 下降、w100 无 NaN/Inf、AMP/反向传播有限、
snapshot 和 resume 正常、checkpoint/summary 记录 loss type 与 object weight。正式
训练使用 GPU（若可用），不秘密修改 learning rate。

每个实验保存 60 个 weights-only snapshots。只在 validation 上用 FP32 离线扫描：
以 centroid PR-AUC macro 为主选择 epoch，再在 validation 上选择 centroid F1 阈值。
第一轮不运行正式 test。表格必须同时记录 strict one-to-one 与 EI legacy-compatible
validation 指标、macro F1、count MAE 及 per-class 结果。

唯一 candidate 进入 locked test，使用 validation 锁定的阈值；test 只运行一次固定
协议，不参与 epoch、object weight 或 threshold 选择。保留既有 focal epoch 58
baseline 作为比较项。

## 结果登记

以下表格待正式训练及 validation offline scan 后填写，禁止手填推测值。

| experiment | loss | object weight | selected epoch | validation threshold | strict P/R/F1 | EI legacy P/R/F1 | macro F1 | count MAE |
|---|---|---:|---:|---:|---|---|---:|---:|
| C1 | weighted_softmax_ce | 1 | pending | pending | pending | pending | pending | pending |
| C2 | ei_weighted_xent_legacy | 10 | pending | pending | pending | pending | pending | pending |
| C3 | ei_weighted_xent_legacy | 30 | pending | pending | pending | pending | pending | pending |
| C4 | ei_weighted_xent_legacy | 100 | pending | pending | pending | pending | pending | pending |
