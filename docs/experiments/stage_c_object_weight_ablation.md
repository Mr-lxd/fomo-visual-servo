# Stage C：EI-style object-weight controlled ablation

状态：C1–C4 正式训练、validation-only selection 和唯一 candidate 的 cleaned-test
locked evaluation 已完成。未重新训练 focal baseline，未用 test 选择 object weight。

## 目的与边界

本阶段只研究 FOMO 输出热力图中 foreground 与 background 的 loss 权重。沿用
`configs/experiments/checkpoint_v2_lite_aug03_locked.yaml` 的数据、模型、增强、
优化器、scheduler、batch、seed、训练轮数和评估协议。C0 是已有 focal baseline，
不重新训练。禁止修改七类 per-class weights、`object_weight` 之外的训练配方、
阈值选择规则或 test 驱动的决策。

Edge Impulse parity evaluator 基础实现提交：
`d10f07e22cb80b61592f46ca963c9de8a156b612`

Stage C loss、配置、训练与 snapshot 实现提交：
`473a795da7de5fba1e0e68ebbd289c9f54184e79`

本文档最后核验/更新提交：见 Git 历史中包含本次 provenance 修正的文档提交，
不在文档内自引用该提交。

## Provenance 分层

以下提交角色保持明确区分，不要求四项相同：

| 字段 | 本阶段含义 | 记录 |
|---|---|---|
| `training_code_commit` | C1–C4 正式训练开始时使用的代码提交 | `8e594831bf07754739fa06568d69272294f5d5bb` |
| `checkpoint_metadata_git_commit` | checkpoint metadata 内实际记录的训练提交 | `8e594831bf07754739fa06568d69272294f5d5bb` |
| `stage_c_implementation_commit` | Stage C loss、配置、训练与 snapshot 的实现提交 | `473a795da7de5fba1e0e68ebbd289c9f54184e79` |
| `evaluation_code_commit` | Stage C.1 运行时当前 HEAD，由各个 evaluation artifact 写入 | 待 Stage C.1 执行时记录 |
| `parity_evaluator_base_commit` | Edge Impulse parity evaluator 基础实现提交 | `d10f07e22cb80b61592f46ca963c9de8a156b612` |
| `documentation_commit` | 本次 provenance 修正文档提交 | 见 Git 历史，不在本文档内自引用 |

checkpoint metadata 中的训练提交是 `8e594831...`，它位于 Stage C 实现提交
`473a795...` 之后，仅包含 Stage C smoke/resume 测试提交；这不改变 checkpoint
的训练代码来源，也不表示需要重新训练。

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
| C1 | weighted_softmax_ce | 1 | 58 | 0.10 | 0.5309/0.2365/0.3272 | 0.5654/0.2473/0.3441 | 0.2034 | 4.8032 |
| C2 | ei_weighted_xent_legacy | 10 | 49 | 0.10 | 0.4581/0.3003/0.3628 | 0.5218/0.3260/0.4013 | 0.2572 | 4.4331 |
| C3 | ei_weighted_xent_legacy | 30 | 59 | 0.20 | 0.4842/0.2860/0.3596 | 0.5363/0.3054/0.3892 | 0.2583 | 4.8346 |
| C4 | ei_weighted_xent_legacy | 100 | 59 | 0.35 | 0.4459/0.3036/0.3613 | 0.5057/0.3291/0.3987 | 0.3270 | 4.5197 |

表中 P/R/F1 是 validation 指标。主选择规则是 validation strict centroid F1：
C2=0.362791 高于 C4=0.361257、C3=0.359613 和 C1=0.327245，因此唯一选择
C2；C4 的 EI legacy validation F1 较高不能改变 primary candidate。

## Locked test

唯一 candidate：

- config：`configs/experiments/loss_ei_object_w10.yaml`
- snapshot：`epoch_049_weights.pt`
- snapshot SHA-256：`654f7c7bfec82e9f00041b75f0dc3e24853ea28fc11bfef19862f5590937033b`
- validation-locked threshold：`0.10`
- cleaning view：`parity-clean-v1`
- cleaned test：63 images，582 GT

| evaluator | TP/FP/FN | P | R | F1 | macro F1 | predictions | GT | count MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local current | 136/226/446 | 0.3757 | 0.2337 | 0.2881 | 0.2156 | 362 | 582 | 5.2698 |
| strict one-to-one | 173/187/409 | 0.4806 | 0.2973 | 0.3673 | 0.2729 | 360 | 582 | 5.3016 |
| EI legacy | 192/168/412 | 0.5333 | 0.3179 | 0.3983 | 0.2911 | 360 | 582 | 5.3016 |

既有 focal epoch58@0.50 cleaned-test baseline 是：strict F1 `0.2781`
（99/31/483，P=0.7615，R=0.1701），EI legacy F1 `0.2897`
（104/26/484，P=0.8000，R=0.1769）。因此 C2 w10 的 locked-test 改变为更高
recall 和更多 prediction：strict F1 +0.0892、EI legacy F1 +0.1086；precision
下降是 object weight 提高 foreground 召回的直接代价。test 结果仅用于报告，未用于
改变 candidate、threshold 或 object weight。

Edge Impulse Studio 当前截图 float32 test 参考为 P=0.63、R=0.36、F1=0.46。
C2 的 EI legacy test 为 P=0.5333、R=0.3179、F1=0.3983，差距分别为
`-0.0967`、`-0.0421`、`-0.0617`。这说明 object-weight ablation 缩小了召回侧
差距，但尚未达到 Studio 结果；仍需将差异分解为模型/训练配方、preprocessing、
解码和 Studio 后端实现差异，不能把 legacy many-to-one 的增益当作模型本身增益。

## 产物位置

正式训练输出位于 `outputs/experiments/stage_c_loss_*`，每组有 60 个
`epoch_snapshots`。validation-only 汇总位于各组的
`validation_scan/stage_c_validation_summary.json`；locked test 仅位于 C2 的
`locked_test_threshold_010/parity_report.json`。这些 outputs 被排除在 Git 提交之外。
