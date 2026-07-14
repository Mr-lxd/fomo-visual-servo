# Stage D2 multi-seed validation design

日期：2026-07-14

本文件是**设计，不是实验结果**。本阶段没有启动新的训练，也没有读取 test
来调参。目的是在进入下一轮工作前，预先固定如何判断 D2 的 validation
信号是否依赖单一随机种子。

## Locked recipe

所有 seed 都从
`configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml` 派生，只允许
改变 `training.seed` 和对应的独立输出目录：

- MobileNetV2 FOMO，width multiplier `0.35`，cut point
  `block_6_expand_relu`，输入 `192`，输出 stride `8`，参数量 `19,208`；
- Edge Impulse MobileNetV2 alpha `0.35` / input `96` H5，只加载 backbone
  到 cut point；FOMO head/classifier 重新初始化；H5 SHA-256 为
  `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`；
- 同一 train/val 数据和 content hash
  `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`；
- 同一 letterbox、增强、loss `ei_weighted_xent_legacy`、object weight `100`、
  AdamW、scheduler、batch、workers、AMP 和 60 epoch schedule；
- 每个 seed 独立保存 60 个 weights-only epoch snapshot；不覆盖已有 D2
  输出，不覆盖 D1/C1/C2 输出。

这些锁定项的依据是
[`stage_d2_config_diff.md`](stage_d2_config_diff.md) 和
[`stage_d2_pretrained_validation.md`](stage_d2_pretrained_validation.md)。

## Predeclared seeds

| seed | 状态 | 输出计划 |
|---:|---|---|
| 42 | 已完成，当前 D2 validation artifact | `outputs/experiments/stage_d2_fomo_ei_w100_pretrained` |
| 123 | 未运行 | `outputs/experiments/stage_d2_multiseed_seed123` |
| 2027 | 未运行 | `outputs/experiments/stage_d2_multiseed_seed2027` |

不得在看到 seed 42 结果后替换 seed 123 或 2027。若资源或外部状态导致某
seed 无法完成，报告缺失原因，不用另一个 seed 填补。

## Per-seed validation protocol

对每个 seed：

1. 固定 Python/torch/CUDA 环境和随机种子，记录 Git commit、配置 fingerprint、
   H5 hash、数据 content hash 和完整环境信息；
2. 只在 train 上训练 60 epoch；
3. 只在 val 上扫描每个 epoch 的 centroid PR-AUC macro；
4. 依据预先锁定的 `centroid_pr_auc_macro` 选择一个 epoch；
5. 仅在该 seed 的 val 上使用既定 strict one-to-one threshold grid，记录
   validation-selected threshold；
6. 保存 strict metrics、EI legacy metrics、per-class metrics、PR-AUC、
   count MAE、prediction/GT count、selected epoch 和 selected threshold；
7. 本轮不访问 test，不用 test 结果选择 seed、epoch、threshold、loss 或
   object weight。

seed 42 的已知 validation 结果是 selected epoch `40`、threshold `0.40`、
strict F1 `0.422235`、strict macro F1 `0.382332`、PR-AUC macro `0.218902`、
count MAE `3.992126`。这段数据来自
[`stage_d2_pretrained_validation.md`](stage_d2_pretrained_validation.md)，不是
对 seed 123/2027 的预测。

## Aggregate decision rule

三 seed 完成后，先只汇总 validation：

| 指标 | 汇总方式 | 角色 |
|---|---|---|
| strict one-to-one F1 | mean ± sample std | primary |
| strict macro F1 | mean ± sample std | primary auxiliary |
| centroid PR-AUC macro | mean ± sample std | epoch-selection stability |
| count MAE | mean ± sample std | count stability |
| precision / recall | mean ± sample std | diagnostic |
| EI legacy F1 / macro F1 | mean ± sample std | Studio-oriented diagnostic |
| selected epoch / threshold | per-seed list、范围、median | selection stability |
| each class F1 | per-class mean ± sample std | class imbalance diagnostic |

首要 validation 判断为 strict F1 的 mean ± sample std，并同时查看每个 seed
是否出现异常类别崩溃。不能只报告最高 seed；不能把不同 seed 的 prediction
或 GT 拼接后当成一个样本级指标。

建议的 validation-positive 条件：

- 三 seed 都能完成且 provenance 完整；
- strict F1 的 sample std 被单独报告，不能用四舍五入隐藏波动；
- D2 mean strict F1 高于此前 locked baseline 的 mean 或至少与 seed 42
  结果一致，且没有单个类别出现未解释的系统性崩溃；
- 若结果不稳定，只记录为“seed-sensitive”，不通过选择最好 seed 来
  宣称 D2 普遍优于 baseline。

这里的“高于 baseline”仍应在同一 validation 协议中验证；此前 C1/C2 test
artifact 只用于历史差异参考，不反向参与本设计。

## Planned artifacts

每个 seed 输出：

- `config.yaml`、config fingerprint、dataset manifest、environment report；
- `training_summary.json`、`history.csv`；
- 60 个 epoch snapshots 及其 SHA-256；
- `validation_scan.json`，包含每 epoch PR-AUC 和选中 epoch；
- `selected_validation_metrics.json`，包含 threshold、strict/EI legacy、
  per-class、count 和 localization；
- `provenance.json`，记录 Git dirty 状态、H5 source/hash、loss/object weight
  和所有锁定配置字段。

汇总文件只追加到新的 multiseed 输出目录，不修改既有
`stage_d2_fomo_ei_w100_pretrained`。在 validation 汇总通过人工确认前，
不创建新的 test matrix，也不启动 object-weight/loss ablation。

## Current decision

当前 D2 的 single-seed validation 和 locked test 均已记录，但多 seed 还没有
执行。因此目前可以说“D2 在 seed 42 上表现为 validation-positive，并在
固定 test protocol 上取得上述结果”，不能说“D2 已通过多 seed 稳定性验证”。
