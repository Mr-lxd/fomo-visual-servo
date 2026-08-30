# FOMO checkpoint selection v2：三划分阈值协议

本项目默认采用标准三划分流程，不要求第四个 calibration split：

```text
Train
  训练模型
Validation
  以 centroid PR-AUC macro 选择 checkpoint
  固定 checkpoint 后，以 centroid F1 调整 confidence threshold
Test
  只使用锁定的 checkpoint 和 validation threshold，评价一次
```

默认协议语义是：

- `selection_split: validation`（当前 YAML 中可写作 `val`）。
- `threshold_tuning_split: validation`，因此
  `selection_and_threshold_tuning_shared: true`。
- `threshold_tuning_independent: false`。这不是把 validation tuning 描述成
  数据泄漏，而是明确记录 validation 同时承担 checkpoint selection 和 threshold
  tuning；test 完全隔离，只用于锁定后的最终评价。
- `best_centroid_pr_auc_macro.pt` 是主 candidate；当前 D2 正式流程固定使用
  validation selection 得到的 seed42 epoch 40 candidate，threshold 为 `0.40`。
- Stage C/C.1 的 epoch 58 focal 结果属于历史基线，只能按其当时的 locked
  protocol 解读，不应与当前 D2 candidate 混写。
- test 禁止 threshold sweep、自动阈值搜索和多 checkpoint 比较。

## 阈值字段

- `postprocess.inference_threshold`：图片/视频推理没有命令行覆盖时的默认阈值。
- `evaluation.checkpoint_threshold`：训练 epoch 期间记录 legacy fixed-threshold
  指标的阈值，不用于 Stage B 的 test。
- `evaluation.checkpoint_selection.threshold_grid`：checkpoint selection v2 和
  validation threshold tuning 共用的网格，默认 `0.05, 0.10, ..., 0.95`。

Stage B 使用 `scripts/tune_validation_threshold.py` 生成：

- `threshold_tuning.json`：保存完整 threshold grid 的 precision、recall、F1、
  checkpoint/source hash、config/dataset/Git provenance 和 FP32 语义；objective
  是 `centroid_f1`，并列时取较低 threshold。
- `locked_test_protocol.json`：锁定 candidate、source epoch、threshold、test
  split 以及上述 provenance。

然后使用 `scripts/evaluate_locked_test.py`。该 wrapper 强制读取 manifest 中的
单个 checkpoint 和 validation threshold，并输出 `final_test_metrics.json` 与
`final_test_metrics.csv`；它不会回写 checkpoint selection 文件。

独立 calibration split 能力可以作为可选高级模式保留，但不是默认协议，也不应在
默认文档中作为 threshold tuning 的必要条件。所有对比模型都应遵循同一原则：每个
模型可以在各自 validation split 上选择各自 threshold；强制所有模型共用 `0.5`
并不公平。最终 test 报告必须同时给出 checkpoint、threshold 和 threshold 来源。

## 指标定义

`centroid_pr_auc_macro` 的完整含义是：

> centroid PR-AUC on the configured threshold grid

PR 点来自配置的 centroid threshold grid，采用项目已有的 observed-point
trapezoidal 规则。它不是 COCO AP、bbox mAP，也不是标准 average precision。

legacy `best_centroid_f1.pt` 仍表示训练期间固定
`evaluation.checkpoint_threshold` 的 centroid F1；`last.pt` 仍是可 resume 的完整
训练 checkpoint。weights-only candidate 只能用于 inference/evaluation，不能 resume。
