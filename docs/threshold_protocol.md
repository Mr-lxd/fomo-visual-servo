# FOMO threshold protocol

项目使用三个明确的阈值语义：

- `postprocess.inference_threshold`：图片和视频推理在未提供命令行覆盖时使用的默认阈值。
- `evaluation.checkpoint_threshold`：每个训练 epoch 计算 centroid 指标、选择 `best_centroid_f1.pt` 的固定阈值。它不受推理阈值变化影响。
- `evaluation.threshold_sweep`：训练结束后只对最终选中的 checkpoint 进行的验证集阈值搜索。

推荐配置：

```yaml
postprocess:
  inference_threshold: 0.5

evaluation:
  checkpoint_threshold: 0.5
  threshold_sweep:
    enabled: true
    minimum: 0.05
    maximum: 0.95
    step: 0.05
```

训练 checkpoint 的规则是：

1. 每个 epoch 使用 `checkpoint_threshold` 计算 centroid precision/recall/F1。
2. `best_centroid_f1.pt` 按该固定阈值下的 centroid F1 严格提升保存。
3. `best_grid_f1.pt` 按 grid F1 严格提升保存。
4. sweep 只生成最终验证报告中的 `final_sweep_best_threshold`，不回写 epoch 选择。
5. `best_val_f1.pt` 保留为兼容别名，metadata 中的 `best_val_f1_alias_target` 表明其实际对应的 criterion 文件。

旧 YAML 的 `postprocess.confidence_threshold` 仍可读取，但会发出 `DeprecationWarning`，且只作为 inference threshold 的兼容输入；它不会隐式改变 checkpoint threshold。
# Checkpoint selection protocol v2

`last.pt`, `best_val_f1.pt`, `best_grid_f1.pt`, and `best_centroid_f1.pt` remain
the legacy full training checkpoints.  In particular, `best_centroid_f1.pt`
continues to mean fixed-threshold (`0.5`) centroid F1; its explicit metadata
name is `fixed_centroid_f1`.

When enabled, epoch snapshots are portable CPU weights-only files.  Offline
selection writes `best_centroid_pr_auc_macro.pt` and
`best_sweep_centroid_f1.pt` as non-resumable inference/evaluation candidates.
They have `resumable: false`; attempting to resume from them is an error.  Use
`last.pt` for resuming.

`centroid_pr_auc_macro` evaluates postprocessed centroids at the configured
threshold grid.  For each foreground class with ground truth, the raw points
are sorted by recall, duplicate recall values retain maximum precision, and
the observed coordinates are integrated with trapezoids.  No AP envelope or
artificial endpoints are added, so this is neither COCO mAP nor interpolated
average precision.  Classes with no ground truth are excluded and the effective
class count is recorded.  Epoch ties choose the earlier epoch, then source
filename; threshold ties choose the lower threshold.
