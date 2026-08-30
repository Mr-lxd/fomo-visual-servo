# checkpoint v2 locked 配置差异报告

## 配置来源

- 基准配置：`configs/experiments/aug03_underwater_conservative.yaml`
- 正式配置：`configs/experiments/checkpoint_v2_lite_aug03_locked.yaml`
- 数据根目录：由 `FOMO_DATASET_ROOT` 提供
- 解析环境：Python 3.10.20，CUDA，NVIDIA GeForce RTX 4060 Laptop GPU

## 允许变化

逐字段解析比较结果显示，除以下字段外没有训练配方差异：

| 字段 | aug03 | locked v2 | 变化原因 |
| --- | --- | --- | --- |
| `experiment.name` | `aug03_underwater_conservative` | `checkpoint_v2_lite_aug03_locked` | 新实验身份 |
| `training.output_dir` | `outputs/experiments/aug03_underwater_conservative` | `outputs/experiments/checkpoint_v2_lite_aug03_locked` | 独立输出目录 |
| `training.epoch_snapshots` | 默认 disabled | `enabled=true`, `format=weights_only`, `interval=1`, `keep_last=null` | v2 每 epoch 保存 snapshot |

locked 配置还显式写出以下 v2 选择协议字段；它们与当前解析默认值一致，不改变训练配方：

- `evaluation.checkpoint_selection.metric: centroid_pr_auc_macro`
- `evaluation.checkpoint_selection.split: val`
- threshold grid：`0.05` 到 `0.95`，步长 `0.05`
- `evaluation.threshold_calibration.enabled: false`

## 已验证保持不变的字段

- Dataset：train/val split、7 类、`preserve` class mode、`keep_first` collision policy
- Model：`mobilenet_v2_lite`、width `0.35`、head `32`、input `192`、stride `8`
- Augmentation：`underwater_conservative` 及全部 preset 参数
- Loss：focal cross entropy、gamma `2.0`、class weights `[1.0, 4.0, ...]`
- Optimizer：AdamW，learning rate `0.001`，weight decay `0.0001`
- Scheduler：StepLR，step size `20`，gamma `0.5`
- Training：batch size `8`、workers `4`、seed `42`、epochs `60`、AMP、checkpoint criterion
- Threshold/postprocess：checkpoint `0.5`、inference `0.5`、centroid matching 和组件后处理

## 预检记录

- Dataset content hash：`0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`
- Dataset manifest files：`1151`
- Train/validation images：`448 / 127`
- DataLoader smoke batch：images `[8,3,192,192]`，targets `[8,24,24]`
- Output directory：不存在，未覆盖已有实验
- 磁盘剩余空间：约 `6.57 GiB`
- `unexpected_difference_fields`：`[]`
