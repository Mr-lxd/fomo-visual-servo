# Stage D1 配置逐字段 diff

基线配置：`configs/experiments/loss_ei_object_w100.yaml`

D1 配置：`configs/experiments/stage_d1_fomo_ei_w100.yaml`

机器可读版本：`docs/experiments/stage_d1_config_diff.json`

逐字段比较结果为 4 项变化，除以下字段外全部相同：

| path | baseline | candidate | reason |
|---|---|---|---|
| `project.name` | `stage_c_loss_ei_object_w100` | `stage_d1_fomo_ei_w100` | experiment identity |
| `experiment.name` | `stage_c_loss_ei_object_w100` | `stage_d1_fomo_ei_w100` | experiment identity |
| `training.output_dir` | `outputs/experiments/stage_c_loss_ei_object_w100` | `outputs/experiments/stage_d1_fomo_ei_w100` | isolated output directory |
| `model.backbone` | `mobilenet_v2_lite` | `mobilenet_v2_fomo` | D1 independent variable |

保持锁定的关键字段包括：

- 192×192 输入、stride=8、width multiplier=0.35、head channels=32；
- `ei_weighted_xent_legacy`、background weight=1、object weight=100、gamma=0；
- `underwater_conservative` augmentation；
- AdamW、learning rate、scheduler、batch size=8、workers=4、seed=42；
- 60 epochs、AMP、snapshot interval、validation selection、threshold grid；
- preprocessing、padding、evaluation 和 matching 配置；
- 不启用 pretrained，使用随机初始化。
