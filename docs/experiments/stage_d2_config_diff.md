# Stage D2 configuration diff

Base: `configs/experiments/stage_d1_fomo_ei_w100.yaml`
Candidate: `configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml`

The candidate is a single-variable initialization experiment. Dataset,
preprocessing, augmentation, loss, object weight, optimizer, scheduler, batch,
workers, seed, 60-epoch schedule, snapshot protocol, validation selection and
threshold protocol remain locked.

| Field | D1 | D2 |
|---|---|---|
| `project.name` | `stage_d1_fomo_ei_w100` | `stage_d2_fomo_ei_w100_pretrained` |
| `model.pretrained` | absent (effective `false`) | `true` |
| `model.pretrained_source` | absent | `${FOMO_PRETRAINED_WEIGHTS}` |
| `model.pretrained_sha256` | absent | `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c` |
| `training.output_dir` | `outputs/experiments/stage_d1_fomo_ei_w100` | `outputs/experiments/stage_d2_fomo_ei_w100_pretrained` |
| `experiment.name` | `stage_d1_fomo_ei_w100` | `stage_d2_fomo_ei_w100_pretrained` |

The H5 artifact is classified as `ei_keras_mobilenet_v2_035_96`. Only the
MobileNetV2 backbone through `block_6_expand_relu` is loaded. The 32-channel
FOMO head and 8-channel classifier use fresh PyTorch initialization. The exact
machine-readable list of locked sections and allowed changes is in
[`stage_d2_config_diff.json`](stage_d2_config_diff.json).
