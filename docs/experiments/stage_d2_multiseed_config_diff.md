# Stage D2 multi-seed configuration diff

Base: `configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml`

The seed-123 and seed-2027 configurations are generated from the D2 config. The
only allowed changes are:

| path | seed 42 | seed 123 | seed 2027 |
|---|---|---|---|
| `project.name` | `stage_d2_fomo_ei_w100_pretrained` | `stage_d2_multiseed_seed123` | `stage_d2_multiseed_seed2027` |
| `training.seed` | `42` | `123` | `2027` |
| `training.output_dir` | `outputs/experiments/stage_d2_fomo_ei_w100_pretrained` | `outputs/experiments/stage_d2_multiseed_seed123` | `outputs/experiments/stage_d2_multiseed_seed2027` |
| `experiment.name` | `stage_d2_fomo_ei_w100_pretrained` | `stage_d2_multiseed_seed123` | `stage_d2_multiseed_seed2027` |

All other resolved YAML fields must remain identical. The machine-readable
field-level comparison is [`stage_d2_multiseed_config_diff.json`](stage_d2_multiseed_config_diff.json).
