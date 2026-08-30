# Stage D2：EI-compatible pretrained initialization validation

日期：2026-07-14

本实验只在原始 `train`/`val` 上完成 60 epoch CUDA 训练和 validation-only
snapshot scan。没有读取 test，没有改变 loss、object weight、预处理、增强、
optimizer、scheduler、batch、workers、seed 或 checkpoint-selection protocol。

## Provenance

| 字段 | 值 |
|---|---|
| implementation commit | `0492706901c93bddcd4cf3ee9e3ab708fed590b5` |
| Git dirty at training start | `false` |
| config | `configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml` |
| config copy SHA-256 | `f6e35683a6df8f98537c4ca870487858a9e089f4d76d965176d51f08fe88fb2e` |
| dataset content hash | `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562` |
| pretrained identity | `ei_keras_mobilenet_v2_035_96` |
| pretrained H5 SHA-256 | `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c` |
| loaded backbone tensors | `95` |
| skipped H5 tensors | `167` |
| missing / unexpected tensors | `0 / 0` |
| initialization policy | backbone through `block_6_expand_relu`; new FOMO head/classifier |
| parameter count | `19,208` |
| training device | CUDA；NVIDIA GeForce RTX 4060 Laptop GPU |
| training time | `1385.052 s` |
| snapshots | `60` |

The machine-readable output is
`outputs/experiments/stage_d2_fomo_ei_w100_pretrained/`; it is intentionally not
committed. The validation summary is
`validation_scan/stage_c_validation_summary.json`.

## Selection protocol and validation result

All 60 FP32 snapshots were scanned on the configured `val` split. The primary
selection metric was `centroid_pr_auc_macro`; the selected snapshot was then
evaluated with the validation-only strict-F1 threshold grid.

| field | result |
|---|---:|
| selected epoch | `40` |
| source snapshot | `epoch_040_weights.pt` |
| centroid PR-AUC macro | `0.2189015308` |
| strict threshold selected on validation | `0.40` |
| strict precision / recall / F1 | `0.4565217391 / 0.3927392739 / 0.4222353637` |
| strict macro F1 | `0.3823315648` |
| strict TP / FP / FN | `357 / 425 / 552` |
| strict predictions / ground truth | `782 / 909` |
| strict mean localization error | `0.0500388263` normalized |
| strict mean absolute count error | `3.9921259843` |
| EI legacy precision / recall / F1 | `0.5063938619 / 0.4142259414 / 0.4556962025` |
| EI legacy macro F1 | `0.4020370025` |
| EI legacy TP / FP / FN | `396 / 386 / 560` |

Per-class strict F1:

| class | F1 |
|---|---:|
| fish | `0.4237288136` |
| jellyfish | `0.5419847328` |
| penguin | `0.3422459893` |
| puffin | `0.4525547445` |
| shark | `0.1944444444` |
| starfish | `0.3684210526` |
| stingray | `0.3529411765` |

The training-time fixed-threshold best epoch was 45, but the locked selection
protocol correctly reports epoch 40 because epoch selection is based on the
offline validation PR-AUC scan. No test result was produced.

## Comparison

| model | initialization | strict F1 | macro F1 | EI legacy F1 | PR-AUC macro | count MAE |
|---|---|---:|---:|---:|---:|---:|
| C4 | lite random | `0.3613` | `0.3270` | `0.3987` | `0.1461` | `4.5197` |
| D1 | FOMO random | `0.3337` | `0.2557` | `0.3714` | `0.1353` | `4.5669` |
| D2 | FOMO EI-compatible pretrained | `0.4222` | `0.3823` | `0.4557` | `0.2189` | `3.9921` |

D2 versus D1 improves strict F1 by `+0.0885`, macro F1 by `+0.1266`, EI legacy
F1 by `+0.0843`, and PR-AUC macro by `+0.0836`; count MAE decreases by `0.5748`.
D2 also exceeds C4 on every listed validation metric. Under the pre-declared
single-seed rule, this is clear validation evidence that the verified EI
pretrained initialization is valuable for the locked recipe.

## Decision and remaining gate

D2 is a validation-positive pretrained-initialization experiment. It is suitable
for a separately approved locked test evaluation, but that test has deliberately
not been run in this stage. The test must use the already approved parity-clean
manifest and must not be used to retune the threshold, select another epoch, or
change the recipe.
