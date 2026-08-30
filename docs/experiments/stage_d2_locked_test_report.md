# Stage D2 locked cleaned-test report

日期：2026-07-14

本报告记录 D2 预训练初始化候选在批准的 `parity-clean-v1` 视图上的一次性
test 评估。没有重新训练，没有修改 loss、class weights、object weight、
backbone、预处理、增强或 checkpoint selection；没有进行 test threshold sweep。

## 结论先行

D2 epoch 40 在清洗后的 63 张 test 上、FP32、validation-selected threshold
`0.40` 的严格一对一 centroid F1 为 **0.451977**，Edge Impulse legacy
many-to-one F1 为 **0.484959**。D2 相比此前 locked focal 和 C2 test artifact
均有更高的 strict F1、strict macro F1 和 EI legacy F1；但这只是固定协议下的
结果比较，不能将 test 结果反用于改阈值或重新选择 epoch。

这里的“复现 Edge Impulse”是**结构和权重初始化来源的复现**，不是 D2
checkpoint 与 EI float32 TFLite 的 bitwise 输出复现：D2 的 FOMO head/classifier
仍是在 PyTorch 中训练得到，EI TFLite 是另一份模型。逐层结构审计见
[`stage_d1_1_architecture_parity.md`](stage_d1_1_architecture_parity.md)。

## D2 locked provenance

| 字段 | 实测值 |
|---|---|
| protocol | `d2_locked_test_v1` |
| config | `configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml` |
| selected epoch | `40` |
| threshold | `0.40`，来源为 validation |
| test split | `test`，63 张 |
| cleaned test GT | `582` |
| checkpoint | `outputs/experiments/stage_d2_fomo_ei_w100_pretrained/epoch_snapshots/epoch_040_weights.pt` |
| checkpoint SHA-256 | `e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d` |
| config fingerprint | `48d40114b90978c03c36971cbc907e5d3d3f0e5ae3e95fd38549ea39dd7d718f` |
| train/val dataset content hash | `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562` |
| cleaning manifest | `outputs/parity_audit/edge_impulse_parity_v1_hashscope/parity-clean-v1.json` |
| cleaning manifest SHA-256 | `003792435ddf1341f4d13846927ee2de0580404fdac368b40cac5a28f3de9d6e` |
| full cleaning view hash | `35b915d3c926425777afb22b0ec684bfd0885f08bbb289133635bde4d584c41c` |
| cleaned test view hash | `d52ee0ffd498a24b5f90e75d6bbecbedd289efab5fdafc3bfeb18ea1518a906c` |
| pretrained source | Edge Impulse MobileNetV2 alpha 0.35 / 96 source H5 |
| pretrained H5 SHA-256 | `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c` |
| pretrained tensors loaded | `95`；missing `0`；unexpected `0` |
| evaluator code commit | `0cbb064f98bdfd3a658b2b0c69b7e9d7fcf3bf32` |
| protocol JSON SHA-256 | `202abb0c9f183f2dd3a71d81c2a4bcd114f6e12470b2dad1f41bc21d23c32ddc` |

The complete machine-readable report is
`outputs/experiments/stage_d2_fomo_ei_w100_pretrained/locked_test_d2/final_test_metrics_d2.json`
（该输出目录按实验策略保持 ignored，不进入 Git）。CSV 和逐图片 match 记录也
保存在同一目录的 `final_test_metrics_d2.csv`、`per_class_test_d2.csv` 和
`metrics.images` 字段中。

## Fixed test metrics

| evaluator | TP | FP | FN | precision | recall | F1 | macro F1 | predictions | GT | count MAE | count bias | localization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| local current (`centroid_in_bbox`) | 182 | 303 | 400 | 0.375258 | 0.312715 | 0.341143 | 0.368448 | 485 | 582 | 4.587302 | -1.539683 | mean 25.264 px; median 19.190 px |
| EI legacy (`many-to-one`) | 266 | 214 | 351 | 0.554167 | 0.431118 | 0.484959 | 0.482510 | 480 | 582 | 4.698413 | -1.619048 | mean 0.057597; median 0.032278 normalized |
| strict one-to-one | 240 | 240 | 342 | 0.500000 | 0.412371 | 0.451977 | 0.468252 | 480 | 582 | 4.698413 | -1.619048 | mean 0.048926; median 0.029504 normalized |

Per-class F1 is in `per_class_test_d2.csv`. The strict one-to-one values are:

| class | F1 |
|---|---:|
| fish | 0.393881 |
| jellyfish | 0.632000 |
| penguin | 0.353846 |
| puffin | 0.400000 |
| shark | 0.431373 |
| starfish | 0.666667 |
| stingray | 0.400000 |

The fixed confidence distributions were also recorded. Local predictions: 485,
min `0.400332`, mean `0.692237`, p50 `0.675292`, p90 `0.958110`, max
`0.996819`. EI-decoded predictions: 480, min `0.400332`, mean `0.686881`,
p50 `0.666559`, p90 `0.953395`, max `0.996819`.

## Comparison with prior fixed artifacts

These are historical artifacts evaluated under their own pre-declared locked
thresholds. D1 test remains **not run** and is intentionally absent.

| model / artifact | test threshold | strict F1 | strict macro F1 | EI legacy F1 | EI legacy macro F1 | count MAE |
|---|---:|---:|---:|---:|---:|---:|
| C1 focal, epoch 58 | 0.35 | 0.383747 | 0.346928 | 0.414097 | 0.367594 | 5.682540 |
| C2 EI-style object weight 10, epoch 49 | 0.10 | 0.367304 | 0.272913 | 0.398340 | 0.291091 | 5.301587 |
| EI float32 TFLite | 0.50 | 0.355769 | 0.190004 | 0.367104 | 0.193530 | 5.746032 |
| D2 EI-pretrained FOMO, epoch 40 | 0.40 | **0.451977** | **0.468252** | **0.484959** | **0.482510** | 4.698413 |

Source artifacts:

- C1: `outputs/experiments/stage_c1_focal_epoch58_test_threshold_035/parity_report.json`;
- C2: `outputs/experiments/stage_c_loss_ei_object_w10/locked_test_threshold_010/parity_report.json`;
- EI float32 TFLite: `outputs/parity_audit/edge_impulse_parity_v1_hashscope/edge_impulse_tflite_threshold_050_dsp_rgb01/parity_report.json`.

Relative to C1, D2 gains `+0.068230` strict F1, `+0.121324` strict macro F1,
`+0.070862` EI legacy F1 and reduces count MAE by `0.984127`. Relative to C2,
the gains are `+0.084674`, `+0.195340`, `+0.086619` and count MAE `-0.603175`,
respectively.

## Difference decomposition

1. **Initialization/model effect.** D2 is the same locked D2 recipe whose only
   intended experiment variable versus D1 is EI-compatible MobileNetV2 H5
   initialization. Its validation-only comparison already showed strict F1
   `0.422235` and macro F1 `0.382332`; the test result is consistent with a
   positive initialization signal, but test is not evidence for changing the
   recipe.
2. **Postprocessor/evaluator effect.** On D2, EI legacy versus strict changes
   TP/FP/FN from `(240,240,342)` to `(266,214,351)` and F1 from `0.451977` to
   `0.484959`. The +`0.032982` is the expected effect of public EI-style
   many-to-one assignment, not an improvement in model weights. The local
   current postprocessor is a different path and yields F1 `0.341143`, so it
   cannot be used as a direct proxy for Studio.
3. **Model versus EI TFLite.** EI TFLite legacy F1 is `0.367104`, below D2
   legacy `0.484959`, but these rows use different models. This gap cannot be
   assigned solely to evaluator semantics. A fair model-level comparison must
   use the exact same TFLite tensor contract and the same evaluator.
4. **Studio comparison.** The current EI Studio float32 test reference is
   P=`0.63`, R=`0.36`, F1=`0.46`. The local EI TFLite replay is P=`0.616000`,
   R=`0.261461`, F1=`0.367104`, giving deltas `-0.014000`, `-0.098539`,
   `-0.092896`. This remaining gap is not solved by changing D2 thresholds;
   it remains a preprocessing/postprocessing/model-export parity question to
   debug from the saved per-image outputs.

The most defensible current attribution is therefore: D2's improvement over C1/C2
is primarily an initialization/model effect under a locked recipe, while the
gap between strict and EI legacy is evaluator semantics. The Studio gap cannot
yet be attributed to the D2 model because D2 is not the EI TFLite model.

## Sources and implementation locations

- D2 model topology and EI H5 loader: `src/fomo_servo/models/mobilenet_v2_fomo.py`,
  `src/fomo_servo/models/pretrained.py`, `src/fomo_servo/models/metadata.py`.
- Locked test protocol and provenance checks: `scripts/evaluate_d2_locked_test.py`.
- Existing local/EI/strict report implementation:
  `scripts/evaluate_parity_local.py` and `src/fomo_servo/evaluation/parity_reporting.py`.
- Manifest-only source/hash verification:
  `src/fomo_servo/evaluation/parity_clean.py`.
- Validation-only D2 selection and threshold artifact:
  [`stage_d2_pretrained_validation.md`](stage_d2_pretrained_validation.md).
- Public source provenance for the transfer H5 is recorded in
  [`stage_d1_1_architecture_parity.md`](stage_d1_1_architecture_parity.md), which
  links Edge Impulse's transfer-learning discussion, pretrained FOMO discussion,
  and the Keras MobileNetV2 source.
