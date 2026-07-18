# Stage E alternative backbone validation

## Protocol and provenance

This is a validation-only engineering candidate comparison. No test split was opened, read, evaluated, or used for threshold/epoch selection. Each candidate used the unchanged D2 protocol: 192px input, stride 8, seven classes, seed 42, 60 epochs, CUDA AMP training, `ei_weighted_xent_legacy` (background/object weights 1/100), AdamW, StepLR, underwater-conservative augmentation, and the existing 0.05–0.95 validation threshold grid. Checkpoint epoch selection is FP32 validation `centroid_pr_auc_macro`; the selected epoch is then fixed before strict one-to-one threshold selection.

| candidate | pretrained source | training duration | snapshots | config fingerprint | selected epoch / snapshot SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| MobileNetV3-Small FOMO | torchvision ImageNet `MobileNet_V3_Small_Weights.IMAGENET1K_V1` | 00:29:23 | 60 | `6c9c83af212d49282bc9d3e42a34dad74dd0b9e926279ee75ae025c56aa2f516` | 54 / `5da8c2c499d7c7eeab68f887761bcaebd6c9dadd531bfc8f7d62c40cce46bd3d` |
| SqueezeNet1.1 FOMO | torchvision ImageNet `SqueezeNet1_1_Weights.IMAGENET1K_V1` | 00:28:54 | 60 | `ec79abcd733e68c1ea6c2fa493cef57881bf3ebe226029bd9085f4bd68208599` | 34 / `47df43feecbcc1683b090d862f8b6e6acabfba58f5ad4ca8271ba0f82ccfc269` |

Both snapshot sets have dataset-content hash `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`, seed 42, training commit `13b18029682e62915f288f9c8c1f1517ee14635f`, no missing/unexpected torchvision tensors, and no NaN/Inf in history. MobileNetV3 history SHA-256 is `b73fe026b66737baf37c9b3afc95f0deaa9f8fcbe1c5e6e7c8573913a00f3c20`; SqueezeNet history SHA-256 is `18855f8d5a06fe8ac745faf3691045207ddc100a68d90f2a3236a772bef3edeb`.

## Validation results

`local current` is the project's configured centroid evaluator; strict and EI legacy use the existing Edge Impulse-compatible reports. Localization values for strict are normalized by original-image dimensions because that is the evaluator's fixed output unit.

| model | epoch | threshold | local P/R/F1 | Strict TP/FP/FN | Strict P/R/F1 | Strict macro F1 | EI legacy P/R/F1 | PR-AUC macro | local count MAE | strict prediction count | strict mean / median localization |
| --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| MobileNetV3-Small | 54 | 0.15 | 0.158809 / 0.281628 / 0.203094 | 333 / 1171 / 576 | 0.221410 / 0.366337 / 0.276005 | 0.220196 | 0.269947 / 0.408451 / 0.325060 | 0.086009 | 5.425197 | 1504 | 0.061748 / 0.048674 |
| SqueezeNet1.1 | 34 | 0.40 | 0.439114 / 0.261826 / 0.328050 | 286 / 236 / 623 | 0.547893 / 0.314631 / 0.399720 | 0.384633 | 0.590038 / 0.328358 / 0.421918 | 0.210454 | 4.433071 | 522 | 0.049182 / 0.032729 |

The generated validation-only artifacts remain ignored under `outputs/experiments/`; they contain the complete FP32 per-epoch and per-image evidence. The committed per-class extract is `stage_e_alternative_backbone_per_class_validation.csv`.

## Accuracy and efficiency comparison

| model | pretrained | params | epoch | threshold | Strict F1 | Macro F1 | EI legacy F1 | PR-AUC | Count MAE | PyTorch CPU ms | ORT CPU ms | ONNX bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 MobileNetV2 FOMO | EI/Keras H5 | 19,208 | 40 | 0.40 | 0.422235 | 0.382332 | 0.455696 | 0.218902 | 3.992126 | 2.888050 | 1.205250 | 102,165 |
| MobileNetV3-Small FOMO | torchvision ImageNet | 6,136 | 54 | 0.15 | 0.276005 | 0.220196 | 0.325060 | 0.086009 | 5.425197 | 0.891800 | 0.390700 | 29,349 |
| SqueezeNet1.1 FOMO | torchvision ImageNet | 79,464 | 34 | 0.40 | 0.399720 | 0.384633 | 0.421918 | 0.210454 | 4.433071 | 12.634750 | 1.646450 | 327,698 |

CPU values are development-PC proxies only: static batch 1 FP32 `[1,3,192,192]`, one thread, 10 warm-up iterations, 50 measured iterations, and ONNX Runtime `CPUExecutionProvider`. They are not Raspberry Pi 5 measurements.

## Predeclared decision rules

- Performance candidate: Strict F1 must exceed D2 by at least 0.01, without clear Macro F1/PR-AUC deterioration. Neither candidate qualifies.
- Efficiency candidate: Strict F1 may trail D2 by at most 0.01 and must improve ORT latency by 20%, parameters by 30%, or ONNX size by 30%. MobileNetV3 has strong efficiency gains but Strict F1 trails by 0.146230; SqueezeNet trails by 0.022515 and is larger/slower. Neither candidate qualifies.
- Do not enter next round: Strict F1 trails D2 by over 0.02 without a qualifying efficiency outcome. Both candidates fall in this category under the declared rules.

This does not establish a causal backbone-only ablation: D2 uses an EI/Keras MobileNetV2 H5, whereas Stage E candidates use torchvision ImageNet weights. There were no multi-seed runs, MobileViT was not implemented, and Raspberry Pi 5 has not been measured. No follow-on training is recommended for these two candidates under the current protocol; MobileViT remains a separate, deferred design question rather than a recommended implementation.
