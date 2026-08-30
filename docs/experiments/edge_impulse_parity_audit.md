# Edge Impulse parity audit（locked checkpoint v2）

日期：2026-07-14。本文只审计，未重新训练、未修改 loss / class weights / object weight、未以 test 调阈值、未替换 epoch 58 checkpoint，亦未覆盖既有正式输出。

## 公平比较协议

本文是历史 epoch58 与 Edge Impulse TFLite 的 parity 审计，最可靠的比较固定为：同一 63 张本地 `test` 图像、FP32、阈值 `0.5`、相同的 `parity-clean-v1` 标签视图。`threshold=0.35` 是 epoch 58 在本地 validation 上已锁定的工作点，仅报告工程工作点，不参与当前 D2 candidate 或 checkpoint 选择。

## Confirmed

### 数据与清洗

原始数据保持只读，物理 split 为 train 448、valid 127、test 63。Edge Impulse Training bucket 对应本地 train；Edge Impulse Test 对应本地 test；本地 valid 未上传，且 EI 内部 20% validation 的文件名单不可获得。

| 项目 | 结果 |
| --- | --- |
| 物理非法 bbox | 2，均在 `test` |
| 非法规则 | `width <= 0 OR height <= 0` |
| 涉及图片 | 2 |
| 两条均为 | `class_id=4` (`shark`) 且 `width=0, height=0` |
| test GT | 584 → 582 |
| shark GT | 38 → 36 |
| train / valid | 未移除任何 bbox；train 的一个空标签是合法无目标样本 |
| 未发现 | 越界、NaN/Inf、非法 class id、重复 bbox、缺失标签、额外字段/数值错误 |

异常原始行、标签 SHA-256、清洗后标签 SHA-256 和行号记录在本地生成的 `outputs/parity_audit/edge_impulse_parity_v1_hashscope/parity-clean-v1.json` 中。实际审计记录位于同一输出目录的 `invalid_label_audit.json` 与 `invalid_label_audit.csv`；这些 ignored outputs 不作为 GitHub 链接发布，可由对应脚本重新生成。

| 视图 | SHA-256 |
| --- | --- |
| 所有 split 的 cleaning view | `35b915d3c926425777afb22b0ec684bfd0885f08bbb289133635bde4d584c41c` |
| cleaned test view | `d52ee0ffd498a24b5f90e75d6bbecbedd289efab5fdafc3bfeb18ea1518a906c` |

运行评估前，`verify_parity_clean_view()` 重新哈希全部被消费的图像、标签和根 `data.yaml`，并再次严格解析虚拟标签；任何文件漂移、清单外非法标签或 hash 不匹配都会在推理前失败。实现见 [parity_clean.py](../../src/fomo_servo/evaluation/parity_clean.py)。

### Edge Impulse ZIP 与实际 TFLite contract

精确检查的文件是用户本地的 `<EI_EXPORT_ZIP>`（PowerShell 路径操作均使用 `-LiteralPath`）；真实用户目录不写入仓库文档。

| 项目 | 实测值 |
| --- | --- |
| ZIP SHA-256 | `697af0ea5bd70f1c5317bc7be03149f5e7779f8de076c5da8452149e577738a8` |
| 唯一 `.tflite` member | `tflite-model/tflite_learn_1052881_7.tflite` |
| TFLite 大小 | 83,284 bytes |
| TFLite SHA-256 | `5ede37a254833c3a97cb13ee6506fb4e7333cade2ba204ee61a2553ab4478c6f` |
| 其他 `.tflite` / int8 模型 | 没有；ZIP 仅含这一 float32 模型 |
| metadata | `model-parameters/model_metadata.h` |
| model variables / classes | `model-parameters/model_variables.h` |
| trained ops | `tflite-model/trained_model_ops_define.h` |

以 `ai_edge_litert.interpreter.Interpreter` 实测（而非仅从 C header 推断）：

| Tensor | 名称 | shape | dtype | quantization |
| --- | --- | --- | --- | --- |
| input | `serving_default_x:0` | `[1, 192, 192, 3]` | `float32` | scale 0 / zero-point 0（未量化） |
| output | `StatefulPartitionedCall:0` | `[1, 24, 24, 8]` | `float32` | scale 0 / zero-point 0（未量化） |

算子包括 `CONV_2D`、`DEPTHWISE_CONV_2D`、`ADD` 和 `SOFTMAX`（以及 LiteRT 的 `DELEGATE`）。每张实际输出的末通道和为 1，故 output 已为 softmax 概率，评估器不会再次 Softmax。input 为 NHWC RGB，输出为 NHWC、background 加七前景类。

导出 metadata 确认 192×192、`EI_CLASSIFIER_RESIZE_FIT_LONGEST`、float32、未量化、无额外 data-normalization 或 image-scaling。关键的是，导出的 `edge-impulse-sdk/classifier/ei_run_dsp.h` 中 `extract_image_features` 将解包 RGB 除以 `255.0f` 后写入模型特征（约第 1144–1189 行）；所以直接喂 `0..255` 给 TFLite 是错误的。正式 TFLite 行使用 RGB、Fit-longest-axis、zero padding、`float32(image)/255`。初次 raw-`0..255` 诊断输出只得到 1 个 FP / F1=0，已保留为非正式调试证据，未覆盖正式 output。

### EI 后处理与 matching

兼容实现是独立模块，不替换本项目既有 `CentroidEvaluator`：[edge_impulse.py](../../src/fomo_servo/evaluation/edge_impulse.py)。其规则为：

- channel 0 是 background；每个前景 channel 独立以 `score >= 0.5` 激活；同类相邻/对角 cell 按 EI cube 逻辑融合；不同类不融合。
- 归一化距离为 `sqrt((dx / width)^2 + (dy / height)^2) <= 0.2`，不除以图像对角线。192 px 输入的精确边界测试覆盖 `(38.4, 0)`、`(0, 38.4)`、`(27.1529, 27.1529)`，边界采用 `<=`。
- `edge_impulse_legacy` 忠实保留公开代码的非标准 many-to-one 行为：同类多个 prediction 可以各自成为同一个最近 GT 的 TP；`strict_one_to_one` 是标准工程评价。

来源锚点：`ei_shared.labels.Centroid.distance_to` 与 `constrained_object_detection.metrics.match_by_near_centroids`；导出 SDK 中 `ei_postprocessing_common.h:67-97` 为 cube overlap，`:100-126` 为 threshold/raster merge，`:131-186` 为 bbox 输出，`:443-477` 为 float32 FOMO 遍历。公开源码不保证与 2026 Studio 云端后端逐字节一致，故仍需要 Studio 逐样本结果才能最终确认。

合成 pytest 覆盖：无目标、单目标、同类相邻 cell、不同类相邻 cell、1 GT + 2 nearby prediction、2 GT + 1 prediction、错误类别、FP/FN、以及 0.2 距离边界。legacy 与 strict 的 TP/FP/FN 差异被显式断言。

### Studio 已确认数值

- EI validation（当前 Quantized int8）：P=0.61、R=0.32、F1≈0.42。
- EI Model Testing（当前 Unoptimized float32，同一 63 图）：P=0.63、R=0.36、F1=0.46。
- 页面 Accuracy=14.29%，不与 non-background object-detection F1 混用。
- threshold=0.5，object tracking disabled。此前 P=0.61/R=0.44/F1=0.51 属于 earlier/other run，不是当前正式结果。

## 固定 test 结果

每个报告含 TP/FP/FN、non-background 聚合 P/R/F1、macro F1、per-class F1、prediction/GT count、定位与计数统计、以及逐图片匹配。
TFLite 报告还保存逐图 `.npy` 原始 probability tensor、zero-pad 预处理图、激活 cell 与融合质心；后续使用 `--raw-output-cache <此前输出目录>` 会在模型 SHA、cleaning hash、阈值和完整 63 图集合一致时复用这些 tensor，不再次调用 TFLite。

- 本地 epoch 58、0.5：`outputs/parity_audit/edge_impulse_parity_v1_hashscope/local_epoch58_threshold_050/parity_report.json`
- 本地 epoch 58、locked 0.35：`outputs/parity_audit/edge_impulse_parity_v1_hashscope/local_epoch58_threshold_035/parity_report.json`
- EI float32 TFLite、0.5：`outputs/parity_audit/edge_impulse_parity_v1_hashscope/edge_impulse_tflite_threshold_050_dsp_rgb01/parity_report.json`

### 2×2：cleaned test、FP32、threshold=0.5

| 模型 | 本地当前 evaluator | EI legacy-compatible evaluator |
| --- | --- | --- |
| 本地 epoch 58 | P=0.6565, R=0.1478, F1=0.2412; TP/FP/FN=86/45/496; 131 predictions | P=0.8000, R=0.1769, F1=0.2897; TP/FP/FN=104/26/484; 130 predictions |
| EI float32 TFLite | P=0.4708, R=0.2079, F1=0.2884; TP/FP/FN=121/136/461; 257 predictions | P=0.6160, R=0.2615, F1=0.3671; TP/FP/FN=154/96/435; 250 predictions |

strict one-to-one 补充：本地 epoch 58 为 P=0.7615/R=0.1701/F1=0.2781（99/31/483）；EI TFLite 为 P=0.5920/R=0.2543/F1=0.3558（148/102/434）。

### 本地 validation-tuned 工作点（不用于 EI parity）

epoch 58 + 本地 evaluator + locked threshold `0.35`：P=0.3669、R=0.1942、F1=0.2539、TP/FP/FN=113/195/469、308 predictions、582 GT。这个数值只表达此前 validation 选择出的本地工作点；没有用 test 改写它。

## Inferred

1. 评估器语义能解释一部分、但不能解释全部差距。对于本地 epoch 58@0.5，local→EI legacy F1 增加 0.0485；对于 EI TFLite@0.5，local→EI legacy 增加 0.0787。
2. 单独看 many-to-one matching，EI TFLite strict→legacy 增加 6 TP（148→154），F1 增加 0.0113；本地 epoch 58 增加 5 TP（99→104），F1 增加 0.0116。其余 evaluator 差异还包含 probability-weighted centroid 与 EI merged-cell bbox centroid、8-connectivity/对角融合、阈值边界、padding=114（本地）对 padding=0（EI）的预处理差异。
3. EI TFLite + EI legacy 与当前 Studio float32 的误差为：P `-0.0140`（0.616 vs 0.63）、R `-0.0985`（0.261 vs 0.36）、F1 `-0.0929`（0.367 vs 0.46）。P 已接近，但 recall 仍明显偏低；因此现阶段更可能是未完全复现 Studio 的图像/标签/云端测试路径或后端细节，而不是仅由 legacy matching 造成。
4. 在本审计的同一 cleaned test 下，EI float32 TFLite 的 EI-legacy F1（0.367）高于本地 epoch 58 的 0.290。因此，现有差距主要不是本地 evaluator 单独造成，也不应据此改 loss/object weight。

## Unknown / blocked

- Edge Impulse 内部 20% validation 的确切文件清单不可得，不能和本地 127 张 valid 做直接 validation 比较。
- Studio Model Testing 没有提供逐图片输出、逐图片 match、实际 padding/resize 中间图或云端后端版本。因此当前不能精确归因 EI TFLite legacy F1 与 Studio F1 的 0.0929 差距。
- Studio 是否保留两条零面积 shark GT 未知；本审计按已批准的客观非法 bbox 规则从二者均使用的本地 view 中排除。即使不清洗也只多 2 个 GT，无法单独解释上述 recall 差。
- 无需因这两条 test-only 非法 bbox 重新训练。

## Ablation gate

2×2 矩阵已经完成，因此“完成 EI parity matrix 后才允许讨论 object-weight ablation”的前置条件已满足。**本任务未启动 ablation**；在 Studio 逐样本证据缺失且 parity gap 尚未完全解释时，是否开始仍须由用户明确授权。
