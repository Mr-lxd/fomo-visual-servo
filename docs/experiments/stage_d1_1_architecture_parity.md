# Stage D1.1：Edge Impulse / local FOMO architecture parity audit

日期：2026-07-14

本审计只使用已存在的 Edge Impulse float32 TFLite、已完成的 D1 validation
artifact 和只读下载的 MobileNetV2 transfer-learning H5。没有运行 D1 test，
没有重新训练，没有修改 loss、class weights、object weight、optimizer、
scheduler、augmentation 或数据集。

## 结论

初始审计发现两类会影响结构/空间一致性的 concrete mismatch：

1. 本地 FOMO head 使用 `ReLU6`，而 EI TFLite 的 1×1 head 使用 fused `RELU`；
2. 本地 stride=2 的 3×3 Conv/DepthwiseConv 使用对称 `padding=1`，而 EI
   `SAME` 在偶数尺寸上使用右/下不对称 padding。

另外，本地 BatchNorm 原先为 PyTorch 默认 `eps=1e-5, momentum=0.1`，而
Edge Impulse 所使用的 Keras MobileNetV2 源码明确为 `epsilon=1e-3,
momentum=0.999`。

这些差异已分别以独立 commit 修正：

- `6ab45a7`：本地 MobileNetV2 FOMO 的 Keras BN 参数和 head `ReLU`；
- `5526e58`：stride=2 的 TensorFlow-SAME 右/下显式 padding。

修正后，D1.1 的结构判定为：**A：backbone 拓扑与 EI 对齐；剩余差异仅为
NHWC/NCHW、BN 在训练图中保留而在 TFLite 中融合，以及 local logits / EI
softmax 这几个有意的接口差异。**

## D1 provenance

当前工作区在审计开始时为 clean，D1 训练 provenance 如下：

| 字段 | 实测值 |
|---|---|
| current HEAD after D1.1 fixes | `5526e58` |
| D1 training commit | `a072452c6dc7013e2d9dccb6f3f3c27c55c528d9` |
| D1 config | `configs/experiments/stage_d1_fomo_ei_w100.yaml` |
| D1 config SHA-256 (output copy) | `6b33eab4780b176d96ec6918dae5e24bbe0686e6aa636ef11ea5b8ea9382d4ae` |
| D1 config fingerprint | `e47b13e8579c6b33e679bc0ba25fb8d85c426a2733a1df82d24774607191f960` |
| dataset content hash | `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562` |
| D1 selected validation snapshot | `epoch_059_weights.pt` |
| selected snapshot SHA-256 | `ff85ed4219d052d37ddad8e306cde39febd7c067072255533e78f497ad82ea12` |
| validation summary SHA-256 | `c7cb9d755a39d00ba7ed4ab8bd8fea33a8fb8e5f96df159c49ca4de2c51a5676` |
| snapshot count | `60` |

## EI TFLite contract

审计对象：

- ZIP：`lxd992712186-project-1-cpp-mcu-v2-impulse-#1.zip`
- ZIP SHA-256：`697af0ea5bd70f1c5317bc7be03149f5e7779f8de076c5da8452149e577738a8`
- ZIP 内唯一模型：`tflite-model/tflite_learn_1052881_7.tflite`
- TFLite size：`83,284` bytes
- TFLite SHA-256：`5ede37a254833c3a97cb13ee6506fb4e7333cade2ba204ee61a2553ab4478c6f`
- interpreter：`ai_edge_litert.interpreter.Interpreter`

实测 tensor contract：

| tensor | name | shape | dtype | layout |
|---|---|---|---|---|
| input | `serving_default_x:0` | `[1,192,192,3]` | `float32` | NHWC RGB |
| output | `StatefulPartitionedCall:0` | `[1,24,24,8]` | `float32` | NHWC |

input/output 均未量化；output 每个 cell 的 8 个通道和为 1，图中存在
`SOFTMAX(beta=1.0)`，所以 EI output 是 background + 7 foreground classes 的
probability，而不是 logits。算子序列为：

```text
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D,
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D,
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D, ADD,
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D,
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D, ADD,
CONV_2D, DEPTHWISE_CONV_2D, CONV_2D, ADD,
CONV_2D, CONV_2D, CONV_2D, SOFTMAX
```

最后一个 `DELEGATE` 是运行时 delegate，不计入模型拓扑。

## Layer mapping

EI shapes 是 NHWC；local shapes 是对应的 NCHW。`C` 表示通道数，所有
Conv dilation 实测为 1，所有 DepthwiseConv depth multiplier 实测为 1。

| EI op / stage | EI NHWC shape transition | local module / PyTorch shape | EI options | status |
|---|---|---|---|---|
| 0 stem | `[1,192,192,3] → [1,96,96,16]` | `backbone.stem` `[1,3,192,192] → [1,16,96,96]` | Conv 3×3, stride 2, SAME, ReLU6 | exact after layout/padding fix |
| 1 block 0 depthwise | `[1,96,96,16] → same` | `blocks_0_to_5.0.block.0` | DW 3×3, stride 1, SAME, ReLU6 | exact |
| 2 block 0 project | `16 → 8` | `blocks_0_to_5.0.block.1/2` | Conv 1×1, linear | exact |
| 3–5 block 1 | `8 → 48 → 48 → 8`, grid `96 → 48` | `blocks_0_to_5.1` | expand 1×1, DW 3×3 stride 2, project 1×1 | exact after SAME padding fix |
| 6–9 block 2 | `8 → 48 → 48 → 8`, grid 48; residual ADD | `blocks_0_to_5.2` | stride 1, residual ADD | exact |
| 10–12 block 3 | `8 → 48 → 48 → 16`, grid `48 → 24` | `blocks_0_to_5.3` | expand 1×1, DW stride 2, project 1×1 | exact after SAME padding fix |
| 13–16 block 4 | `16 → 96 → 96 → 16`, grid 24; residual ADD | `blocks_0_to_5.4` | stride 1, residual ADD | exact |
| 17–20 block 5 | `16 → 96 → 96 → 16`, grid 24; residual ADD | `blocks_0_to_5.5` | stride 1, residual ADD | exact |
| 21 block 6 cut | `16 → 96`, `[1,24,24,16] → [1,24,24,96]` | `backbone.block_6_expansion` | Conv 1×1 + BN + ReLU6 | exact cut point |
| 22 FOMO head | `96 → 32`, grid 24 | `head.0` + `head.1` | Conv 1×1 VALID + fused RELU | fixed in `6ab45a7` |
| 23 classifier | `32 → 8`, `[1,24,24,32] → [1,24,24,8]` | `head.2` | Conv 1×1 VALID, linear | exact |
| 24 output | same shape | local `forward` returns logits | Softmax beta 1.0 in EI graph | intentional interface difference |

The local implementation is in
[`src/fomo_servo/models/mobilenet_v2_fomo.py`](../../src/fomo_servo/models/mobilenet_v2_fomo.py):

- block specifications and cut point: lines 12–25 and 138–188;
- explicit TensorFlow-SAME padding: lines 45–88;
- Keras-compatible BatchNorm settings: lines 74, 125 and 181;
- FOMO head and classifier: lines 231–240;
- raw-logit output contract: lines 191–198 and 246–259.

## Padding and layout details

EI FlatBuffer options report `SAME` for the stem and both stride-2 depthwise
operators. For an even input and a 3×3, stride-2 operator, TensorFlow SAME adds
one pixel on the right and bottom. The local implementation now applies
`F.pad(features, (0, 1, 0, 1))` before a zero-padding Conv2d; this is tested by
`test_ei_stride_two_same_padding_is_right_bottom_asymmetric`.

For stride-1 3×3 operators, local symmetric padding 1 is equivalent to SAME.
For 1×1 operators, both graphs use no spatial padding. Local tensors remain
NCHW for PyTorch/ONNX; an EI H5/TFLite weight importer must transpose:

- Conv2d: Keras/TFLite OHWI or HWIO → PyTorch OIHW;
- DepthwiseConv2d: Keras `[H,W,C,1]` → PyTorch `[C,1,H,W]`;
- BatchNorm gamma/beta/moving mean/moving variance: channel order unchanged.

## Parameter count reconciliation

| representation | count |
|---|---:|
| local trainable backbone | 15,840 |
| local trainable head | 3,368 |
| local trainable total | 19,208 |
| EI TFLite Conv/Depthwise weight+bias tensor elements | 18,336 |
| EI TFLite raw float parameter bytes | 73,344 |
| EI TFLite file size | 83,284 bytes |

The `18,336` value is not a contradictory architecture count. The local training
graph keeps 19 BatchNorm layers with trainable gamma/beta and uses bias-free
backbone Conv2d; TFLite folds BN into Conv weights/biases. The local graph also
keeps the two head biases. After BN fusion, the local inference-equivalent
Conv/Depthwise weight+bias count is 18,336. The remaining TFLite bytes are
FlatBuffer graph/operator/tensor metadata, not trainable parameters.

## Pretrained source audit

The Edge Impulse-supported transfer-learning source was located at:

`https://cdn.edgeimpulse.com/transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5`

Read-only download evidence:

- size: `7,149,872` bytes;
- SHA-256: `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`;
- H5 contains Keras MobileNetV2 alpha `0.35`, RGB Conv1 `[3,3,3,16]`,
  `mobl0` through `mobl6`, and the exact cut-point weight
  `mobl6_conv_6_expand/kernel:0` with shape `[1,1,16,96]`;
- H5 also contains later MobileNetV2 blocks and the ImageNet `Logits` layer;
  those tensors must not be loaded into the FOMO head/classifier.

The source is not a trained FOMO classifier and does not contain the seven-class
head. The approved D2 loading policy is therefore:

1. load only the backbone tensors through block 6 expansion;
2. transpose Keras tensors to PyTorch layout explicitly;
3. load BN gamma/beta and moving statistics with strict shape checks;
4. initialize the 32-channel FOMO head and 8-channel classifier locally;
5. record source URL, SHA-256, loaded keys, skipped keys and initialization policy.

Edge Impulse's public explanation says FOMO uses a MobileNetV2 base and selects
pretrained MobileNetV2 models by alpha. The public Edge Impulse transfer-learning
discussion gives the exact CDN path above. The Keras MobileNetV2 source documents
the alpha/input-size variants, the block sequence, `epsilon=1e-3`,
`momentum=0.999`, and ReLU6 backbone activations. See:

- [Edge Impulse: pretrained FOMO](https://forum.edgeimpulse.com/t/pretrained-fomo/7952)
- [Edge Impulse: exact MobileNetV2 transfer-learning path](https://forum.edgeimpulse.com/t/transfer-learning-locally-using-ipython-notebook-error-in-weight-path/3350)
- [Keras Applications MobileNetV2 source](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py)

The Keras Applications repository carries an MIT license, but the downloaded H5
artifact itself has no embedded license field. The project must retain the MIT
notice and record the artifact provenance; it must not describe the H5 as a
torchvision checkpoint or as a proprietary EI FOMO checkpoint.

## D1.1 decision and D2 gate

**Approved to proceed to D2 after the loader/config implementation and smoke
test.** The source is classified as `ei_keras_mobilenet_v2_035_96`, an official
Edge Impulse transfer-learning backbone source, not a generic torchvision
approximation.

D1 remains a validation-only experiment at commit `a072452...`; its snapshots
were produced before the two architecture-alignment fixes and must not be
re-evaluated under the new implementation. No D1 test result is created or
overwritten. D2 may change only experiment name/output directory and the
pretrained source enabled/path/hash fields; loss, object weight, data, seed,
training schedule, checkpoint-selection protocol and evaluator remain locked.
