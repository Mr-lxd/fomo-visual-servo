# MobileNetV2 FOMO Backbone Design

## 1. Scope and experiment boundary

This change adds `mobilenet_v2_fomo` beside the existing
`mobilenet_v2_lite`. It does not replace, rename, or alter the existing model's
state-dict layout. The first experiment changes only model identity fields and
the experiment/output names relative to
`configs/experiments/aug03_underwater_conservative.yaml`.

The following remain locked: dataset and split, augmentation preset and
parameters, input size, output stride, labels and collision policy, focal loss,
class weights, optimizer, scheduler, batch size, workers, AMP, seed, 60 epochs,
early-stopping setting, checkpoint threshold, inference threshold, postprocess,
centroid matching, checkpoint selection, and evaluator.

`pretrained=false` is mandatory for the first experiment. The implementation
contains no download path. A request for `pretrained=true` fails explicitly
until a separately designed local-weight source exists.

## 2. Evidence from the current model

For `width_multiplier=0.35`, seven foreground classes, and a 192×192 RGB input,
the current `mobilenet_v2_lite` resolves as follows:

| Section | Repeats | First stride | Actual output channels | Spatial output |
|---|---:|---:|---:|---:|
| 3×3 stem | 1 | 2 | 16 | 96×96 |
| `(t=1,c=16)` lite stage | 1 | 1 | 8 | 96×96 |
| `(t=6,c=24)` lite stage | 2 | 2 | 8 | 48×48 |
| `(t=6,c=32)` lite stage | 3 | 2 | 16 | 24×24 |
| `(t=6,c=64)` lite stage | 2 | 1 | 24 | 24×24 |

Measured contracts:

- output stride: 8;
- cut/output channels: 24;
- output shape: `[B,8,24,24]`;
- backbone trainable parameters: 28,080;
- head trainable parameters: 1,064;
- total trainable parameters: 29,144;
- convolution MACs at batch 1: 28,809,216;
- FLOPs under the `2 FLOPs per MAC` convention: 57.62 MFLOPs.

The old head is `24→32→8`. Its parameter count is
`24×32+32 + 32×8+8 = 1,064`.

## 3. Standard MobileNetV2 block numbering and the Edge Impulse mapping

The standard MobileNetV2 stage definition is:

| Stage | Expansion `t` | Base output `c` | Repeats `n` | First stride `s` | Keras/Edge block IDs |
|---|---:|---:|---:|---:|---|
| stem | — | 32 | 1 | 2 | — |
| 1 | 1 | 16 | 1 | 1 | block 0 |
| 2 | 6 | 24 | 2 | 2 | blocks 1–2 |
| 3 | 6 | 32 | 3 | 2 | blocks 3–5 |
| 4 | 6 | 64 | 4 | 2 | blocks 6–9 |
| 5 | 6 | 96 | 3 | 1 | blocks 10–12 |
| 6 | 6 | 160 | 3 | 2 | blocks 13–15 |
| 7 | 6 | 320 | 1 | 1 | block 16 |

Edge Impulse's `block_6_expand_relu` is not the projection output of the sixth
completed block and is not a generic sequential module index. It is the
expansion activation inside block 6, the first block of the `(t=6,c=64,n=4,s=2)`
stage.

The exact execution sequence at the cut is:

```text
stem
→ complete block 0
→ complete blocks 1–2
→ complete blocks 3–5
→ block 6 expansion Conv2d(1×1, stride=1)
→ block 6 expansion BatchNorm2d
→ block 6 expansion ReLU6             ← cut_point output
→ [excluded] block 6 depthwise Conv2d(3×3, stride=2)
→ [excluded] block 6 projection
```

At alpha 0.35, standard `make_divisible(..., 8)` channel rounding gives:

| Point | Channels | Output stride at activation |
|---|---:|---:|
| stem output | 16 | 2 |
| block 0 projection | 8 | 2 |
| block 2 projection | 8 | 4 |
| block 5 projection | 16 | 8 |
| block 6 expansion ReLU6 | `16×6 = 96` | 8 |
| block 6 depthwise output, excluded | 96 | 16 |

Thus a 192×192 input produces `[B,96,24,24]` at the cut point. This agrees
with Edge Impulse's documented default 1/8 reduction and its alpha-0.35 example
showing 96 channels at `block_6_expand_relu`.

## 4. Chosen implementation

The environment does not contain `torchvision`. The project will implement the
standard trunk internally with ordinary PyTorch modules; no dependency will be
added.

Create `src/fomo_servo/models/mobilenet_v2_fomo.py` containing:

- immutable standard block specifications with explicit block IDs 0–6;
- a private standard Conv–BN–ReLU6 primitive;
- a complete standard inverted residual implementation for blocks 0–5;
- a separately named `block_6_expansion` module containing only the 1×1
  expansion Conv, BatchNorm, and ReLU6;
- `MobileNetV2FOMOBackbone`, exposing `output_channels=96`,
  `output_stride=8`, and `cut_point="block_6_expand_relu"`;
- `MobileNetV2FOMONet`, with the requested 1×1 FOMO classifier head.

The new head is exactly:

```text
Conv2d(96, 32, kernel_size=1, bias=true)
ReLU6
Conv2d(32, 1+N, kernel_size=1, bias=true)
```

For seven foreground classes its parameter count is
`96×32+32 + 32×8+8 = 3,368`.

The expected trainable counts from the standard definitions are:

- backbone: 15,840;
- head: 3,368;
- total: 19,208.

The expected convolution cost for `[1,3,192,192]` is 23,583,744 MACs, or
47.17 MFLOPs under the `2 FLOPs per MAC` convention. Stage A will verify these
figures by executing the model, not accept them merely from arithmetic.

The model validates fixed float32 RGB `[B,3,S,S]` input and returns only raw
logits `[B,1+N,S/8,S/8]`. It performs no normalization, softmax, thresholding,
connected components, or centroid decoding.

## 5. Factory and backward compatibility

`build_fomo_model` remains the public factory. It dispatches:

- `mobilenet_v2_lite` to the existing `FOMONet` without changing its module
  names, parameters, initialization, output, or state-dict keys;
- `mobilenet_v2_fomo` to `MobileNetV2FOMONet`;
- all other names to a diagnostic `ModelConfigurationError` listing supported
  backbones.

The existing `FOMONet`, `MobileNetV2LiteBackbone`, and old public imports remain
available. Old YAML files omit the new model identity fields and resolve to
`cut_point=lite_stride8_output` and `pretrained=false`. Existing old
checkpoints continue to load because the old model is still constructed and no
old state-dict key changes.

## 6. Configuration and metadata

`ModelConfig` gains:

- `cut_point: str`;
- `pretrained: bool`.

The formal new config explicitly sets:

```yaml
model:
  backbone: mobilenet_v2_fomo
  width_multiplier: 0.35
  head_channels: 32
  input_size: 192
  output_stride: 8
  cut_point: block_6_expand_relu
  pretrained: false
```

The parser rejects a mismatched cut point for `mobilenet_v2_fomo`. The model
constructor rejects output strides other than 8. `pretrained=true` raises a
clear error before any model download or network call.

A model-description helper records the following without changing model
state:

- backbone name;
- width multiplier;
- cut point;
- cut-point input channels;
- cut-point output channels;
- head channels;
- pretrained flag;
- initialization policy (`pytorch_module_defaults`);
- backbone, head, and total trainable parameter counts.

This mapping is added to every new checkpoint, `training_summary.json`, and
`experiment_metadata.json`. Loaders tolerate its absence so historical
checkpoints remain valid.

## 7. Experimental configuration lock

Create
`configs/experiments/model01_mobilenet_v2_fomo_aug03.yaml` from aug03. The only
allowed resolved differences are:

- experiment/project name;
- output directory;
- `model.backbone`;
- `model.cut_point`;
- `model.pretrained`;
- derived model identity metadata.

The output directory is
`outputs/experiments/model01_mobilenet_v2_fomo_aug03`. A test compares resolved
dataclasses after removing only model, experiment name, output directory, and
source-path identity. Every other value must match aug03 exactly.

## 8. Initialization and error behavior

With `pretrained=false`, the new model uses the initialization performed by
each stock PyTorch module constructor. The implementation must not run a
custom initialization pass, alter head biases, or emulate another framework's
initialization recipe. Metadata records
`initialization=pytorch_module_defaults`. Random initialization remains
controlled by the existing global training seed.

There is no implicit fallback between backbones, no network access, and no
exception suppression. Invalid backbone, cut point, stride, input geometry, or
pretrained request fails before training.

## 9. Test strategy

Tests first establish failures for the missing model, fields, factory branch,
and metadata. They then cover:

- unchanged old factory behavior and synthetic old-checkpoint reload;
- new `[1,3,192,192] → [1,8,24,24]` logits contract;
- explicit block 0–6 mapping and cut after expansion ReLU6;
- stride 8 and 96 channels at alpha 0.35;
- exact `96→32→8` head;
- no softmax in forward;
- finite forward/backward and gradients;
- CPU/CUDA shape agreement and CUDA AMP smoke when available;
- state-dict save/load equality;
- unknown backbone diagnostics;
- no download when pretrained is false and explicit rejection when true;
- stable model metadata and parameter counts;
- optional ONNX export and ONNX Runtime consistency with explicit dependency
  skips.

## 10. Complexity and smoke protocol

A temporary Stage A diagnostic uses batch 1 and identical `[0,1]` RGB inputs
for both models. Convolution MACs are counted from actual Conv2d output shapes;
FLOPs are reported as twice MACs. Latency uses sufficient warmup, synchronized
CUDA timing, and the same iteration count. It reports:

- trainable/backbone/head parameters;
- cut-point channels;
- MACs and FLOPs;
- weights-only serialized size and available full checkpoint size;
- CPU FP32 batch-1 latency;
- CUDA FP32 batch-1 latency;
- CUDA peak allocated memory.

After the complete test suite, compile check, and Git diff check pass, a
temporary two-epoch config changes only epochs and smoke output directory. The
CUDA smoke validates AMP, finite loss/backward, validation, augmentation stats,
checkpoint save/load and model metadata, resume epoch metadata, and stable
memory. Temporary diagnostics and smoke configs are removed before reporting.

Stage A then pauses with no commit and no formal 60-epoch training. Stage B is
performed only after explicit user approval.

## 11. Rejected alternatives

1. Installing and slicing torchvision MobileNetV2 is rejected because
   torchvision is absent and a raw `features[index]` cut would not represent
   the internal block-6 expansion activation.
2. Truncating after block 5 projection is rejected because it produces 16
   channels, not the required 96-channel expansion feature.
3. Completing block 6 and truncating after its projection is rejected because
   its stride-2 depthwise convolution changes output stride from 8 to 16.

## 12. Source basis

- Edge Impulse FOMO documentation:
  <https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo>
- MobileNetV2 stage definitions: Sandler et al., “MobileNetV2: Inverted
  Residuals and Linear Bottlenecks,” CVPR 2018.
