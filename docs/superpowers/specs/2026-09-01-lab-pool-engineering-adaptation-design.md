# Lab Pool Engineering Adaptation Design

## Scope and identity

This work produces `lab_pool_adaptation_seed42_e20`, a lab-pool engineering
model for Raspberry Pi hardware closed-loop validation. It does not replace or
redefine `baseline-d2-v1`, change the D2 architecture, or create a publication
benchmark. The implementation lives on
`experiment/lab-pool-adaptation-v1` and is not merged to `main` in this phase.

## Immutable data boundary

`datasets_processed/lab_pool_v1` remains an immutable annotation asset. A
deterministic tool builds `datasets_processed/lab_pool_v1_d2_trainonly` from
the source images and LabelMe JSON annotations. It emits a seven-class D2
`data.yaml`, linked or copied train images, regenerated YOLO labels, and a
conversion manifest containing source/generated hashes and per-shape mapping
records.

The fixed mapping is:

- `jellyfish -> 1` (`jellyfish`)
- `fish -> 0` (`fish`)
- `tuna -> 0` (`fish`)
- `reflection tuna -> background`
- `reflection jellyfish -> background`

The converter clamps normalized coordinates only when the excess is within a
small declared numerical epsilon. Larger geometry violations fail without
partially publishing a destination view. `images/test` and the frozen
`pool-20260831-005/raw.avi` are not copied or decoded by the converter.

## Train-only contract

`dataset.validation_split: null` selects train-only mode. The engine creates no
validation dataset or loader, performs no validation or threshold sweep, writes
no best-checkpoint aliases, and exposes no fabricated validation metric. Early
stopping is invalid in this mode. Existing non-null validation configurations
retain their current behavior.

Train-only runs retain `history.csv`, resumable `last.pt`, training summary and
provenance. The only selected deployment candidate is the predeclared final
epoch weights snapshot. For this run it is `epoch_020_weights.pt`.

## Initialization contract

`training.initialize_from` and `training.initialize_sha256` describe a strict,
weights-only initialization operation distinct from resume. The source must be
the D2 seed42 epoch40 snapshot with SHA-256
`e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d`.
Only `model_state` is loaded with `strict=True`; optimizer, scheduler, scaler,
RNG, epoch and history are not restored, so the adaptation starts at epoch 1.
`initialize_from` and `resume` are mutually exclusive. Initialization path,
hash and source checkpoint metadata are recorded in output provenance.

## Fixed training protocol

- unchanged D2 MobileNetV2-FOMO, seven foreground classes plus background
- input 192, stride 8, width multiplier 0.35, head channels 32
- `model.pretrained: false`
- `underwater_conservative`, unchanged
- `ei_weighted_xent_legacy`, background weight 1, object weight 100
- seed 42, batch 8, 20 epochs, 27 steps/epoch, 540 expected updates
- AdamW, learning rate `1e-4`, weight decay `1e-4`, no scheduler
- no validation, early stopping, sweep or post-hoc checkpoint selection
- inherited engineering inference threshold `0.40`

## Freeze and deployment gate

Before held-out inference, record config SHA, training-view manifest SHA,
source checkpoint SHA, epoch20 checkpoint SHA, seed, epoch and threshold. Export
a separately named lab-pool ONNX artifact through the existing formal exporter,
checker and PyTorch/ORT parity path without overwriting baseline artifacts.

Only after epoch20 and ONNX are frozen and parity passes may the held-out video
be decoded for the first time. Its output is an annotated video plus existing
per-frame telemetry for manual hardware-validation review. No precision,
recall, F1 or mAP is reported because test labels do not exist.

## Test strategy

Tests cover deterministic conversion and manifest hashes, approved mapping,
numerical clamp versus hard failure, empty/background labels, train-only
control flow, absence of fabricated best metrics, strict initialization and
hash rejection, resume regression, and unchanged train+validation behavior.
Targeted tests, full pytest, `git diff --check`, and compileall must pass before
CUDA training starts.
