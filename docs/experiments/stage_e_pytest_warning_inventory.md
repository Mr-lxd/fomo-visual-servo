# Stage E pytest warning inventory

The final Stage E code verification baseline reported `301 passed, 34 warnings`. The warnings were reviewed before the results-documentation commit; none hid a test failure or invalidated the fixed-shape ONNX/ORT parity checks.

| count | category | source | impact | action |
| ---: | --- | --- | --- | --- |
| 16 | Python `DeprecationWarning` | legacy inline-augmentation YAML fixtures in `tests/test_aug01_color.py`, `tests/test_aug02_color_hflip.py`, `tests/test_augmentation.py`, `tests/test_augmentation_suite.py`, and `tests/test_config.py` | Intentional compatibility coverage; no numeric or export effect | Retain: removing it would stop testing supported legacy configuration behavior. |
| 6 | PyTorch `TracerWarning` | shape guards in `src/fomo_servo/models/fomo.py` during two fixed-input ONNX tests | The Python guards become trace-time constants; exported graphs are deliberately fixed 192px inputs and parity tests pass | No change in this stage. A future dynamic-shape export would need export-specific validation. |
| 6 | PyTorch `TracerWarning` | shape guards in `src/fomo_servo/models/mobilenet_v2_fomo.py` during two fixed-input ONNX tests | Same fixed-input trace behavior; no numerical mismatch | No change in this stage. |
| 6 | PyTorch ONNX constant-folding `UserWarning` | PyTorch internals while exporting Slice nodes in MobileNetV2 ONNX smoke/parity tests | Optimization-only warning; ONNX checker and ONNX Runtime logits parity pass | Third-party behavior; do not upgrade dependencies merely to force zero warnings. |

The 34 warnings are therefore either intentional compatibility coverage, fixed-shape tracing limitations explicitly covered by export parity tests, or third-party optimization notices. There is no Stage E project-code warning that is an obvious correctness defect or an imminent behavior-change deprecation.
