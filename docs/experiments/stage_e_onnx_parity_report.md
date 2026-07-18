# Stage E ONNX / ONNX Runtime parity

环境：PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、ONNX 1.22.0、ONNX Runtime 1.23.2；opset 17、固定 FP32 `[1,3,192,192]`、`images` 输入、`logits` 输出、CPUExecutionProvider。所有模型通过 ONNX checker，输出均为 `[1,8,24,24]`；使用项目阈值 `rtol=1e-4`、`atol=1e-5` 时均通过。

| model | max abs error | mean abs error | ONNX bytes | nodes |
| --- | ---: | ---: | ---: | ---: |
| D2 MobileNetV2 FOMO, EI H5 pretrained | 1.76e-06 | 2.63e-07 | 102,165 | 127 |
| MobileNetV3-Small FOMO pretrained | 2.10e-06 | 2.21e-07 | 29,349 | 19 |
| SqueezeNet1.1 FOMO pretrained | 5.72e-06 | 1.11e-06 | 327,698 | 44 |

动态 batch 未启用，模型按 Raspberry Pi batch=1 固定输入部署约束导出。随后使用统一 `scripts/benchmark_backbones.py` 重新以真实、经 SHA-256 验证的 D2 H5 导出，D2 的最新一致性数据为 max `1.76e-06`、mean `2.63e-07`、ONNX `102,165` bytes；详细的三模型 CPU 基准见 `stage_e_cpu_benchmark_report.md`。
