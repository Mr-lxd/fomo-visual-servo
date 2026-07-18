# Stage E CPU / ONNX benchmark

执行日期：2026-07-18。此表为开发主机 CPU 的工程筛选数据，不代表 Raspberry Pi 5 的实测延迟。所有模型均为 FP32、固定 `[1,3,192,192]`、batch=1、PyTorch CPU 与 ONNX Runtime `CPUExecutionProvider`、单线程、预热 10 次后测量 50 次的中位数；基准脚本没有读取任何 train、validation 或 test 图像。

| candidate | parameters | ONNX bytes | PyTorch CPU median (ms) | ORT CPU median (ms) | PyTorch/ORT max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| D2 MobileNetV2 FOMO, EI H5 pretrained | 19,208 | 102,165 | 2.888050 | 1.205250 | 1.758337e-06 |
| MobileNetV3-Small FOMO, torchvision ImageNet pretrained | 6,136 | 29,349 | 0.891800 | 0.390700 | 2.145767e-06 |
| SqueezeNet1.1 FOMO, torchvision ImageNet pretrained | 79,464 | 327,698 | 12.634750 | 1.646450 | 6.675720e-06 |

三者均通过 ONNX checker，输出固定为 `[1,8,24,24]`，并满足 `rtol=1e-4`、`atol=1e-5` 的 PyTorch/ONNX Runtime logits 一致性门。D2 H5 从交接文档给定来源下载到已忽略的本地缓存，并校验 SHA-256 `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c` 后加载。

结论：在本开发主机的该协议下，MobileNetV3-Small 是最小且最快的候选；SqueezeNet1.1 的 ONNX Runtime 延迟仍可接受，但模型体积和 PyTorch CPU 延迟显著更高。最终部署取舍仍必须在 Raspberry Pi 5 上复测。
