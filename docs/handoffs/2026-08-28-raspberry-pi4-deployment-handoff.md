# FOMO Visual Servo：Raspberry Pi 4 部署最终交接

生成日期：2026-08-28
最终更新：2026-08-30
仓库：`fomo-visual-servo`
用途：记录 D2 正式模型从 checkpoint、ONNX 导出、Windows parity 到 Raspberry Pi 4 ORT、USB UVC 摄像头和 VNC preview 的最终工程状态。

## 1. Milestone 结论

本阶段已经完成并实际验证：

- D2 MobileNetV2-FOMO seed42 epoch40 正式 checkpoint 锁定。
- 固定 batch=1、opset 17 的正式 ONNX 导出、`onnx.checker` 和 PyTorch/ORT raw-logits parity。
- PyTorch 与 ORT 共享 preprocessing、NumPy postprocessing、坐标反变换和完整 detection pipeline parity。
- 部署前 review Finding 1–4 已解决；Finding 5 `metrics.__dir__()` deferred，不阻塞 ORT 部署。
- Raspberry Pi 4 ARM64 / Python 3.13 最小 ORT runtime、静态图片、预录视频、跨平台 parity 和 benchmark。
- train-only 的 4 positive + 2 negative deterministic real-data smoke test。
- USB UVC `/dev/video0` 实时推理、annotated MP4、CSV、JSONL，以及 VNC 桌面实时 preview。

以下边界在整个阶段保持不变：没有重新训练，没有访问 test split 重新选择模型、epoch 或 threshold，正式 threshold 始终为 `0.40`，正式 checkpoint/ONNX contract 没有修改。checkpoint、ONNX、数据集、smoke media、camera outputs、cache 和 Pi venv 不进入 Git。

## 2. Git 收口与稳定 main integration

Raspberry Pi milestone 原始提交来自 `feature/fomo-backbone-ablation-v1` 的 `32169d6054861dece2b37cb149e75eb2ab160492`。稳定 main integration 在重新读取远端 graph 后按以下边界建立：

- integration branch：`feature/fomo-main-integration-v1`
- 起点：`a8f2e38dcf95000c3b5970bb0cc6cddd618cc6a3`，包含 Stage E 之前完整的 D2 training/evaluation/provenance 历史。
- deployment cherry-pick：`7cc1d19d465941780a08a4c68cfdd08c99b811f4`，由原始 `32169d6` 无冲突生成。
- Stage E commits `13b18029682e62915f288f9c8c1f1517ee14635f` 与 `4f0aadf939573dfcbc97aa3e3ac3efad76fa17a4` 不进入稳定 integration，但完整保留在原 `feature/fomo-backbone-ablation-v1` 历史中。
- integration 使用 linked worktree 隔离完成；原 Stage E checkout、远程 branch 和用户未跟踪文件均未修改或删除。
- integration branch 只允许正常 push 到同名 origin；本 handoff 不表示 `main` 已合并、tag 已创建或原 branch 已重写。
- 工作树包含本 milestone 的 ONNX exporter、ORT predictor、共享推理语义、Finding 1–4、smoke selector、camera/VNC CLI、launcher、requirements、测试和文档。
- `docs/handoffs/2026-07-18-github-draft-pr-continuation-handoff.md` 是 milestone 前已有的用户文件，保持未修改且不纳入本次 commit。

`.gitignore` 明确或通过 `/outputs/` 排除：

- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.ort`、`*.h5`、`*.tflite`
- `/outputs/`、`/cache/`、`/inference_cache/`
- `/data/`、`/datasets/`
- `/smoke_media/`、`/camera_runs/`

提交和 PR 前必须再次以 staged/branch diff 为准复核文件、敏感信息和禁止扩展名；本节记录不能代替下一会话的实时 Git 检查。

## 3. 正式模型合同与 provenance

### 3.1 模型合同

| 项目 | 正式值 |
| --- | --- |
| architecture | MobileNetV2 FOMO |
| width multiplier | `0.35` |
| cut point | `block_6_expand_relu` |
| head channels | `32` |
| parameter count | `19,208` |
| seed / epoch | `42` / `40` |
| validation threshold | `0.40` |
| input | `images`, RGB FP32 `[1,3,192,192]`, range `[0,1]` |
| output | `logits`, FP32 `[1,8,24,24]`, raw logits |
| output stride | `8` |
| ONNX opset | `17` |
| classes | fish, jellyfish, penguin, puffin, shark, starfish, stingray |

正式 checkpoint（本地忽略产物）：

```text
outputs/experiments/stage_d2_fomo_ei_w100_pretrained/epoch_snapshots/epoch_040_weights.pt
```

- size：143,430 bytes
- SHA-256：`e8c242f4af2b87b70fea2a516352f28e70bf438161eeb7d092231ed46c976a1d`
- metadata：epoch 40、seed 42、19,208 parameters、weights-only `model_state`

正式 ONNX 与 sidecar（本地忽略产物）：

```text
outputs/deployment/d2_seed42_epoch40/d2_mobilenet_v2_fomo_seed42_epoch40.onnx
outputs/deployment/d2_seed42_epoch40/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json
```

- ONNX size：104,412 bytes
- ONNX SHA-256：`3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08`
- sidecar SHA-256：`9ef9b98d6692d44d71271764ffcadfcbf24a6e668d8b66b1ffd1b91360845aa0`

### 3.2 Sidecar / publication contract

导出器在发布前验证 export YAML、source experiment config SHA、checkpoint SHA/metadata、strict state dict、参数量、固定 I/O shape、opset、ONNX checker 和 ORT parity。sidecar 记录 artifact/config/checkpoint provenance、ONNX SHA/size、I/O contract、postprocess contract 和 parity tolerances。

正式 ONNX 与 sidecar 在目标目录内 staging，完整回读并交叉验证 SHA、size 和 provenance 后才发布。普通发布异常使用备份和回滚，达到 exception-safe/fail-safe publication；该设计不宣称两个 `os.replace` 之间具有进程终止、系统崩溃或掉电条件下的双文件 crash-atomicity。

## 4. Windows 开发机验证

### 4.1 ONNX 与 logits parity

- `onnx.checker`：passed。
- export parity：`rtol=1e-4`、`atol=1e-5`。
- deterministic export input 最大绝对误差：`3.528594970703125e-05`。
- deterministic export input 平均绝对误差：`4.7044127313711215e-06`。

### 4.2 完整 pipeline parity

`local_pipeline_parity.json` 包含 1 张固定图片和 6 个预录视频帧，共 7/7 passed：

- preprocessing tensor 和 letterbox transform 一致。
- PyTorch/ORT logits 全部在正式 tolerance 内；全记录最大绝对误差 `0.00013828277587890625`。
- detection count、class、confidence、centroid 和原图坐标一致；最大 detection 数值误差 `0.0`。
- 固定图片来自 `tests/fixtures/yolo_micro/images/train/landscape.jpg`。
- 6 帧视频由 micro fixture 的 train 图片按确定性顺序生成，只用于 pipeline regression/parity，不用于模型性能评价。

### 4.3 Code review findings

1. Finding 1 resolved：ONNX/sidecar staging、完整验证、备份与异常回滚。
2. Finding 2 resolved：Windows 输出路径组件尾部 `.`/空格拒绝；exporter 与 CLI 共享 `path_safety.py`。
3. Finding 3 resolved：PyTorch predictor 对非法输入继续公开 `InferenceError`，并保留内部 `PreprocessingError` exception chain。
4. Finding 4 resolved：ORT/NumPy import 路径不主动 import/probe torch；torch 只在 PyTorch-specific postprocess 入口延迟加载。
5. Finding 5 deferred：lazy metrics exports 在首次访问前不完整出现在 `dir(fomo_servo.metrics)`；只影响 introspection/autocomplete，不阻塞部署。

最终收口的新鲜验证结果：

- 稳定 integration 定向 pytest（checkpoint/config、ONNX/ORT、imports、parity/postprocess、image/video/camera、bundle、smoke selector、display）：`195 passed, 13 warnings in 59.38s`。
- 排除 Stage E 后的稳定 integration 全量 pytest：`373 passed, 34 warnings`。
- `git diff --check`：退出码 0；只有 Windows LF/CRLF 提示。
- 正式 `onnx.checker`：passed，opset 17，I/O shape 保持 `[1,3,192,192] -> [1,8,24,24]`。
- checkpoint/ONNX/sidecar SHA 与本 handoff 记录完全一致。
- 重新执行 1 图 + 6 视频帧完整 pipeline parity：7/7 passed，最大 logits 绝对误差 `0.00013828277587890625`，最大 detection 数值误差 `0.0`。

## 5. Raspberry Pi 4 最小部署

### 5.1 Hardware / OS

| 项目 | 实测 |
| --- | --- |
| board | Raspberry Pi 4 Model B Rev 1.4，8GB |
| OS | Debian GNU/Linux 13.5 / Raspberry Pi OS 64-bit |
| kernel | `6.18.34+rpt-rpi-v8` |
| architecture | `aarch64` / ARM64 |
| Python | `3.13.5` |
| memory | 7.6GiB，总可用约 7.2GiB |
| root filesystem | 58G，可用约 49G（部署验收时） |
| throttling | 所有记录检查均为 `0x0` |

访问凭据、identity 文件和网络地址属于 operator-local 配置，不写入仓库。部署 bundle 与 venv 的已验证路径为：

```text
/home/pi/fomo-ort-d2-epoch40
/home/pi/venvs/fomo-ort-d2-epoch40
/home/pi/venvs/fomo-ort-d2-epoch40-preview
```

### 5.2 Runtime profiles

Headless profile：

```text
numpy==2.5.2
onnxruntime==1.29.0
opencv-python-headless==5.0.0.93
```

Preview profile：

```text
numpy==2.5.2
onnxruntime==1.29.0
opencv-python==5.0.0.93
```

共享 transitive versions：`flatbuffers==25.12.19`、`packaging==26.3`、`protobuf==7.36.0`。preview OpenCV build 报告 `GUI: QT5 (5.15.19)`；Wayland/labwc、WayVNC 和 XWayland 已实装验证。

两个 venv 均通过 `pip check`。bundle 静态 import closure 和真实运行验证均确认不依赖 torch、torchvision、models、training、checkpoint、CUDA 或数据集；ORT session 显式使用 `CPUExecutionProvider`。

### 5.3 静态图片与预录视频 parity

固定图片 `media/landscape.jpg`：

- image SHA-256：`b2f8c9835da70dab5f7dde95658145696bfd9dc7db59804e22246b733d5e30a1`
- Windows/Pi preprocessing tensor SHA 完全一致：`ed16b346ce27afb889167060df2e651d977ecbc673256c1a2d2a6a08dafc79a8`
- Pi logits `[1,8,24,24]` FP32；最大绝对误差 `3.4332275390625e-05`，平均绝对误差 `8.015479579626117e-06`
- Windows/Pi detection count 均为 0

预录视频 `media/local_validation_train_fixtures.avi`：6 帧、320x240、5 FPS、MJPG。

- ARM64 与 Windows OpenCV 对部分 MJPG 通道存在已量化的最多 1 LSB decoder rounding difference，没有通过放宽 logits tolerance 掩盖。
- 同一 AVI 完整 pipeline 的 6/6 帧 detection count、class 和坐标一致，全部为 0 detections。
- Windows 保存的 1 图 + 6 帧精确 NCHW 输入在 Pi 重放后 7/7 logits parity passed；最大绝对误差 `4.1961669921875e-05`。
- 10 次顺序重放共 60 帧全部完成，结束温度 56.4°C，`throttled=0x0`。

### 5.4 Real-data smoke test

选择脚本只扫描显式指定的 train split，并按 POSIX 相对路径排序；正式 sidecar threshold `0.40`、模型、epoch 和 seed 不能由 CLI 覆盖或重新选择。manifest 记录 image/model/sidecar SHA、完整模型合同和 expected detections。

- train-only candidates：448；扫描 78 张后确定性获得最先 4 张 positive 与最先 2 张 negative control。
- Windows positive detection counts：23、12、20、21；Pi 完全一致且都保持非空。
- 两张 negative control 在 Windows/Pi 均为 0 detections。
- class ID/name、confidence、`original_x/original_y` 均通过 manifest tolerance。
- 最低边界 positive confidence `0.41577598452568054` 在 Pi 仍高于正式 threshold `0.40`。
- 6 张图片及 `smoke_test_manifest.json` 仅复制到 Pi bundle；数据集、图片和生成 manifest 均不提交 Git。

这些样本只用于 deployment regression/parity，不属于模型性能评价，也没有访问 test split。

### 5.5 Benchmark

每组 50 次 warm-up、1000 次计时，单位 ms：

| ORT mode | stage | Median | P90 | P95 |
| --- | --- | ---: | ---: | ---: |
| default，intra=0/inter=0 | preprocess | 0.974 | 1.024 | 1.030 |
| default，intra=0/inter=0 | inference | 10.629 | 10.796 | 11.116 |
| default，intra=0/inter=0 | postprocess | 1.902 | 1.963 | 1.974 |
| default，intra=0/inter=0 | end-to-end | 13.528 | 13.714 | 13.987 |
| single，intra=1/inter=1 | preprocess | 0.973 | 1.027 | 1.038 |
| single，intra=1/inter=1 | inference | 16.942 | 17.042 | 17.126 |
| single，intra=1/inter=1 | postprocess | 1.909 | 1.945 | 1.952 |
| single，intra=1/inter=1 | end-to-end | 19.836 | 19.936 | 20.017 |

- default RSS：约 86.3MiB -> 90.4MiB；最大约 90.3MiB。
- single RSS：约 85.8MiB -> 89.6MiB；最大约 89.6MiB。
- default 温度：53.556°C -> 63.783°C；single：57.452°C -> 58.426°C。
- 两组 `get_throttled` 均为 `0x0`，swap 未使用。

这些是固定图片 pipeline benchmark，不是摄像头或 VNC preview FPS。

## 6. USB UVC camera 与 VNC preview

### 6.1 Camera contract

```text
device: /dev/video0
driver: uvcvideo
camera: PC Camera
negotiated format: 640x480 YUYV 4:2:2
camera input rate: 25 FPS
```

`LatestFrameReader` 使用单槽 replacement buffer；显示或推理变慢时丢弃旧帧，不形成 backlog。`--max-frames` 和 `--duration-seconds` 默认均为 `None`，因此不改变原有视频 EOF / 摄像头无限模式。

### 6.2 VNC operator acceptance

用户已在 VNC 桌面 Terminal 中实际确认 `--display` 成功打开实时 annotated-frame preview，并同时生成可回读的 MP4、CSV 和 JSONL。实际 run：

```text
camera_runs/20260830-164412/
```

- MP4：640x480、25 FPS、1,080 frames、19,392,539 bytes
- CSV / JSONL：各 1,080 records
- processed source indices：0 -> 1448，体现单槽 latest-frame 丢弃旧帧而不积压
- telemetry 中 model SHA 全部为正式 ONNX SHA
- MP4 SHA-256：`3ead8e707ef6a968e5ed4bfa87e610d88f044ccdbf9c9579ff1232dd6b540cbe`
- CSV SHA-256：`b2d712ebf0f415ee4ef359a028cc6616baf0df49ede7a8d1b7b1b318f6909e7e`
- JSONL SHA-256：`d30ba13f0d302ecd9dd4739edaa7d6ef6479442eff2fe82fb00d5ea787c7e786`

自动化测试另外覆盖 `q`、`Q`、Esc、正常/异常资源释放、headless build、缺少桌面 session 和 HighGUI runtime error。普通 SSH shell 不猜测或硬编码 `DISPLAY`，而是在打开 camera/output 前给出从 VNC desktop terminal 启动的诊断。

### 6.3 正式 preview 命令

必须在 VNC 桌面中的 Terminal 执行：

```bash
cd /home/pi/fomo-ort-d2-epoch40
run_id="$(date +%Y%m%d-%H%M%S)"
/home/pi/venvs/fomo-ort-d2-epoch40-preview/bin/python run.py predict_video \
  --onnx artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx \
  --onnx-report artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json \
  --source 0 \
  --display \
  --output-video "camera_runs/${run_id}/camera-overlay.mp4" \
  --output-csv "camera_runs/${run_id}/camera-telemetry.csv" \
  --output-jsonl "camera_runs/${run_id}/camera-telemetry.jsonl"
```

命令不传 `--confidence-threshold`，因此使用 sidecar 的正式 `0.40`；不传受控结束参数时持续运行，直到 `q`/`Q`/Esc 或正常终端中断。

### 6.4 Headless smoke

正式 headless venv 已完成 30-frame camera smoke：640x480、25 FPS、MP4/CSV/JSONL 各 30 帧/记录，camera 正常释放；温度 47.2°C -> 52.5°C，`throttled=0x0`。preview venv 在不传 `--display` 时也完成 5-frame smoke，证明 GUI profile 不改变 headless inference semantics。

摄像头拍摄显示器/桌面时观察到明显 domain shift 和 false positive。这不是 PyTorch/ORT、Windows/Pi 或 camera pipeline parity 问题，不能作为正式模型精度评价；真实目标域性能必须后续使用真实摄像头和水下场景验证。

## 7. Source / bundle 边界

进入 Git 的部署内容只包括源码、scripts、tests、launcher、export config、两份 Pi requirements、README 和 handoff。Pi bundle 运行闭包包括正式 ONNX、sidecar、ORT predictor、preprocessing、NumPy postprocessing、geometry、metrics/video 必需模块以及图片/视频 CLI；bundle 和所有运行产物继续位于忽略的 `outputs/` 与 Pi 文件系统。

`run.py` 从自身位置导出 bundle root，并仅延迟导入 allowlisted `predict_image`/`predict_video` 入口；没有固定 `/home/pi/...` 源码依赖。

## 8. Deferred / 下一阶段边界

明确 deferred，不应扩大本 milestone：

- Finding 5 `metrics.__dir__()` introspection compatibility。
- 真实水下/目标域模型性能验证。
- CSI camera。
- systemd。
- 长时间 daemon stability test。

后续工作不得重新训练或利用 test split 重选模型/epoch/threshold，不得改变正式 ONNX、sidecar、preprocessing/postprocessing 或 threshold `0.40`，除非开启经过单独批准的新模型 milestone。
