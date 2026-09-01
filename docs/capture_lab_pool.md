# Lab Pool Dataset Capture Tool（实验室水池数据采集）

独立采集工具：Raspberry Pi 4 + USB UVC 摄像头 + HDMI 显示屏 + 键盘即可完成
实验室水池数据采集。**不运行任何 FOMO / ONNX Runtime 推理，不加载模型，
不做 detection。** 无需 Wi-Fi / SSH / VNC / 云端。

## 数据 provenance 定义

本工具采集的数据定义为：

> **LAB POOL ENGINEERING VALIDATION DATASET**

用途链路：实验室环境 retrain/fine-tune → Pi 部署 → FOMO detection →
visual servo → hardware-control verification。

它**不是**最终真实水域的论文数据集。后续真实水域数据必须建立独立的
dataset version，不得与本次数据混用同一版本号。

## 现场硬件

- Raspberry Pi 4；
- 当前已验证的 USB UVC 摄像头；
- HDMI 实验室显示器；
- 键盘和鼠标；
- 稳定的 Raspberry Pi 电源；
- 可选 USB SSD（长时间录制优先，挂载后把 `--output-root` 指向 SSD）。

采集全程可以离线运行，不需要 Wi-Fi、SSH、VNC 或云服务。Pi 本地桌面只
用于打开 Terminal 和显示 OpenCV preview。

## 出发前与开录前检查

1. 将本轮源码同步到 Pi bundle；至少必须同时包含 `run.py`、
   `scripts/capture_dataset.py` 和完整的 `src/fomo_servo/capture/`。旧的
   deployment bundle 原本只包含推理闭包，单独替换 `run.py` 不够。
2. 确认 USB 摄像头被识别：`ls -l /dev/video0`。若已安装 `v4l2-ctl`，可
   额外运行 `v4l2-ctl --list-devices`；该工具不是本项目的新依赖。
3. 运行 `date -Is`，确认系统日期、时间和时区正确。session 日期目录来自
   Pi 系统时钟；错误时钟会造成错误的 provenance。
4. 运行 `df -h .`（使用 USB SSD 时对挂载点运行），确认目标存储位置和
   剩余空间。不要把 SSD 挂载状态留到开始录制后再猜。
5. 固定摄像头，记录安装高度、朝向和视角；确认线缆不会进入目标运动区。
6. 先看 preview，确认画面方向、实际分辨率/FPS、目标覆盖范围、焦点、曝光
   和白平衡。工具只记录能够可靠读取的控制值，不会替用户猜测控制状态。
7. 明确本次 `--scene`、`--target`、`--notes`，不同条件拆成不同 session。

## 启动命令（Raspberry Pi 上）

```bash
cd /home/pi/fomo-ort-d2-epoch40
/home/pi/venvs/fomo-ort-d2-epoch40-preview/bin/python run.py capture_dataset \
  --source 0 \
  --output-root datasets_raw/lab_pool \
  --display --fullscreen \
  --width 640 --height 480 --fps 25 \
  --scene "pool-clear-water-front-view" \
  --target "fish-target" \
  --notes "fixed-camera session" \
  --min-free-gb 10
```

该 venv 路径和 `640x480 @ 25 FPS` 来自此前已完成的 Pi 4 USB UVC / preview
milestone。它们是本轮推荐的首个现场试录设置，不代表本轮已经在 Pi 上验收
capture 工具。若输出到 USB SSD，仅把 `--output-root` 改成已确认挂载的绝对
目录，例如 `/media/pi/LAB_POOL_SSD/datasets_raw/lab_pool`；不要在挂载失败时
退回到一个同名的根分区目录。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--source` | 摄像头 index（`0`）或设备节点（`/dev/video0`） |
| `--output-root` | 数据根目录（建议 `datasets_raw/lab_pool`） |
| `--session-prefix` | session ID 前缀；默认取 output-root 叶名最后一个 `_` 段（`lab_pool` → `pool`） |
| `--display` | 本地 OpenCV 预览窗口（普通窗口） |
| `--fullscreen` | 全屏预览（隐含 `--display`），现场 HDMI 采集用 |
| `--record` | 启动即录制；headless 模式（无 `--display`）总是自动录制 |
| `--width/--height/--fps` | 请求的采集参数（只 set，不 resize；实际值以 observed 为准） |
| `--frame-interval-seconds N` | 运行期间每 N 秒自动保存一张原始 JPEG（默认关闭） |
| `--min-free-gb N` | 剩余空间低于 N GB 时启动与运行中持续告警（默认 5；不自动删除任何数据） |
| `--max-frames N` / `--duration-seconds N` | 受控结束条件（可选，用于无人值守） |
| `--scene/--target/--notes` | 人工备注，写入 metadata |

## 现场操作流程（HDMI）

1. 接好摄像头、HDMI 屏、键盘，Pi 开机进入本地图形桌面，打开终端。
2. 用上面的命令启动（`--display --fullscreen`）。窗口标题为
   **FOMO Dataset Capture**，左下角常驻快捷键提示。
3. 按键操作：

   | 按键 | 动作 |
   | --- | --- |
   | `SPACE` | 开始 / 停止当前 recording |
   | `N` | 结束当前 session，自动建立下一个 session 目录 |
   | `S` | 保存当前原始帧为 JPEG |
   | `Q` / `ESC` | 安全退出（flush/close 所有文件后结束） |

4. HUD 显示：`CAMERA / RES / FPS / FREE SPACE / SESSION / STATUS
   (PREVIEW 或 ● REC) / REC TIME / VIDEO FRAMES / SAVED IMAGES`，
   低磁盘时追加醒目 `WARNING: LOW DISK SPACE`。
5. 主要数据资产是 `raw.avi`（分段时为 `raw_002.avi`...）。回到 PC 后统一
   抽帧、去模糊/去重、标注。

推荐的完整现场顺序：Pi 开机 → 接好并固定摄像头 → 打开本地 Terminal →
执行上面的命令 → 检查 preview/HUD → `SPACE` 开始录制 → `SPACE` 停止 →
`N` 创建下一个条件的 session → 重复录制 → `Q` 安全退出 → 检查文件。

### Session 采集设计

每次只改变少量条件，并用 `N` 分开记录。建议覆盖：

- 距离、视角、目标位置、静止/运动目标；
- 明暗变化、反光、阴影和不同背景；
- 固定相机、移动相机以及机器人运动造成的画面变化；
- 气泡、悬浮颗粒、线缆和水面扰动；
- **negative/background session**：空水池、池壁/池底、反光、气泡、线缆、
  机器人本体部件、手/工具、阴影和无目标运动。

negative session 对减少 false positive 很重要，不应只采“目标始终居中”的
正样本。视频是主要资产；`S` snapshot 和定时 snapshot 只作少量现场记录，
不要用高频 JPEG 代替连续视频。

## 目录结构

```text
datasets_raw/
└── lab_pool/
    └── 20260831/
        ├── pool-20260831-001/
        │   ├── raw.avi            # 第一段原始视频
        │   ├── raw_002.avi        # （同 session 内再次 SPACE 续录的分段）
        │   ├── frames/
        │   │   ├── frame_000001.jpg
        │   │   └── ...
        │   └── metadata.json
        ├── pool-20260831-002/
        └── pool-20260831-003/
```

- session ID 单调递增（`max(NNN)+1`），已存在的编号永不复用、永不覆盖。
- 新的一天自动从 `-001` 重新开始（按 `output_root/<YYYYMMDD>/` 分组）。
- `N` 键或程序结束都会为当前 session 写完整的 `metadata.json`。

## 原始数据语义

`raw.avi` 与 `frames/*.jpg` 保存**未经任何处理的原始帧**：

- 不 resize（除非相机本身协商失败，此时以 observed 分辨率原样写入）；
- 不加 overlay / 文字 / detection（HUD 只画在预览副本上）；
- 不做颜色转换、letterbox、归一化——BGR 原样落盘。

Codec 选择：**MJPG in AVI**。理由：OpenCV VideoWriter 在树莓派上对
MJPG/AVI 的写入路径最稳定（纯帧内 JPEG 编码，无系统 ffmpeg/x264 依赖，
无跨平台解码差异）；H.264/MP4 依赖系统编码器可用性。代价是体积大于
H.264（见下节估算）。

## 磁盘占用估算（MJPG，30 fps）

以下数字仅为 **synthetic/local calibration estimate**：Windows 开发机用
OpenCV 默认 MJPG writer 写入合成噪声帧和平滑渐变帧。它不是实验室水池
录像实测，也不能预测某次真实采集的精确码率。

| 分辨率 | 上界（噪声） | 下界（平滑） |
| --- | ---: | ---: |
| 640x480 | 约 413 MB/min / 24.8 GB/h | 约 11 MB/min / 0.64 GB/h |
| 1280x720 | 约 1.23 GB/min / 73.8 GB/h | 约 19 MB/min / 1.13 GB/h |
| 1920x1080 | 约 2.76 GB/min / 165.5 GB/h | 约 37 MB/min / 2.21 GB/h |

因此当前唯一可诚实引用的 `640x480` 标定范围约为 **0.64–24.8 GB/h**，跨度
很大。首次现场采集必须先录 1 分钟，再用该 session 的
`actual_video_size_bytes / recording_duration_seconds` 判断真实现场速率。
metadata 会给出 `actual_storage_gb_per_hour`；如果文件大小或有效录制时长
无法可靠取得，该字段为 `null`。`--min-free-gb` 默认 5，现场建议 10 以上。

## metadata.json schema（v1）

```json
{
  "schema_version": 1,
  "kind": "fomo_capture_session",
  "dataset_role": "LAB POOL ENGINEERING VALIDATION DATASET",
  "status": "completed | interrupted | failed",
  "end_reason": "user_quit | user_new_session | window_closed | max_frames | duration_reached | keyboard_interrupt | camera_read_failure | video_writer_open_failure | video_writer_write_failure | video_writer_release_failure | frame_size_mismatch | snapshot_write_failure | capture_error | unexpected_exception",
  "session_id": "pool-20260831-001",
  "scene": "...", "target": "...", "notes": "...",
  "start_time_utc": "...", "end_time_utc": "...", "duration_seconds": 0.0,
  "source": "/dev/video0",
  "camera_backend": "V4L2 | null",
  "requested_width": null, "requested_height": null, "requested_fps": null,
  "observed_width": 640, "observed_height": 480,
  "observed_fps": 25.0, "observed_fourcc": "YUYV",
  "measured_capture_fps": 24.7,
  "video_container": "avi", "codec": "MJPG",
  "video_filename": "raw.avi",
  "video_files": [{"filename": "raw.avi", "frame_count": 0, "duration_seconds": 0.0, "container_fps": 25.0, "file_size_bytes": 0}],
  "video_frame_count": 0,
  "snapshot_count": 0,
  "recording_duration_seconds": 0.0,
  "actual_video_size_bytes": 0,
  "actual_storage_gb_per_hour": null,
  "frame_interval_seconds": null,
  "frames_directory": "frames",
  "platform": {"platform": "...", "machine": "...", "python": "...", "opencv": "..."},
  "camera_controls": {
    "exposure": null, "auto_exposure": null, "gain": null,
    "white_balance": null, "auto_white_balance": null,
    "focus": null, "autofocus": null,
    "brightness": null, "contrast": null, "saturation": null
  }
}
```

`camera_controls` 遵循 no-guessing 规则：读不到或 OpenCV 返回
not-available 哨兵值（`0` / `-1`）时一律记 `null`，绝不猜测。metadata
始终输出上述固定 control key；未知 control 名、字符串、布尔值、NaN/Inf
等错误类型不会被伪装成可靠实测值。

## 数据安全

- `Q` / `ESC` / Ctrl+C / 窗口关闭 / 相机读失败 / 写入失败 / 任何异常：
  `try/finally` 保证 VideoWriter release（已录分段可播放）、
  VideoCapture release、窗口销毁，并在退出路径写入带
  `status=interrupted|failed` 与既有统计的 `metadata.json`。
- 磁盘低空间只告警，绝不自动删除旧数据。
- Git 边界：`datasets_raw/` 整目录已加入 `.gitignore`；仓库只包含采集
  源码、测试、文档与 schema 示例（本文档），真实数据永不入库。

## 退出后核验

先在 Terminal 中定位最近的 session，然后检查目录和 JSON：

```bash
find datasets_raw/lab_pool -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
/home/pi/venvs/fomo-ort-d2-epoch40-preview/bin/python -m json.tool \
  datasets_raw/lab_pool/<YYYYMMDD>/<SESSION_ID>/metadata.json >/dev/null
```

再用 Pi 桌面播放器打开 `raw.avi`，逐项人工检查：文件可播放、分辨率/时长
合理、视频中没有 HUD/REC/FPS/帮助文字；打开 `frames/*.jpg` 时同样不应有
任何 overlay。确认后再关机或拔出 USB SSD。

## 未来 train/val/test split 原则（重要）

**禁止**把抽帧后的图片随机切分为 train/val/test：相邻视频帧高度相似，
随机切分会造成 leakage，使验证指标虚高。

正确做法：按 **capture session**（`pool-YYYYMMDD-NNN`）或
**source video**（单个 `raw*.avi`）为最小单元做 group split——同一
session/source video 的所有帧必须落在同一个 split。建议在 session
数量足够时再划分（例如 8 个 session：5 train / 2 val / 1 test），并
优先按日期分层，避免同一天的光照/水位条件只出现在单一 split。

## 与项目其它部分的关系

本工具是独立的数据采集能力，不触碰也不改变：
`predict_image` / `predict_video` / ONNX export / formal model contract /
threshold / training / evaluation / robot-control 代码。capture 模块只使用
Pi preview profile 已有的 OpenCV/NumPy 运行时与 Python 标准库，不新增依赖，
并且没有 torch、onnxruntime 或 model 导入路径。

## Raspberry Pi acceptance checklist（同步后执行）

- [ ] 不连接 Wi-Fi、不使用 SSH/VNC，Pi 本地 Terminal 可启动工具。
- [ ] `/dev/video0` 能打开，HUD 显示正确的 camera、resolution 和 FPS。
- [ ] HDMI 普通 preview 正常。
- [ ] `--fullscreen` 全屏正常，键盘事件仍可用。
- [ ] `SPACE` 开始录制时明确显示 `REC`，再次按下能停止并关闭分段。
- [ ] `raw.avi` 可播放，帧数和 metadata 基本一致。
- [ ] `raw.avi` **没有** REC/FPS/session/help/timestamp/detection overlay。
- [ ] `S` 保存的 snapshot **没有**任何 overlay。
- [ ] `metadata.json` 可被 `python -m json.tool` 解析，requested/observed、
  camera controls、状态、计数和实际存储字段可信；未知值为 `null`。
- [ ] 连续按 `N` 会依次创建新的单调 session ID，旧目录和旧文件未改变。
- [ ] `Q`、`ESC` 和 Ctrl+C 都能释放 camera/writer 并完成 metadata。
- [ ] 把 `--min-free-gb` 临时设为高于当前可用空间时，Terminal/HUD 能看到
  `LOW DISK SPACE`，且工具不删除旧数据。
- [ ] 断开摄像头或用受控 fake/故障条件验证时，退出码非零，metadata 标为
  `failed`，不会显示“正常完成”。
- [ ] USB SSD（若使用）实际写入目标挂载点，没有因掉挂载写回 Pi 根分区。

本轮只在 Windows 开发机完成静态/单元测试；以上硬件验收在代码下一次同步
到 Raspberry Pi 4 后执行，未执行前不得宣称 capture 已在 Pi 现场验证。
