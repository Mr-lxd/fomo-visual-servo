# 会话数据包工具（规范 17）

三个工具离线工作，依赖现有 numpy、OpenCV 和 matplotlib；不用 torch、模型、Console 或 Pi 连接。只读输入，数据包输出目录必须不存在。开发样本与冻结测试会话不用于调整算法或参数。

## 会话登记

把 [sessions.csv 模板](templates/sessions.csv) 复制到数据根目录的 `sessions.csv`，每次 Record 登记一行。列为 `session_id,date,location_water,lighting,turbidity,targets,backend,actions,split,notes`，对应会话 ID、日期、地点/水体、光照、浑浊度、类别×数量、后端、动作、划分、备注。`split` 只能是 `dev`、`test` 或 `desk`。划分在录制前决定；两个开发样本不登记。

## build_session_package.py

```powershell
python scripts/build_session_package.py --raw <raw/session-id> --sessions <sessions.csv> --output <packages/session-id>
```

raw 目录包含 `frame_index.csv`、它列出的 AVI 分段、一个 `motion*.csv`、可选一个 `visual*.csv` 与 `metadata.json`。索引列为 `segment,segment_frame_index,frame_id,capture_timestamp_ns`。默认会话 ID 来自 raw 目录名，可用 `--session-id` 指定。

- 默认 `--stride 12`，按索引位置从第 0 帧均匀抽帧；`--keyframes 123 456` 补帧，重合帧也标为 `keyframe`。逐帧解码所有分段，核对每段索引数与实际可解码帧数，没有模型选帧或 overlay。
- `--only motion` / `--only video` 分别用于阶段 A 组件验证。`--dev-sample` 允许未登记的开发样本，`registration=null`，不把它补写进登记表；这不是生产数据的登记方式。
- MCU→Pi 时钟：默认 `--fit-window-ms 1000`。先按 `batch_seq` 去重，以批次 `mcu_tx_ms` 对 `pi_rx_ms` 拟合；每窗选择 `pi_rx_ms-a*mcu_tx_ms` 最小的批次，对这些下包络点拟合斜率，迭代三轮，最终截距取全部批次的最小接收偏移。再把 a、b 应用于样本的 `mcu_unwrapped_ms`，得到 `offline_pi_ms`，避免把批内样本排队时间混入拟合。批次发送时间须递增，当前不展开跨 32 位回绕的发送时间。这个截距仍含最小传输/传感器延迟，不能当独立同步真值。原始 CSV 最后的计数器行不参与拟合，但保留在 `imu.csv`。
- 输出 `frames/` JPEG、`frames.csv`、`imu.csv`、`session.json`。`imu.csv` 保留全部原始行，加 `offline_pi_ms`；计数器行该列为空。
- `frames.csv` 列：`frame_id,capture_ts_ns,sampling,image_file,gyro_x,gyro_y,gyro_z,roll,pitch,gait_phase,backend,active_mode,target_mode,motion_state,depth_cal_m,depth_age_ms,depth_match_dt_ms,nearest_imu_dt_ms`。gyro 为 deg/s，roll/pitch 为 deg。原 motion CSV 的 `gait_phase` 实为 `[0,2π)` 弧度（u16×2π/65536）；内部转为 turns 处理 1→0 回绕，再转回弧度，保持单位。有效标记必须在两侧都为 1；范围外不外推，列留空。状态取帧时刻之前最近的样本。
- 深度优先同 `frame_id`，否则在有深度值的 visual 行中选择最近 `capture_ts_ns`，仅接受绝对时间差 ≤100 ms，超出则深度相关列留空。`depth_age_ms` 保留原 visual 值；`depth_match_dt_ms` 记录已接受匹配的绝对时间差（ms），同 ID 行缺时间戳时该列留空；没有 depth 行则留空。近邻匹配不代表深度新鲜，使用时仍需检查 age。
- `session.json` 记录登记行、每个输入的路径/SHA-256、拟合参数/残差/相对在线映射差异、frame_id 跳号和缺帧数、全部视频帧 50 ms 内 IMU 覆盖率、抽样帧深度覆盖率、计数器起止和增量。计数器起值可能是录制前累计，不能把起值叫作本次会话丢片数。还记录 Forward 相位回绕插值所得 Pi 时钟周期。

帧 ID 跳号只能统计首末已记录帧之间的缺口；首帧前和末帧后的相机丢帧无法从此索引推断。`nearest_imu_dt_ms` 为绝对时间差。

## camera_imu_lag.py

```powershell
python scripts/camera_imu_lag.py --package <packages/session-id> --start 5 --end 25 --output <lag-report-dir>
```

`start/end` 单位为秒，从数据包第一张抽样帧起算。对相邻灰度 JPEG 作全图相位相关；`atan(dx/focal_length)/dt` 得到水平角速度估计，与 `gyro_z` 作带符号相关。默认 focal_length 为图像宽度，是未标定的角速度代理；已标定时传 `--focal-length-px`。左转 gyro_z>0，背景右移 dx>0。

正的 lag 意味着相机落后 IMU：`camera(t)≈gyro_z(t-lag)`。默认扫描 ±500 ms、步长 5 ms，可用 `--max-lag-ms` / `--lag-step-ms` 改。所有偏移使用同一个重叠区间；时间落在 IMU 之外不参与计算。无变化或不足三个共同样本时输出 `lag_ms=null`，不制造测量值；峰在边界/无正相关时标注状态。

输出 `camera_imu_alignment.png`、`camera_motion.csv` 和 `camera_imu_lag.json`。相关峰仅是估计，5 ms 网格不是测量精度。**阶段 B 的手转段必须以 `--stride 1` 打包**；每 12 帧抽一帧不能支撑一帧（40 ms）的时间差验收。阶段 A 拼接数据的相关数值没有物理意义。

## check_annotations.py

```powershell
python scripts/check_annotations.py --annotations <images/train> --report <annotation-report.json>
python scripts/check_annotations.py --annotations <new-session-annotations> --report <annotation-report.json> --strict
```

只读取明确指定目录的 JSON，不递归扫描、不打开图片、不改标注，跳过 X-AnyLabeling 的 `attributes.json`。

检查 5 个源类别、visibility、非 ignore fish/tuna 的 orientation、head/tail 标签和数量、axial/ignore 禁点、group_id 唯一对应、点的画面边界/所属框以及 point_state。与 07b 一样允许框边 2 px 的标注误差，但画面边界不放宽。full 的 side/oblique 框须有完整头尾；部分可见框允许缺点，头尾都在画面外时也允许零点，不要求猜位置。完整头尾对只计非 ignore、非 axial 的有效标签组合。

默认兼容旧 07a/07b：省略的 visibility 按 X-AnyLabeling 默认 full 解析；多 fish/tuna 的无 group_id 点，仅当落在唯一框内时可对应。这两类情况记入 warnings。**新数据请使用 `--strict`**，把缺显式 visibility 和多目标无 group_id 视为问题。属性优先读 `attributes`，再识别合法 description 值。

JSON 报告包括类别×visibility 框数、完整头尾对数、问题清单和警告。存在问题时 exit 1（报告仍会保存）；无问题 exit 0。旧训练标注里的 2 个 ignore 框有点是已知问题，不由此脚本修改。

## 阶段 A / B 边界

阶段 A 用非同步的真实视频和 motion 分别验证，再在结果目录创建平移时间戳的副本验证完整输出路径；原始开发样本哈希保留，合成数据明确标识，不登记为实际会话、不报告真实相机–IMU lag。

阶段 B 等整机装配完成：用户录制，Claude Code 拷文件并登记 `desk`，再对真实同步会话运行全流程（手转段 stride=1），核对 IMU/深度覆盖、缺帧、时间差。当前交付不宣称阶段 B 的 ≤40 ms 或深度路径实测通过。
