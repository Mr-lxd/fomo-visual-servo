# FOMO checkpoint selection v2：阶段 A 交接文档

> 生成日期：2026-07-13
> 用途：供没有当前聊天上下文的新 Codex/ChatGPT 会话直接继续工作。
> 边界：本阶段修改尚未 commit、尚未 push；本文件不代表正式训练或正式模型选择已经完成。

## 1. 项目概况

### 仓库、目标和研究问题

- 本地仓库：D:\DL_Project\fomo-visual-servo
- GitHub 远程：https://github.com/Mr-lxd/fomo-visual-servo.git
- 项目目标：建立可修改、可训练、可导出、最终部署到 Raspberry Pi 5 的 PyTorch FOMO 项目，用于轻量化水下目标视觉伺服。
- 当前研究问题：旧 best_centroid_f1.pt 按 fixed threshold=0.5 选择；训练后 threshold sweep 可能发现另一个 epoch 或 threshold 更优。因此 checkpoint selection、threshold calibration、final test evaluation 必须分离。

### 环境和数据

- Windows 笔记本，Python 3.10，PyTorch CUDA 训练环境；最终部署为 Raspberry Pi 5 + ONNX Runtime CPU。
- dataset 根目录由环境变量 FOMO_DATASET_ROOT 提供，源码和 YAML 不写死本机绝对路径。
- 本次检查的 PowerShell 进程中 FOMO_DATASET_ROOT 未设置；明确检查的 C:\Users\laixindong\Desktop\archive\aquarium_pretrain 存在，包含 train、valid、test、data.yaml。后续命令需在同一进程设置：
~~~powershell
$env:FOMO_DATASET_ROOT="C:\Users\laixindong\Desktop\archive\aquarium_pretrain"
~~~
- dataset content hash：
  0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562

### 模型、增强和阈值

- 默认/locked baseline 模型仍为 mobilenet_v2_lite。
- mobilenet_v2_fomo 仅作为更轻量的可选 backbone；当前 model01 使用它，但本阶段没有修改模型拓扑。
- 当前已有正式实验的 augmentation preset 为 underwater_conservative；locked baseline aug00_none_locked 为无增强。v2 不调增强。
- fixed checkpoint threshold=0.5；inference threshold=0.5。
- 当前研究重点是 checkpoint selection 与 threshold calibration 协议，不是继续调增强或测试 pretrained。

## 2. 当前 Git 状态

所有 Git 命令均使用：
~~~powershell
git -c safe.directory=D:/DL_Project/fomo-visual-servo <command>
~~~

实际结果：

- branch：feature/fomo-checkpoint-selection-v2
- HEAD：e2945bd526f920ed11bb9453e675f9618b3ec327
- working tree：不干净。
- 阶段 A 修改未 commit，未执行 push。
- origin：https://github.com/Mr-lxd/fomo-visual-servo.git

tracked 修改：
~~~text
 M docs/threshold_protocol.md
 M src/fomo_servo/config.py
 M src/fomo_servo/evaluation/__init__.py
 M src/fomo_servo/metrics/__init__.py
 M src/fomo_servo/metrics/centroid.py
 M src/fomo_servo/training/engine.py
 M tests/test_config.py
 M tests/test_training_engine.py
~~~

新增但未 tracked 的阶段 A 文件：
~~~text
?? docs/superpowers/plans/2026-07-13-checkpoint-selection-v2.md
?? docs/superpowers/specs/2026-07-13-checkpoint-selection-v2-design.md
?? scripts/audit_checkpoint_selection_v2.py
?? scripts/evaluate_epoch_snapshots.py
?? src/fomo_servo/evaluation/epoch_snapshots.py
?? src/fomo_servo/metrics/pr_auc.py
?? src/fomo_servo/training/snapshots.py
?? tests/test_checkpoint_selection_v2.py
~~~

tracked 部分的实际 git diff --stat：
~~~text
 docs/threshold_protocol.md            |  21 ++++
 src/fomo_servo/config.py              | 174 ++++++++++++++++++++++++++++++++++
 src/fomo_servo/evaluation/__init__.py |  20 +++-
 src/fomo_servo/metrics/__init__.py    |   6 ++
 src/fomo_servo/metrics/centroid.py    |   3 +
 src/fomo_servo/training/engine.py     |  81 +++++++++++++++-
 tests/test_config.py                  |  67 +++++++++++++
 tests/test_training_engine.py         |  39 ++++++++
 8 files changed, 409 insertions(+), 2 deletions(-)
~~~
注意：git diff --stat 默认不统计未 tracked 文件。

最近 5 个 commit：
~~~text
e2945bd feat: add MobileNetV2 FOMO backbone
83390d4 feat: add preset-based augmentation suite
2d03df0 feat: add horizontal flip augmentation
178ad3e feat: add color jitter augmentation
72ac415 feat: add configurable augmentation pipeline
~~~

## 3. 阶段 A 的目标和边界

旧协议中，best_centroid_f1.pt 按 validation fixed threshold=0.5 的 centroid F1 严格提升保存；训练结束后的 threshold sweep 可能发现另一个 epoch 或 threshold 更优。阶段 A 增加可复现的 weights-only epoch snapshot 和离线 FP32 selection。

本阶段没有改变：

- dataset、split、dataset hash；
- backbone、model width、head、input size、output stride；
- heatmap 标签生成、collision policy；
- augmentation preset 和增强参数；
- class weights、focal loss、focal gamma；
- optimizer、scheduler、batch size、num_workers；
- seed、epochs、AMP 训练协议；
- fixed checkpoint threshold、inference threshold；
- centroid matching、postprocess 基本语义和既有 evaluator 检测定义。

best_centroid_f1.pt、best_grid_f1.pt、best_val_f1.pt、last.pt 的旧文件语义保留。新写入的 best_centroid_f1.pt metadata 使用 fixed_centroid_f1，但仍是完整训练 checkpoint。

## 4. 本阶段实际修改文件

- src/fomo_servo/config.py：新增 EpochSnapshotConfig、CheckpointSelectionConfig、ThresholdCalibrationConfig；解析 snapshot、selection、calibration YAML；snapshot 默认 disabled。
- src/fomo_servo/training/engine.py：epoch 完成后按 interval 保存 snapshot；training summary 增加 v2 字段；candidate/weights-only snapshot resume 时提前报错；旧恢复路径保留。
- src/fomo_servo/training/snapshots.py：config fingerprint、CPU state 提取、snapshot schema、source hash、candidate 构造、原子写入和校验。
- src/fomo_servo/evaluation/epoch_snapshots.py：严格 FP32 split logit collection、offline metrics report、确定性 epoch selection、calibration split guard；评估 Dataset 显式 augmentation=None。
- src/fomo_servo/metrics/pr_auc.py：centroid PR-AUC、raw points、per-class、macro/micro AUC 和 effective class count。
- src/fomo_servo/metrics/centroid.py：per-class metrics 额外保留 TP/FP/FN。
- src/fomo_servo/evaluation/__init__.py、src/fomo_servo/metrics/__init__.py：导出新增接口。
- scripts/evaluate_epoch_snapshots.py：扫描 snapshot，输出 CSV/JSON/selection summary，并可生成 candidate。
- scripts/audit_checkpoint_selection_v2.py：审计 aug03/model01 六个旧 checkpoint。
- tests/test_checkpoint_selection_v2.py：覆盖 schema、state_dict、hash、atomic replace、inference load、resume rejection、PR-AUC、确定性选择、calibration guard、FP32/no-autocast。
- tests/test_config.py、tests/test_training_engine.py：覆盖 v2 配置、interval、legacy checkpoint 保留和 disabled snapshot。
- docs/threshold_protocol.md：补充 v2 与旧阈值协议边界。
- docs/superpowers/specs/2026-07-13-checkpoint-selection-v2-design.md：设计说明。
- docs/superpowers/plans/2026-07-13-checkpoint-selection-v2.md：实施计划。

## 5. Checkpoint 类型与 schema

### 5.1 完整训练 checkpoint

last.pt、best_val_f1.pt、best_grid_f1.pt、best_centroid_f1.pt 是完整训练 checkpoint，包含 model_state、optimizer_state、scheduler_state、scaler_state、epoch、best metrics、RNG state 等恢复状态，可以 resume。v2 不会把 candidate 伪造成完整 checkpoint，也不会猜测缺失状态。

### 5.2 Epoch snapshot

路径：<experiment-output>/epoch_snapshots/epoch_XXX_weights.pt

~~~text
checkpoint_kind: epoch_snapshot
weights_only: true
resumable: false
format: weights_only
model_state: CPU state_dict
epoch: positive integer
model_metadata: model identity and parameter metadata
parameter_count
config_fingerprint
dataset_content_hash
git_commit_sha
seed
augmentation_preset
checkpoint_threshold
~~~

snapshot 不含 optimizer、scheduler、GradScaler、RNG、history 或其他完整训练状态，因此不能 resume。默认 training.epoch_snapshots.enabled=false；启用后按 interval 保存，keep_last 可选保留最近几个。

### 5.3 Inference/evaluation candidate

候选文件：best_centroid_pr_auc_macro.pt、best_sweep_centroid_f1.pt。候选从已发布 source snapshot 读取、校验并重建，不是字节复制；保留 source 的 model state 和 provenance，并新增：

~~~text
checkpoint_kind: inference_candidate
weights_only: true
resumable: false
source_snapshot: epoch_XXX_weights.pt
source_snapshot_sha256: SHA-256(source snapshot bytes)
selected_epoch: integer
selection_metric: centroid_pr_auc_macro | max_centroid_f1_over_thresholds
selection_metric_value: float
selection_split: string
selection_dtype: float32
selection_details:
  threshold_grid: [...]
  integration: trapezoidal_observed_recall_no_envelope
  macro_effective_class_count: integer
pr_auc_threshold_grid: [...]
pr_auc_integration: trapezoidal_observed_recall_no_envelope
pr_auc_macro_effective_class_count: integer
~~~

原 snapshot 的 model_metadata、config_fingerprint、dataset_content_hash、git_commit_sha、seed、augmentation_preset、checkpoint_threshold 继续保留。candidate 可以 inference/evaluate，但 resume 会明确说明缺少 optimizer、scheduler、scaler 和完整训练状态，并建议使用 last.pt。

### 5.4 原子发布

snapshot/candidate 均在目标目录写临时文件，执行 flush、os.fsync 后用 os.replace 发布；失败时清理临时文件并保留上下文，不暴露半写入目标。

## 6. PR-AUC 精确定义

项目统一称为：centroid PR-AUC on the configured threshold grid。实际路径是 threshold sweep、centroid matching、centroid_pr_auc。

- PR 点来自 evaluation.checkpoint_selection.threshold_grid，默认 0.05、0.10、...、0.95。
- 每个 threshold 重新运行 centroid postprocess/matching 得到 precision、recall。
- 不是 COCO AP、bbox mAP，也不是标准 average precision。
- 不能称为 threshold-insensitive：grid 范围、步长和内容会改变点及 observed recall domain。
- raw (threshold, precision, recall) 按 threshold 升序保存；积分时按 recall 升序排序。
- duplicate recall 坐标保留最大 precision。
- 使用 trapezoidal 积分；不启用 precision envelope；不补 (0,1)、(1,0) 等人工端点。
- 积分 domain 是实际 grid 产生的 observed recall 范围。
- 有 GT 但只有一个 distinct recall 坐标时 AUC 为 0.0；无 GT 类别为 null 且不进入 macro。
- 有 GT 但无预测时 recall 全为 0，AUC 为 0.0。
- macro_effective_class_count 是具有 GT 且参与平均的 foreground class 数；macro 是这些 AUC 的算术平均。
- micro 使用 aggregate PR 点执行相同积分；aggregate 无 GT 时为 null。
- metrics JSON 保存 per-class raw curve 和 micro raw curve；candidate 保存 grid、积分方式和 macro effective class count。

## 7. 确定性选择规则

- CLI 文件扫描先用字典序；最终选择使用 report 中的数值 epoch。
- 只选择有限 metric 中的最高值。
- metric 并列时较早 epoch 优先；epoch 仍相同时 source snapshot 文件名字典序优先。
- max_centroid_f1_over_thresholds 对应 sweep centroid F1；centroid_pr_auc_macro 对应 macro PR-AUC。
- 既有 threshold sweep 的并列规则是较低 threshold 优先，排序 key 为 (-centroid_f1, threshold)。
- 因此选择不依赖目录返回顺序和 threshold 输入顺序，可复现。

## 8. 阶段 A 测试与质量检查

本次交接前重新执行：
~~~powershell
conda run --no-capture-output -n fomo-servo-train python -m pytest -q
conda run --no-capture-output -n fomo-servo-train python -m compileall src scripts
git -c safe.directory=D:/DL_Project/fomo-visual-servo diff --check
~~~

实际结果：212 passed, 4 skipped, 16 warnings；compileall 成功；diff check 成功。4 个 skipped 是因缺少 onnx 和 onnxruntime，不能写成 passed。

## 9. 旧 checkpoint FP32 回归审计

脚本：scripts/audit_checkpoint_selection_v2.py。

审计六个旧 checkpoint：

~~~text
outputs/experiments/aug03_underwater_conservative/best_centroid_f1.pt
outputs/experiments/aug03_underwater_conservative/best_grid_f1.pt
outputs/experiments/aug03_underwater_conservative/last.pt
outputs/experiments/model01_mobilenet_v2_fomo_aug03/best_centroid_f1.pt
outputs/experiments/model01_mobilenet_v2_fomo_aug03/best_grid_f1.pt
outputs/experiments/model01_mobilenet_v2_fomo_aug03/last.pt
~~~

使用 CPU FP32。输出：

- outputs/experiments/checkpoint_selection_v2/existing_checkpoint_audit.csv
- outputs/experiments/checkpoint_selection_v2/existing_checkpoint_audit.json

实际耗时约 31.79 秒；按六个 checkpoint 平均耗时估算，60 snapshot 纯 CPU FP32 扫描约 5.3 分钟。

历史 payload 指标与新离线 FP32 指标有约 1.7e-4～1.65e-3 差异。根因已核实：历史训练期 validation 使用 CUDA AMP；v2 强制 FP32、禁用 autocast；v2 与当前 scripts/evaluate.py FP32 结果一致；审计保留真实 delta，没有伪造一致。

## 10. 已完成、未完成和未验证事项

### 已完成且有代码/测试证据

- v2 配置 schema 和默认 disabled；
- 每 epoch weights-only snapshot；
- identity metadata、sanitized config fingerprint、dataset hash；
- candidate、source hash、atomic publication；
- candidate inference/evaluation load；
- candidate resume 明确拒绝；
- centroid PR-AUC observed-grid trapezoidal 实现；
- raw curve、threshold grid、macro effective class count 输出；
- calibration split guard 和显式 optimistic split reuse；
- legacy full checkpoint 兼容性测试；
- 六个旧 checkpoint CPU FP32 审计；
- pytest、compileall、diff check。

### 尚未完成或尚未提供端到端证据

- 没有新的正式 60 epoch；
- 没有真实 60 epoch snapshot 全扫描；
- 没有正式完整全 epoch candidate 选择；
- 没有独立 calibration split 正式运行；
- 没有正式 final test evaluation；
- 没有安装 onnx/onnxruntime；
- 尚未 commit；
- 尚未 push。

当前有 2 epoch synthetic CPU snapshot 单元测试，但尚未完成真实 CLI 级的“训练 → 多 snapshot → 离线扫描 → candidate → candidate evaluation → resume rejection → last.pt resume”端到端 smoke。单元测试通过不等同于完整工作流验收。

## 11. 风险与复核重点

- PR-AUC 是 configured threshold grid 上的 observed-point trapezoidal area；不同 checkpoint 可能有不同 observed recall range。
- 不补端点使该指标与标准 AP/COCO mAP 不同，跨实验解释依赖 grid。
- metric 仍受 grid 范围和步长影响。
- FP32 与历史 AMP 指标不可要求逐位一致；比较必须记录 dtype/protocol。
- 混合 snapshot 扫描必须检查 model identity、config fingerprint、dataset content hash。
- 必须检查 epoch 是否 off-by-one；必须按 epoch 数值排序，不能只按字符串。
- candidate 的 model_state 必须与 source snapshot 完全一致，并核对 source SHA-256。
- candidate 可 evaluate 但不可 resume；不要伪造 optimizer/scheduler/scaler。
- last.pt resume 必须兼容。
- outputs 审计文件通常被 .gitignore 忽略，不应为提交强行加入 Git。

## 12. 下一阶段建议

下一步唯一任务：阶段 A.1：端到端 smoke test 与实际 diff 验收。

建议顺序：

1. 审查实际 git diff 和新增文件；
2. 使用临时目录、小 fixture 和 2～3 epoch 专用配置；
3. 验证每个 snapshot 的 epoch、metadata、state_dict；
4. 运行离线 FP32 全 snapshot 扫描；
5. 生成两个 candidate；
6. 使用 candidate 执行 inference/evaluation；
7. 确认 candidate resume 明确拒绝；
8. 确认 last.pt 仍可 resume；
9. 构造 model identity/config fingerprint/dataset hash 不一致的混合 snapshot 测试；
10. 再次运行 pytest、compileall、diff check；
11. 人工审查后单独 commit 阶段 A。

之后才考虑正式 60 epoch locked 训练。

下一阶段仍不得：改训练配方、测试 pretrained、调增强参数、改模型、改 threshold 定义、push。

## 13. 关键路径索引

### 核心源码

- [配置解析](../../src/fomo_servo/config.py)
- [训练引擎](../../src/fomo_servo/training/engine.py)
- [snapshot/candidate](../../src/fomo_servo/training/snapshots.py)
- [offline evaluator](../../src/fomo_servo/evaluation/epoch_snapshots.py)
- [PR-AUC](../../src/fomo_servo/metrics/pr_auc.py)
- [centroid metrics](../../src/fomo_servo/metrics/centroid.py)

### CLI 和测试

- [snapshot evaluator](../../scripts/evaluate_epoch_snapshots.py)
- [legacy audit](../../scripts/audit_checkpoint_selection_v2.py)
- [v2 tests](../../tests/test_checkpoint_selection_v2.py)
- [config tests](../../tests/test_config.py)
- [training tests](../../tests/test_training_engine.py)

### 设计、协议、配置和输出

- [v2 design](../superpowers/specs/2026-07-13-checkpoint-selection-v2-design.md)
- [v2 implementation plan](../superpowers/plans/2026-07-13-checkpoint-selection-v2.md)
- [threshold protocol](../threshold_protocol.md)
- [AGENTS.md](../../AGENTS.md)
- configs/experiments/aug00_none_locked.yaml
- configs/experiments/model01_mobilenet_v2_fomo_aug03.yaml
- outputs/experiments/aug03_underwater_conservative/
- outputs/experiments/model01_mobilenet_v2_fomo_aug03/
- outputs/experiments/checkpoint_selection_v2/
- 阶段 A.1 建议临时输出：outputs/experiments/checkpoint_selection_v2_smoke/

## 14. 新会话启动消息

下面内容可直接复制到全新的 ChatGPT/Codex 会话：

~~~text
仓库：D:\DL_Project\fomo-visual-servo

请先阅读：
1. AGENTS.md
2. docs/handoffs/2026-07-13-checkpoint-selection-v2-stage-a-handoff.md
3. docs/superpowers/specs/2026-07-13-checkpoint-selection-v2-design.md
4. docs/superpowers/plans/2026-07-13-checkpoint-selection-v2.md
5. docs/threshold_protocol.md

当前实际状态：
- branch：feature/fomo-checkpoint-selection-v2
- HEAD：e2945bd526f920ed11bb9453e675f9618b3ec327
- working tree：不干净；阶段 A 修改未 commit，未 push。
- 默认模型：mobilenet_v2_lite。
- mobilenet_v2_fomo 只是可选轻量 backbone；本阶段不改模型。
- 历史实验 augmentation preset：underwater_conservative；locked baseline aug00_none_locked 为 disabled。
- fixed checkpoint threshold=0.5，inference threshold=0.5。
- dataset 通过 FOMO_DATASET_ROOT 注入；dataset content hash 为 0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562。

阶段 A 已实现：
- v2 epoch weights-only snapshot schema 和 atomic write；
- inference/evaluation candidate best_centroid_pr_auc_macro.pt 与 best_sweep_centroid_f1.pt；
- candidate 明确 weights_only=true、resumable=false，resume 会说明缺少 optimizer/scheduler/scaler/full state；
- offline FP32 evaluator、PR-AUC、calibration split guard、六 checkpoint audit；
- 旧 last.pt/best_val_f1.pt/best_grid_f1.pt/best_centroid_f1.pt 协议保持兼容。

PR-AUC 准确定义：
- PR 点来自 configured centroid threshold grid；
- 不是 COCO AP、bbox mAP 或标准 AP；
- raw points 按 recall 排序，duplicate recall 保留最高 precision；
- observed-point trapezoidal integral；无 precision envelope，不补人工端点；
- 无 GT 类别排除 macro，有 GT 无预测返回 0；
- 保存 raw curve 和 macro effective class count；
- 仍受 threshold grid 范围和步长影响。

质量结果：
- pytest：212 passed, 4 skipped；skip 原因是缺少 onnx/onnxruntime；
- compileall 成功；diff check 成功。
- 六个旧 checkpoint CPU FP32 audit 约 31.79 秒；历史 AMP payload 与 FP32 offline 结果有约 1.7e-4～1.65e-3 差异，v2 与当前 scripts/evaluate.py 的 FP32 结果一致。

尚未完成：
- 没有新的正式 60 epoch；
- 没有真实 60 epoch snapshot 全扫描；
- 没有正式 calibration split；
- 没有正式 final test evaluation；
- 没有完整 CLI 级训练→snapshot→scan→candidate→evaluate→resume rejection→last.pt resume smoke。

本会话唯一下一项任务：阶段 A.1 端到端 smoke test 与实际 diff 验收。

不可改变变量：dataset/split/hash、backbone/model width/head、input_size、output_stride、heatmap labels、collision policy、augmentation preset/参数、class weights、focal loss/gamma、optimizer、scheduler、batch size、num_workers、seed、epochs、AMP protocol、checkpoint threshold、inference threshold、centroid matching、postprocess 基本语义和既有 evaluator。

要求：
1. 首先只阅读本交接文档并用不超过 20 行复述项目状态；
2. 检查实际 Git branch、HEAD、status 和 diff；
3. 再整理阶段 A.1 可执行计划；
4. 不立即修改代码；
5. 不运行正式 60 epoch；
6. 不测试 pretrained；
7. 不 push。
~~~
