# 实验与发布索引

本页是仓库内可复核的实验入口。原始 dataset、checkpoint、H5、TFLite、ONNX 和 `outputs/` 不提交到 Git；文档只保留 protocol、关键指标、hash 和可重生成的本地输出路径。

## 统一评估协议

训练使用 YAML 配置；validation 先按 `centroid_pr_auc_macro` 选择 snapshot，再在同一 validation split 上按 strict one-to-one `centroid_f1` 选择 threshold。locked test 只读取已经锁定的 epoch 和 validation threshold，禁止 sweep、自动选择或比较多个 checkpoint。Strict one-to-one 是正式主指标，EI legacy 只用于 Edge Impulse parity 解释。

协议定义：[threshold_protocol.md](../threshold_protocol.md)。

## 阶段索引

| 阶段 | 研究问题与唯一变化 | 主要结论 | 文档 | 配置/入口 | 是否运行 test | provenance |
| --- | --- | --- | --- | --- | --- | --- |
| Stage A / checkpoint selection | checkpoint 选择是否应与训练期固定阈值分离 | 建立 PR-AUC 选 epoch、validation 调 threshold、test 锁定评价的三划分协议 | [checkpoint v2 diff](../checkpoint_v2_locked_config_diff.md)、[threshold protocol](../threshold_protocol.md) | [checkpoint v2 config](../../configs/experiments/checkpoint_v2_lite_aug03_locked.yaml) | locked protocol 后才运行 | `5cedeaba`、`ad30925a`、`82ebf19c` |
| Stage B / locked evaluation | 锁定唯一 candidate 后，test 是否只评价一次 | wrapper 强制单 checkpoint、单 validation threshold 和 provenance | [D2 locked report](stage_d2_locked_test_report.md) | [locked evaluator](../../scripts/evaluate_d2_locked_test.py) | 仅锁定 candidate | `ad30925a`、`505f970b` |
| EI parity / parity-clean-v1 | 本地 evaluator 能否解释 Edge Impulse 指标差异 | 增加 fail-closed cleaning manifest、EI legacy 与 strict evaluator、TFLite parity 入口 | [EI parity audit](edge_impulse_parity_audit.md) | [parity evaluator](../../scripts/evaluate_edge_impulse_tflite.py) | 使用历史 locked test；不用于当前 D2 选择 | `d10f07e2` |
| Stage C / object weight | EI-style object weight 是否改善前景召回 | C2 w10 是当时 primary candidate；结论必须区分 fixed parity 与 validation-tuned focal baseline | [Stage C ablation](stage_c_object_weight_ablation.md) | `configs/experiments/loss_ei_object_w10.yaml` | 仅唯一 candidate | `473a795d`、`6932d903`、`01b7c3f9` |
| Stage C.1 / threshold fairness | 统一 validation threshold tuning 后，object-weight 比较是否仍成立 | focal baseline 与 C2 均应各自使用 validation 选择 threshold；不能把 fixed 0.5 与 tuned 结果混比 | [Stage C ablation](stage_c_object_weight_ablation.md) | `scripts/tune_focal_stage_c1.py` | locked candidate only | `e926c03f`、`01b7c3f9` |
| D1 / random-init FOMO | random-init 轻量 FOMO backbone 是否优于旧 baseline | D1 建立 stride-8 MobileNetV2 FOMO baseline | [D1 backbone](stage_d1_backbone_alignment.md) | [D1 config](../../configs/experiments/stage_d1_fomo_ei_w100.yaml) | D1 test 未运行 | `a072452c`、`2808ab0f` |
| D1.1 / architecture parity | 本地 backbone 是否与 Edge Impulse topology 对齐 | 修正 same padding、BatchNorm 与 head 对齐；进入 pretrained gate | [D1.1 parity](stage_d1_1_architecture_parity.md) | [FOMO model](../../src/fomo_servo/models/mobilenet_v2_fomo.py) | validation-only | `6ab45a7`、`5526e58d`、`08626c69` |
| EI pretrained initialization | Edge Impulse H5 是否可安全加载到本地 backbone | 95 tensors loaded，missing/unexpected 为 0；只初始化 backbone，head 保持 PyTorch 默认 | [pretrained validation](stage_d2_pretrained_validation.md) | [D2 config](../../configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml) | gate 前 validation-only | `04927069`、`7b2b9566` |
| D2 validation | pretrained initialization 的固定 seed42 candidate 表现如何 | seed42 选 epoch40、threshold0.40，validation Strict F1 0.422235 | [D2 multiseed report](stage_d2_multiseed_validation_report.md) | [D2 config](../../configs/experiments/stage_d2_fomo_ei_w100_pretrained.yaml) | seed42 locked test 已单独完成 | `04927069`、`7b2b9566` |
| D2 locked test | seed42 candidate 在 parity-clean test 上的 locked 指标是什么 | Strict F1 0.451977；EI legacy F1 0.484959；test 不参与选择 | [D2 locked report](stage_d2_locked_test_report.md) | [D2 locked evaluator](../../scripts/evaluate_d2_locked_test.py) | 是，仅 seed42 一次 | `505f970b`、`0cbb064f`、`bf3e3c76`、`6849f9c9` |
| D2 multi-seed validation | seed42 结论是否依赖单个随机 seed | 固定 42/123/2027；Strict F1 `0.418215 ± 0.005413`；123/2027 不运行 test | [multi-seed report](stage_d2_multiseed_validation_report.md)、[design](stage_d2_multiseed_validation_design.md) | [seed123](../../configs/experiments/stage_d2_multiseed_seed123.yaml)、[seed2027](../../configs/experiments/stage_d2_multiseed_seed2027.yaml) | 否，validation-only | `c39dd6bb`、`38d6330f` |
| D2 ONNX / Pi 4 deployment | 锁定的 D2 candidate 能否形成可移植 ORT/camera runtime | opset 17 ONNX、Windows/Pi parity、USB UVC 与 VNC preview 已验证 | [Pi 4 deployment handoff](../handoffs/2026-08-28-raspberry-pi4-deployment-handoff.md) | [export config](../../configs/export/d2_seed42_epoch40_onnx.yaml)、[bundle launcher](../../run.py) | 不用于模型选择或性能重评 | `7cc1d19d` |

## 当前正式候选

- 模型：MobileNetV2 FOMO，alpha `0.35`，输入 `192×192 RGB`，输出 `[B,8,24,24]`，stride `8`。
- 初始化：外部 Edge Impulse/Keras pretrained H5，仅用于 backbone；SHA-256 `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`。
- candidate：D2 seed42 / epoch40 / validation threshold `0.40`。
- 主指标：D2 seed42 locked-test strict one-to-one F1 `0.451977`。
- 稳定性：D2 multi-seed validation strict F1 `0.418215 ± 0.005413`。
- 正式 ONNX：opset `17`，固定 `[1,3,192,192] -> [1,8,24,24]` raw logits，SHA-256 `3dea74511bf2c44844192e75594fd53d4c4ce941f8b53b15767e020832bf9b08`。
- 部署状态：Raspberry Pi 4 ARM64 / Python 3.13 上的 ORT CPU、静态图片、预录视频、real-data smoke test、USB camera 与 VNC preview 已验证；该结果不代表机器人控制闭环或最终论文方法已经完成。

## 稳定主线与科研分支边界

`feature/fomo-main-integration-v1` 只集成当前已验证的 D2 baseline 与 Raspberry Pi 4 deployment milestone。Stage E MobileNetV3/SqueezeNet 对比属于有效但未进入正式系统的科研探索，其 `13b1802`、`4f0aadf` 历史继续保留在 `feature/fomo-backbone-ablation-v1`；“不进入稳定 main”不等于删除实验或 provenance。

## 后续事项

1. 从稳定 main 开展 camera → perception → target selection → visual servo/control → actuator/hardware 的机器人闭环。
2. 新的 baseline 创新、模型对比和消融优先使用 `experiment/*` 分支，并保持训练/validation/test 边界。
3. 完成实验室环境与真实环境验证、投稿版本冻结，以及最终代码与数据集开源方案。
4. Raspberry Pi 5 batch=1 CPU 延迟、内存、功耗和长期运行稳定性尚未实测。
5. pretrained H5 的 license/redistribution status 仍需从来源方确认；不加入仓库或 Release。
