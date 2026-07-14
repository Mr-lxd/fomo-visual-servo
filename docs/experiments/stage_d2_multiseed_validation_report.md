# Stage D2 多 seed validation-only 稳定性报告

本实验固定使用 seed `42`、`123`、`2027`。三组都使用 D2 配方，FP32 扫描 60 个 weights-only snapshot，以 validation `centroid_pr_auc_macro` 选择 epoch，再用既有 strict one-to-one evaluator 在 validation 上选择阈值。seed `123` 和 `2027` 没有访问 test split；seed `42` 的 locked test 在本实验前已完成，未修改或重跑。

## 结果

| seed | selected epoch | threshold | Strict P | Strict R | Strict F1 | Macro F1 | EI legacy F1 | PR-AUC macro | Count MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 40 | 0.40 | 0.456522 | 0.392739 | 0.422235 | 0.382332 | 0.455696 | 0.218902 | 3.992126 |
| 123 | 43 | 0.55 | 0.480234 | 0.360836 | 0.412060 | 0.383682 | 0.443084 | 0.210315 | 3.826772 |
| 2027 | 35 | 0.50 | 0.536752 | 0.345435 | 0.420348 | 0.312365 | 0.444444 | 0.180958 | 3.842520 |

三组的 mean ± sample std（`ddof=1`）为：

| metric | mean | sample std | min | max |
| --- | ---: | ---: | ---: | ---: |
| selected epoch | 39.333333 | 4.041452 | 35 | 43 |
| threshold | 0.483333 | 0.076376 | 0.40 | 0.55 |
| Strict F1 | 0.418215 | 0.005413 | 0.412060 | 0.422235 |
| Macro F1 | 0.359460 | 0.040790 | 0.312365 | 0.383682 |
| EI legacy F1 | 0.447742 | 0.006922 | 0.443084 | 0.455696 |
| PR-AUC macro | 0.203392 | 0.019897 | 0.180958 | 0.218902 |
| Count MAE | 3.887139 | 0.091262 | 3.826772 | 3.992126 |

训练耗时：seed `123` wall time 1388.04 s（23 分 08.04 s），seed `2027` wall time 1398.41 s（23 分 18.41 s）。两组均完成 60 epoch、60 个 snapshot、CUDA AMP 训练。

## 类别稳定性

以下为 selected validation strict one-to-one F1 的三 seed 汇总；完整 precision、recall、F1 统计在输出目录的 `per_class_stability.csv` 中。

| class | F1 mean | F1 sample std | min | max |
| --- | ---: | ---: | ---: | ---: |
| fish | 0.432084 | 0.018485 | 0.419251 | 0.453271 |
| jellyfish | 0.504299 | 0.032695 | 0.483516 | 0.541985 |
| penguin | 0.323385 | 0.023633 | 0.296875 | 0.342246 |
| puffin | 0.431550 | 0.050790 | 0.373626 | 0.468468 |
| shark | 0.232394 | 0.063374 | 0.194444 | 0.305556 |
| starfish | 0.272743 | 0.174414 | 0.071429 | 0.378378 |
| stingray | 0.319763 | 0.128647 | 0.177778 | 0.428571 |

少数类尤其是 starfish、stingray 的 seed 敏感性明显；shark 的 F1 较低但三组相对接近。Macro F1 的波动主要由少数类组成，而整体 Strict F1 仍然紧密聚集。

## 解释边界

- seed42 的 Strict F1 `0.422235` 并未明显高于 seed123 和 seed2027，因此结论不依赖只报告 seed42。
- 三个 Strict F1 都高于 D1 random validation `0.3337`，也高于 focal validation `0.3463`。
- 按预先声明的规则，这支持 EI pretrained initialization 相对这两个 validation 基线具有稳定收益；但三组仍只有三个 seed，不据此构造过度解释的置信区间。
- Count MAE 范围为 `3.826772–3.992126`，未显示相对 seed42 的明确改善。
- seed 不是被优化的超参数；本实验不搜索更多 seed，也不使用 test 选择 seed、epoch 或 threshold。

## 固定 provenance

- Dataset content hash：`0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`
- Edge Impulse pretrained H5 SHA-256：`a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`
- loss：`ei_weighted_xent_legacy`
- background weight：`1`
- object weight：`100`
- augmentation：`underwater_conservative`
- input/output：`[B, 3, 192, 192] -> [B, 8, 24, 24]`
- 训练环境：Python 3.10.20、PyTorch 2.5.1+cu121、RTX 4060 Laptop GPU、CUDA AMP

seed123 的正式训练 commit 为 `c39dd6bb635fa7d0332f67b641c545e1be09c558`。seed2027 使用 `24286ba50a22088aab1acbcfcbe36472d4de332a`，该 commit 只修复 CUDA `map_location` 将 CPU/CUDA RNG ByteTensor 搬错设备导致的 resume 恢复失败；没有改动 fresh-training 的模型、loss、数据增强、optimizer、scheduler、batch、workers、epoch、checkpoint selection 或 threshold 规则。该 resume 修复已通过 CUDA 和 CPU 回归测试，并实际恢复两个新 run 的 `last.pt` 到 epoch 61。

每 seed 的详细产物：

- validation 逐图片明细：`outputs/experiments/stage_d2_multiseed_seed123/validation_scan/stage_c_validation_summary.json`、`stage_d2_multiseed_seed2027/...`
- 聚合 JSON/CSV/Markdown：`outputs/experiments/stage_d2_multiseed_validation/aggregate.json`、`aggregate.csv`、`aggregate.md`
- per-class 稳定性：`outputs/experiments/stage_d2_multiseed_validation/per_class_stability.csv`
- provenance：`outputs/experiments/stage_d2_multiseed_validation/seed_123_provenance.json`、`seed_2027_provenance.json`

本阶段没有进行 object-weight/loss ablation，也没有 push。
