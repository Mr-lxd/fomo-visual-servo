# Stage D1：Edge Impulse 对齐 backbone 实验

状态：D1 已完成唯一一次 60 epoch CUDA 正式训练和 validation-only snapshot scan；
尚未运行 test，等待 validation 结果批准。

## 研究问题与锁定变量

D1 以 C4 为基线，只改变 backbone：

- C4：`mobilenet_v2_lite`、EI legacy loss、object weight=100；
- D1：`mobilenet_v2_fomo`、EI legacy loss、object weight=100。

其余变量保持不变：192×192 输入、stride=8、width multiplier=0.35、head
channels=32、`underwater_conservative` augmentation、AdamW、scheduler、batch
size=8、workers=4、seed=42、60 epochs、AMP、snapshot interval、validation
selection、threshold grid、preprocessing、padding 和 matching。

逐字段配置证明：
`docs/experiments/stage_d1_config_diff.json` 和
`docs/experiments/stage_d1_config_diff.md`。

## Provenance

| 字段 | 值 |
|---|---|
| D1 config | `configs/experiments/stage_d1_fomo_ei_w100.yaml` |
| training code/config commit | `a072452c6dc7013e2d9dccb6f3f3c27c55c528d9` |
| backbone | `mobilenet_v2_fomo` |
| loss | `ei_weighted_xent_legacy` |
| object weight | 100 |
| pretrained | false；random initialization |
| dataset content hash | `0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562` |
| model parameters | 19,208 |
| output shape | `[B, 8, 24, 24]` |
| training device | CUDA；NVIDIA GeForce RTX 4060 Laptop GPU |
| training time | 1380.216 s |
| snapshots | 60 |

训练 snapshot metadata 记录的 backbone、loss、object weight、dataset hash 和
Git commit 均与 D1 配置一致。

## D1 validation 结果

选择协议与 C4 相同：先在全部 60 个 validation snapshot 上以
`centroid_pr_auc_macro` 选择 epoch，再在 validation 上按 strict one-to-one
centroid F1 选择阈值。数据为原始 validation 127 张，FP32，未读取 test。

| model | backbone | loss | object weight | selected epoch | threshold | strict P/R/F1 | macro F1 | EI legacy P/R/F1 | PR-AUC macro | count MAE |
|---|---|---|---:|---:|---:|---|---:|---|---:|---:|
| C4 | mobilenet_v2_lite | EI legacy | 100 | 59 | 0.35 | 0.4459/0.3036/0.3613 | 0.3270 | 0.5057/0.3291/0.3987 | 0.1461 | 4.5197 |
| D1 | mobilenet_v2_fomo | EI legacy | 100 | 59 | 0.30 | 0.3688/0.3047/0.3337 | 0.2557 | 0.4221/0.3316/0.3714 | 0.1353 | 4.5669 |

C4→D1 差异：

- strict F1：`-0.0275`；
- strict macro F1：`-0.0712`；
- EI legacy F1：`-0.0273`；
- PR-AUC macro：`-0.0108`；
- count MAE：`+0.0472`。

因此，在当前 locked 配方和 validation 协议下，D1 没有显示出
`mobilenet_v2_fomo` 能解释剩余 Edge Impulse 差距；相反，C4 的 validation
结果更好。该结论只来自 validation，不能用 test 重新选择 backbone 或阈值。

## D1 strict per-class validation

| class | TP | FP | FN | precision | recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| fish | 152 | 291 | 307 | 0.3431 | 0.3312 | 0.3370 | 0.1284 |
| jellyfish | 58 | 71 | 97 | 0.4496 | 0.3742 | 0.4085 | 0.2661 |
| penguin | 24 | 49 | 80 | 0.3288 | 0.2308 | 0.2712 | 0.0543 |
| puffin | 22 | 26 | 52 | 0.4583 | 0.2973 | 0.3607 | 0.0784 |
| shark | 20 | 36 | 37 | 0.3571 | 0.3509 | 0.3540 | 0.1654 |
| starfish | 0 | 1 | 27 | 0.0000 | 0.0000 | 0.0000 | 0.1019 |
| stingray | 1 | 0 | 32 | 1.0000 | 0.0303 | 0.0588 | 0.1523 |

D1 相对 C4 只改善 jellyfish 的 strict F1（`0.3362 → 0.4085`）；fish、penguin、
puffin、shark、starfish 和 stingray 均下降。D1 参数量从 C4 的 29,144 降至
19,208，但当前 validation 结果不支持保留该 backbone 作为性能优先方案。

## 产物

- 训练目录：`outputs/experiments/stage_d1_fomo_ei_w100/`
- validation summary：`outputs/experiments/stage_d1_fomo_ei_w100/validation_scan/stage_c_validation_summary.json`
- 60 个 snapshot：`outputs/experiments/stage_d1_fomo_ei_w100/epoch_snapshots/`

本阶段未创建 D1 test 结果，也未修改任何既有 C4 或 parity 输出。
