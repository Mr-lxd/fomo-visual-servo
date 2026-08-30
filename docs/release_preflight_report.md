# GitHub 发布前审计报告

审计日期：2026-07-14。范围是本地仓库整理、provenance、敏感信息、Markdown、配置/代码发布阻塞项和质量门。没有训练、没有新的 test 集评估、没有 merge/rebase/reset/clean/stash，没有 push，也没有创建 PR。

## 1. GitHub 与当前分支

| 项目 | 结果 |
| --- | --- |
| origin | `https://github.com/Mr-lxd/fomo-visual-servo.git` |
| 默认分支 | `main` / `origin/main` |
| 当前分支 | `feature/fomo-checkpoint-selection-v2` |
| 审计基线 HEAD | `8dc0e2d92b3f4be30f9679930906e361ecc9a0e1` |
| 当前分支 tracking branch | 无；远程没有同名 feature branch |
| 相对 `origin/main` | ahead `33`，behind `0` |
| 同名 PR | 无 |
| GitHub CLI | `gh 2.96.0`，已认证；token 内容未写入仓库或报告 |

`git fetch --all --prune` 已执行；只更新了本地远程引用，没有 pull 或自动 merge/rebase。

## 2. 本地分支审计

`unique vs current` 表示该分支有多少提交不在当前 feature 分支；`unique vs default` 表示该分支有多少提交不在 `origin/main`。

| branch | HEAD | upstream | relative current | merged into current | merged into default | unique vs current | unique vs default | recommendation |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `codex-fomo-pre-augmentation-baseline` | `af4e69b` | none | behind 31 | yes | no | 0 | 2 | 保留历史引用，是否删除另行批准 |
| `experiment/fomo-aug01-color` | `178ad3e` | none | behind 28 | yes | no | 0 | 5 | 保留历史引用，是否删除另行批准 |
| `experiment/fomo-aug02-color-hflip` | `2d03df0` | none | behind 27 | yes | no | 0 | 6 | 保留历史引用，是否删除另行批准 |
| `experiment/fomo-augmentation-v1` | `72ac415` | none | behind 29 | yes | no | 0 | 4 | 保留历史引用，是否删除另行批准 |
| `experiment/fomo-official-mobilenetv2` | `e2945bd` | none | behind 25 | yes | no | 0 | 8 | 保留历史引用，是否删除另行批准 |
| `feature/fomo-augmentation-suite` | `83390d4` | none | behind 26 | yes | no | 0 | 7 | 保留历史引用，是否删除另行批准 |
| `feature/fomo-checkpoint-selection-v2` | `38d6330` | none | current | yes | no | 0 | 33 | 建议经批准后作为发布分支 |
| `main` | `8d7c392` | `origin/main` | behind 33 | yes | yes | 0 | 0 | 保持默认分支不变 |

所有其他本地分支的提交都已包含在当前 feature 分支中，没有发现需要 cherry-pick 的遗漏正式提交，也没有发现 detached-only 的重要提交。本轮没有删除或合并任何分支。

## 3. Provenance

完整阶段表见 [commit_provenance.md](experiments/commit_provenance.md)。关键当前候选 provenance：

- D2 pretrained implementation/training：`0492706901c93bddcd4cf3ee9e3ab708fed590b5`
- D2 seed42 locked evaluator：`505f970b900bf1effaf8d7d569c9ae371789dcb5`
- D2 multi-seed configs：`c39dd6bb635fa7d0332f67b641c545e1be09c558`
- CUDA resume RNG fix：`24286ba50a22088aab1acbcfcbe36472d4de332a`
- D2 multi-seed report：`38d6330ffd2213fda294374cf14ad8bba7d8f0fe`
- 本轮仓库整理提交：`8dc0e2d92b3f4be30f9679930906e361ecc9a0e1`

所有上述 SHA 都是当前 feature branch 历史中的 commit。seed42 的训练、evaluation、文档提交已分开记录；seed123 的正式训练 commit 为 `c39dd6bb`，seed2027 使用 `24286ba5`，后者只处理 resume RNG state 的设备归属，不改变 fresh-training 数值路径。

## 4. Tracked、ignored 与大文件

- tracked 文件数：`676`。
- tracked 且大于 1 MB 的文件：无。
- `outputs/`、`.pytest_cache/`、Python `__pycache__`、egg-info 均为 ignored；dataset、checkpoint、H5、TFLite、ONNX、ZIP 和日志没有被 Git 跟踪。
- Git object database：约 `1.01 MiB`，无 pack、garbage 或不可达对象报告。
- 本轮补强 `.gitignore`：`*.h5`、`*.hdf5`、`*.zip`、`/cache/`、`/inference_cache/`。
- 未删除用户文件，也未取消任何错误跟踪项。

## 5. Secret 与隐私扫描

仓库没有安装 `gitleaks`、`trufflehog` 或 `detect-secrets`，因此使用了 `git grep` 的有限扫描：API key、GitHub token、access token、password assignment、private key、Edge Impulse key、`.env`、邮箱和用户目录路径均未发现命中。

历史 handoff 和计划文档中的真实 Windows 用户目录数据/H5/ZIP 路径已改为 `<DATASET_ROOT>`、`<EI_EXPORT_ZIP>` 等占位符。`safe.directory=<REPO_ROOT>` 仅作为 Git 命令示例保留；源码、YAML 和训练脚本没有机器绝对路径。

## 6. Markdown 与发布文档

- 仓库内 Markdown 链接保持相对路径；external URLs 保留为 external URLs。
- ignored `outputs/` 不再伪装成 GitHub 可访问链接，改为代码格式的本地生成路径。
- 根 README 已更新当前 D2 seed42/epoch40/threshold0.40 候选、D2 multi-seed validation、locked-test、EI legacy 语义、H5 SHA、安装和 ONNX optional 依赖。
- 新增 [实验索引](experiments/README.md) 和 [D2 model card](model_card_d2_seed42.md)。
- 历史 C1/C2/epoch58 与 EI parity 文档已明确标注为历史结果，不再声称是当前最佳候选。
- 所有本地 Markdown link checker 结果为 `0` 个 broken local links。

## 7. 当前正式候选

- 架构：MobileNetV2 FOMO alpha `0.35`，输入 `192×192 RGB`，输出 `[B,8,24,24]`，stride `8`。
- 初始化：外部 Edge Impulse/Keras MobileNetV2 H5，仅用于 backbone；SHA-256 `a94030b8c5e6811c60b93c8b6888d2f309dc112008bd14f0963e8c5473201c2c`。
- candidate：D2 seed42 / epoch40 / validation threshold `0.40`。
- validation multi-seed Strict F1：`0.418215 ± 0.005413`，seed `42/123/2027`，validation-only。
- locked-test seed42 Strict F1：`0.451977`；EI legacy F1：`0.484959`。
- EI legacy 仅用于 parity，不是正式主指标。
- 详细限制：[model_card_d2_seed42.md](model_card_d2_seed42.md)。

## 8. Pretrained 来源与许可

来源 URL：<https://cdn.edgeimpulse.com/transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5>。

该 H5 是 Edge Impulse transfer-learning 使用的 Keras MobileNetV2 alpha 0.35 / input 96 artifact，不是包含 FOMO 七类 head 的完整分类器。本项目只加载 block 6 expansion 之前的 backbone tensors，head/classifier 使用本地初始化。Keras Applications 的来源说明包含 MIT notice；下载 H5 本身没有 embedded license field，因此 `license/redistribution status requires confirmation`。H5 不进入 GitHub 仓库或 Release。

## 9. 代码与配置审查

- config schema 支持环境变量路径，D2 配置没有本机绝对路径。
- pretrained loader 验证 source SHA、shape、loaded/missing/unexpected tensor provenance。
- resume RNG fix 已对 CPU/CUDA ByteTensor 状态做显式类型检查和 CPU API 归一化。
- strict evaluator 与 EI legacy evaluator 是独立模式。
- parity-clean manifest 在输入文件 hash、manifest 外非法标签和 cleaning view hash 不一致时 fail closed。
- D2 locked-test CLI 拒绝 threshold override、split override 和 sweep。
- validation-only snapshot evaluator 使用配置的 validation split，不访问 test。
- ONNX/ONNX Runtime import 仅在对应可选测试/部署路径中使用；缺失时测试显式 skip，不改变核心依赖。

## 10. 质量检查

| command | result |
| --- | --- |
| 定向 config/locked/pretrained/cleaning/resume tests | `83 passed, 1 warning` |
| `python -m pytest -q` | `287 passed, 4 skipped, 16 warnings` |
| `python -m compileall -q src scripts` | pass |
| `git diff --check` | pass |
| ONNX skips | 4；原因仅为当前环境未安装 `onnx`/`onnxruntime` |

没有为消除 ONNX skip 安装依赖，也没有运行新的 test 集评估。

## 11. 发布建议与阻塞项

建议推送分支：`feature/fomo-checkpoint-selection-v2`。

建议 PR base：`main`。

建议 Draft PR 标题：

`docs: prepare FOMO D2 candidate for review`

建议正文摘要：

> This draft records the D2 MobileNetV2 FOMO candidate, locked-test protocol, EI parity boundaries, and fixed-seed validation stability. The candidate is seed42/epoch40 with validation threshold 0.40; strict locked-test F1 is 0.451977 and D2 multi-seed validation Strict F1 is 0.418215 ± 0.005413. Dataset, weights, H5, TFLite, ONNX, ZIP, and outputs remain external and untracked. Raspberry Pi 5 measurements and pretrained artifact redistribution status remain open.

发布阻塞项：

1. 用户尚未批准 `git push`；本轮必须停止在 push 之前。
2. 当前 feature branch 没有远程 tracking branch，后续推送需要用户确认远程目标。
3. H5 artifact 的 license/redistribution status 尚未明确确认。
4. ONNX/ONNX Runtime optional dependencies 当前未安装，导出/运行时一致性测试尚未在本机执行。
5. Raspberry Pi 5 的实际 CPU 延迟、内存、功耗和稳定性尚未实测。

## 12. 当前 Git 状态

生成本报告前工作树 clean；本报告将作为单独的本地文档提交。未 push，未创建 PR。
