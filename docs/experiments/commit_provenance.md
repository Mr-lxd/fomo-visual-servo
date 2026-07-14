# Experiment commit provenance

本表只记录已存在于当前 feature branch 历史中的提交。训练输出、dataset、checkpoint、H5、TFLite、ONNX、ZIP 和缓存均未提交。`implementation commit` 表示源码/配置变更，`evaluation commit` 表示 evaluator 或锁定协议，`documentation commit` 表示结果或协议文档；三者不应互相替代。

| stage | purpose | implementation commit | evaluation commit | documentation commit | output committed? |
| --- | --- | --- | --- | --- | --- |
| Stage A / checkpoint selection | 建立 validation PR-AUC epoch selection 与 threshold 分离 | `5cedeaba41049f1b1ba73eb23e48cc9ce6dd7c9a` | `ad30925aaf05097b996152cafd867ca608004527` | `ad30925aaf05097b996152cafd867ca608004527` | no |
| Stage B / locked evaluation | 固定 candidate、validation threshold 后只运行一次 test | `82ebf19cb5946e820424cfb6077b95e1574a95d9` | `ad30925aaf05097b996152cafd867ca608004527` | `ad30925aaf05097b996152cafd867ca608004527` | no |
| EI parity / parity-clean-v1 | fail-closed cleaning manifest、EI legacy evaluator、TFLite parity | `d10f07e22cb80b61592f46ca963c9de8a156b612` | `d10f07e22cb80b61592f46ca963c9de8a156b612` | `d10f07e22cb80b61592f46ca963c9de8a156b612` | no |
| Stage C / object weight | EI-style object weight 1/10/30/100 ablation | `473a795da7de5fba1e0e68ebbd289c9f54184e79` | `473a795da7de5fba1e0e68ebbd289c9f54184e79` | `6932d9039b087ff439220c2e620b6fba7c7fb38b`、`01b7c3f94f4b968f127b2283042b6b1a118dc12e` | no |
| Stage C.1 / threshold fairness | focal baseline validation threshold audit | `e926c03fd89d91dc9b7d6aa1d6ca6ba067607b54` | `e926c03fd89d91dc9b7d6aa1d6ca6ba067607b54` | `01b7c3f94f4b968f127b2283042b6b1a118dc12e` | no |
| D1 / random-init FOMO | random-init MobileNetV2 FOMO backbone experiment | `a072452c6dc7013e2d9dccb6f3f3c27c55c528d9` | `a072452c6dc7013e2d9dccb6f3f3c27c55c528d9` | `2808ab0ff4baca680f1c94de86fbe465d8d1891c` | no |
| D1.1 / architecture parity | same-padding、BatchNorm、head 与 EI topology 对齐 | `6ab45a76825d79c9a0587e248a670c49513d7ddb`、`5526e58d586920213c96b09fcde0b4266df2244c` | `5526e58d586920213c96b09fcde0b4266df2244c` | `08626c69cbbab27f8e935bb7897c19aab1ab9ca0` | no |
| EI pretrained initialization | H5 backbone loader、hash 校验、95 tensor mapping | `0492706901c93bddcd4cf3ee9e3ab708fed590b5` | `0492706901c93bddcd4cf3ee9e3ab708fed590b5` | `7b2b9566976eb0ab342c4eba8e3f97087df3f49f` | no |
| D2 locked test | seed42 epoch40 parity-clean locked evaluation | `505f970b900bf1effaf8d7d569c9ae371789dcb5` | `505f970b900bf1effaf8d7d569c9ae371789dcb5`、`bf3e3c76129ed4627903308c6709696e0acbfc71`、`0cbb064f98bdfd3a658b2b0c69b7e9d7fcf3bf32` | `6849f9c97879dc036a37e3ed17014ae247fd5969` | no |
| D2 multi-seed validation | fixed seeds 42/123/2027 validation-only stability | `c39dd6bb635fa7d0332f67b641c545e1be09c558` | `c39dd6bb635fa7d0332f67b641c545e1be09c558` | `38d6330ffd2213fda294374cf14ad8bba7d8f0fe` | no |
| CUDA resume RNG fix | normalize CPU/CUDA ByteTensor states after `map_location=device` | `24286ba50a22088aab1acbcfcbe36472d4de332a` | `24286ba50a22088aab1acbcfcbe36472d4de332a` | `38d6330ffd2213fda294374cf14ad8bba7d8f0fe` | no |
| Current D2 documentation | multi-seed report and current candidate record | no new training code | no new evaluation | `38d6330ffd2213fda294374cf14ad8bba7d8f0fe` | no |

## Verification

每个 SHA 均应满足 `git cat-file -t <sha> == commit`，且均已位于当前 feature branch 的祖先历史中。D2 seed123 的正式训练 commit 是 `c39dd6bb`；seed2027 使用 `24286ba5`，后者只修复 resume RNG 设备归属，不改变 fresh-training 数值路径。seed42 locked-test 的 evaluation commit 与 training commit 分开记录，避免把 evaluation 修复误写成训练 provenance。
