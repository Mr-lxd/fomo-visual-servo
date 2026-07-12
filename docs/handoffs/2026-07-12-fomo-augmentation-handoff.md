# FOMO 数据增强阶段交接

## 1. 仓库状态

- 仓库名称：`fomo-visual-servo`
- 当前分支：`experiment/fomo-augmentation-v1`
- 当前 HEAD：`af4e69bda4d948967ca0437f7f01dfd647783af8`
- baseline 分支：`codex-fomo-pre-augmentation-baseline`
- baseline 两笔 commit：
  - `a487facebba6e07d9754a2e59a1df41dda3124a9`
  - `af4e69bda4d948967ca0437f7f01dfd647783af8`
- Git 工作区应保持干净。
- 除非用户明确要求，禁止 push。

## 2. 已完成功能

当前 baseline 已包含以下功能：

- YOLOv5 数据读取与单类/多类映射；
- letterbox 预处理与坐标反变换；
- stride=8 centroid heatmap 标签生成；
- MobileNetV2-lite FOMO 模型；
- CPU/CUDA 训练与 CUDA AMP；
- grid metrics；
- centroid postprocess 与 centroid 评价；
- 8 邻域 connected components；
- validation threshold sweep；
- `TargetTracker`；
- 图片和视频推理；
- sequence metrics；
- manual/auto class weights；
- experiment metadata；
- dataset content manifest；
- checkpoint threshold 与 inference threshold 拆分。

模型 forward 只输出 logits。softmax、阈值、连通域、质心坐标变换和目标选择均在模型外执行。

## 3. Locked baseline 协议

配置文件：

`configs/experiments/aug00_none_locked.yaml`

必须保持以下不变量：

- `input_size=192`；
- `output_stride=8`；
- `class_mode=preserve`；
- 7 个 foreground classes；
- `class_weights=[1,4,4,4,4,4,4,4]`；
- focal cross entropy，`gamma=2.0`；
- AdamW，learning rate `0.001`，weight decay `0.0001`；
- StepLR，step size `20`，gamma `0.5`；
- batch size `8`；
- AMP=true，`amp_initial_scale=256.0`；
- seed=`42`；
- epochs=`60`；
- early stopping disabled；
- `checkpoint_threshold=0.5`；
- `inference_threshold=0.5`；
- threshold sweep 只在训练结束后执行；
- augmentation 未启用。

## 4. 数据集身份

dataset content hash：

`0576cb20e7adb94e0d57db4a44ce226cc7ad75cd366259e4366380e5d7e25562`

数据根目录只能通过环境变量 `${FOMO_DATASET_ROOT}` 提供。文档、配置和源码不得写入本机数据集绝对路径。

## 5. Locked baseline 结果

统一 FP32 evaluator 对 `best_centroid_f1.pt` 的结果：

- epoch=`40`；
- centroid precision=`0.3290`；
- centroid recall=`0.1672`；
- centroid F1=`0.2217`；
- grid F1=`0.1364`；
- mean localization error=`43.85 px`；
- median localization error=`32.03 px`；
- mean count bias=`-3.52`；
- count MAE=`4.73`；
- final sweep threshold=`0.5`。

per-class centroid F1：

- fish=`0.2528`；
- jellyfish=`0.3273`；
- penguin=`0.1583`；
- puffin=`0.1270`；
- shark=`0.1188`；
- starfish=`0.0606`；
- stingray=`0.0000`。

## 6. 评价规则

- 训练 checkpoint selection 使用 AMP validation；
- checkpoint threshold 固定为 `0.5`；
- 最终统一报告使用 FP32 evaluator；
- threshold sweep 不参与 epoch checkpoint selection；
- AMP 与 FP32 centroid F1 存在约 `0.00275` 差异；
- 数据增强实验阶段暂不改变该评价协议。

因此，训练日志中的 checkpoint 选择指标和最终 FP32 复评结果必须标注各自的评价路径，不得混为同一个数值。

## 7. 下一阶段实验顺序

1. augmentation framework，全部 disabled，不训练；
2. `aug01_color`，只加入 color jitter；
3. `aug02_color_hflip`，再加入 horizontal flip；
4. `aug03_degrade`，再加入 blur/noise；
5. `aug04_affine`，再加入 mild scale/translation。

当前阶段禁止加入：

- Mosaic；
- MixUp；
- CutMix；
- random crop；
- vertical flip；
- 大角度 rotation。

## 8. 下一项任务

下一项任务仅为建立可配置 augmentation framework，但所有增强保持 disabled。

验收条件：

- augmentation 只作用于 train split；
- validation/test 永远不增强；
- `enabled=false` 时与原始 dataset 输出逐元素一致；
- 几何增强必须发生在 letterbox 和 heatmap 生成之前；
- 随机性受现有 seed 与 worker seed 控制；
- 增加可视化脚本；
- 不执行正式训练。

## 9. 禁止同时修改的变量

数据增强框架阶段不得同时修改：

- backbone；
- model width；
- input size；
- stride；
- 标签生成；
- class weights；
- loss；
- optimizer；
- scheduler；
- batch size；
- epochs；
- seed；
- threshold protocol；
- centroid matching；
- checkpoint selection；
- AMP protocol。

## 10. 重要文件索引

- `AGENTS.md`：项目边界、张量约定、部署约束和质量门槛；
- `configs/experiments/aug00_none_locked.yaml`：locked baseline 的完整配置；
- `src/fomo_servo/config.py`：YAML 解析、校验和 threshold 语义；
- `src/fomo_servo/datasets/yolo.py`：YOLOv5 数据集、letterbox 和 heatmap 标签；
- `src/fomo_servo/datasets/collate.py`：训练 batch 组装；
- `src/fomo_servo/training/engine.py`：训练循环、固定阈值 checkpoint selection 和 history；
- `src/fomo_servo/evaluation/validation.py`：统一 validation evaluator 与 threshold sweep；
- `src/fomo_servo/postprocess/`：softmax、阈值、connected components 和 Detection；
- `src/fomo_servo/experiments.py`：Git、dataset manifest 和 experiment metadata；
- `outputs/experiments/aug00_none_locked/training_summary.json`：locked 训练摘要；
- `outputs/experiments/locked_baseline_comparison.csv`：checkpoint 横向评价表。

`outputs/` 文件不进入 Git，但可在本机用于核对训练和评价结果。

## 11. 测试基线

当前测试基线：

`109 passed, 2 skipped`

跳过项目：

- `onnx`；
- `onnxruntime`。

跳过原因是当前环境未安装对应依赖。数据增强框架必须在不依赖真实大型数据集的前提下继续通过现有 pytest 基线。

## 12. 新会话启动检查

新会话必须首先：

1. 阅读 `AGENTS.md`；
2. 阅读本交接文档；
3. 确认当前分支；
4. 确认当前 HEAD；
5. 确认工作区干净；
6. 概述将保持不变的实验变量；
7. 只执行当前指定阶段。

在用户明确要求之前，不得开始 augmentation 实现、训练或 push。
