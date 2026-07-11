# Roboflow YOLO 数据布局适配设计

## 已确认的数据集

真实数据集目录由 `FOMO_DATASET_ROOT` 指定，目录结构为 Roboflow YOLO 导出格式：

```text
aquarium_pretrain/
  data.yaml
  train/images/  train/labels/
  valid/images/  valid/labels/
  test/images/   test/labels/
```

`data.yaml` 中类别顺序为 fish、jellyfish、penguin、puffin、shark、starfish、stingray。已观测 train 448、valid 127、test 63 张图，且每个 split 有同数标签文件。

## 决策

原始数据保持原地不动。数据集 loader 继续优先支持现有项目布局 `images/<split>`、`labels/<split>`，同时增加 Roboflow 布局 `<split>/images`、`<split>/labels`。当逻辑 split 为 `val` 时，会额外尝试 Roboflow 常用目录名 `valid`。

不依赖 `data.yaml` 中 Roboflow 相对 train/val 路径；这些路径在当前文件夹层级下为 `../train/images`，无法可靠地作为项目路径语义。类别 `names` 仍从同一 `data.yaml` 读取。

## 配置

新增 `configs/aquarium_pretrain_192.yaml`，数据根目录写入用户确认的实际位置，使用：

```yaml
dataset:
  root: "${FOMO_DATASET_ROOT}"
  train_split: train
  validation_split: val
  classes: [fish, jellyfish, penguin, puffin, shark, starfish, stingray]
  class_mode: preserve
```

模型输出为 8 通道：background 加 7 个前景类。loss class weights 必须为 8 个数。

## 验证

新增 pytest 使用临时 Roboflow 格式目录和已有小 JPEG fixture，验证 `split="train"` 解析 `train/images`，`split="val"` 解析 `valid/images`，并生成预期 class-index heatmap。真实数据只做只读布局检查，不写入或修改。

## 自检

- 原始数据路径仅出现在新增用户配置，不写入 Python 源码。
- 保持项目布局与既有测试兼容。
- 多类配置的类别顺序与真实 data.yaml 完全一致。
