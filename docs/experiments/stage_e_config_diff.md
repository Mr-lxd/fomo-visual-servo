# Stage E 配置差异

两份 Stage E YAML 均以 `stage_d2_fomo_ei_w100_pretrained.yaml` 的数据、192 输入、stride 8、增强、EI weighted cross entropy、权重 1/100、AdamW、StepLR、batch 8、60 epochs、seed 42、AMP、snapshot、validation PR-AUC 选择和 threshold grid 为固定协议。

允许变化仅为 project/experiment 名称、output 目录、`model.backbone` 与各自官方本地 ImageNet 权重 provenance。两份新配置的权重文件位于已忽略缓存中，并由环境变量指定；它们不是 D2 H5 的替代或再分发。

这是工程候选比较，不是纯 backbone 单变量因果消融：D2 使用 EI/Keras MobileNetV2 H5，新候选使用 torchvision 官方 ImageNet 权重。
