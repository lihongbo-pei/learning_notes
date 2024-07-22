# Vision Transformer (ViT)

## 附录

训练策略

批量大小：batch size = 4096

学习率预热：在训练的前 10,000 步中使用学习率预热。

梯度剪裁：在训练 ImageNet 数据集时，应用全局范数为 1 的梯度剪裁。

训练分辨率：224