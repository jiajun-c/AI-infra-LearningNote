# 大模型训练

这里按一次训练的时间顺序组织阅读；并行策略的机制统一从 [分布式训练](../10-dist/README.md) 进入。

1. **目标与预算**：[预训练](./pre-training/README.md) → [Scaling Law](./scalingLaw/README.md) → [训练数据](../03-llm/train/dataset/README.md)。
2. **单步训练**：[训练 step](./pretrain-step/README.md) → [交叉熵损失](./loss/cross-entropy-loss/README.md) → [梯度累加](./gradAccStep/README.md) → [Checkpoint](./checkpoint/README.md)。
3. **多卡训练**：[并行概览](./parallel/README.md) → [DP](../10-dist/dp/README.md) → [DDP](../10-dist/DDP/README.md) → [FSDP / ZeRO](../10-dist/README.md)。
4. **后训练**：[SFT](./post-training/SFT/README.md) → [对齐概览](./post-training/Alignment/README.md) → [DPO](./post-training/Alignment/DPO/README.md) / [RLHF](./post-training/Alignment/RLHF/README.md)。

专题：[采样](./sample/README.md)、[梯度 Bucket](./gradBucket/READEME.md)、[模型优化器](../03-llm/optimizer/)。
