# 分布式训练

这个目录整理大模型训练中的分布式并行策略。建议按下面顺序学习：

```text
DP -> DDP -> FSDP / ZeRO -> TP -> PP -> EP -> Hybrid Parallel
```

## 学习入口

| 主题 | 说明 |
| --- | --- |
| [DP](./dp/README.md) | 单机多卡数据并行，理解“切数据、不切模型”的基本思想 |
| [DDP](./DDP/README.md) | 多进程数据并行，通过 AllReduce 同步梯度 |
| [FSDP](./fsdp/README.md) | 参数、梯度、优化器状态分片，降低单卡显存压力 |
| [ZeRO](./zero/README.md) | 训练状态分片：从 ZeRO-1 到 ZeRO-3 理解参数、梯度、优化器状态如何切分 |
| [分布式转置](./trans/README.md) | 从矩阵转置理解 All-to-All / All-Gather 等通信形态 |

## 主线问题

学习每种并行策略时，都可以问三个问题：

```text
1. 切了什么？
2. 通信发生在哪里？
3. 省了什么资源，又引入了什么代价？
```

例如：

```text
DP / DDP: 切数据，通信梯度
FSDP / ZeRO: 切参数、梯度、优化器状态，通信参数和梯度分片
TP: 切矩阵乘法，通信 partial result 或 activation
PP: 切层，通信 micro-batch activation
EP: 切 expert，通信 token dispatch / combine
```
