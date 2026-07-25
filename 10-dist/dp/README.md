# 数据并行 DP

Data Parallel 是最朴素的数据并行方式：每张卡持有一份完整模型，把一个 mini-batch 切成多份，分别在多张 GPU 上前向和反向，最后把梯度汇总后更新参数。

## 1. 核心思想

```text
同一个模型复制到多张 GPU
不同 GPU 处理不同数据 shard
梯度汇总到主卡
主卡更新参数
再把新参数同步到其他卡
```

一次训练 step 可以理解成：

```text
mini-batch
  -> scatter 到多张 GPU
  -> 每张 GPU 独立 forward/backward
  -> gather gradients 到主 GPU
  -> 主 GPU 做 optimizer step
  -> broadcast updated parameters
```

## 2. PyTorch DataParallel

PyTorch 中可以通过 `nn.DataParallel` 使用单进程多线程的数据并行：

```python
import torch
import torch.nn as nn

model = NeuralNetwork().to("cuda:0")
model = nn.DataParallel(model)
```

`DataParallel` 会在一次 forward 中做几件事：

```text
1. replicate: 把模型复制到多张 GPU
2. scatter: 把输入 batch 切分到多张 GPU
3. parallel_apply: 多张 GPU 并行执行 forward
4. gather: 把输出收集到主 GPU
5. backward: 梯度回传并汇总到主 GPU
```

## 3. 通信模式

DP 的通信更偏中心化：

```text
主 GPU 负责参数源和梯度汇总
其他 GPU 与主 GPU 交换数据/梯度
```

这会导致主 GPU 压力更大：

```text
主 GPU 显存占用更高
主 GPU 通信更重
多卡扩展性较差
```

## 4. 局限

`nn.DataParallel` 一般只适合简单的单机多卡实验，不适合作为大模型训练主力方案。

主要问题：

- 单进程多线程，Python 侧容易受到 GIL 和调度开销影响。
- 主 GPU 负载更重，容易成为瓶颈。
- 每次 forward 都涉及模型复制和输入分发。
- 不适合多机训练。
- 扩展性明显弱于 DDP。

## 5. DP 和 DDP 的区别

| 维度 | DataParallel | DistributedDataParallel |
| --- | --- | --- |
| 进程模型 | 单进程多线程 | 多进程 |
| 适用范围 | 单机多卡 | 单机多卡 / 多机多卡 |
| 梯度同步 | 汇总到主 GPU | 各进程 AllReduce |
| 通信形态 | 中心化 | 去中心化 |
| 扩展性 | 较差 | 更好 |
| 推荐程度 | 简单实验 | 训练首选 |

## 6. 小结

DP 的价值在于帮助理解数据并行的基本概念：

```text
切数据，不切模型
每张卡都有完整模型
每张卡算局部梯度
最终需要得到全局等价梯度
```

但是实际训练中，尤其是 LLM pre-training，应优先使用 DDP、FSDP 或 ZeRO 这类更可扩展的方案。
