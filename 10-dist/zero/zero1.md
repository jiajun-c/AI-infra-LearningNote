# zero1

## 1. ZeRO-1 是什么

首先说明一下其基准

- 模型权重：6bytes（`fp16 weight 2B` + `fp32 master weight 4B`）
- 梯度：2bytes（`fp16 .grad`）
- 优化器状态：8bytes（`fp32 m 4B` + `fp32 v 4B`，Adam）
- 合计：16 bytes / param（ZeRO 论文里那个经典数字）

然后zero1切分了优化器的状态，只需要每次用的时候去 allGather一下

所以在训练中，对于一个参数，需要存储16B的数据

## 2. 实现模式

### 2.1 前向

在前向的过程中，不需要进行通信，都是在这个卡上

### 2.2 反向

在反向的过程中，需要先做一次reduce-scatter去同步梯度，因为每个卡上只保存了部分的优化器状态，同时在使用优化器去参数进行更新的时候，仅需要All-gather一次就可以获得到完整的优化器状态。

所以完整的通信量为 $\frac{2 S *(n-1)}{n}(前向reduceScatter 模型的梯度）+ \frac{2S*(n-1)}{n}(AllGather权重）$

### 2.3 对模型参数的影响

变成了12/N + 4
