# 分布式数据并行 DDP

Distributed Data Parallel 是 PyTorch 中更推荐的数据并行方式。它用多进程代替 `nn.DataParallel` 的单进程多线程，并通过 collective communication 同步梯度，可以同时支持单机多卡和多机多卡。

## 1. DDP 解决什么问题

DP 的瓶颈在于主 GPU 需要承担额外的 scatter/gather 和梯度汇总压力。DDP 的核心改进是：

```text
每张 GPU 一个独立进程
每个进程持有完整模型副本
每个进程处理不同数据 shard
反向传播时通过 AllReduce 同步梯度
每个进程本地执行 optimizer step
```

只要所有进程初始参数相同、每一步使用同步后的全局梯度更新，那么每个进程上的模型参数就会保持一致。

## 2. 训练流程

一次 DDP 训练 step：

```text
1. 每个 rank 读取不同 mini-batch shard
2. 每个 rank 独立 forward
3. 每个 rank 独立计算 loss
4. backward 产生本地梯度
5. DDP 对梯度 bucket 触发 AllReduce
6. 每个 rank 得到平均后的全局梯度
7. 每个 rank 执行 optimizer.step()
```

注意：DDP 并不是把梯度发到 rank 0 再更新，而是所有 rank 通过 AllReduce 得到相同梯度，然后各自更新本地模型。

## 3. 初始化与 rank

DDP 的行为和 MPI 类似，每个进程都有自己的 rank：

```text
rank: 当前进程的全局编号
world_size: 总进程数
local_rank: 当前节点内的 GPU 编号
rank 0: 通常负责日志、checkpoint、主控逻辑
```

典型初始化：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

model = Model().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## 4. 参数与 buffer 同步

DDP 要求各个 rank 的模型参数在训练开始时一致。

构建 DDP 时，会把 rank 0 的 module states 广播到其他 rank：

```text
parameters
buffers
```

训练中：

- parameters 通过同步梯度后各自执行 optimizer step 来保持一致。
- buffers 例如 BatchNorm running mean/var，可以按配置在 forward 前同步。

## 5. 梯度同步与通信重叠

DDP 的关键优化是梯度 bucket 和计算通信重叠。

反向传播不是等所有梯度都算完才通信，而是：

```text
某一组参数的梯度 ready
-> 放入 bucket
-> bucket 满或 ready
-> 立即启动 AllReduce
-> 同时继续计算更早层的 backward
```

这样可以把一部分通信时间藏在 backward 计算后面。

如果把 backward 看成从最后一层往前算：

```text
Layer N gradient ready -> AllReduce bucket 1
Layer N-1 backward still running
Layer N-2 backward still running
...
```

这就是 DDP 比朴素数据并行扩展性更好的原因之一。

## 6. 通信模式：AllReduce

DDP 默认通过 AllReduce 得到全局平均梯度。

假设有 4 个 rank，每个 rank 得到局部梯度：

```text
g0, g1, g2, g3
```

AllReduce 后，每个 rank 都得到：

```text
g = (g0 + g1 + g2 + g3) / 4
```

然后每个 rank 都执行同样的参数更新：

```text
w = w - lr * g
```

因此所有 rank 的参数保持一致。

## 7. DDP 和 DP 的区别

| 维度 | DP | DDP |
| --- | --- | --- |
| 进程模型 | 单进程多线程 | 多进程 |
| 设备范围 | 通常单机多卡 | 单机多卡 / 多机多卡 |
| 梯度同步 | 主 GPU 汇总 | AllReduce |
| 通信瓶颈 | 主 GPU 容易成为瓶颈 | 通信更均衡 |
| 性能 | 较弱 | 更好 |
| PyTorch 推荐 | 不推荐大规模训练 | 推荐 |

## 8. DDP 的局限

DDP 只切分数据，不切分模型：

```text
每张 GPU 都必须放得下完整模型参数、梯度和优化器状态
```

因此当模型变大时，DDP 会遇到显存瓶颈。LLM pre-training 通常会继续引入：

```text
ZeRO / FSDP: 切参数、梯度、优化器状态
Tensor Parallel: 切矩阵乘法
Pipeline Parallel: 切层
Sequence Parallel: 切序列维度
```

## 9. 小结

DDP 是理解大模型分布式训练的第一块地基：

```text
切数据，不切模型
多进程，每进程一张 GPU
反向传播时 AllReduce 梯度
每个 rank 本地更新模型
通过 bucket 实现计算通信重叠
```

后续 FSDP、ZeRO、TP、PP 等方法，本质上都是在 DDP 的基础问题上继续回答：

```text
如果完整模型、梯度、优化器状态放不下一张卡，应该怎么切？
```
