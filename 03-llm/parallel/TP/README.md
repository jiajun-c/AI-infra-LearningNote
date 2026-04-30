# 张量并行

## 1. 朴素张量并行/模型并行

在模型并行中，会将模型拆分到不同的节点上，例如对于一个6层的网络而言，在两个设备上进行张量并行的时候，每个设备只拥有三层。如果一个模型可以被放入到一张卡中，那么其速度将会比模型并行更快


## 2. 张量并行

张量并行是一种更细粒度的模型并行方法，其中分为行并行和列并行，行并行是在行的维度进行切分，列并行是在列维度进行切分。

如下所示，使用`init_device_mesh` 调用8个设备进行计算，针对 `w1` 层进行使用列并行，针对`w2` 层进行使用行并行，通常对于矩阵乘法这么处理较为简单。

```python3
from torch.distributed.tensor.parallel import  parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh

m = Model(...)
tp_mesh = init_device_mesh("cuda", (8,))
m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
```

## 3. 序列并行

序列并行是张量并行的一种变种，它在序列维度上对`nn.LayerNorm`或者`nn.RMSNorm` 进行分割

## 4. 并行损失函数计算

在张量并行中，损失函数的计算也需要进行并行化处理。

### 4.1 Cross Entropy Loss 并行化

对于 Cross Entropy Loss，当词汇表被切分到不同的 GPU 上时，需要收集所有 GPU 上的 logits 来计算全局的 softmax 和 loss。

```python
# Megatron-LM 风格的实现
vocab_start_index = rank * vocab_chunk_size
vocab_end_index = vocab_start_index + vocab_chunk_size

# 获取当前 GPU 的 vocab 范围
target_mask = (target >= vocab_start_index) & (target < vocab_end_index)
local_target = target - vocab_start_index

# 计算本地 loss，然后通过 all-reduce 聚合
local_loss = cross_entropy(local_logits, local_target)
global_loss = all_reduce(local_loss, op='sum')
```

### 4.2 梯度聚合

在反向传播时，来自不同设备的梯度需要进行聚合：

```python
# All-reduce 聚合梯度
for param in model.parameters():
    if param.is_tensor_parallel:
        all_reduce(param.grad, op='sum')
```

## 5. 3D 并行

将数据并行、张量并行和流水线并行结合起来：

- **数据并行**: 复制模型，分发数据
- **张量并行**: 切分单层参数
- **流水线并行**: 切分网络层

```
数据并行度 = DP
张量并行度 = TP
流水线并行度 = PP

总 GPU 数 = DP × TP × PP
```
