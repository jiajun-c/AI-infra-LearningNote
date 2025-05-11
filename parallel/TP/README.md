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

