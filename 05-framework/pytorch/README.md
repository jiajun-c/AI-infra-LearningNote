# PyTorch 框架

PyTorch 是一个开源的深度学习框架，提供了灵活的张量计算和神经网络构建能力。

## 目录

- [Tensor 操作](./tensor/)
- [计算图](./graph/)
- [分布式训练](./dist/)
- [梯度机制](./grad/)
- [优化器](./optimizer/)
- [装饰器](./decorator/)

## 1. 核心特性

### 1.1 动态计算图

PyTorch 使用动态计算图（Define-by-Run），允许在运行时构建计算图。

### 1.2 自动微分

通过 `autograd` 模块实现自动微分。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()

print(x.grad)  # [2., 4., 6.]
```

## 2. 常用模块

| 模块 | 功能 |
|------|------|
| `torch.nn` | 神经网络层 |
| `torch.optim` | 优化器 |
| `torch.utils.data` | 数据加载 |
| `torch.distributed` | 分布式训练 |

## 3. 性能优化

- CUDA Graph
- 混合精度训练 (AMP)
- torch.compile (PyTorch 2.0+)
