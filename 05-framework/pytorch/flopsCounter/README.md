# FLOPs 计数

## FlopCounterMode

PyTorch 内置的 `FlopCounterMode` 可以统计模型的浮点运算量（FLOPs）。

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode(display=False, depth=None)
with flop_counter:
    model(inp)
total_flops = flop_counter.get_total_flops()
```

封装成工具函数：

```python
def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    istrain = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops
```

## FLOPs 单位

| 单位 | 数量级 |
|------|--------|
| MFLOPs | $10^6$ |
| GFLOPs | $10^9$ |
| TFLOPs | $10^{12}$ |
| PFLOPs | $10^{15}$ |

> **FLOPs**（计算量）vs **FLOPS**（算力，每秒浮点数）：两者相除得到理论执行时间。

## 实测：ResNet18

```
forward:           3,628,146,688  ≈  3.63 GFLOPs
forward+backward: 10,648,412,160  ≈ 10.65 GFLOPs
```

backward 约为 forward 的 **2.93x**，符合理论值（反向传播 ≈ 前向的 2x，合计 3x）。
