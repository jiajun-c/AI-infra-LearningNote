# Tensor 操作

## 1. 广播

使用`repeat_interleave`可以在一个选择的维度上进行广播

```cpp
import torch

x = torch.tensor([[1, 2], 
                  [3, 4]]) # Shape: (2, 2)

# 在 dim=0 (行) 上每行重复 2 次
out = torch.repeat_interleave(x, repeats=2, dim=0)
print(out)
```