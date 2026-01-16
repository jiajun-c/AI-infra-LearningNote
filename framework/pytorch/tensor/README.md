# Tensor 操作

## 1. 广播

使用`repeat_interleave`可以在一个选择的维度上进行广播

```python
import torch

x = torch.tensor([[1, 2], 
                  [3, 4]]) # Shape: (2, 2)

# 在 dim=0 (行) 上每行重复 2 次
out = torch.repeat_interleave(x, repeats=2, dim=0)
print(out)
# 输出shape
```

输入[0, 1, 2], repeat也是数组为 [1, 2, 3]，那么输出为[0, 1, 1, 2, 2, 2]
```python
y = torch.arange(0, 3)
outy = torch.repeat_interleave(y, repeats=torch.tensor([1, 2, 3], dtype=torch.long))
print(outy)
```