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

## tensor version

当我们通过torch.view 来获取一个tensor的视图时，其实着两个指向的是同一片地址空间，对一个进行修改另外一个就要进行修改

当对一个tensor进行了修改之后，其version会进行自增，同时顺藤摸瓜将其他关联的视图也全部进行了更新

在反向求导的场景下，该版本管理可以帮助我们追踪修改，防止报错，在推理场景下可以关闭


## nn.Parameter

nn.Parameter 是 Tensor中的一个子类，其用于存储需要进行更新和训练的权重，当把一个`nn.Parameter` 赋值到`nn.Moduele`的属性中，其会把它自动添加到Module的参数列表中

如下所示仅会输出 `weight` 这么一个参数变量

```python
import torch
import torch.nn as nn

class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = torch.empty(10, 20)
        initial_data = torch.randn(10, 20)  # 第一步: 先创建一个普通的 n*m Tensor
        self.weight = torch.nn.Parameter(initial_data)

d = Demo()

for name, param in d.named_parameters():
    print(name) 
# 输出: my_param
# (注意：my_tensor 不在这里！)

# 检查梯度
print(f"Parameter requires_grad: {d.weight.requires_grad}") # True
print(f"Tensor requires_grad: {d.data.requires_grad}")   # False
```