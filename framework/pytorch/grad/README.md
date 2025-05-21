# 梯度/导数相关

## 手动求导

在pytorch中，有些操作是无法在计算图上进行自动的微分操作，如返回离散值的函数，ReLU这种条件函数，以及一些无法进行求导的函数。在此时我们可以继承 `torch.autograd.Function` 类，并实现 `forward` 和 `backward` 方法，以手动实现导数。

如下所示，Exp类继承`torch.autograd.Function`类，然后其中实现了一个`forward`和`backward`方法。同时两个函数需要传递进一个ctx参数，用于保存`forward`的结果。

```python
import torch

class Exp(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, _ = ctx.saved_tensors
        return grad_output * result
    
input = torch.tensor([1.0, 2.0, 3.0])
print(Exp.apply(input))
```
