# 量化感知训练(QAT)

在模型输入前加上QuantStub()，在模型输出后加上DequantStub()。目的是将输入从float32量化为int8，将输出从int8反量化为float32。

## 1. 插入节点

这部分较为简单，在进行forward的时候，先对其进行量化，最后进行反量化即可。
```python3
class net(nn.Module):
    def __init__(self):
        self.backbone = mobilenet()

     def forward(self, x):
        x = self.backbone(x) 
        return x

# 修改后
from torch.quantization import QuantStub, DeQuantStub
class Q_net(nn.Module):
    def __init__(self):
        self.backbone = mobilenet()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

     def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x) 
        x = self.dequant(x)
        return x
```


## 2. 伪量化

对加法等操作加上伪量化节点，因为在进行部分运算的时候，容易出现溢出。所以需要进行量化->计算->反量化的过程。
其中实际的运算仍然使用原有的数据类型进行计算。

```python3
class Q_ResBlock(nn.Module):
    def __init__(self):
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x, residual):
        return self.skip_add.add(x, residual)  # 伪量化加法
```

对于其他的如ReLU，将其替换为ReLU6，将大于6的阈值进行截断，在保证了一定精度的同时降低了计算量

