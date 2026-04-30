# 数据并行

数据并行针对的场景是单机多卡，其核心思想是将大规模的数据集划分为若干个小的数据集，并使用多个计算卡对分别进行计算。

- 前向传播：将min-batch数据平均分配到每个GPU上，并将模型和优化器复制到每个NPU上，保证各个GPU的x 的模型和优化器是完全相同的
- 损失计算和反向传播：前向传播完成后，每个GPU计算模型损失并进行反向传播，得到梯度后，将梯度传递到某个GPU上进行累加，更新模型的参数和优化器状态

在pytorch中可以通过`nn.DataParallel`来实现模型的DP

缺点：当模型的
```python3
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
 
    def forward(self, x):
        x = self.flatten(x)
        # print(f"shape: {x.size()}")
        logits = self.linear_relu_stack(x)
        return logits
 
model = NeuralNetwork().to(device)
model = nn.DataParallel(model)
```
