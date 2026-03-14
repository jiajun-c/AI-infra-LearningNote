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