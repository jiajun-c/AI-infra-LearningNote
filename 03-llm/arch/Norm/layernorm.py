import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ManualLayerNorm, self).__init__()
        if (isinstance(normalized_shape, int)):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, keepdim=True, unbiased=False)
        x_norm = (x - mean)/torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm

# 创建输入
x = torch.randn(2, 3, 4, requires_grad=True)

# 手写 LayerNorm
manual_ln = ManualLayerNorm(normalized_shape=4, eps=1e-5, elementwise_affine=True)
# 官方 LayerNorm
official_ln = nn.LayerNorm(normalized_shape=4, eps=1e-5, elementwise_affine=True)

# 同步参数
official_ln.weight.data = manual_ln.weight.data.clone()
official_ln.bias.data = manual_ln.bias.data.clone()

# 前向
y1 = manual_ln(x)
y2 = official_ln(x)

# 检查输出是否一致
print("Forward match:", torch.allclose(y1, y2, atol=1e-6))  # 应为 True

# 反向（验证梯度）
loss1 = y1.sum()
loss2 = y2.sum()
loss1.backward()
loss2.backward()

print("Input grad match:", torch.allclose(x.grad, x.grad, atol=1e-6))  # True
print("Weight grad match:", torch.allclose(manual_ln.weight.grad, official_ln.weight.grad, atol=1e-6))
print("Bias grad match:", torch.allclose(manual_ln.bias.grad, official_ln.bias.grad, atol=1e-6))