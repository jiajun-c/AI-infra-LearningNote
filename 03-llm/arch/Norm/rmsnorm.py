import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor):
        # 确定归一化的维度：最后 len(normalized_shape) 个维度
        norm_dims = tuple(range(-len(self.normalized_shape), 0))
        # 计算 RMS：sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=norm_dims, keepdim=True) + self.eps)
        # 归一化
        x_normed = x / rms
        if self.elementwise_affine:
            x_normed = x_normed * self.weight

        return x_normed
    
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor):
        # 确定归一化的维度：最后 len(normalized_shape) 个维度
        norm_dims = tuple(range(-len(self.normalized_shape), 0))
        rms = torch.sqrt(x.pow(2).mean(dim=norm_dims, keepdim=True) + self.eps)
        x_normed = x / rms
        if self.elementwise_affine:
            x_normed = x_normed * self.weight
        return x_normed

# ----------------------------
# 测试函数
# ----------------------------
def test_rmsnorm():
    torch.manual_seed(42)
    
    # 测试用例：(input_shape, normalized_shape)
    test_cases = [
        ((2, 3), 3),
        ((4, 5, 6), 6),
        ((2, 8, 10, 12), (10, 12)),
        ((1, 768), 768),
    ]
    
    for input_shape, norm_shape in test_cases:
        print(f"\nTesting input shape {input_shape} with normalized_shape {norm_shape}")
        
        # 创建模块
        rmsnorm = RMSNorm(norm_shape, eps=1e-6, elementwise_affine=True)
        x = torch.randn(input_shape, requires_grad=True)
        
        # 前向
        y = rmsnorm(x)
        
        # 手动计算验证
        if isinstance(norm_shape, int):
            norm_dims = (-1,)
        else:
            norm_dims = tuple(range(-len(norm_shape), 0))
        rms_manual = torch.sqrt(x.pow(2).mean(dim=norm_dims, keepdim=True) + 1e-6)
        y_manual = x / rms_manual * rmsnorm.weight
        
        # 检查数值一致性
        assert torch.allclose(y, y_manual, atol=1e-6), "Forward pass mismatch!"
        print("✅ Forward pass matches manual computation.")
        
        # 检查输出是否真的被归一化（RMS ≈ 1，忽略 weight）
        y_unscaled = y / rmsnorm.weight  # 移除 weight 影响
        rms_out = torch.sqrt(y_unscaled.pow(2).mean(dim=norm_dims, keepdim=True))
        assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-5), "Output RMS not ~1!"
        print("✅ Output RMS ≈ 1 (after removing weight).")
        
        # 梯度检查
        loss = y.sum()
        loss.backward()
        assert x.grad is not None and x.grad.shape == x.shape, "Gradient not computed correctly!"
        if rmsnorm.weight is not None:
            assert rmsnorm.weight.grad is not None, "Weight gradient missing!"
        print("✅ Gradients computed successfully.")

    print("\n🎉 All tests passed!")

# ----------------------------
# 运行测试
# ----------------------------
if __name__ == "__main__":
    test_rmsnorm()