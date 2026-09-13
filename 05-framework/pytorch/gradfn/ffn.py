import argparse
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x: torch.Tensor) -> torch.Tensor:
    """精确版 GELU: 0.5 * x * (1 + erf(x / sqrt(2)))，与 F.gelu 默认实现一致。"""
    return 0.5 * x * (1.0 + torch.erf(x * math.sqrt(0.5)))


def gelu_grad(x: torch.Tensor) -> torch.Tensor:
    """GELU 的导数: Φ(x) + x * φ(x)，其中 Φ 是标准正态 CDF、φ 是 PDF。"""
    cdf = 0.5 * (1.0 + torch.erf(x * math.sqrt(0.5)))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf

def _flat(t: torch.Tensor) -> torch.Tensor:
    """[..., F] -> [N, F]。

    参数的梯度要对所有前导维度求和，而 x 可能是 1-D（单样本 [I]）/ 2-D（[B, I]）/
    3-D（[B, T, I]）。统一拍平成 2D 再算，反向公式就不必分情况讨论。

    这同时也是个真会踩的坑：若直接写 grad_y.sum(dim=0)，1-D 输入下 grad_y 形状是
    [O]，求和会退化成标量，b2 的梯度形状就错了；`t.mT` 对 1-D 张量更是直接报错
    （tensor.mT is only supported on matrices）。
    """
    return t.reshape(-1, t.shape[-1])

class FFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        pre = x @ w1 + b1  # [..., H]   线性层 1
        h = gelu(pre)  # [..., H]   激活
        y = h @ w2 + b2  # [..., O]   线性层 2
        # 只保存反向真正需要的东西。b1/b2 的梯度是求和，无需保存。
        ctx.save_for_backward(x, w1, pre, h, w2)
        return y
        pass

    @staticmethod
    def backward(ctx, grad_y):
        x, w1, pre, h, w2 = ctx.saved_tensors

        # ---- 线性层 2 的反向 ----
        grad_w2 = _flat(h).mT @ _flat(grad_y)  # [H, O]
        grad_b2 = _flat(grad_y).sum(dim=0)  # [O]

        # ---- 激活的反向 ----
        grad_h = grad_y @ w2.mT  # [..., H]
        grad_pre = grad_h * gelu_grad(pre)  # [..., H]  逐元素乘

        # ---- 线性层 1 的反向 ----
        flat_grad_pre = _flat(grad_pre)
        grad_w1 = _flat(x).mT @ flat_grad_pre  # [I, H]
        grad_b1 = flat_grad_pre.sum(dim=0)  # [H]
        grad_x = grad_pre @ w1.mT  # [..., I]

        # 顺序、个数必须与 forward 的输入 (x, w1, b1, w2, b2) 一一对应
        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2
    
