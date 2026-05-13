"""
Step 6: FlashAttention 反向传播实现

注意: 这是一个简化版实现, 用于理解算法
完整实现参考 Triton 官方教程

运行方式: python flash_attention_backward.py
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# PyTorch 参考: 标准 Attention 反向传播
# ============================================================

def attention_backward_torch(dO, Q, K, V):
    """
    标准 Attention 反向传播 (PyTorch 实现)

    用于验证 FlashAttention 反向传播的正确性
    """
    d = Q.size(-1)
    scale = 1.0 / math.sqrt(d)

    # 前向传播 (需要保存中间结果)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    O = torch.matmul(attn, V)

    # 反向传播
    # dV = attn.T @ dO
    dV = torch.matmul(attn.transpose(-2, -1), dO)

    # d(attn) = dO @ V.T
    d_attn = torch.matmul(dO, V.transpose(-2, -1))

    # d(scores) = attn * (d_attn - sum(attn * d_attn))
    # 这是 softmax 的反向传播
    d_scores = attn * (d_attn - (attn * d_attn).sum(dim=-1, keepdim=True))

    # dQ = d_scores @ K
    dQ = torch.matmul(d_scores, K) * scale

    # dK = d_scores.T @ Q
    dK = torch.matmul(d_scores.transpose(-2, -1), Q) * scale

    return dQ, dK, dV, O


# ============================================================
# FlashAttention 反向传播 (朴素实现)
# ============================================================

def flash_attention_backward_naive(dO, Q, K, V, O):
    """
    FlashAttention 反向传播 - 朴素实现

    核心思想: 重新计算注意力分数, 而不是存储

    Args:
        dO: 输出梯度 [N, d]
        Q: Query [N, d]
        K: Key [N, d]
        V: Value [N, d]
        O: 前向传播输出 [N, d]

    Returns:
        dQ, dK, dV: 输入梯度
    """
    N, d = Q.shape
    BLOCK_M = 64
    BLOCK_N = 64

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # 预计算 D = rowsum(dO * O)
    # 这是 softmax 反向传播的优化项
    D = (dO * O).sum(dim=-1)  # [N]

    scale = 1.0 / math.sqrt(d)

    # ========== 核心循环: 遍历 K, V 分块 ==========

    for j in range(0, N, BLOCK_N):
        Kj = K[j:j+BLOCK_N]   # [BLOCK_N, d]
        Vj = V[j:j+BLOCK_N]   # [BLOCK_N, d]

        # 当前 K, V 分块的梯度累加器
        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        # 遍历所有 Q 分块
        for i in range(0, N, BLOCK_M):
            Qi = Q[i:i+BLOCK_M]    # [BLOCK_M, d]
            dOi = dO[i:i+BLOCK_M]  # [BLOCK_M, d]
            Di = D[i:i+BLOCK_M]    # [BLOCK_M]

            # ========== 关键: 重新计算注意力分数 ==========

            # 计算 S = Q @ K.T / sqrt(d)
            Sij = torch.matmul(Qi, Kj.T) * scale  # [BLOCK_M, BLOCK_N]

            # 计算 P = softmax(S)
            # 使用数值稳定的 softmax
            m_ij = Sij.max(dim=-1, keepdim=True).values
            Pij = torch.exp(Sij - m_ij)
            l_ij = Pij.sum(dim=-1, keepdim=True)
            Pij = Pij / l_ij  # 归一化

            # ========== 计算梯度 ==========

            # dV = P.T @ dO
            dVj = dVj + torch.matmul(Pij.T, dOi)

            # dP = dO @ V.T
            dPij = torch.matmul(dOi, Vj.T)  # [BLOCK_M, BLOCK_N]

            # dS = P * (dP - D)
            # 这是 softmax 反向传播的简化形式
            dSij = Pij * (dPij - Di.unsqueeze(-1))  # [BLOCK_M, BLOCK_N]

            # dQ = dS @ K
            dQ[i:i+BLOCK_M] = dQ[i:i+BLOCK_M] + torch.matmul(dSij, Kj) * scale

            # dK = dS.T @ Q
            dKj = dKj + torch.matmul(dSij.T, Qi) * scale

        # 写回当前 K, V 分块的梯度
        dK[j:j+BLOCK_N] = dKj
        dV[j:j+BLOCK_N] = dVj

    return dQ, dK, dV


# ============================================================
# FlashAttention 前向传播 (简化版)
# ============================================================

def flash_attention_forward_naive(Q, K, V):
    """FlashAttention 前向传播 - 朴素实现"""
    N, d = Q.shape
    BLOCK_M = 64
    BLOCK_N = 64

    O = torch.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)

    for i in range(0, N, BLOCK_M):
        Qi = Q[i:i+BLOCK_M]

        # Online Softmax 初始化
        m_i = torch.full((Qi.shape[0],), float('-inf'), device=Qi.device, dtype=Qi.dtype)
        l_i = torch.zeros(Qi.shape[0], device=Qi.device, dtype=Qi.dtype)
        O_i = torch.zeros(Qi.shape[0], d, device=Qi.device, dtype=Qi.dtype)

        for j in range(0, N, BLOCK_N):
            Kj = K[j:j+BLOCK_N]
            Vj = V[j:j+BLOCK_N]

            # 计算注意力分数
            Sij = torch.matmul(Qi, Kj.T) * scale

            # Online Softmax 更新
            m_ij = Sij.max(dim=-1).values
            m_new = torch.maximum(m_i, m_ij)

            alpha = torch.exp(m_i - m_new)

            # Pij = exp(Sij - m_new), 注意不是 exp(Sij - m_ij)!
            Pij = torch.exp(Sij - m_new.unsqueeze(-1))
            l_ij = Pij.sum(dim=-1)
            l_new = alpha * l_i + l_ij

            # 正确公式: O_new = alpha * O_old + Pij @ Vj
            O_i = alpha.unsqueeze(-1) * O_i + torch.matmul(Pij, Vj)

            m_i = m_new
            l_i = l_new

        # 最终归一化
        O[i:i+BLOCK_M] = O_i / l_i.unsqueeze(-1)

    return O


# ============================================================
# 测试
# ============================================================

def test_backward_correctness():
    """测试反向传播正确性"""
    print("=" * 60)
    print("测试 FlashAttention 反向传播正确性")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (128, 64),
        (256, 64),
        (512, 64),
    ]

    for N, d in test_cases:
        Q = torch.randn(N, d, device='cuda', dtype=torch.float32, requires_grad=True)
        K = torch.randn(N, d, device='cuda', dtype=torch.float32, requires_grad=True)
        V = torch.randn(N, d, device='cuda', dtype=torch.float32, requires_grad=True)
        dO = torch.randn(N, d, device='cuda', dtype=torch.float32)

        # PyTorch 参考实现
        dQ_ref, dK_ref, dV_ref, O_ref = attention_backward_torch(dO, Q, K, V)

        # FlashAttention 实现
        O_flash = flash_attention_forward_naive(Q, K, V)
        dQ_flash, dK_flash, dV_flash = flash_attention_backward_naive(dO, Q, K, V, O_flash)

        # 检查正确性
        dQ_diff = (dQ_ref - dQ_flash).abs().max().item()
        dK_diff = (dK_ref - dK_flash).abs().max().item()
        dV_diff = (dV_ref - dV_flash).abs().max().item()
        O_diff = (O_ref - O_flash).abs().max().item()

        threshold = 1e-4
        status = "✓ PASS" if max(dQ_diff, dK_diff, dV_diff, O_diff) < threshold else "✗ FAIL"

        print(f"  N={N:4d}, d={d:2d}:")
        print(f"    O   diff: {O_diff:.2e}")
        print(f"    dQ  diff: {dQ_diff:.2e}")
        print(f"    dK  diff: {dK_diff:.2e}")
        print(f"    dV  diff: {dV_diff:.2e}")
        print(f"    {status}")

    print()


def visualize_backward():
    """可视化反向传播过程"""
    print("=" * 60)
    print("FlashAttention 反向传播可视化")
    print("=" * 60)

    # 小规模示例
    N, d = 8, 4
    torch.manual_seed(42)

    Q = torch.randn(N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(N, d, device='cuda', dtype=torch.float32)
    dO = torch.randn(N, d, device='cuda', dtype=torch.float32)

    # 前向传播
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(Q, K.T) * scale
    attn = F.softmax(scores, dim=-1)
    O = torch.matmul(attn, V)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"attn (注意力矩阵) shape: {attn.shape}")

    # 预计算 D
    D = (dO * O).sum(dim=-1)
    print(f"\nD = rowsum(dO * O) shape: {D.shape}")
    print(f"D values: {D.tolist()}")

    # 计算梯度
    d_attn = torch.matmul(dO, V.T)
    d_scores = attn * (d_attn - D.unsqueeze(-1))

    print(f"\nd_scores shape: {d_scores.shape}")
    print(f"d_scores[0] (第一行): {d_scores[0].tolist()}")

    # 使用分块计算
    print("\n--- 分块计算 dV ---")
    BLOCK_N = 4
    dV_full = torch.zeros_like(V)

    for j in range(0, N, BLOCK_N):
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]
        Pj = attn[:, j:j+BLOCK_N]

        dVj = torch.matmul(Pj.T, dO)
        dV_full[j:j+BLOCK_N] = dVj

        print(f"  Block j={j}: P.T @ dO -> dV[{j}:{j+BLOCK_N}]")

    # 验证
    dV_ref = torch.matmul(attn.T, dO)
    diff = (dV_full - dV_ref).abs().max().item()
    print(f"\n分块计算与完整计算的差异: {diff:.2e}")


def main():
    test_backward_correctness()
    visualize_backward()

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. FlashAttention 反向传播通过重新计算注意力分数")
    print("     避免了存储 O(N²) 的中间结果")
    print("  2. 核心公式:")
    print("     - D = rowsum(dO * O)")
    print("     - dS = P * (dP - D)")
    print("     - dQ = dS @ K, dK = dS.T @ Q, dV = P.T @ dO")
    print("  3. 内存: O(N²) -> O(N)")
    print("=" * 60)


if __name__ == "__main__":
    main()