"""
Step 4: Online Softmax 验证

运行方式: python online_softmax.py
"""

import torch
import torch.nn.functional as F
import math


def standard_attention(q, k, v):
    """标准 Attention 作为参考"""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def online_softmax_attention_naive(Q, K, V, BLOCK_N=64):
    """
    Online Softmax Attention - 朴素实现

    Q: [BLOCK_M, d] - 只处理一个 Q block
    K: [N, d]
    V: [N, d]

    这个实现用于理解算法，实际 Triton 实现见 Step 5
    """
    BLOCK_M, d = Q.shape
    N = K.shape[0]

    # 初始化状态
    m_i = torch.full((BLOCK_M,), float('-inf'), device=Q.device, dtype=Q.dtype)
    l_i = torch.zeros(BLOCK_M, device=Q.device, dtype=Q.dtype)
    O_i = torch.zeros(BLOCK_M, d, device=Q.device, dtype=Q.dtype)

    scale = 1.0 / math.sqrt(d)

    # 遍历 K, V 分块
    for j in range(0, N, BLOCK_N):
        # 当前 K, V 块
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]

        # 计算注意力分数
        Sj = torch.matmul(Q, Kj.T) * scale  # [BLOCK_M, BLOCK_N]

        # 当前块的 rowmax
        m_ij = Sj.max(dim=-1).values  # [BLOCK_M]

        # 新的 rowmax
        m_i_new = torch.maximum(m_i, m_ij)

        # 重新缩放的因子
        # 关键: P 必须用 m_i_new 计算，不是 m_ij!
        p_i = torch.exp(Sj - m_i_new.unsqueeze(-1))  # [BLOCK_M, BLOCK_N]

        # 更新 l
        l_i_new = torch.exp(m_i - m_i_new) * l_i + p_i.sum(dim=-1)

        # 更新 O
        # O_new = (exp(m_old - m_new) * O_old * l_old + p @ V) / l_new
        O_i = (torch.exp(m_i - m_i_new).unsqueeze(-1) * O_i * l_i.unsqueeze(-1) + \
               torch.matmul(p_i, Vj)) / l_i_new.unsqueeze(-1)

        # 更新状态
        m_i = m_i_new
        l_i = l_i_new

    return O_i


def online_softmax_attention_v2(Q, K, V, BLOCK_N=64):
    """
    Online Softmax - 更清晰的实现

    使用 FlashAttention 论文中的公式
    """
    BLOCK_M, d = Q.shape
    N = K.shape[0]

    # 初始化
    m_i = torch.full((BLOCK_M,), float('-inf'), device=Q.device, dtype=Q.dtype)
    l_i = torch.zeros(BLOCK_M, device=Q.device, dtype=Q.dtype)
    O_i = torch.zeros(BLOCK_M, d, device=Q.device, dtype=Q.dtype)

    scale = 1.0 / math.sqrt(d)

    for j in range(0, N, BLOCK_N):
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]

        # 计算 Q @ Kj.T
        qk = torch.matmul(Q, Kj.T) * scale

        # 当前块的 rowmax
        m_ij = qk.max(dim=-1).values

        # 新的 rowmax
        m_i_new = torch.maximum(m_i, m_ij)

        # 重新缩放因子
        p_i = torch.exp(qk - m_i_new.unsqueeze(-1))  # [M, BLOCK_N]

        # 更新 l
        l_i_new = torch.exp(m_i - m_i_new) * l_i + p_i.sum(dim=-1)

        # 更新 O
        # O_i_new = exp(m_i - m_i_new) * O_i * l_i / l_i_new + (p_i @ Vj) / l_i_new * l_i_new
        # 简化:
        # O_i_new = (exp(m_i - m_i_new) * O_i * l_i + p_i @ Vj) / l_i_new

        O_i_new = (torch.exp(m_i - m_i_new).unsqueeze(-1) * O_i * l_i.unsqueeze(-1) + \
                   torch.matmul(p_i, Vj)) / l_i_new.unsqueeze(-1)

        # 更新状态
        m_i = m_i_new
        l_i = l_i_new
        O_i = O_i_new

    return O_i


def test_correctness():
    """测试 Online Softmax 的正确性"""
    print("=" * 60)
    print("测试 Online Softmax 正确性")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (16, 64, 64),
        (32, 128, 64),
        (64, 256, 64),
        (128, 512, 64),
    ]

    for BLOCK_M, N, d in test_cases:
        Q = torch.randn(BLOCK_M, d, device='cuda', dtype=torch.float32)
        K = torch.randn(N, d, device='cuda', dtype=torch.float32)
        V = torch.randn(N, d, device='cuda', dtype=torch.float32)

        # 标准实现
        ref = standard_attention(Q, K, V)

        # Online Softmax v1
        out_v1 = online_softmax_attention_naive(Q, K, V, BLOCK_N=64)
        diff_v1 = (ref - out_v1).abs().max().item()

        # Online Softmax v2
        out_v2 = online_softmax_attention_v2(Q, K, V, BLOCK_N=64)
        diff_v2 = (ref - out_v2).abs().max().item()

        status_v1 = "✓ PASS" if diff_v1 < 1e-4 else "✗ FAIL"
        status_v2 = "✓ PASS" if diff_v2 < 1e-4 else "✗ FAIL"

        print(f"  Q[{BLOCK_M:3d},{d}] K,V[{N:3d},{d}]:")
        print(f"    v1 max_diff = {diff_v1:.2e} {status_v1}")
        print(f"    v2 max_diff = {diff_v2:.2e} {status_v2}")

    print()


def test_different_block_sizes():
    """测试不同分块大小"""
    print("=" * 60)
    print("测试不同分块大小")
    print("=" * 60)

    torch.manual_seed(42)

    BLOCK_M, N, d = 64, 256, 64
    Q = torch.randn(BLOCK_M, d, device='cuda', dtype=torch.float32)
    K = torch.randn(N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(N, d, device='cuda', dtype=torch.float32)

    ref = standard_attention(Q, K, V)

    for BLOCK_N in [16, 32, 64, 128]:
        out = online_softmax_attention_v2(Q, K, V, BLOCK_N=BLOCK_N)
        diff = (ref - out).abs().max().item()
        status = "✓ PASS" if diff < 1e-4 else "✗ FAIL"
        print(f"  BLOCK_N={BLOCK_N:3d}: max_diff = {diff:.2e} {status}")

    print()


def visualize_algorithm():
    """可视化 Online Softmax 算法过程"""
    print("=" * 60)
    print("Online Softmax 算法可视化")
    print("=" * 60)

    # 小规模示例
    d = 4
    Q = torch.tensor([[1.0, 0.0, 0.5, -0.5]], device='cuda', dtype=torch.float32)
    K = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ], device='cuda', dtype=torch.float32)
    V = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ], device='cuda', dtype=torch.float32)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")

    # 标准实现
    ref = standard_attention(Q, K, V)
    print(f"\n标准 Attention 输出: {ref[0].tolist()}")

    # 逐步跟踪 Online Softmax
    print("\n--- Online Softmax 逐步跟踪 ---")

    m = torch.tensor([float('-inf')], device='cuda', dtype=torch.float32)
    l = torch.tensor([0.0], device='cuda', dtype=torch.float32)
    O = torch.zeros(1, 2, device='cuda', dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)

    BLOCK_N = 2
    for j in range(0, 4, BLOCK_N):
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]

        qk = torch.matmul(Q, Kj.T) * scale
        print(f"\nBlock {j//BLOCK_N}: K[{j}:{j+BLOCK_N}], V[{j}:{j+BLOCK_N}]")
        print(f"  Q @ K.T / sqrt(d) = {qk[0].tolist()}")

        mj = qk.max(dim=-1).values
        m_new = torch.maximum(m, mj)
        print(f"  old m = {m[0].item():.4f}, new m = {m_new[0].item():.4f}")

        p = torch.exp(qk - m_new.unsqueeze(-1))
        print(f"  exp(scores - m_new) = {p[0].tolist()}")

        l_new = torch.exp(m - m_new) * l + p.sum(dim=-1)
        print(f"  old l = {l[0].item():.4f}, new l = {l_new[0].item():.4f}")

        O_new = (torch.exp(m - m_new).unsqueeze(-1) * O * l.unsqueeze(-1) + \
                 torch.matmul(p, Vj)) / l_new.unsqueeze(-1)
        print(f"  O = {O_new[0].tolist()}")

        m = m_new
        l = l_new
        O = O_new

    print(f"\n最终输出: {O[0].tolist()}")
    print(f"参考输出: {ref[0].tolist()}")
    print(f"误差: {(ref - O).abs().max().item():.2e}")


def main():
    test_correctness()
    test_different_block_sizes()
    visualize_algorithm()

    print("\n" + "=" * 60)
    print("总结:")
    print("  - Online Softmax 通过增量更新实现分块计算")
    print("  - 关键: 维护 m (max) 和 l (sum) 状态")
    print("  - 更新公式: l_new = exp(m_old - m_new) * l_old + exp(m_j - m_new) * l_j")
    print("=" * 60)


if __name__ == "__main__":
    main()