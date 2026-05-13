"""
Step 5: FlashAttention 完整实现

运行方式: python flash_attention.py

参考: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def flash_attn_kernel(
    # 指针
    Q, K, V, Out,
    # 步长
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # 维度
    Z, H, N_CTX, D_HEAD,
    # Meta 参数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    FlashAttention Forward Kernel

    每个 program instance 处理:
    - 一个 batch (z)
    - 一个 head (h)
    - 一个 Q block (m)
    """
    # ========== 1. 确定 program instance 处理的范围 ==========

    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    off_m = tl.program_id(1)

    Q_BLOCK_START = off_m * BLOCK_M

    # ========== 2. 计算 Q block 的地址 ==========

    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh

    Q_row_offsets = Q_BLOCK_START + tl.arange(0, BLOCK_M)
    Q_col_offsets = tl.arange(0, BLOCK_DMODEL)

    Q_ptrs = q_base_ptr + Q_row_offsets[:, None] * stride_qm + Q_col_offsets[None, :] * stride_qk
    Q_mask = Q_row_offsets[:, None] < N_CTX

    # ========== 3. 加载 Q block 到 SRAM ==========

    q = tl.load(Q_ptrs, mask=Q_mask, other=0.0).to(tl.float32)

    # ========== 4. 初始化 Online Softmax 状态 ==========

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    scale = 1.0 / math.sqrt(BLOCK_DMODEL)

    # ========== 5. 遍历 K, V blocks ==========

    k_base_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_base_ptr = V + off_z * stride_vz + off_h * stride_vh

    lo = 0
    hi = N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        # 加载 K block
        K_row_offsets = start_n + tl.arange(0, BLOCK_N)
        K_ptrs = k_base_ptr + K_row_offsets[None, :] * stride_kn + Q_col_offsets[:, None] * stride_kk
        K_mask = K_row_offsets[None, :] < N_CTX

        k = tl.load(K_ptrs, mask=K_mask, other=0.0).to(tl.float32)

        # 计算 Q @ K.T
        qk = tl.dot(q, k) * scale

        # 添加 causal mask
        # causal_mask = K_row_offsets[None, :] <= Q_row_offsets[:, None]
        # qk = tl.where(causal_mask, qk, float('-inf'))

        # Online Softmax
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.math.exp(m_i - m_i_new)
        # beta 不需要单独计算，p 已经是 exp(qk - m_i_new)

        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        l_i_new = alpha * l_i + l_ij  # 注意: l_ij 已经是 exp(qk - m_new) 的和

        # 加载 V block
        V_row_offsets = start_n + tl.arange(0, BLOCK_N)
        V_ptrs = v_base_ptr + V_row_offsets[:, None] * stride_vn + Q_col_offsets[None, :] * stride_vk
        V_mask = V_row_offsets[:, None] < N_CTX

        v = tl.load(V_ptrs, mask=V_mask, other=0.0).to(tl.float32)

        # 累加输出
        # 正确公式: acc_new = alpha * acc_old + p @ V
        # p 已经是 exp(qk - m_new)，不需要再乘 beta
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # 更新状态
        m_i = m_i_new
        l_i = l_i_new

    # ========== 6. 写回结果 ==========
    # 最终归一化
    acc = acc / l_i[:, None]

    O_base_ptr = Out + off_z * stride_oz + off_h * stride_oh
    O_row_offsets = Q_BLOCK_START + tl.arange(0, BLOCK_M)
    O_ptrs = O_base_ptr + O_row_offsets[:, None] * stride_om + Q_col_offsets[None, :] * stride_ok
    O_mask = O_row_offsets[:, None] < N_CTX

    tl.store(O_ptrs, acc.to(tl.float16), mask=O_mask)


def flash_attention(q, k, v):
    """
    FlashAttention Host 函数

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4

    batch, heads, seq_len, head_dim = q.shape

    # 输出必须是 float16 或 bfloat16
    output = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (batch * heads, triton.cdiv(seq_len, BLOCK_M))

    # 确保 head_dim 是编译时常量支持的值
    assert head_dim in {16, 32, 64, 128}, f"head_dim must be 16, 32, 64 or 128, got {head_dim}"

    flash_attn_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
    )

    return output


# ============================================================
# 标准 Attention 作为参考
# ============================================================

def standard_attention(q, k, v):
    """标准 Attention 实现"""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ============================================================
# 测试
# ============================================================

def test_correctness():
    """测试正确性"""
    print("=" * 60)
    print("测试 FlashAttention 正确性")
    print("=" * 60)

    torch.manual_seed(42)

    test_cases = [
        (1, 1, 128, 64),
        (1, 2, 256, 64),
        (2, 4, 512, 64),
        (1, 8, 1024, 64),
    ]

    for batch, heads, seq_len, head_dim in test_cases:
        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

        # 标准 Attention
        ref = standard_attention(q.float(), k.float(), v.float()).half()

        # Flash Attention
        out = flash_attention(q, k, v)

        diff = (ref - out).abs().max().item()
        status = "✓ PASS" if diff < 1e-2 else "✗ FAIL"
        print(f"  ({batch}, {heads}, {seq_len}, {head_dim}): max_diff = {diff:.2e} {status}")

    print()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='Time (ms)',
        plot_name='flash-attention-performance',
        args={'M': 4096, 'H': 16, 'D': 64},
    )
)
def benchmark(M, N, H, D, provider):
    """性能测试"""
    q = torch.randn(1, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(1, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(1, H, N, D, device='cuda', dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: standard_attention(q.float(), k.float(), v.float()),
            quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attention(q, k, v),
            quantiles=quantiles
        )

    return ms, max_ms, min_ms


def main():
    test_correctness()

    print("=" * 60)
    print("性能基准测试")
    print("=" * 60)
    benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()