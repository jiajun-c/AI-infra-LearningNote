import torch
import triton
import triton.language as tl


@triton.jit
def flashAttentionV2(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qd,  # Q strides: [batch, head, seq, dim]
    stride_kz, stride_kh, stride_kn, stride_kd,   # K strides
    stride_vz, stride_vh, stride_vn, stride_vd,   # V strides
    stride_oz, stride_oh, stride_om, stride_od,   # O strides
    num_heads,
    seq_len,
    scale,                                         # softmax 缩放因子 1/sqrt(d)
    BLOCK_M: tl.constexpr,   # Q 的分块大小
    BLOCK_N: tl.constexpr,   # KV 的分块大小
    BLOCK_D: tl.constexpr,   # 隐藏层维度 (Head Dim)
):
    # program_id(0) -> Q 序列方向的分块索引
    # program_id(1) -> batch * num_heads 联合索引
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // num_heads
    off_h = off_bh % num_heads

    # 计算当前 Q/K/V/O 在该 batch-head 下的基地址偏移
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh

    # --- A. 加载当前 Q 分块 ---
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offset_d = tl.arange(0, BLOCK_D)                        # [BLOCK_D]

    # Q 指针: (BLOCK_M, BLOCK_D)
    q_ptrs = Q + q_offset + offset_m[:, None] * stride_qm + offset_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offset_m[:, None] < seq_len, other=0.0)

    # --- 初始化在线 softmax 状态 ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 行最大值，初始化为 -inf
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 行求和（分母）
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)        # 输出累加器

    # --- B. 遍历所有 KV 分块 ---
    for start_n in range(0, seq_len, BLOCK_N):
        offset_n = start_n + tl.arange(0, BLOCK_N)              # [BLOCK_N]

        # K^T: 需要 (BLOCK_D, BLOCK_N)，所以指针按 (d, n) 索引
        k_ptrs = K + k_offset + offset_d[:, None] * stride_kd + offset_n[None, :] * stride_kn
        # V:   需要 (BLOCK_N, BLOCK_D)
        v_ptrs = V + v_offset + offset_n[:, None] * stride_vn + offset_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offset_n[None, :] < seq_len, other=0.0)
        v = tl.load(v_ptrs, mask=offset_n[:, None] < seq_len, other=0.0)

        # QK^T: (BLOCK_M, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale
        # 越界位置设为 -inf，防止 softmax 中 exp(0)=1 污染结果
        qk = tl.where(offset_n[None, :] < seq_len, qk, float("-inf"))

        # 在线 safe softmax
        m_ij = tl.max(qk, 1)                                   # 当前块的行最大值
        m_i_new = tl.maximum(m_i, m_ij)                         # 更新全局最大值
        alpha = tl.exp(m_i - m_i_new)                           # 缩放因子
        p = tl.exp(qk - m_i_new[:, None])                       # 当前块的分子项
        l_i_new = alpha * l_i + tl.sum(p, 1)                    # 更新全局分母和

        # --- C. 累加输出 O (完全在 SRAM 中进行) ---
        acc = acc * alpha[:, None]                               # 先用缩放因子修正旧的累加结果
        acc += tl.dot(p.to(v.dtype), v)                          # 加上新计算出的这部分 V

        # 更新状态，进入下一轮 KV 循环
        l_i = l_i_new
        m_i = m_i_new

    # --- D. 最终归一化并写回 ---
    acc = acc / l_i[:, None]
    o_ptrs = O + o_offset + offset_m[:, None] * stride_om + offset_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=offset_m[:, None] < seq_len)


# ===================== Host wrapper =====================
def flash_attention_v2(Q, K, V):
    """
    Flash Attention V2 的 Python wrapper.

    Args:
        Q: (batch, num_heads, seq_len, head_dim)
        K: (batch, num_heads, seq_len, head_dim)
        V: (batch, num_heads, seq_len, head_dim)
    Returns:
        O: (batch, num_heads, seq_len, head_dim)
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    batch, num_heads, seq_len, head_dim = Q.shape
    assert K.shape == V.shape == Q.shape

    O = torch.empty_like(Q)
    scale = head_dim ** -0.5

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = head_dim  # head_dim 必须是 2 的幂

    # grid: (seq_len 方向的分块数, batch * num_heads)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)

    flashAttentionV2[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        num_heads,
        seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return O


# ===================== 测试代码 =====================
def naive_attention(Q, K, V):
    """标准 Attention 实现作为参考 (PyTorch)"""
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    # Q @ K^T -> (batch, heads, seq, seq)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    # attn @ V -> (batch, heads, seq, dim)
    out = torch.matmul(attn, V)
    return out


def test_flash_attention_v2():
    """全面测试 Flash Attention V2 的正确性"""
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 60)
    print("Flash Attention V2 正确性测试")
    print("=" * 60)

    test_configs = [
        # (batch, num_heads, seq_len, head_dim, dtype, 描述)
        (1, 1, 64, 64, torch.float16, "最小配置: 1 batch, 1 head, seq=BLOCK"),
        (2, 4, 128, 64, torch.float16, "基础配置: 多 batch, 多 head"),
        (1, 8, 256, 64, torch.float16, "中等序列长度: seq=256"),
        (2, 4, 512, 64, torch.float16, "较长序列: seq=512"),
        (1, 1, 128, 128, torch.float16, "大 head_dim=128"),
        (4, 8, 64, 64, torch.float16, "大 batch=4, 多 head=8"),
        (1, 1, 96, 64, torch.float16, "非对齐序列: seq=96 (非 BLOCK_M 倍数)"),
        (2, 2, 200, 64, torch.float16, "非对齐序列: seq=200"),
    ]

    all_passed = True

    for i, (batch, num_heads, seq_len, head_dim, dtype, desc) in enumerate(test_configs):
        print(f"\n--- 测试 {i + 1}: {desc} ---")
        print(f"    shape=({batch}, {num_heads}, {seq_len}, {head_dim}), dtype={dtype}")

        Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # 参考实现 (PyTorch)
        ref_out = naive_attention(Q, K, V)

        # Flash Attention V2 (Triton)
        tri_out = flash_attention_v2(Q, K, V)

        # 比较结果
        # fp16 精度下 atol 需要放宽一些
        atol = 1e-2
        rtol = 1e-2
        is_close = torch.allclose(tri_out, ref_out, atol=atol, rtol=rtol)

        max_diff = (tri_out - ref_out).abs().max().item()
        mean_diff = (tri_out - ref_out).abs().mean().item()

        status = "✅ 通过" if is_close else "❌ 失败"
        print(f"    {status}  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if not is_close:
            all_passed = False
            print(f"    ⚠️  误差超出容忍范围 (atol={atol}, rtol={rtol})")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试未通过，请检查实现。")
    print("=" * 60)

    return all_passed


# ===================== 性能对比 Benchmark =====================
def benchmark_flash_vs_naive():
    """Flash Attention V2 vs 标准 Attention 性能对比"""
    device = "cuda"

    print("\n" + "=" * 70)
    print("Flash Attention V2 vs Naive Attention 性能对比")
    print("=" * 70)

    # 固定参数
    batch = 4
    num_heads = 8
    head_dim = 64
    dtype = torch.float16

    # 测试不同的序列长度 (128 ~ 8192)
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"\n配置: batch={batch}, num_heads={num_heads}, head_dim={head_dim}, dtype={dtype}")
    print(f"{'seq_len':>8} | {'Naive (ms)':>12} | {'Flash V2 (ms)':>14} | {'加速比':>8} | {'Naive 显存 (MB)':>16} | {'Flash 显存 (MB)':>16} | {'显存节省':>10}")
    print("-" * 110)

    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # ---- Benchmark Naive Attention ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(3):
            _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()

        _ = naive_attention(Q, K, V)
        torch.cuda.synchronize()
        naive_peak_mem = torch.cuda.max_memory_allocated()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        n_iters = 20
        start_event.record()
        for _ in range(n_iters):
            _ = naive_attention(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        naive_ms = start_event.elapsed_time(end_event) / n_iters

        # ---- Benchmark Flash Attention V2 ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(3):
            _ = flash_attention_v2(Q, K, V)
        torch.cuda.synchronize()

        _ = flash_attention_v2(Q, K, V)
        torch.cuda.synchronize()
        flash_peak_mem = torch.cuda.max_memory_allocated()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(n_iters):
            _ = flash_attention_v2(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        flash_ms = start_event.elapsed_time(end_event) / n_iters

        # ---- 汇总 ----
        speedup = naive_ms / flash_ms
        naive_mem_mb = naive_peak_mem / (1024 * 1024)
        flash_mem_mb = flash_peak_mem / (1024 * 1024)
        mem_saving = (1 - flash_peak_mem / naive_peak_mem) * 100 if naive_peak_mem > 0 else 0

        print(f"{seq_len:>8} | {naive_ms:>10.3f}ms | {flash_ms:>12.3f}ms | {speedup:>7.2f}x | {naive_mem_mb:>14.1f}MB | {flash_mem_mb:>14.1f}MB | {mem_saving:>8.1f}%")

    # ---- Triton benchmark (绘图) ----
    print("\n" + "=" * 70)
    print("使用 triton.testing 进行详细 benchmark 并生成图表...")
    print("=" * 70)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],                                    # x 轴: 序列长度
            x_vals=[2**i for i in range(7, 14)],                    # 128 ~ 8192
            line_arg="provider",                                    # 不同实现用不同线
            line_vals=["naive", "flash_v2"],
            line_names=["Naive Attention", "Flash Attention V2"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="ms",                                            # y 轴: 毫秒
            plot_name=f"flash-attention-v2-fwd-batch{batch}-head{num_heads}-dim{head_dim}-{dtype}",
            args={"batch": batch, "num_heads": num_heads, "head_dim": head_dim, "dtype": dtype},
        )
    )
    def bench_attention(seq_len, provider, batch, num_heads, head_dim, dtype):
        Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        quantiles = [0.5, 0.2, 0.8]  # 中位数、P20、P80
        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_attention(Q, K, V), quantiles=quantiles)
        elif provider == "flash_v2":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attention_v2(Q, K, V), quantiles=quantiles)
        return ms, min_ms, max_ms

    bench_attention.run(print_data=True, save_path=".")


if __name__ == "__main__":
    test_flash_attention_v2()
    benchmark_flash_vs_naive()
