import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math

# ==========================================
# 1. 你的 Triton Kernel 代码 (保持不变)
# ==========================================


@triton.jit
def _flash_attn_nd_kernel(
    Q, K, V, Out,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    d: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    q_ptrs = Q + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)   # 128 * 128 * 4 = 64KB BLOCK_M * D

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 128 * 4 = 0.5KB
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # 128 * 4 = 0.5KB
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)             # 128 * 128 * 4 = 64KB BLOCK_M * D

    for start_n in tl.range(0, N, BLOCK_N, num_stages=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0) # 64 * 128 * 4 = 32KB BLOCK_N * D
        
        qk = tl.dot(q, k) # 注意: 这里 k 应该转置，但 Triton dot 自动处理布局，通常需要注意 Layout
        # 修正: 上面你的代码直接 dot(q, k)。如果 k 是 [BLOCK_N, d]，dot(q, k) 会报错 [M, d] @ [N, d]。
        # Triton 的 tl.dot(A, B) 要求 A的最后一维 == B的第一维。
        # 这里需要转置 k。
        # k = tl.trans(k) # <--- 必须加上转置才能进行 [M,d] @ [d,N] = [M,N]

        qk *= sm_scale
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd) # 64 * 64 * 4 = 32KB BLOCK_N * DD
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
        
        # P [M, N], V [N, d] -> [M, d]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N)

@triton.jit
def flashAttnV1(
    Q, K, V, O,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, D)
    # (Block_m, 1)
    # (1, Block_d)
    q_ptrs = Q + (offsets_m[:, None] * stride_qn + offsets_d[None, :] * stride_qd)
    
    # (BLOCK_M, D)
    q = tl.load(q_ptrs, offsets_m[:, None] < N, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # 128 * 4 = 0.5KB
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # 128 * 4 = 0.5KB
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)  
    
    for i in tl.range(0, N, BLOCK_N):
        offsets_n = i + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offsets_n[None, :] * stride_kn + offsets_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=offsets_n[None, :] < N, other=0.0) # 64 * 128 * 4 = 32KB BLOCK_N * D
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        m_ij = tl.max(qk, 1)
        p_ij = tl.exp(qk - m_ij[:, None])
        
        lij = tl.sum(p_ij, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + lij
        acc = acc * alpha[:, None]
        p = tl.exp(qk - m_new[:, None])

        v_ptrs = V + (offsets_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd) # 64 * 64 * 4 = 32KB BLOCK_N * DD
        v = tl.load(v_ptrs, mask=offsets_n[:, None] < N, other=0.0)
        
        # P [M, N], V [N, d] -> [M, d]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    o_ptrs = O + (offsets_m[:, None] * stride_on + offsets_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offsets_m[:, None] < N)
    

def flash_attention_nd(q, k, v):
    N, d = q.shape
    BLOCK_M = 128
    BLOCK_N = 64
    sm_scale = 1.0 / (d ** 0.5)
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), )
    
    _flash_attn_nd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        d=d, N=N
    )
    return o

# ==========================================
# 2. 性能测试工具 (Triton Benchmark)
# ==========================================

# 设置 Benchmark 配置
configs = [
    triton.testing.Benchmark(
        x_names=['N'],      # 变量名为 N (序列长度)
        x_vals=[1024, 2048, 4096, 8192, 16384], # 不同的 N 值
        line_arg='provider', # 线条对比参数
        line_vals=['naive', 'torch_sdpa', 'triton'], 
        line_names=['Naive PyTorch', 'PyTorch SDPA', 'Your Triton'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms', # y轴单位
        plot_name='attention-performance',
        args={'d': 128}, # 固定 Head Dimension
    )
]

@triton.testing.perf_report(configs)
def benchmark(N, d, provider):
    # 初始化数据
    q = torch.randn((N, d), device='cuda', dtype=torch.float16)
    k = torch.randn((N, d), device='cuda', dtype=torch.float16)
    v = torch.randn((N, d), device='cuda', dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'naive':
        # ⚠️ 注意：Naive 模式在 N 很大时会 OOM，这里做个保护
        if N > 8192: 
            return float('inf'), float('inf'), float('inf')
        
        def naive_impl():
            scale = 1.0 / (d ** 0.5)
            scores = torch.matmul(q, k.t()) * scale
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)
        
        ms, min_ms, max_ms = triton.testing.do_bench(naive_impl, quantiles=quantiles)
        
    elif provider == 'torch_sdpa':
        # PyTorch SDPA 需要 3D/4D 输入，我们 fake 一个 batch 维度
        q_b = q.unsqueeze(0).unsqueeze(0) # [1, 1, N, d]
        k_b = k.unsqueeze(0).unsqueeze(0)
        v_b = v.unsqueeze(0).unsqueeze(0)
        
        def sdpa_impl():
            return F.scaled_dot_product_attention(q_b, k_b, v_b)
            
        ms, min_ms, max_ms = triton.testing.do_bench(sdpa_impl, quantiles=quantiles)
        
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attention_nd(q, k, v), quantiles=quantiles)
        
    return ms, max_ms, min_ms

# ==========================================
# 3. 显存测试工具
# ==========================================
def benchmark_memory(N, d):
    print(f"\n--- Memory Benchmark (N={N}, d={d}) ---")
    
    # 1. 准备数据 (这部分内存不应该计入算法的开销)
    q = torch.randn((N, d), device='cuda', dtype=torch.float16)
    k = torch.randn((N, d), device='cuda', dtype=torch.float16)
    v = torch.randn((N, d), device='cuda', dtype=torch.float16)
    
    # 理论输入大小 (仅供参考)
    input_size_mb = (N * d * 2 * 3) / 1024**2
    print(f"Input Tensors Size (Q+K+V): {input_size_mb:.2f} MB")

    # ==========================================
    # 1. Naive Memory Test
    # ==========================================
    if N <= 8192:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # [关键步骤] 记录操作前的基准显存 (此时只包含 q, k, v)
        base_mem = torch.cuda.memory_allocated()
        
        # --- 执行 Naive Attention ---
        scale = 1.0 / (d ** 0.5)
        scores = torch.matmul(q, k.t()) * scale # O(N^2) 显存爆炸点
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        # ---------------------------
        
        # 计算峰值增量
        max_mem = torch.cuda.max_memory_allocated()
        mem_naive = (max_mem - base_mem) / 1024**2
        
        print(f"Naive PyTorch (Overhead): {mem_naive:.2f} MB")
        
        del scores, probs, out, scale
    else:
        print("Naive PyTorch: OOM (Skipped)")

    # ==========================================
    # 2. Triton Memory Test
    # ==========================================
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # [关键步骤] 记录基准显存
    base_mem = torch.cuda.memory_allocated()
    
    # --- 执行 Triton Kernel ---
    out_tri = flash_attention_nd(q, k, v)
    # -------------------------
    
    # 计算峰值增量
    max_mem = torch.cuda.max_memory_allocated()
    mem_tri = (max_mem - base_mem) / 1024**2
    
    print(f"Your Triton   (Overhead): {mem_tri:.2f} MB")
def test_correctness():
    print("\n--- Correctness Test ---")
    torch.manual_seed(0)
    
    # 1. 设置参数
    # 故意设置 BLOCK_N != d 来测试维度匹配是否正确
    # 注意：如果你的 kernel 没加 tl.trans(k)，这里 D=128, BLOCK_N=64 会直接报错
    N = 1024
    d = 128 
    dtype = torch.float16
    device = "cuda"

    print(f"Config: N={N}, d={d}, dtype={dtype}")

    # 2. 准备数据
    q = torch.randn((N, d), device=device, dtype=dtype)
    k = torch.randn((N, d), device=device, dtype=dtype)
    v = torch.randn((N, d), device=device, dtype=dtype)

    # 3. 运行 Triton Kernel
    print("Running Triton implementation...")
    try:
        tri_out = flash_attention_nd(q, k, v)
    except Exception as e:
        print(f"\n❌ Triton Execution Crushed: {e}")
        print("Tip: 检查是否在 tl.dot(q, k) 前忘记了 tl.trans(k)？")
        print("Tip: 检查 tl.dot 的维度是否为 [M, d] @ [d, N]")
        return

    # 4. 运行 PyTorch Reference (Naive)
    print("Running PyTorch Reference...")
    # 这里的逻辑必须是绝对正确的数学公式
    sm_scale = 1.0 / (d ** 0.5)
    # Q: [N, d], K.T: [d, N] -> Scores: [N, N]
    scores = torch.matmul(q, k.t()) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v)

    # 5. 数值比对
    # 允许一定的误差 (FP16 精度有限，且 FlashAttn 的累加顺序不同)
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    
    diff = torch.abs(tri_out - ref_out)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n--- Result Analysis ---")
    print(f"Max Difference:  {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")

    if max_diff < atol:
        print("✅ Test PASSED: Results match within tolerance.")
    else:
        print("❌ Test FAILED: Results differ significantly.")
        # 打印部分数据帮助调试
        print("\nFirst 5 elements (Triton):")
        print(tri_out[0, :5])
        print("\nFirst 5 elements (Reference):")
        print(ref_out[0, :5])

if __name__ == "__main__":
    test_correctness()
    # 之后的 benchmark 代码...
# if __name__ == "__main__":
#     # 1. 运行显存测试
#     benchmark_memory(N=4096, d=128)
#     benchmark_memory(N=8192, d=128) # 这里 Naive 可能会非常大
    
#     # 2. 运行速度测试
#     print("\n--- Speed Benchmark (Running...) ---")
#     benchmark.run(print_data=True, show_plots=False)