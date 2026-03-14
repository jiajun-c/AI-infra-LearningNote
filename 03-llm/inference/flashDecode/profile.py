import torch
import triton
import triton.language as tl
import math

# =====================================================================
# 1. Baseline: 标准 Decoding Kernel (没有序列切分，单 Block 跑完全程)
# =====================================================================
@triton.jit
def baseline_decode_kernel(
    Q, K, V, Out,
    seq_len,
    stride_k_seq, stride_k_d,
    stride_v_seq, stride_v_d,
    D: tl.constexpr, BLOCK_SEQ: tl.constexpr
):
    bh_id = tl.program_id(0) # Batch * Head
    
    # 定位当前 Head 的 Q
    q_ptrs = Q + bh_id * D + tl.arange(0, D)
    q = tl.load(q_ptrs)
    
    # 定位 K 和 V 的起始指针
    k_ptrs = K + bh_id * seq_len * D + tl.arange(0, BLOCK_SEQ)[:, None] * stride_k_seq + tl.arange(0, D)[None, :] * stride_k_d
    v_ptrs = V + bh_id * seq_len * D + tl.arange(0, BLOCK_SEQ)[:, None] * stride_v_seq + tl.arange(0, D)[None, :] * stride_v_d
    
    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    
    # 痛点：一个 Block 必须串行遍历整个 Seq Len
    for start_n in range(0, seq_len, BLOCK_SEQ):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Q * K^T
        qk = tl.sum(q[None, :] * k, axis=1) # [BLOCK_SEQ]
        qk = qk * 1.44269504 # log2(e) for fast math if needed, simplified here
        
        # Online Softmax 逻辑
        m_ij = tl.max(qk, axis=0)
        new_m_i = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - new_m_i)
        beta = tl.exp(qk - new_m_i)
        
        l_i = l_i * alpha + tl.sum(beta, axis=0)
        
        # O = O * alpha + P * V
        beta_expanded = beta[:, None]
        acc = acc * alpha + tl.sum(beta_expanded * v, axis=0)
        
        m_i = new_m_i
        k_ptrs += BLOCK_SEQ * stride_k_seq
        v_ptrs += BLOCK_SEQ * stride_v_seq
        
    out = acc / l_i
    out_ptrs = Out + bh_id * D + tl.arange(0, D)
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))

# =====================================================================
# 2. Flash-Decoding Stage 1: 分块计算局部 Attention
# =====================================================================
@triton.jit
def flash_decode_stage1_kernel(
    Q, K, V, 
    Mid_O, Mid_M, Mid_L,
    seq_len,
    stride_k_seq, stride_k_d,
    stride_v_seq, stride_v_d,
    D: tl.constexpr, BLOCK_SEQ: tl.constexpr
):
    bh_id = tl.program_id(0)
    chunk_id = tl.program_id(1) # 核心：在 Seq 维度并行
    
    q_ptrs = Q + bh_id * D + tl.arange(0, D)
    q = tl.load(q_ptrs)
    
    start_n = chunk_id * BLOCK_SEQ
    k_ptrs = K + bh_id * seq_len * D + start_n * stride_k_seq + tl.arange(0, BLOCK_SEQ)[:, None] * stride_k_seq + tl.arange(0, D)[None, :] * stride_k_d
    v_ptrs = V + bh_id * seq_len * D + start_n * stride_v_seq + tl.arange(0, BLOCK_SEQ)[:, None] * stride_v_seq + tl.arange(0, D)[None, :] * stride_v_d
    
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    
    qk = tl.sum(q[None, :] * k, axis=1) 
    
    # 局部统计量
    m_i = tl.max(qk, axis=0)
    p = tl.exp(qk - m_i)
    l_i = tl.sum(p, axis=0)
    
    p_expanded = p[:, None]
    o_i = tl.sum(p_expanded * v, axis=0)
    
    # 写入 Global Memory Workspace
    offset = bh_id * tl.num_programs(1) + chunk_id
    tl.store(Mid_M + offset, m_i)
    tl.store(Mid_L + offset, l_i)
    
    out_ptrs = Mid_O + offset * D + tl.arange(0, D)
    tl.store(out_ptrs, o_i)

# =====================================================================
# 3. Flash-Decoding Stage 2: 全局归约 (Reduction)
# =====================================================================
@triton.jit
def flash_decode_stage2_kernel(
    Mid_O, Mid_M, Mid_L, Out,
    num_chunks, D: tl.constexpr
):
    bh_id = tl.program_id(0)
    
    global_m = -float('inf')
    global_l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    
    # 读取 Stage 1 的局部结果并合并
    for chunk_id in range(num_chunks):
        offset = bh_id * num_chunks + chunk_id
        
        m_i = tl.load(Mid_M + offset)
        l_i = tl.load(Mid_L + offset)
        o_ptrs = Mid_O + offset * D + tl.arange(0, D)
        o_i = tl.load(o_ptrs)
        
        new_global_m = tl.maximum(global_m, m_i)
        
        alpha = tl.exp(global_m - new_global_m)
        beta = tl.exp(m_i - new_global_m)
        
        global_l = global_l * alpha + l_i * beta
        acc = acc * alpha + o_i * beta
        
        global_m = new_global_m
        
    out = acc / global_l
    out_ptrs = Out + bh_id * D + tl.arange(0, D)
    tl.store(out_ptrs, out.to(Out.dtype.element_ty))

# =====================================================================
# 4. PyTorch Wrapper
# =====================================================================
def flash_decode(q, k, v, BLOCK_SEQ=256):
    B, H, seq_len, D = k.shape
    num_chunks = seq_len // BLOCK_SEQ
    
    # 分配 Stage 1 的 Workspace
    mid_o = torch.empty((B, H, num_chunks, D), device=q.device, dtype=torch.float32)
    mid_m = torch.empty((B, H, num_chunks), device=q.device, dtype=torch.float32)
    mid_l = torch.empty((B, H, num_chunks), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    
    grid_stage1 = (B * H, num_chunks)
    flash_decode_stage1_kernel[grid_stage1](
        q, k, v, mid_o, mid_m, mid_l,
        seq_len, k.stride(2), k.stride(3), v.stride(2), v.stride(3),
        D=D, BLOCK_SEQ=BLOCK_SEQ
    )
    
    grid_stage2 = (B * H,)
    flash_decode_stage2_kernel[grid_stage2](
        mid_o, mid_m, mid_l, out,
        num_chunks, D=D
    )
    return out

def baseline_decode(q, k, v, BLOCK_SEQ=256):
    B, H, seq_len, D = k.shape
    out = torch.empty_like(q)
    
    grid = (B * H,)
    baseline_decode_kernel[grid](
        q, k, v, out,
        seq_len, k.stride(2), k.stride(3), v.stride(2), v.stride(3),
        D=D, BLOCK_SEQ=BLOCK_SEQ
    )
    return out

# =====================================================================
# 5. Benchmark 性能分析
# =====================================================================
configs = [
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2**i for i in range(12, 18)], # 4K 到 128K
        line_arg='provider',
        line_vals=['baseline', 'flash_decode'],
        line_names=['Baseline Decoding', 'Flash-Decoding'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='Latency (ms)',
        plot_name='decoding-performance',
        args={'B': 1, 'H': 32, 'D': 128}
    )
]

@triton.testing.perf_report(configs)
def benchmark(B, H, D, seq_len, provider):
    q = torch.randn((B, H, D), device='cuda', dtype=torch.float16)
    k = torch.randn((B, H, seq_len, D), device='cuda', dtype=torch.float16)
    v = torch.randn((B, H, seq_len, D), device='cuda', dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'baseline':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: baseline_decode(q, k, v), quantiles=quantiles)
    elif provider == 'flash_decode':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_decode(q, k, v), quantiles=quantiles)
        
    return ms, min_ms, max_ms

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True)