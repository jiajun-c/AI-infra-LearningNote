import torch
import triton
import triton.language as tl

@triton.jit
def flashAttentionNDKernel(Q, K, V, Out,
                           stride_qn, stride_qd,
                           stride_kn, stride_kd,
                           stride_vn, stride_vd,
                           stride_on, stride_od,
                           sm_scale,
                           BLOCK_M: tl.constexpr, # Q 分块的大小
                           BLOCK_N: tl.constexpr, # KV 分块的大小
                           D: tl.constexpr,
                           N: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, D)
    
    q_ptrs = Q + (offsets_m[:, None] * stride_qn + offsets_d[None, :] * stride_qd) 
    q = tl.load(q_ptrs, maks=offsets_m[:, None] < N, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # 遍历KV
    for start_n in range(0, N, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        
        k_ptrs    = K + (offsets_n[None, :] * stride_kn + offsets_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=offsets_n[None, :] < N, other=0.0)
        
        # [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        m_ij  = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:,None])
        
        alpha = tl.exp(m_i - m_new)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        v_ptrs = V + (offsets_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask = offsets_n[:, None] < N, other=0.0)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    
    o_ptrs = Out + (offsets_m[:, None] * stride_on + offsets_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offsets_m[:, None] < N)
    
        
        
        
    