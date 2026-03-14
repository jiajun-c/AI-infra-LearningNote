import triton
import triton.language as tl

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
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)  
    
    for i in tl.range(0, N, BLOCK_N):
        offsets_n = i + tl.arange(0, BLOCK_N)
        k_ptrs = K + + (offsets_n[:, None] * stride_kn + offsets_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, offsets_n[:, None] < N, other=0.0)
        qk = tl.dot(q, k)
        qk *= sm_scale
        
        m_ij = tl.max(qk, 1)
        p_ij = tl.exp(qk - m_ij)
        
        lij = tl.sum(p_ij, 1)
        m_new = max(m_new, m_i)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + lij
        acc = acc * alpha[: None]
        
        v_ptrs = V + (offsets_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd) # 64 * 64 * 4 = 32KB BLOCK_N * DD
        v = tl.load(v_ptrs, mask=offsets_n[:, None] < N, other=0.0)
        
        # P [M, N], V [N, d] -> [M, d]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    o_ptrs = O + (offsets_m[:, None] * stride_on + offsets_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offsets_m[:, None] < N)
    
        
        
        