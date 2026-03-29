import torch
import triton
import triton.language as tl

@triton.jit
def flashAttentionV2(Q, K, V, O,
                     stride_qz, stride_qh, stride_qm, stride_qd,
                     stride_kz, stride_kh, stride_kn, stride_kd,
                     stride_vz, stride_vh, stride_vn, stride_vd,
                     stride_oz, stride_oh, stride_om, stride_od,
                     num_heads,
                     seq_len,
                     scale,
                     IS_CAUSAL: tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // num_heads
    off_h = off_bh % num_heads

    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh

    offset_m = start_m * BLOCK_M + tl.range(0, BLOCK_M)
    offset_d = tl.range(0, BLOCK_D)

    q_ptrs = Q + q_offset + offset_m[:, None] * stride_qm + offset_d[None, :]
    q_mask = offset_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 行上的最大值
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # 行上求和
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 输出
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # causal mask: 只需迭代到当前Q block能看到的最大K位置
    # 因为 Q[i] 只能 attend 到 K[j] where j <= i
    # 所以 K 的上界是 (start_m + 1) * BLOCK_M，而非 seq_len
    if IS_CAUSAL:
        end_n = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
    else:
        end_n = seq_len

    for start_n in range(0, end_n, BLOCK_N):
        offset_n = start_n + tl.range(0, BLOCK_N)
        mask_n = offset_n < seq_len
        k_ptrs = K + k_offset + offset_n[:, None] * stride_kn + offset_d[None, :]
        v_ptrs = V + v_offset + offset_n[:, None] * stride_vn + offset_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk *= scale

        # causal mask: 将 Q 行号 < K 列号 的位置设为 -inf
        # offset_m[i] 是 Q 的第 i 行的绝对位置
        # offset_n[j] 是 K 的第 j 列的绝对位置
        # 当 offset_m[i] < offset_n[j] 时, 即未来 token, 需要被 mask 掉
        if IS_CAUSAL:
            causal_mask = offset_m[:, None] >= offset_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_ptrs = O + o_offset + offset_m[:, None] * stride_om + offset_d[None, :]
    tl.store(o_ptrs, acc, mask=offset_m[:, None] < seq_len)
    