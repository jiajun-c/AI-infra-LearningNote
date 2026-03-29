"""
Flash Attention V2 with KV Cache (Triton 实现)

=== 核心思想 ===

推理阶段分两步:
  1. Prefill (首次): 处理完整 prompt，Q/K/V 序列长度相同，等价于标准 FlashAttention
  2. Decode (逐 token): 每次只生成 1 个新 token
     - Q: (batch, num_heads, 1, head_dim)           ← 只有当前 token
     - K_cache: (batch, num_heads, cache_len, head_dim)  ← 累积的所有历史 K
     - V_cache: (batch, num_heads, cache_len, head_dim)  ← 累积的所有历史 V

  Decode 时 Q 只有一行，所以 QK^T 的结果是 (1, cache_len) 的向量。
  不需要 online softmax 的分块重缩放（因为 BLOCK_M=1 时行最大值就是标量），
  但 KV cache 可能很长（几千~几万 token），仍需要分块遍历 KV 来节省显存。

=== 与标准 FlashAttention V2 的关键区别 ===

  ┌──────────────────┬──────────────────────┬──────────────────────────┐
  │                  │   标准 Flash Attn V2  │  Flash Attn + KV Cache   │
  ├──────────────────┼──────────────────────┼──────────────────────────┤
  │ Q 序列长度        │  seq_len (完整)       │  1 (decode) / N (prefill)│
  │ KV 序列长度       │  = Q 序列长度         │  cache_len (可能很长)     │
  │ KV 来源          │  当前输入计算          │  从预分配的 cache 读取     │
  │ 额外操作          │  无                   │  写入新 KV 到 cache       │
  │ BLOCK_M          │  64 (多行)            │  decode 时 = 1            │
  └──────────────────┴──────────────────────┴──────────────────────────┘
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# Kernel 1: Flash Attention V2 Decode with KV Cache
#           Q: (batch, num_heads, 1, head_dim)  只有当前 1 个 token
#           K_cache / V_cache: (batch, num_heads, max_cache_len, head_dim)
# =====================================================================
@triton.jit
def flash_attn_decode_kernel(
    Q, K_cache, V_cache, O,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    num_heads,
    cache_len,       # 当前有效的 cache 长度（不含新 token，已经 append 过了）
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    每个 program 处理一个 (batch, head) 对。
    Q 只有 1 行，所以不需要在 M 维度分块。
    沿 KV cache 的 N 维度分块遍历，执行 online softmax。
    """
    off_bh = tl.program_id(0)                           # batch * num_heads 联合索引
    off_b = off_bh // num_heads
    off_h = off_bh % num_heads

    # 基地址偏移
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh

    # 加载 Q: (1, BLOCK_D) -> 展平为 (BLOCK_D,) 方便计算
    offset_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q + q_offset + 0 * stride_qm + offset_d * stride_qd     # Q 只有 seq=0
    q = tl.load(q_ptrs)                                               # [BLOCK_D]

    # 初始化 online softmax 状态 (标量，因为 Q 只有 1 行)
    m_i = -float("inf")                    # 全局行最大值
    l_i = 0.0                              # 全局分母和
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)   # 输出累加器 [BLOCK_D]

    # 分块遍历 KV cache
    for start_n in range(0, cache_len, BLOCK_N):
        offset_n = start_n + tl.arange(0, BLOCK_N)     # [BLOCK_N]
        mask_n = offset_n < cache_len

        # 加载 K 块: (BLOCK_D, BLOCK_N) —— 转置形式
        k_ptrs = K_cache + k_offset + offset_d[:, None] * stride_kd + offset_n[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)     # [BLOCK_D, BLOCK_N]

        # Q @ K^T: (BLOCK_D,) @ (BLOCK_D, BLOCK_N) -> (BLOCK_N,)
        qk = tl.sum(q[:, None] * k, axis=0)                      # [BLOCK_N]
        qk *= scale
        # 越界置为 -inf
        qk = tl.where(mask_n, qk, float("-inf"))

        # online softmax
        m_ij = tl.max(qk, 0)                                     # 当前块最大值 (标量)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)                                 # [BLOCK_N]
        l_i_new = alpha * l_i + tl.sum(p, 0)

        # 加载 V 块: (BLOCK_N, BLOCK_D)
        v_ptrs = V_cache + v_offset + offset_n[:, None] * stride_vn + offset_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)     # [BLOCK_N, BLOCK_D]

        # 累加: acc = alpha * acc + P @ V
        acc = acc * alpha
        acc += tl.sum(p[:, None] * v, axis=0)                    # [BLOCK_D]

        m_i = m_i_new
        l_i = l_i_new

    # 归一化并写回
    acc = acc / l_i
    o_ptrs = O + o_offset + 0 * stride_om + offset_d * stride_od
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty))


# =====================================================================
# Kernel 2: Flash Attention V2 Prefill with KV Cache
#           Q: (batch, num_heads, seq_len, head_dim)
#           复用 v2.py 中的 prefill kernel，同时写入 KV cache
# =====================================================================
@triton.jit
def flash_attn_prefill_kernel(
    Q, K, V, O,
    K_cache, V_cache,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_kcz, stride_kch, stride_kcn, stride_kcd,
    stride_vcz, stride_vch, stride_vcn, stride_vcd,
    num_heads,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Prefill: 标准 FlashAttention V2 + 写入 KV cache"""
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // num_heads
    off_h = off_bh % num_heads

    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    kc_offset = off_b * stride_kcz + off_h * stride_kch
    vc_offset = off_b * stride_vcz + off_h * stride_vch

    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_d = tl.arange(0, BLOCK_D)

    # 加载 Q 分块
    q_ptrs = Q + q_offset + offset_m[:, None] * stride_qm + offset_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offset_m[:, None] < seq_len, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offset_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + k_offset + offset_d[:, None] * stride_kd + offset_n[None, :] * stride_kn
        v_ptrs = V + v_offset + offset_n[:, None] * stride_vn + offset_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=offset_n[None, :] < seq_len, other=0.0)
        v = tl.load(v_ptrs, mask=offset_n[:, None] < seq_len, other=0.0)

        # --- 写入 KV cache (只在第一个 Q 分块时写，避免重复写入) ---
        if start_m == 0:
            # K -> K_cache
            kc_ptrs = K_cache + kc_offset + offset_n[None, :] * stride_kcn + offset_d[:, None] * stride_kcd
            tl.store(kc_ptrs, k, mask=offset_n[None, :] < seq_len)
            # V -> V_cache (注意 v 是 (BLOCK_N, BLOCK_D))
            vc_ptrs = V_cache + vc_offset + offset_n[:, None] * stride_vcn + offset_d[None, :] * stride_vcd
            tl.store(vc_ptrs, v, mask=offset_n[:, None] < seq_len)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale
        qk = tl.where(offset_n[None, :] < seq_len, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_ptrs = O + o_offset + offset_m[:, None] * stride_om + offset_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=offset_m[:, None] < seq_len)


# =====================================================================
# KV Cache 管理类
# =====================================================================
class KVCache:
    """
    预分配固定大小的 KV Cache.

    内存布局: (batch, num_heads, max_seq_len, head_dim)
    通过 cache_len 追踪当前已填入的 token 数量。
    """

    def __init__(self, batch: int, num_heads: int, max_seq_len: int,
                 head_dim: int, dtype=torch.float16, device="cuda"):
        self.batch = batch
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # 预分配缓冲区
        self.k_cache = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                                   dtype=dtype, device=device)
        self.cache_len = 0  # 当前已填入的 token 数

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        将新的 K, V 追加到 cache 末尾。
        k_new, v_new: (batch, num_heads, new_len, head_dim)
        """
        new_len = k_new.shape[2]
        assert self.cache_len + new_len <= self.max_seq_len, \
            f"KV Cache 溢出: {self.cache_len} + {new_len} > {self.max_seq_len}"
        self.k_cache[:, :, self.cache_len:self.cache_len + new_len, :] = k_new
        self.v_cache[:, :, self.cache_len:self.cache_len + new_len, :] = v_new
        self.cache_len += new_len

    def reset(self):
        """清空 cache"""
        self.cache_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


# =====================================================================
# Host Wrapper: Prefill
# =====================================================================
def flash_attn_prefill(Q, K, V, kv_cache: KVCache):
    """
    Prefill 阶段: 处理完整 prompt，同时填充 KV Cache.

    Args:
        Q, K, V: (batch, num_heads, seq_len, head_dim)
        kv_cache: KVCache 实例
    Returns:
        O: (batch, num_heads, seq_len, head_dim)
    """
    batch, num_heads, seq_len, head_dim = Q.shape
    O = torch.empty_like(Q)
    scale = head_dim ** -0.5

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = head_dim

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)

    flash_attn_prefill_kernel[grid](
        Q, K, V, O,
        kv_cache.k_cache, kv_cache.v_cache,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        kv_cache.k_cache.stride(0), kv_cache.k_cache.stride(1),
        kv_cache.k_cache.stride(2), kv_cache.k_cache.stride(3),
        kv_cache.v_cache.stride(0), kv_cache.v_cache.stride(1),
        kv_cache.v_cache.stride(2), kv_cache.v_cache.stride(3),
        num_heads,
        seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    kv_cache.cache_len = seq_len  # prefill 后 cache 长度 = prompt 长度
    return O


# =====================================================================
# Host Wrapper: Decode (逐 token 生成)
# =====================================================================
def flash_attn_decode(Q_new, K_new, V_new, kv_cache: KVCache):
    """
    Decode 阶段: 处理单个新 token.

    步骤:
      1. 将新的 K, V append 到 cache
      2. 用 Q_new 对整个 cache 做 attention

    Args:
        Q_new: (batch, num_heads, 1, head_dim)     当前新 token 的 Q
        K_new: (batch, num_heads, 1, head_dim)     当前新 token 的 K
        V_new: (batch, num_heads, 1, head_dim)     当前新 token 的 V
        kv_cache: KVCache 实例
    Returns:
        O: (batch, num_heads, 1, head_dim)
    """
    batch, num_heads, _, head_dim = Q_new.shape

    # Step 1: append 新 KV 到 cache
    kv_cache.append(K_new, V_new)
    cache_len = kv_cache.cache_len

    # Step 2: attention decode
    O = torch.empty_like(Q_new)
    scale = head_dim ** -0.5

    BLOCK_N = 64
    BLOCK_D = head_dim

    grid = (batch * num_heads,)

    flash_attn_decode_kernel[grid](
        Q_new, kv_cache.k_cache, kv_cache.v_cache, O,
        Q_new.stride(0), Q_new.stride(1), Q_new.stride(2), Q_new.stride(3),
        kv_cache.k_cache.stride(0), kv_cache.k_cache.stride(1),
        kv_cache.k_cache.stride(2), kv_cache.k_cache.stride(3),
        kv_cache.v_cache.stride(0), kv_cache.v_cache.stride(1),
        kv_cache.v_cache.stride(2), kv_cache.v_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        num_heads,
        cache_len,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return O


# =====================================================================
# 参考实现 (PyTorch, 不使用 Triton)
# =====================================================================
def naive_attention(Q, K, V):
    """标准 Attention (无 cache)"""
    scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def naive_decode_with_cache(Q_new, K_cache, V_cache, cache_len):
    """
    参考: decode 时 Q_new 对 cache 做 attention.
    Q_new: (batch, heads, 1, dim)
    K_cache: (batch, heads, max_len, dim)  只取前 cache_len 列
    V_cache: (batch, heads, max_len, dim)
    """
    scale = Q_new.shape[-1] ** -0.5
    K = K_cache[:, :, :cache_len, :]
    V = V_cache[:, :, :cache_len, :]
    attn = torch.matmul(Q_new, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


# =====================================================================
# 测试代码
# =====================================================================
def test_flash_attn_with_kvcache():
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 70)
    print("Flash Attention V2 + KV Cache 正确性测试")
    print("=" * 70)

    all_passed = True

    # ---- 测试配置 ----
    test_configs = [
        # (batch, num_heads, prompt_len, decode_steps, head_dim, 描述)
        (1, 1, 64, 10, 64,   "最小配置"),
        (2, 4, 128, 20, 64,  "多 batch, 多 head"),
        (1, 8, 256, 50, 64,  "长 prompt + 50 步 decode"),
        (2, 4, 96, 30, 64,   "非对齐 prompt (seq=96)"),
        (1, 2, 64, 10, 128,  "大 head_dim=128"),
    ]

    for ci, (batch, num_heads, prompt_len, decode_steps, head_dim, desc) in enumerate(test_configs):
        print(f"\n--- 测试 {ci + 1}: {desc} ---")
        print(f"    batch={batch}, heads={num_heads}, prompt={prompt_len}, "
              f"decode_steps={decode_steps}, dim={head_dim}")

        max_seq_len = prompt_len + decode_steps + 16  # 留一些余量
        dtype = torch.float16

        # ========== A. Prefill 测试 ==========
        Q = torch.randn(batch, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, num_heads, prompt_len, head_dim, device=device, dtype=dtype)

        # Triton prefill
        kv_cache = KVCache(batch, num_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        tri_prefill = flash_attn_prefill(Q, K, V, kv_cache)

        # 参考 prefill
        ref_prefill = naive_attention(Q, K, V)

        atol, rtol = 1e-2, 1e-2
        prefill_ok = torch.allclose(tri_prefill, ref_prefill, atol=atol, rtol=rtol)
        max_diff = (tri_prefill - ref_prefill).abs().max().item()
        status = "✅" if prefill_ok else "❌"
        print(f"    [Prefill] {status}  max_diff={max_diff:.6f}")
        if not prefill_ok:
            all_passed = False

        # ========== B. Decode 逐步测试 ==========
        # 用 naive 方式构建参考 cache
        ref_K_all = K.clone()
        ref_V_all = V.clone()

        decode_ok = True
        max_decode_diff = 0.0
        for step in range(decode_steps):
            Q_new = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=dtype)
            K_new = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=dtype)
            V_new = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=dtype)

            # Triton decode
            tri_out = flash_attn_decode(Q_new, K_new, V_new, kv_cache)

            # 参考 decode: 将 K_new, V_new 拼到历史后面
            ref_K_all = torch.cat([ref_K_all, K_new], dim=2)
            ref_V_all = torch.cat([ref_V_all, V_new], dim=2)
            ref_out = naive_attention(Q_new, ref_K_all, ref_V_all)

            diff = (tri_out - ref_out).abs().max().item()
            max_decode_diff = max(max_decode_diff, diff)
            if not torch.allclose(tri_out, ref_out, atol=atol, rtol=rtol):
                decode_ok = False

        status = "✅" if decode_ok else "❌"
        print(f"    [Decode x{decode_steps}] {status}  max_diff={max_decode_diff:.6f}")
        if not decode_ok:
            all_passed = False

    # ========== C. 显存对比 ==========
    print(f"\n{'='*70}")
    print("显存对比: KV Cache vs 无 Cache (每步重新计算)")
    print(f"{'='*70}")

    batch, num_heads, prompt_len, head_dim = 2, 8, 512, 64
    decode_steps = 100
    max_seq = prompt_len + decode_steps + 16
    dtype = torch.float16

    # 方案 A: 使用 KV Cache (预分配固定大小)
    kv_cache_mem = 2 * batch * num_heads * max_seq * head_dim * 2  # K + V, fp16=2bytes
    # 方案 B: 无 cache，每步存完整 KV
    # 第 t 步需要 (prompt + t) 长度的 K, V
    no_cache_peak_mem = 2 * batch * num_heads * (prompt_len + decode_steps) * head_dim * 2

    print(f"  配置: batch={batch}, heads={num_heads}, prompt={prompt_len}, "
          f"decode={decode_steps}, dim={head_dim}")
    print(f"  KV Cache 预分配: {kv_cache_mem / 1024 / 1024:.1f} MB (固定)")
    print(f"  无 Cache 峰值:   {no_cache_peak_mem / 1024 / 1024:.1f} MB")
    print(f"  KV Cache 核心优势: 避免每步重新计算所有 K/V，计算量从 O(n^2) 降为 O(n)")

    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试未通过，请检查实现。")
    print(f"{'='*70}")

    return all_passed


# =====================================================================
# 性能对比 Benchmark
# =====================================================================
def benchmark_decode():
    """对比 decode 阶段: Flash Attn + KV Cache vs 每步重算 naive attention"""
    device = "cuda"

    print(f"\n{'='*70}")
    print("Decode 性能对比: Flash Attn + KV Cache vs Naive (每步重算)")
    print(f"{'='*70}")

    batch, num_heads, head_dim = 2, 8, 64
    dtype = torch.float16
    prompt_len = 128

    cache_lens = [128, 256, 512, 1024, 2048, 4096]

    print(f"\n配置: batch={batch}, heads={num_heads}, dim={head_dim}, prompt={prompt_len}")
    print(f"{'cache_len':>10} | {'Naive (ms)':>12} | {'Flash+Cache (ms)':>16} | {'加速比':>8}")
    print("-" * 60)

    for cache_len in cache_lens:
        max_seq = cache_len + 64

        # 构造数据
        Q_new = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=dtype)
        K_full = torch.randn(batch, num_heads, cache_len, head_dim, device=device, dtype=dtype)
        V_full = torch.randn(batch, num_heads, cache_len, head_dim, device=device, dtype=dtype)

        # 方案 A: naive 每步重算
        def naive_fn():
            return naive_attention(Q_new, K_full, V_full)

        # 方案 B: Flash Attn decode (cache 已填好)
        kv_cache = KVCache(batch, num_heads, max_seq, head_dim, dtype=dtype, device=device)
        kv_cache.k_cache[:, :, :cache_len, :] = K_full
        kv_cache.v_cache[:, :, :cache_len, :] = V_full
        kv_cache.cache_len = cache_len

        def flash_fn():
            # 不 append，直接用已有 cache 做 decode
            O = torch.empty_like(Q_new)
            scale = head_dim ** -0.5
            BLOCK_N = 64
            BLOCK_D = head_dim
            grid = (batch * num_heads,)
            flash_attn_decode_kernel[grid](
                Q_new, kv_cache.k_cache, kv_cache.v_cache, O,
                Q_new.stride(0), Q_new.stride(1), Q_new.stride(2), Q_new.stride(3),
                kv_cache.k_cache.stride(0), kv_cache.k_cache.stride(1),
                kv_cache.k_cache.stride(2), kv_cache.k_cache.stride(3),
                kv_cache.v_cache.stride(0), kv_cache.v_cache.stride(1),
                kv_cache.v_cache.stride(2), kv_cache.v_cache.stride(3),
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                num_heads,
                cache_len,
                scale,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )
            return O

        quantiles = [0.5, 0.2, 0.8]
        naive_ms, _, _ = triton.testing.do_bench(naive_fn, quantiles=quantiles)
        flash_ms, _, _ = triton.testing.do_bench(flash_fn, quantiles=quantiles)

        speedup = naive_ms / flash_ms if flash_ms > 0 else float("inf")
        print(f"{cache_len:>10} | {naive_ms:>10.3f}ms | {flash_ms:>14.3f}ms | {speedup:>7.2f}x")


if __name__ == "__main__":
    test_flash_attn_with_kvcache()
    benchmark_decode()