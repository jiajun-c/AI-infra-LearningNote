# FlashAttention V2 实现

一个Standard Attention的形式如下所示

$softmax(Q \times K^T) V$

假设输入为 (N, D) 的形式，其需要的中间存储开销为 $2*N*N$，随着序列长度的增加其平方增长

FlashAttention中通过引入`online softmax`的机制，使得其可以每次计算出一块的结果。下面将会通过`triton`的flashAttention v2的代码来解释分块策略

![alt text](image-1.png)


在输入为 (N, D)的情况下，选择大小为 (Block_M x Block_N)

在并行维度上对Q的N维度进行并行，因为这样并行时不同块之间是完全无数据依赖的，可以在D维度进行softmax操作

如下所示获取到每个Q的分块，然后内存对K进行循环

```python
    pid = tl.program_id(0)
    
    # 2. 初始化 Q 的指针
    # offs_m: Q 的行索引范围 [pid*BLOCK_M, (pid+1)*BLOCK_M]
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d) # 特征维度索引

    q_ptrs = Q + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask = offs_m[:, None] < N, other=0.0)
```

初始化统计量

```python
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)
```

遍历K和V，对于K按照 (Block_N, d) 的shape进行分块

```python
k_ptrs = K + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)
```

然后将Q和K进行相乘，再进行放缩

```python
qk = tl.dot(q, k)
qk *= sm_scale
```

按照原先的流程此时需要进行softmax的操作，但是此时我们仅得到了一个中间的部分结果，输出shape为 (BLOCK_M, BLOCK_N)，完整的输出为(BLOCK_M, D)

采用online softmax的方式进行数据的更新

```python
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        
        alpha = tl.exp(m_i - m_new)
        
        # 更新分母 l
        l_i = l_i * alpha + tl.sum(p, 1)
```

将softmax的分子和v进行相乘(BLOCK_M, BLOCK_N) x (BLOCK_N, D)，然后最后再除以online softmax得到的分母。

完整的代码如下所示

```python
import torch
import triton
import triton.language as tl

# ==========================================
# Part 1: Triton Kernel (针对 [N, d] 输入)
# ==========================================

@triton.jit
def _flash_attn_nd_kernel(
    Q, K, V, Out,              # 指针
    stride_qn, stride_qd,      # Q 的 Strides (N维度, d维度)
    stride_kn, stride_kd,      # K 的 Strides
    stride_vn, stride_vd,      # V 的 Strides
    stride_on, stride_od,      # Output 的 Strides
    sm_scale,                  # Softmax scaling factor
    BLOCK_M: tl.constexpr,     # Q 分块大小
    BLOCK_N: tl.constexpr,     # K/V 分块大小
    d: tl.constexpr,           # 特征维度 (Head Dim)
    N: tl.constexpr            # 序列总长度
):
    # 1. Grid 索引
    # pid 对应 Q 矩阵在 N 维度上的切片索引
    pid = tl.program_id(0)
    
    # 2. 初始化 Q 的指针
    # offs_m: Q 的行索引范围 [pid*BLOCK_M, (pid+1)*BLOCK_M]
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d) # 特征维度索引

    # offs_m [BLOCK_M, 1]
    # offs_d [1, BLOCK_D]
    # Q 指针计算: Base + (Row * Stride_Row) + (Col * Stride_Col)
    q_ptrs = Q + (offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd)

    # 3. 加载 Q (驻留 SRAM)
    # mask 确保不读取超过 N 的行
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # 4. 初始化统计量 (m, l, acc)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)

    # 5. 遍历 K 和 V (对应算法中的 j 循环)
    # 我们遍历整个 N 长度
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # 计算 K 指针并加载
        k_ptrs = K + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)

        # 计算 QK^T
        # 输出的shape为[BLOCK_M, BLOCK_N]
        # 
        qk = tl.dot(q, k)
        qk *= sm_scale

        # --- Online Softmax 逻辑 (与论文一致) ---
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        
        alpha = tl.exp(m_i - m_new)
        
        # 更新分母 l
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # 修正旧的 acc
        acc = acc * alpha[:, None]
        
        # 计算 P 并加载 V
        
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
        
        # 累加 P * V
        acc += tl.dot(p.to(tl.float16), v)
        
        # 更新 max
        m_i = m_new

    # 6. 最终归一化与写回
    acc = acc / l_i[:, None]
    
    o_ptrs = Out + (offs_m[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N)


# ==========================================
# Part 2: Python Wrapper ([N, d])
# ==========================================

def flash_attention_nd(q, k, v):
    # 输入形状检查: [N, d]
    assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2
    N, d = q.shape
    
    # Block 大小配置
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Softmax scaling
    sm_scale = 1.0 / (d ** 0.5)
    
    # 输出张量
    o = torch.empty_like(q)
    
    # Grid: 只需要在 N 方向切分
    grid = (triton.cdiv(N, BLOCK_M), )
    
    # print(q.stride(0), q.stride(1))
    _flash_attn_nd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        d=d,
        N=N
    )
    
    return o

# ==========================================
# Part 3: 验证
# ==========================================

def test_nd_shape():
    torch.manual_seed(42)
    dtype = torch.float16
    device = "cuda"

    # 定义形状 [N, d]
    N = 2048
    d = 64
    
    print(f"Testing simple [N, d] layout with N={N}, d={d}")

    q = torch.randn((N, d), dtype=dtype, device=device)
    k = torch.randn((N, d), dtype=dtype, device=device)
    v = torch.randn((N, d), dtype=dtype, device=device)

    # 1. Triton 结果
    tri_out = flash_attention_nd(q, k, v)

    # 2. PyTorch Reference (模拟 [N, d])
    # 标准 Attention 公式: Softmax(Q @ K.T / scale) @ V
    # Q: [N, d], K.T: [d, N] -> Scores: [N, N]
    scores = torch.matmul(q, k.t()) * (1.0 / (d**0.5))
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v)

    # 3. 对比
    diff = torch.abs(ref_out - tri_out)
    max_diff = diff.max().item()
    
    print(f"Max Difference: {max_diff}")
    if max_diff < 1e-2:
        print("✅ Test Passed! Strictly matches [N, d] logic.")
    else:
        print("❌ Test Failed.")

if __name__ == "__main__":
    test_nd_shape()
```