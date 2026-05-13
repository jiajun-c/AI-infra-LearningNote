# Step 5: FlashAttention 完整实现

现在我们将前面学到的所有知识整合起来，实现完整的 FlashAttention。

## 1. 实现概览

```
FlashAttention = Tiling + Online Softmax + Triton Kernel

核心流程:
1. 每个 program instance 处理一个 Q block
2. 遍历所有 K, V blocks
3. 使用 Online Softmax 增量更新结果
4. 最终输出写入 HBM
```

## 2. Kernel 实现

```python
import triton
import triton.language as tl
import torch
import math

@triton.jit
def flash_attn_kernel(
    # 指针
    Q, K, V, Out,
    # 步长 (用于计算地址)
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

    # 获取 block 索引
    off_hz = tl.program_id(0)  # batch * num_heads + head
    off_z = off_hz // H
    off_h = off_hz % H
    off_m = tl.program_id(1)  # Q block 索引

    # Q block 的起始行
    Q_BLOCK_START = off_m * BLOCK_M

    # ========== 2. 计算 Q block 的地址 ==========

    # Q 的基地址: [batch, head, seq, dim]
    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh

    # Q block 的行偏移
    Q_row_offsets = Q_BLOCK_START + tl.arange(0, BLOCK_M)

    # Q block 的列偏移 (head_dim 维度)
    Q_col_offsets = tl.arange(0, BLOCK_DMODEL)

    # Q block 的指针: [BLOCK_M, BLOCK_DMODEL]
    Q_ptrs = q_base_ptr + Q_row_offsets[:, None] * stride_qm + Q_col_offsets[None, :] * stride_qk

    # Q block 的 mask (处理不满 BLOCK_M 的情况)
    Q_mask = Q_row_offsets[:, None] < N_CTX

    # ========== 3. 加载 Q block 到 SRAM ==========

    q = tl.load(Q_ptrs, mask=Q_mask, other=0.0)  # [BLOCK_M, BLOCK_DMODEL]

    # ========== 4. 初始化 Online Softmax 状态 ==========

    # 最大值初始化为 -inf
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # 归一化因子初始化为 0
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 输出累加器
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 缩放因子: 1 / sqrt(d)
    scale = 1.0 / math.sqrt(BLOCK_DMODEL)

    # ========== 5. 遍历 K, V blocks ==========

    # K, V 的基地址
    k_base_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_base_ptr = V + off_z * stride_vz + off_h * stride_vh

    # 遍历所有 K, V blocks
    lo = 0
    hi = N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        # ------- 5.1 加载 K block -------

        K_row_offsets = start_n + tl.arange(0, BLOCK_N)
        K_ptrs = k_base_ptr + K_row_offsets[None, :] * stride_kn + Q_col_offsets[:, None] * stride_kk
        K_mask = K_row_offsets[None, :] < N_CTX

        k = tl.load(K_ptrs, mask=K_mask, other=0.0)  # [BLOCK_DMODEL, BLOCK_N]

        # ------- 5.2 计算 Q @ K.T -------

        # [BLOCK_M, BLOCK_N] = [BLOCK_M, D] @ [D, BLOCK_N]
        qk = tl.dot(q, k) * scale

        # 添加 causal mask (如果需要)
        # ...

        # ------- 5.3 Online Softmax -------

        # 当前块的 rowmax
        m_ij = tl.max(qk, axis=1)  # [BLOCK_M]

        # 新的 rowmax
        m_i_new = tl.maximum(m_i, m_ij)

        # 重新缩放因子
        alpha = tl.exp(m_i - m_i_new)  # 旧状态的缩放
        beta = tl.exp(m_ij - m_i_new)  # 当前块的缩放

        # 更新归一化因子
        p = tl.exp(qk - m_i_new[:, None])  # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, axis=1)  # [BLOCK_M]
        l_i_new = alpha * l_i + beta * l_ij

        # ------- 5.4 加载 V block -------

        V_row_offsets = start_n + tl.arange(0, BLOCK_N)
        V_ptrs = v_base_ptr + V_row_offsets[:, None] * stride_vn + Q_col_offsets[None, :] * stride_vk
        V_mask = V_row_offsets[:, None] < N_CTX

        v = tl.load(V_ptrs, mask=V_mask, other=0.0)  # [BLOCK_N, BLOCK_DMODEL]

        # ------- 5.5 累加输出 -------

        # 缩放旧累加器
        acc = acc * alpha[:, None]

        # 加上当前块的贡献
        # acc += beta[:, None] * (p / l_ij[:, None]) @ v
        # 但为了数值稳定性，我们用另一种方式
        acc = acc + (beta / l_i_new)[:, None] * tl.dot(p, v)

        # 更新状态
        m_i = m_i_new
        l_i = l_i_new

    # ========== 6. 写回结果 ==========

    # 最终归一化 (虽然已经在累加时做了，但为了正确性再检查)
    # acc 已经是归一化后的结果

    O_base_ptr = Out + off_z * stride_oz + off_h * stride_oh
    O_row_offsets = Q_BLOCK_START + tl.arange(0, BLOCK_M)
    O_ptrs = O_base_ptr + O_row_offsets[:, None] * stride_om + Q_col_offsets[None, :] * stride_ok
    O_mask = O_row_offsets[:, None] < N_CTX

    tl.store(O_ptrs, acc, mask=O_mask)
```

## 3. Host 函数

```python
def flash_attention(q, k, v):
    """
    FlashAttention Host 函数

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]

    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    # 检查输入
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4

    batch, heads, seq_len, head_dim = q.shape

    # 分配输出
    output = torch.empty_like(q)

    # 分块参数
    BLOCK_M = 128
    BLOCK_N = 64

    # Grid 配置
    # dim 0: batch * heads
    # dim 1: seq_len / BLOCK_M
    grid = (
        batch * heads,
        triton.cdiv(seq_len, BLOCK_M),
    )

    # 启动 kernel
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
```

## 4. 完整实现见本目录下的 `flash_attention.py`

## 5. 关键优化点

### 5.1 使用 `tl.dot` 进行矩阵乘法

```python
qk = tl.dot(q, k)  # 而不是 q @ k
```

`tl.dot` 会使用 Tensor Core 进行加速。

### 5.2 内存访问优化

```python
# Q 加载一次，复用多次
q = tl.load(Q_ptrs, mask=Q_mask, other=0.0)

# K, V 分块加载
for start_n in range(0, N_CTX, BLOCK_N):
    k = tl.load(...)
    v = tl.load(...)
```

### 5.3 数值稳定性

```python
# 所有 exp 的参数都是负数或零
alpha = tl.exp(m_i - m_i_new)  # m_i <= m_i_new, 所以 <= 0
```

## 6. 性能对比

运行 `flash_attention.py` 会对比：
- PyTorch 标准 Attention
- FlashAttention (Triton)

在长序列上，FlashAttention 会显著更快且内存占用更少。

## 7. 下一步

→ [Step 6: 反向传播](../step6_backward/README.md)