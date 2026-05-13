# Step 3: 分块计算 (Tiling)

分块计算是 FlashAttention 的核心思想。通过将大矩阵分成小块，我们可以在 SRAM 中完成计算，避免频繁的 HBM 访问。

## 1. 什么是 Tiling？

```
Tiling = 将大矩阵分成小块 (tiles)，逐块处理

示例: 将 [N, d] 的矩阵分成 [BLOCK_SIZE, d] 的小块

大矩阵: [4096, 64]
分成 64 个块: 每块 [64, 64]
```

## 2. 为什么 Tiling 有效？

```
┌────────────────────────────────────────────────────┐
│ 问题: N×N 的注意力矩阵太大，无法放入 SRAM          │
└────────────────────────────────────────────────────┘

解决方案: 不存储完整的 N×N 矩阵!

计算流程:
┌─────────┐   ┌─────────┐
│ Q_block │   │ K_block │
│ [B, d]  │   │ [B, d]  │
└────┬────┘   └────┬────┘
     │             │
     └──────┬──────┘
            ↓
      Q @ K.T = [B, B]  ← 小矩阵，可以放入 SRAM!
            ↓
       softmax
            ↓
      @ V_block
            ↓
       输出增量更新
```

## 3. Tiling 的关键参数

```python
# FlashAttention 的分块参数
BLOCK_M = 128  # Q 的分块大小 (行数)
BLOCK_N = 64   # K, V 的分块大小 (行数)

# 为什么 BLOCK_M ≠ BLOCK_N?
# - Q 分块只处理一次 (外层循环)
# - K, V 分块需要遍历多次 (内层循环)
# - 不同的分块大小可以优化内存访问模式
```

## 4. Tiling 计算流程

```python
# 伪代码: FlashAttention 的分块计算

def flash_attention_tiled(Q, K, V, BLOCK_M, BLOCK_N):
    """
    Q: [N, d]
    K: [N, d]
    V: [N, d]
    """
    N, d = Q.shape
    output = zeros(N, d)

    # 外层循环: 遍历 Q 的分块
    for i in range(0, N, BLOCK_M):
        Q_block = Q[i:i+BLOCK_M]  # [BLOCK_M, d]

        # 初始化累加器
        acc = zeros(BLOCK_M, d)
        l_i = zeros(BLOCK_M, 1)  # 归一化因子
        m_i = ones(BLOCK_M, 1) * (-inf)  # 最大值

        # 内层循环: 遍历 K, V 的分块
        for j in range(0, N, BLOCK_N):
            K_block = K[j:j+BLOCK_N]  # [BLOCK_N, d]
            V_block = V[j:j+BLOCK_N]  # [BLOCK_N, d]

            # 在 SRAM 中计算注意力
            scores = Q_block @ K_block.T  # [BLOCK_M, BLOCK_N]
            scores = scores / sqrt(d)

            # Online softmax 更新 (Step 4 详细讲解)
            m_i_new = max(m_i, max(scores, dim=-1))
            # ... (Online Softmax 公式)

            acc = acc + ...

        output[i:i+BLOCK_M] = acc / l_i

    return output
```

## 5. 内存访问分析

```
标准 Attention:
  读取: Q, K, V, scores, attn_weights
  写入: scores, attn_weights, output
  HBM 访问: O(N²)

Tiled Attention:
  读取: Q (N/BLOCK_M 次), K, V (N/BLOCK_N 次，每次处理 BLOCK_M 行)
  写入: output
  HBM 访问: O(N * d² / BLOCK_SIZE) ≈ O(N) for fixed d
```

## 6. Triton 实现 Tiling

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention kernel

    每个 program instance 处理一个 (batch, head) 的一个 Q block
    """
    # 获取当前处理的 batch, head, block 索引
    off_bz = tl.program_id(0)  # batch
    off_h = tl.program_id(1)   # head
    off_m = tl.program_id(2)   # Q block 索引

    # 计算 Q block 的起始位置
    Q_block_start = off_m * BLOCK_M
    Q_offsets = Q_block_start + tl.arange(0, BLOCK_M)

    # ... 详见 Step 5 完整实现
```

## 7. Grid 启动配置

```python
def flash_attention(Q, K, V):
    batch, heads, seq_len, head_dim = Q.shape

    BLOCK_M = 128
    BLOCK_N = 64

    # Grid 配置
    # - dim 0: batch size
    # - dim 1: num heads
    # - dim 2: seq_len / BLOCK_M (Q block 数量)
    grid = (
        batch,
        heads,
        triton.cdiv(seq_len, BLOCK_M),
    )

    flash_attention_kernel[grid](
        Q, K, V, Out,
        # strides...
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
```

## 8. 可视化理解

```
序列长度 N=16, BLOCK_M=4, BLOCK_N=4

Q 分块:
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ Q0 │ │ Q1 │ │ Q2 │ │ Q3 │  每个 Q block 是 [4, d]
└────┘ └────┘ └────┘ └────┘

K, V 分块:
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ K0 │ │ K1 │ │ K2 │ │ K3 │
└────┘ └────┘ └────┘ └────┘
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ V0 │ │ V1 │ │ V2 │ │ V3 │
└────┘ └────┘ └────┘ └────┘

计算 Q1 的输出:
  for Kj, Vj in [K0,V0, K1,V1, K2,V2, K3,V3]:
      scores = Q1 @ Kj.T  # [4, 4] 在 SRAM 中
      softmax(scores)
      accumulate += scores @ Vj

不存储 16×16 的完整注意力矩阵!
```

## 9. 下一步

理解了 Tiling 后，下一步学习 **Online Softmax**，这是让分块计算正确工作的关键技术。

→ [Step 4: Online Softmax](../step4_online_softmax/README.md)