# Step 6: FlashAttention 反向传播

FlashAttention 的反向传播比前向传播更复杂，需要重新计算注意力分数。关键洞察是：**我们可以在反向传播时重新计算中间结果，而不是存储它们**。

## 1. 反向传播的挑战

标准 Attention 的反向传播需要：

```python
# 前向传播
scores = Q @ K.T / sqrt(d)    # [N, N]
attn = softmax(scores)         # [N, N]
output = attn @ V              # [N, d]

# 反向传播需要
# dQ, dK, dV = backward(d_output, Q, K, V, attn)
```

问题：我们需要 `attn` 矩阵 (N×N) 来计算梯度，但 FlashAttention 没有存储它！

## 2. FlashAttention 的解决方案：Recomputation

**核心思想**：在反向传播时重新计算注意力分数，而不是存储。

```
前向传播:
  不存储 N×N 的 attn 矩阵
  只存储必要的统计量 (m, l) 或重新计算

反向传播:
  重新计算 Q @ K.T 和 softmax
  计算梯度
```

### 内存对比

```
标准 Attention:
  前向: 存储 attn [N, N]
  反向: 使用存储的 attn
  总内存: O(N²)

FlashAttention:
  前向: 不存储 attn
  反向: 重新计算 attn
  总内存: O(N)
```

## 3. 梯度公式

### 3.1 输出对输入的梯度

设 `O = softmax(Q @ K.T / sqrt(d)) @ V`，求 `dQ`, `dK`, `dV`：

```python
# dV 的梯度
dV = attn.T @ dO    # [N, d]

# d(attn) 的梯度
d_attn = dO @ V.T   # [N, N]

# d(scores) 的梯度 (softmax 反向传播)
# softmax(s)_i = p_i
# dL/ds_j = sum_i (dL/dp_i * dp_i/ds_j)
#         = p_j * (dL/dp_j - sum_i(p_i * dL/dp_i))
P = softmax(scores)
d_scores = P * (d_attn - (P * d_attn).sum(dim=-1, keepdim=True))

# dQ, dK 的梯度
dQ = d_scores @ K / sqrt(d)   # [N, d]
dK = d_scores.T @ Q / sqrt(d) # [N, d]
```

### 3.2 简化公式

```python
# 更高效的计算方式
D = (dO * O).sum(dim=-1)  # [N]

d_scores = P * (d_attn - D.unsqueeze(-1))
```

## 4. FlashAttention 反向传播算法

```python
def flash_attention_backward(dO, Q, K, V, O):
    """
    FlashAttention 反向传播

    核心思想: 重新计算注意力分数, 而不是存储
    """
    N, d = Q.shape
    BLOCK_M = 128
    BLOCK_N = 64

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # 预计算 D = rowsum(dO * O)
    D = (dO * O).sum(dim=-1)

    scale = 1.0 / math.sqrt(d)

    # 外层循环: 遍历 K, V 分块
    for j in range(0, N, BLOCK_N):
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]
        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        # 内层循环: 遍历 Q 分块
        for i in range(0, N, BLOCK_M):
            Qi = Q[i:i+BLOCK_M]
            dOi = dO[i:i+BLOCK_M]
            Di = D[i:i+BLOCK_M]

            # 重新计算注意力分数
            Sij = Qi @ Kj.T * scale
            Pij = softmax(Sij)

            # 计算 dV
            dVj += Pij.T @ dOi

            # 计算 dP
            dPij = dOi @ Vj.T
            dSij = Pij * (dPij - Di.unsqueeze(-1))

            # 计算 dQ, dK
            dQ[i:i+BLOCK_M] += dSij @ Kj * scale
            dKj += dSij.T @ Qi * scale

        dK[j:j+BLOCK_N] = dKj
        dV[j:j+BLOCK_N] = dVj

    return dQ, dK, dV
```

## 5. Triton Kernel 实现

反向传播需要两个 kernel：

### 5.1 Kernel 1: 预处理

```python
@triton.jit
def flash_attn_bwd_preprocess_kernel(
    O, DO, D,
    stride_om, stride_ok,
    stride_dom, stride_dok,
    stride_dm,
    N, D_HEAD,
    BLOCK_M: tl.constexpr,
):
    """计算 D = rowsum(dO * O)"""
    off_m = tl.program_id(0)

    O_ptrs = O + off_m * BLOCK_M * stride_om + tl.arange(0, D_HEAD)[None, :]
    DO_ptrs = DO + off_m * BLOCK_M * stride_dom + tl.arange(0, D_HEAD)[None, :]

    # 累加器
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, D_HEAD, tl.cdiv(D_HEAD, 1)):
        o = tl.load(O_ptrs)
        do = tl.load(DO_ptrs)
        acc += tl.sum(o * do, axis=1)

    D_ptrs = D + off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(D_ptrs, acc)
```

### 5.2 Kernel 2: 主循环

```python
@triton.jit
def flash_attn_bwd_kernel(
    # 指针
    Q, K, V, DO, DQ, DK, DV,
    # 步长
    stride_qz, stride_qh, stride_qm, stride_qk,
    # ...
    # 维度
    Z, H, N_CTX, D_HEAD,
    # Meta 参数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """FlashAttention 反向传播主 kernel"""
    # 类似前向传播，但需要:
    # 1. 重新计算 P = softmax(Q @ K.T)
    # 2. 计算 dQ, dK, dV
    # ...
```

## 6. 完整实现

完整实现见本目录下的 `flash_attention_backward.py`。

关键点：

1. **原子累加**：`dQ` 需要原子加，因为多个 block 可能更新同一个位置

```python
tl.store(DQ_ptrs, dq_new, mask=mask)  # 需要原子操作
```

2. **重新计算而非存储**：反向传播时重新计算注意力分数

3. **分块策略**：反向传播的分块顺序与前向传播不同

## 7. 性能对比

```
标准 Attention Backward:
  - 需要存储 attn 矩阵: O(N²) 内存
  - 计算量: O(N²)

FlashAttention Backward:
  - 内存: O(N)
  - 计算量: O(N²) (重新计算)
  - 但由于 SRAM 访问更快，实际更快
```

## 8. 扩展：FlashAttention-2

FlashAttention-2 进一步优化了：

1. **减少非矩阵乘法操作**：更多使用 `tl.dot`
2. **更好的并行化**：调整 block 分配策略
3. **序列并行**：支持序列维度并行

## 9. 总结

```
FlashAttention 核心技术:

1. Tiling (分块)
   - 将大矩阵分成小块
   - 在 SRAM 中完成计算

2. Online Softmax
   - 增量更新 softmax 结果
   - 避免存储完整的注意力矩阵

3. Recomputation (反向传播)
   - 重新计算而非存储中间结果
   - 用计算换内存

最终效果:
   - 内存: O(N²) → O(N)
   - 速度: 更快 (减少 HBM 访问)
```

## 参考资料

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)