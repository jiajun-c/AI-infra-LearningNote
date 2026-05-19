# Step 1: 理解标准 Attention 的内存瓶颈

在开始写 FlashAttention 之前，我们需要先理解 **为什么** 需要它。

## 1. 标准 Attention 公式

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

其中：
- Q: [batch, heads, seq_len, head_dim] 查询矩阵
- K: [batch, heads, seq_len, head_dim] 键矩阵
- V: [batch, heads, seq_len, head_dim] 值矩阵
- Q @ K.T: [batch, heads, seq_len, seq_len] 注意力分数矩阵

## 2. PyTorch 标准实现

```python
import torch
import torch.nn.functional as F

def standard_attention(q, k, v):
    """
    标准 Attention 实现 
    q, k, v: [batch, heads, seq_len, head_dim]
    """
    # Step 1: 计算 Q @ K.T，得到注意力分数
    # [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
    # = [batch, heads, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: 缩放
    scores = scores / (q.size(-1) ** 0.5)

    # Step 3: Softmax
    # 这里需要读取完整的 scores 矩阵
    attn_weights = F.softmax(scores, dim=-1)

    # Step 4: 加权求和
    # [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, head_dim]
    # = [batch, heads, seq_len, head_dim]
    output = torch.matmul(attn_weights, v)

    return output
```

## 3. 内存访问分析

假设 `seq_len = 4096`, `head_dim = 64`, `dtype = float16`:

```
Q, K, V 各自大小: 4096 * 64 * 2 bytes = 512 KB

中间矩阵:
- scores: 4096 * 4096 * 2 bytes = 32 MB
- attn_weights: 4096 * 4096 * 2 bytes = 32 MB
```

### GPU 内存层级

```
┌─────────────────────────────────────┐
│           HBM (显存)                │  ← 大但慢 (~2 TB/s)
│   大小: 40-80 GB                    │
│   存储: Q, K, V, scores, attn_weights │
└─────────────────────────────────────┘
              ↕ (数据搬运)
┌─────────────────────────────────────┐
│           SRAM (片上缓存)            │  ← 小但快 (~20 TB/s)
│   大小: ~20 MB per SM               │  ← 关键约束!
└─────────────────────────────────────┘
```

### 问题所在

```python
# 标准 Attention 的内存访问模式:

# 1. Q, K 从 HBM 读入 → 计算 Q@K.T
scores = q @ k.T           # scores 写入 HBM (32 MB)

# 2. scores 从 HBM 读入 → 计算 softmax
attn = softmax(scores)     # attn_weights 写入 HBM (32 MB)

# 3. attn_weights 从 HBM 读入 → 计算 @ V
output = attn @ v          # output 写入 HBM

# 总计: 写入 64 MB 中间结果到 HBM，然后再读出来
```

**关键问题**：
- `scores` 和 `attn_weights` 都是 N×N 的大矩阵
- 必须写入 HBM，因为 SRAM 放不下 (32 MB > 20 MB)
- 这导致了大量 HBM 读写，成为性能瓶颈

## 4. 内存复杂度分析

```
标准 Attention: O(N²) 内存
- scores 矩阵: N × N
- attn_weights 矩阵: N × N

对于长序列 (如 N=16K):
- N² = 256M 元素
- float16: 512 MB 仅存储一个注意力矩阵!
```

## 5. FlashAttention 的核心洞察

FlashAttention 的核心思想：

```
❌ 不存储完整的 N×N 注意力矩阵

✅ 分块计算，每次只在 SRAM 中计算一小块
✅ 使用 Online Softmax 增量更新结果
```

### 内存对比

```
标准 Attention:
  HBM 读写: O(N² + N·d)  ← 随序列长度平方增长

FlashAttention:
  HBM 读写: O(N·d²/block_size)  ← 线性增长!
```

## 6. 可视化理解

```
标准 Attention (N=16):

Q: [16, d]     K: [16, d]     V: [16, d]
      ↓              ↓
      └──────┬───────┘
             ↓
    scores: [16, 16]  ← 必须存储整个矩阵
             ↓
    attn: [16, 16]    ← 必须存储整个矩阵
             ↓
      └──────┬───────┘
             ↓
    output: [16, d]


FlashAttention (分块, block_size=4):

Q 分成 4 块: [Q0, Q1, Q2, Q3], 每块 [4, d]
K 分成 4 块: [K0, K1, K2, K3], 每块 [4, d]
V 分成 4 块: [V0, V1, V2, V3], 每块 [4, d]

for Qi in [Q0, Q1, Q2, Q3]:
    for Kj, Vj in [(K0,V0), (K1,V1), (K2,V2), (K3,V3)]:
        # 在 SRAM 中计算 Qi @ Kj.T = [4, 4] ← 很小!
        # 用 online softmax 增量更新
        # 不存储中间的 [4, 4] 矩阵到 HBM
```

## 7. 实验验证

运行本目录下的 `benchmark.py` 来验证：

```bash
python step1_problem/benchmark.py
```

你会看到：
- 不同序列长度下的内存占用
- HBM 读写量对比
- 执行时间对比

## 下一步

理解了问题之后，下一步学习 **Triton Softmax Kernel**，这是 FlashAttention 的基础构建块。

→ [Step 2: Softmax Kernel](../step2_softmax/README.md)