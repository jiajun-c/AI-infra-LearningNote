# Step 4: Online Softmax

Online Softmax（在线 Softmax）是 FlashAttention 的核心技术。它允许我们在**不存储完整注意力矩阵**的情况下，增量地计算 softmax。

## 1. 问题引入

假设我们分块计算注意力：

```python
# 分块计算 Q @ K.T
for j in range(0, N, BLOCK_N):
    scores_j = Q_block @ K[j:j+BLOCK_N].T  # [BLOCK_M, BLOCK_N]

    # 问题: 如何正确计算 softmax?
    # softmax 需要知道所有 scores, 但我们只有当前块!
```

**问题**：softmax 需要对**所有位置**求和，但我们一次只处理一个块。

## 2. 标准 Softmax 回顾

```python
def softmax(x):
    # x: [N]
    m = max(x)           # 最大值
    exp_x = exp(x - m)   # 减去最大值保证数值稳定
    l = sum(exp_x)       # 归一化因子
    return exp_x / l     # softmax 输出
```

如果分块计算，我们需要**增量更新** `m` 和 `l`。

## 3. Online Softmax 公式

### 3.1 核心思想

假设我们已经处理了前 j-1 个块，得到：
- `m_old`: 前面的最大值
- `l_old`: 前面的归一化因子

现在处理第 j 个块，得到：
- `m_new = max(m_old, m_j)` (新的最大值)
- 需要更新 `l`

### 3.2 归一化因子更新公式

```
关键公式:
l_new = exp(m_old - m_new) * l_old + exp(m_j - m_new) * l_j
```

**推导**：
```
设 x = [x1, x2, ..., x_n, x_{n+1}, ..., x_m]

原始 softmax:
softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))
             = exp(x_i - m) / l

其中 m = max(x), l = sum(exp(x_j - m))

分割 x 为两块:
  块1: [x1, ..., x_n], m1 = max(x1...xn), l1 = sum(exp(xi - m1))
  块2: [x_{n+1}, ..., x_m], m2 = max(x_{n+1}...xm), l2 = sum(exp(xi - m2))

合并:
  m = max(m1, m2)

  l = sum(exp(xi - m))
    = sum_{i=1}^{n} exp(xi - m) + sum_{i=n+1}^{m} exp(xi - m)
    = exp(m1 - m) * sum_{i=1}^{n} exp(xi - m1) + exp(m2 - m) * sum_{i=n+1}^{m} exp(xi - m2)
    = exp(m1 - m) * l1 + exp(m2 - m) * l2
```

### 3.3 输出更新公式

```
O_new = exp(m_old - m_new) * O_old + exp(m_j - m_new) * (scores_j @ V_j)
```

## 4. Online Softmax 算法

```python
def online_softmax_attention(Q_block, K, V, BLOCK_N):
    """
    Online Softmax Attention

    Q_block: [BLOCK_M, d]
    K: [N, d]
    V: [N, d]
    """
    BLOCK_M, d = Q_block.shape
    N = K.shape[0]

    # 初始化
    m = torch.full((BLOCK_M,), float('-inf'))  # 最大值
    l = torch.zeros(BLOCK_M)                    # 归一化因子
    O = torch.zeros(BLOCK_M, d)                 # 输出累加器

    # 遍历 K, V 分块
    for j in range(0, N, BLOCK_N):
        K_block = K[j:j+BLOCK_N]  # [BLOCK_N, d]
        V_block = V[j:j+BLOCK_N]  # [BLOCK_N, d]

        # 计算当前块的注意力分数
        scores = Q_block @ K_block.T / math.sqrt(d)  # [BLOCK_M, BLOCK_N]

        # 当前块的 max 和 sum
        m_j = scores.max(dim=-1).values              # [BLOCK_M]
        l_j = torch.exp(scores - m_j.unsqueeze(-1)).sum(dim=-1)  # [BLOCK_M]

        # 更新全局 max
        m_new = torch.maximum(m, m_j)

        # 更新归一化因子
        # l_new = exp(m_old - m_new) * l_old + exp(m_j - m_new) * l_j
        l_new = torch.exp(m - m_new) * l + torch.exp(m_j - m_new) * l_j

        # 更新输出
        # O_new = exp(m_old - m_new) * O_old + exp(m_j - m_new) * softmax(scores) @ V
        alpha = torch.exp(m - m_new).unsqueeze(-1)  # [BLOCK_M, 1]
        beta = torch.exp(m_j - m_new).unsqueeze(-1)  # [BLOCK_M, 1]

        # 计算当前块的贡献
        scores_softmax = torch.exp(scores - m_new.unsqueeze(-1)) / l_new.unsqueeze(-1).clamp(min=1e-10)
        current_contrib = torch.matmul(
            torch.exp(scores - m_new.unsqueeze(-1)) / l_j.unsqueeze(-1).clamp(min=1e-10),
            V_block
        )

        O = alpha * O * (l / l_new).unsqueeze(-1) + beta * current_contrib * (l_j / l_new).unsqueeze(-1)

        # 更新状态
        m = m_new
        l = l_new

    # 最终归一化
    O = O / l.unsqueeze(-1)

    return O
```

## 5. 简化版公式（更高效）

实际实现中，我们使用更高效的公式：

```python
# 更新公式
m_new = max(m_old, m_j)

# 缩放因子
scale_old = exp(m_old - m_new)
scale_j = exp(m_j - m_new)

# 更新 l
l_new = scale_old * l_old + scale_j * l_j

# 更新 O (分两步更清晰)
# Step 1: 缩放旧的 O
O_old_scaled = scale_old * O_old

# Step 2: 计算当前块贡献并累加
# softmax_ij = exp(scores_ij - m_j) / l_j
# contribution = softmax_ij @ V_j * l_j * scale_j
#             = exp(scores_ij - m_j) @ V_j * scale_j
P_j = exp(scores - m_j)  # [BLOCK_M, BLOCK_N]
O_new = O_old_scaled + scale_j * (P_j @ V_block)
```

## 6. 数值稳定性

```python
# 关键: 所有 exp 的参数都是 ≤ 0 的!
# exp(m_old - m_new) ≤ 1  因为 m_old ≤ m_new
# exp(m_j - m_new) ≤ 1    因为 m_j ≤ m_new
# exp(scores - m_new) ≤ 1 因为 scores ≤ m_new (至少对于当前块)
```

这保证了数值稳定性，不会出现 `exp(很大正数)` 的溢出问题。

## 7. 完整示例代码

```python
import torch
import math

def online_softmax_attention_naive(Q, K, V, BLOCK_N=64):
    """
    Online Softmax Attention - 朴素实现

    用于理解算法，实际实现见 Step 5
    """
    BLOCK_M, d = Q.shape
    N = K.shape[0]

    # 初始化状态
    m = torch.full((BLOCK_M,), float('-inf'), device=Q.device)
    l = torch.zeros(BLOCK_M, device=Q.device)
    O = torch.zeros(BLOCK_M, d, device=Q.device)

    scale = 1.0 / math.sqrt(d)

    for j in range(0, N, BLOCK_N):
        # 当前 K, V 块
        Kj = K[j:j+BLOCK_N]
        Vj = V[j:j+BLOCK_N]

        # 计算注意力分数
        Sj = (Q @ Kj.T) * scale  # [BLOCK_M, BLOCK_N]

        # 当前块的 max
        mj = Sj.max(dim=-1).values

        # 更新全局 max
        m_new = torch.maximum(m, mj)

        # 缩放因子
        alpha = torch.exp(m - m_new)  # 旧状态的缩放
        beta = torch.exp(mj - m_new)  # 当前块的缩放

        # 更新归一化因子
        Pj = torch.exp(Sj - mj.unsqueeze(-1))  # 未归一化的概率
        lj = Pj.sum(dim=-1)  # 当前块的 sum
        l_new = alpha * l + beta * lj

        # 更新输出
        O = alpha.unsqueeze(-1) * O * (l / l_new).unsqueeze(-1) + \
            beta.unsqueeze(-1) * (Pj @ Vj) * (lj / l_new).unsqueeze(-1)

        # 更新状态
        m = m_new
        l = l_new

    return O
```

## 8. 验证正确性

运行本目录下的 `online_softmax.py` 验证 Online Softmax 的正确性。

## 9. 下一步

掌握了 Online Softmax 后，我们可以开始实现完整的 FlashAttention。

→ [Step 5: FlashAttention 实现](../step5_flash_attn/README.md)