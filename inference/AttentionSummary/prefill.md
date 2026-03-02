# Attention 机制的演进

> 从标准 Attention 到 KV Cache 优化

## 基本约定

本文档中所有张量形状均为 `[B, L, D]`，分别对应 `batch_size`、`seq_len`、`hidden_dim`。

## 标准 Attention

Attention 的核心公式：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于不带 KV Cache 的 Attention，不存在 Prefill 或 Decode 阶段的区分，时间复杂度均为 $O(B \cdot L^2)$。

### 标准 Attention 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 线性映射
        q = self.q_proj(x).view(batch_size, seq_len, self.embed_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.embed_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.embed_dim)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Causal Mask - 阻止模型看到未来的 Token
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

        # 计算注意力输出
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)

        return output

# 测试
if __name__ == "__main__":
    model = StandAttention(embed_dim=128)
    dummy_input = torch.randn(2, 10, 128)
    output = model(dummy_input)
    print(f"输入形状：{dummy_input.shape}, 输出形状：{output.shape}")
```

## KV Cache：从 Prefill 到 Decode

### 为什么需要 KV Cache？

观察 Causal Mask 的注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{\text{causal}}\right)V$$

在自回归生成过程中，每个新 Token 的生成都需要关注之前所有 Token。由于 Causal Mask 的存在，之前计算过的 K 和 V 可以复用，无需重复计算。

**KV Cache 的核心思想**：将之前计算的 K 和 V 保存下来，新 Token 只需计算自己的 K 和 V，然后与历史缓存拼接即可。

### KV Cache 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardMultiHeadAttentionKVCache(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, past_key_value=None, use_cache=False):
        """
        x shape: [batch_size, seq_len, hidden_dim]
        past_key_value: (past_K, past_V) 元组，用于 Decode 阶段
        """
        batch_size, q_len, _ = x.shape

        # 1. 计算当前输入的 Q, K, V
        q = self.q_proj(x).view(batch_size, q_len, self.hidden_dim)
        k = self.k_proj(x).view(batch_size, q_len, self.hidden_dim)
        v = self.v_proj(x).view(batch_size, q_len, self.hidden_dim)

        # 2. KV Cache 拼接
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        current_key_value = (k, v) if use_cache else None
        kv_len = k.size(1)

        # 3. 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

        # Causal Mask - Prefill 阶段需要遮蔽，Decode 阶段不需要
        if q_len > 1:
            mask = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device),
                diagonal=kv_len - q_len + 1
            )
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # 4. 输出投影
        output = self.o_proj(context)

        return output, current_key_value

# 测试：Prefill + Decode 流程
if __name__ == "__main__":
    model = StandardMultiHeadAttentionKVCache(hidden_dim=128)

    # Prefill 阶段：处理初始 prompt
    prompt = torch.randn(1, 4, 128)
    output_prefill, past_kv = model(prompt, use_cache=True)
    print(f"Prefill 输出：{output_prefill.shape}, KV Cache: {past_kv[0].shape}")

    # Decode 阶段：单 Token 生成
    next_token = torch.randn(1, 1, 128)
    output_decode, _ = model(next_token, past_key_value=past_kv, use_cache=True)
    print(f"Decode 输出：{output_decode.shape}")
```

## 多头注意力 (Multi-Head Attention)

### 为什么需要多头？

单头 Attention 只能从一个子空间捕捉信息。**多头注意力**通过并行计算多个注意力头，使模型能够同时关注不同位置的不同类型信息。

### 多头注意力实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 投影并重塑：[B, S, H*D] -> [B, H, S, D]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        # Causal Mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # 重塑回 [B, S, H*D]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output

if __name__ == "__main__":
    model = MultiHeadAttention(num_heads=8, embed_dim=128)
    prompt = torch.randn(1, 4, 128)
    output = model(prompt)
    print(f"输出形状：{output.shape}")
```

### 带 KV Cache 的多头注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_kv=None):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 投影并重塑：[B, S, H*D] -> [B, H, S, D]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV Cache 拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v)
        kv_len = k.size(2)

        # 注意力计算
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=q.device),
                diagonal=kv_len - seq_len + 1
            )
            scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output, past_kv

if __name__ == "__main__":
    model = MultiHeadAttentionKVCache(num_heads=8, embed_dim=128)

    # Prefill
    prompt = torch.randn(1, 4, 128)
    out, kv_cache = model(prompt, None)
    print(f"Prefill 输出：{out.shape}, KV Cache: {kv_cache[0].shape}")

    # Decode
    token = torch.randn(1, 1, 128)
    out, _ = model(token, kv_cache)
    print(f"Decode 输出：{out.shape}")
```

## KV Cache 优化：从 MHA 到 MQA

### KV Cache 显存分析

在多头注意力中，KV Cache 的大小与 `num_heads` 成正比：

$$\text{KV Cache Size} = 2 \times B \times L \times \text{num\_heads} \times \text{head\_dim}$$

对于长序列生成，KV Cache 可能成为显存瓶颈。

### Multi-Query Attention (MQA)

**MQA 的核心思想**：多个 Query 头共享同一组 Key 和 Value 头。

- **Q**: 保持 `num_heads` 个头的完整维度
- **K, V**: 只保留 1 个头

这样 KV Cache 的用量减少到原来的 $1/\text{num\_heads}$。

### MQA 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Q 保持完整维度
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # K, V 只有 1 个头
        self.k_proj = nn.Linear(embed_dim, self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_kv=None):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Q: [B, H, S, D]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: [B, 1, S, D] - 只有 1 个头
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1, 2)

        # KV Cache 拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v)
        kv_len = k.size(2)

        # 注意力计算 - 利用广播机制
        # q: [B, H, S, D] @ k^T: [B, 1, D, L] -> [B, H, S, L]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=q.device),
                diagonal=kv_len - seq_len + 1
            )
            scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # attn_weights: [B, H, S, L] @ v: [B, 1, L, D] -> [B, H, S, D]
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.o_proj(context)
        return output, past_kv

if __name__ == "__main__":
    print("=== MQA 显存节省验证 ===")
    model = MultiQueryAttention(num_heads=8, embed_dim=128)

    # Prefill
    prompt = torch.randn(1, 4, 128)
    out, kv_cache = model(prompt, None)
    print(f"Prefill 输出：{out.shape}")
    print(f"KV Cache 形状：K={kv_cache[0].shape}, V={kv_cache[1].shape}")
    print(f"-> 第二维（头数）为 1，显存占用降至 MHA 的 1/8!")

    # Decode
    token = torch.randn(1, 1, 128)
    out, _ = model(token, kv_cache)
    print(f"Decode 输出：{out.shape}")
```

## 总结：Attention 演进路线

| 变体 | KV Cache 大小 | 优点 | 缺点 |
|------|---------------|------|------|
| **Standard Attention** | 无 | 实现简单 | 无法用于自回归生成 |
| **Multi-Head Attention (MHA)** | $2 \times B \times L \times H \times D$ | 多子空间信息捕捉 | KV Cache 显存占用大 |
| **Multi-Query Attention (MQA)** | $2 \times B \times L \times D$ | KV Cache 节省 $H$ 倍 | 表达能力略有损失 |

### 后续演进方向

- **GQA (Grouped-Query Attention)**：MQA 和 MHA 的折中方案，Q 分成多组，每组共享一个 K/V 头
- **FlashAttention**：通过 IO 感知优化，减少 HBM 访问，提升计算效率
- **Chunked Prefill**：将长 Prefill 分块处理，平衡显存和延迟
