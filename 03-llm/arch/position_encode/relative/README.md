# 相对位置编码

## 1. 概述

相对位置编码（Relative Position Encoding）相比绝对位置编码，能够更好地捕捉 token 之间的相对位置关系。

## 2. RoPE (Rotary Position Embedding)

RoPE 通过旋转矩阵来实现位置编码，具有以下优点：

- 相对位置感知
- 线性计算复杂度
- 外推性好

### 2.1 RoPE 数学原理

对于位置 $m$ 的 token，其 query 向量为 $q_m$，位置 $n$ 的 key 向量为 $k_n$，RoPE 操作定义为：

$$q_m = R_{\Theta,m} q, \quad k_n = R_{\Theta,n} k$$

其中 $R_{\Theta,m}$ 是旋转矩阵。

### 2.2 RoPE 实现

```python
import torch

def apply_rope(q, k, freqs_cis):
    """
    应用 RoPE 位置编码
    freqs_cis: 预计算的频率复数张量
    """
    q_ = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.to(q.device)
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)

    return q_out.type_as(q), k_out.type_as(k)
```

## 3. ALiBi (Attention with Linear Biases)

ALiBi 通过注意力权重中添加与距离成比例的偏置来实现位置编码。

## 4. 其他相对位置编码方法

### 4.1 Transformer-XL 相对位置编码

### 4.2 T5 相对位置编码

## 5. 对比分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| RoPE | 效率高，外推性好 | 实现稍复杂 |
| ALiBi | 简单，长度外推性好 | 需要调参 |
| T5 | 灵活 | 参数量大 |
