# Dropout

## 1. 定义

Dropout 是一种正则化技术，用于防止神经网络过拟合。其核心思想是在**训练时**以概率 $p$ 随机将一部分神经元的输出置零，使得每个 mini-batch 训练的实际是原网络的**一个子网络**，从而减少神经元之间的共适应（co-adaptation），提升泛化能力。

### 数学表达

训练时，对输入 $\mathbf{x}$ 的每个元素 $x_i$：

$$x_i' = \begin{cases} 0 & \text{概率 } p \\ \frac{x_i}{1-p} & \text{概率 } 1-p \end{cases}$$

其中除以 $1-p$ 称为 **inverted dropout**，目的是保持输出的期望值不变。这样在**推理时无需做任何缩放**，直接使用原始网络即可。

> 另一种等价的写法是训练时不缩放，推理时乘以 $1-p$。PyTorch/TensorFlow 均采用 inverted dropout，推理时无需额外操作。

### 训练 vs 推理

| 阶段 | Dropout 行为 |
| ---- | ------------ |
| 训练 | 以概率 $p$ 随机丢弃神经元，保留下来的神经元输出缩放 $1/(1-p)$ |
| 推理 | **关闭** Dropout，使用完整网络，不做丢弃也不做缩放 |

### 为什么有效

- **Bagging 集成视角**：每次 dropout 相当于训练一个不同的子网络（共享参数），推理时相当于指数级数量子网络的集成平均
- **破坏共适应**：神经元不能依赖特定其他神经元的存在，被迫学习更鲁棒的特征
- **噪声正则化**：引入噪声迫使网络学习到的表示更加稳定

---

## 2. 使用

### PyTorch 基础用法

```python
import torch.nn as nn

# p 为丢弃概率，常用 p=0.1 或 p=0.5
dropout = nn.Dropout(p=0.1)

# 训练时必须切换到 train 模式
model.train()

# 推理时必须切换到 eval 模式（dropout 不生效）
model.eval()
```

> **注意**：`nn.Dropout` 的行为由 `model.train()` / `model.eval()` 控制。如果在 eval 模式下调 loss，dropout 不会生效，会导致训练/推理结果不一致，这是常见踩坑点。

### Transformer 中的 Dropout 位置

Transformer 中通常在以下几个地方使用 dropout：

```text
LayerNorm
    ↓
Attention (attention dropout)
    ↓
    + ← residual dropout (又名 output dropout)
    ↓
LayerNorm
    ↓
FFN/MLP
    ↓
    + ← residual dropout
```

具体来说：

| 位置 | 说明 | 典型 p 值 |
| ---- | ---- | --------- |
| **Attention dropout** | 对 attention weights 做 dropout，随机遮蔽部分注意力关系 | 0.0 ~ 0.1 |
| **Residual dropout** | 在残差连接加法后、下一层 LayerNorm 前做 dropout | 0.1 |
| **FFN dropout** | 在 FFN 中间层（激活函数后）做 dropout | 0.1 |
| **Embedding dropout** | 对词嵌入层做 dropout | 0.1 |

以原版 Transformer 为例：

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # 1. Self-Attention
        attn_out = self.attn(x)
        x = x + self.dropout(attn_out)  # residual dropout

        # 2. FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)    # residual dropout
        return x
```

### Dropout 比例选择建议

| p 值 | 适用场景 |
| ---- | -------- |
| 0.1 | Transformer（LLaMA、GPT 等），残差和 FFN 层 |
| 0.2 | 较大模型、数据量充足的场景 |
| 0.5 | 全连接层、CNN 最后的分类层 |
| 0.0 | 不适用 dropout（BN 或其他正则替代） |

### 变体简介

| 变体 | 说明 |
| ---- | ---- |
| **SpatialDropout** | 按通道维度整体丢弃（如 CNN 的 feature map 通道），而非逐元素丢弃 |
| **DropConnect** | 丢弃的是权重连接而不是激活值，即随机将权重矩阵的部分元素置零 |
| **Variational Dropout** | 所有时间步共享同一个 dropout mask（常用于 RNN），PyTorch 中已废弃 |
| **DropPath / Stochastic Depth** | 随机丢弃整个网络层（跳过整个 block），常用于 ViT 等深层网络 |

### 现代大模型中的趋势

- **GPT-1**: 全量使用 dropout（p=0.1）
- **GPT-2**: dropout 比例降得很低
- **GPT-3 / LLaMA**: 仅在部分位置使用极少 dropout（或完全不用），原因是：
  - 数据量足够大，过拟合不是主要问题
  - 预训练阶段的 dropout 可能损害下游微调性能
  - 大规模训练时 dropout 带来的噪声反而不利于收敛
  - 使用 LayerNorm、weight decay、ema 等其他正则手段替代
