# Ulysses

Ulysses 是 DeepSpeed 提出的序列并行方案。与 Megatron-SP（按 token 序列维度切分）和 Ring Attention（环状 P2P 传递 KV）不同，Ulysses 在 **head 维度**进行切分，通过 all-to-all 通信在 head 并行与序列并行之间来回切换。

## 1. 切分公式

设输入为 $X \in \mathbb{R}^{s \times h}$，其中 $s$ 为序列长度，$h$ 为 hidden size。有 $P$ 个 GPU，总 attention head 数为 $H$，head dim 为 $d = h / H$。

### 1.1 初始状态：按 Head 切分

将 attention heads 均匀分到 $P$ 个 GPU，每个 GPU $i$ 拥有 $H/P$ 个 head：

$$
H_i = \frac{H}{P}, \quad d_h = \frac{h}{H} = d
$$

每个 GPU 持有完整的序列 $s$，但只负责 $H/P$ 个 head 的 hidden dim：

$$
X_i \in \mathbb{R}^{s \times \frac{h}{P}} = \mathbb{R}^{s \times H_i d}
$$

QKV 投影矩阵也按 head 切分，每个 GPU 只存自己那部分 head 的权重：

$$
W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{h \times H_i d}
$$

投影后每个 GPU 得到自己所辖 head 的 Q、K、V：

$$
Q_i = X_i W_Q^i \in \mathbb{R}^{s \times H_i \times d}
\quad
K_i = X_i W_K^i \in \mathbb{R}^{s \times H_i \times d}
\quad
V_i = X_i W_V^i \in \mathbb{R}^{s \times H_i \times d}
$$

这一步**无需任何通信**，每个 GPU 独立完成。

### 1.2 All-to-All：Head 并行 → 序列并行

投影完成后，Q、K、V 都是「所有 token、部分 head」。要计算 attention，需要每个 token 看到**所有 head 的上下文**。

all-to-all 将数据从 $(s, H/P, d)$ 重排为 $(s/P, H, d)$：

```text
all-to-all 前 (GPU i):           all-to-all 后 (GPU i):
┌─────────────────────┐          ┌─────────────────────┐
│ s tokens            │          │ H heads (全部)       │
│ H/P heads           │   ──→    │ s/P tokens           │
│                     │          │                     │
│ GPU i 有全序列长度   │          │ GPU i 有全部 head    │
│ 但只有 H/P 个 head   │          │ 但只有 s/P 个 token  │
└─────────────────────┘          └─────────────────────┘
```

形式化地，每个 GPU 把自己的 Q 沿序列维度切成 $P$ 份，每份 $s/P$ 个 token：

$$
Q_i = [Q_i^{(0)}; Q_i^{(1)}; \ldots; Q_i^{(P-1)}], \quad Q_i^{(j)} \in \mathbb{R}^{(s/P) \times H_i \times d}
$$

all-to-all 通信：GPU $i$ 把第 $j$ 块 $Q_i^{(j)}$ 发给 GPU $j$，同时从所有 GPU 接收属于自己的第 $i$ 块：

$$
\text{send: } Q_i^{(j)} \rightarrow \text{GPU}_j, \quad \forall j \in [0, P-1]
$$
$$
\text{recv: } Q_j^{(i)} \leftarrow \text{GPU}_j, \quad \forall j \in [0, P-1]
$$

all-to-all 后 GPU $i$ 的 Q 为：

$$
Q_i^{\text{seq}} = \text{Concat}(Q_0^{(i)}, Q_1^{(i)}, \ldots, Q_{P-1}^{(i)}) \in \mathbb{R}^{(s/P) \times H \times d}
$$

对 $K$、$V$ 执行相同的 all-to-all，得到 $K_i^{\text{seq}}, V_i^{\text{seq}} \in \mathbb{R}^{(s/P) \times H \times d}$。

单个 GPU 每方向通信量：

$$
\text{comm\_volume} = \frac{s}{P} \times \frac{H}{P} \times d \times P = \frac{s H d}{P} = \frac{s h}{P}
$$

Q、K、V 三者合计：$3sh/P$。两个 all-to-all 往返后总通信量为 $6sh/P$。

### 1.3 本地 Attention 计算

all-to-all 后，GPU $i$ 拥有 $s/P$ 个 token 的完整 Q、K、V（所有 head）。在每个 GPU 上独立计算 self-attention，**无需额外通信**：

对于 head $h \in [0, H)$，GPU $i$ 上的 token 范围 $[i \cdot s/P,\; (i+1) \cdot s/P)$：

$$
\text{Attention}_i^{(h)} = \text{softmax}\left(\frac{Q_i^{(h)} K_i^{(h)\top}}{\sqrt{d}}\right) V_i^{(h)} \in \mathbb{R}^{(s/P) \times d}
$$

对所有 head 拼接得到 attention 输出：

$$
O_i^{\text{seq}} = [\text{Attention}_i^{(0)}; \ldots; \text{Attention}_i^{(H-1)}] \in \mathbb{R}^{(s/P) \times H \times d}
$$

### 1.4 All-to-All（反向）：序列并行 → Head 并行

attention 计算完成后，需要恢复回「所有 token、部分 head」的布局，以便做 output projection 和后续 MLP：

第二个 all-to-all 与第 1.2 节互为逆操作：

$$
O_i^{\text{seq}} \in \mathbb{R}^{(s/P) \times H \times d} \;\xrightarrow{\text{all-to-all}}\; O_i \in \mathbb{R}^{s \times H_i \times d}
$$

即把每个 GPU 上「$s/P$ 个 token 的全部 head 输出」重新切分为「$s$ 个 token 的部分 head 输出」。

### 1.5 Output Projection

每个 GPU 用自己的 output 权重矩阵做投影：

$$
\text{Output}_i = O_i W_O^i \in \mathbb{R}^{s \times (h/P)}
$$

最后每张卡持有 $s$ 个 token 的 $h/P$ 维 hidden state，与输入时的切分状态一致。

## 2. 通信与计算流程总览

```text
时间 ──────────────────────────────────────────────────►

GPU 0 (head 0..H/P-1):
  [ QKV投影 ] ── all-to-all ── [ Attention(s/P tokens) ] ── all-to-all ── [ Output投影 ]
       │              │                    │                     │               │
GPU 1  │   无通信     │    3sh/P         │      无通信          │    3sh/P     │
...    │              │    per GPU        │                      │   per GPU    │
GPU P-1│              │                    │                     │               │

        ← head-parallel →  ← sequence-parallel →  ← head-parallel →
```

## 3. 与其他方案的对比

参见 [CP 并行总览](../README.md) 中的对比表。
