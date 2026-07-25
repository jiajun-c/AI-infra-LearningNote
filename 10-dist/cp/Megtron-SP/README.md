# Megatron-SP

Megatron-LM 的序列并行（Sequence Parallelism），在 **token 维度**切分输入序列，与 LayerNorm/Dropout 等 per-token 操作天然兼容。

## 1. 切分公式

设输入为 $X \in \mathbb{R}^{s \times h}$，有 $P$ 个 GPU

### 1.1 初始状态：按 Token 切分

输入 $X$ 沿序列维度均分到 $P$ 个 GPU，每个 GPU $i$ 持有 $s/P$ 个 token 的完整 hidden state：

$$
X_i \in \mathbb{R}^{(s/P) \times h}
$$

即在 token 维度切分，hidden 维度**完整保留**（与 Ulysses 按 head 切分恰好正交）。

### 1.2 Per-Token 操作（零通信）

LayerNorm、Dropout 都是 per-token 操作，在每个 GPU 上本地计算即可：

$$
\text{LN}_i = \text{LayerNorm}(X_i) \in \mathbb{R}^{(s/P) \times h}
$$

**无需任何通信**。

### 1.3 通信操作：gather → 计算 → reduce-scatter

线性层（Attention 内的 QKV 投影、MLP 内的 FFN）需要完整序列上下文。Megatron-SP 的通信模式为「先 gather 再 scatter」：

**Step 1 — all-gather**：将各 GPU 的 token chunk 拼回完整序列：

$$
X = \text{AllGather}(X_0, X_1, \ldots, X_{P-1}) \in \mathbb{R}^{s \times h}
$$

每个 GPU 都拿到完整的 $s$ 个 token。

**Step 2 — 本地计算**：在完整序列上做矩阵乘法：

$$
Y_i = X \cdot W^i \in \mathbb{R}^{s \times h_{\text{out}}}
$$

如果是 Tensor Parallelism 混合场景，$W^i$ 是 TP 切分后的权重；纯 SP 场景 $W^i$ 就是完整权重。

**Step 3 — reduce-scatter**：计算完以后只保留自己负责的那段 token：

$$
Y_i^{\text{local}} = \text{ReduceScatter}(Y_0, Y_1, \ldots, Y_{P-1}) \in \mathbb{R}^{(s/P) \times h_{\text{out}}}
$$

reduce-scatter 将各 GPU 的结果沿 token 维度求和后再分片。

### 1.4 通信量

一次 all-gather 或 reduce-scatter 的通信量约为 $sh$（每个 GPU 收/发 $(P-1)/P \cdot sh \approx sh$）。

每个 Transformer layer 有两次 gather-scatter 对：

|位置|操作|通信量|
|---|---|---|
|Attention 前|all-gather $X$|$\approx sh$|
|Attention 后|reduce-scatter|$\approx sh$|
|MLP 前|all-gather $X$|$\approx sh$|
|MLP 后|reduce-scatter|$\approx sh$|

**总计 $\approx 4sh$ per layer per GPU**。

## 2. 一次 Forward Pass 的完整流程

```text
输入: X_i (s/P, h)                        输出: X_i' (s/P, h)

 [LayerNorm]──── 本地，无通信
     │
     ▼
 [all-gather]─── token 维度拼回完整 s
     │            通信: sh
     ▼
 [Attention]──── 本地计算（或用 TP 进一步切分）
     │
     ▼
 [reduce-scatter]── token 维度结果分回各 GPU
     │            通信: sh
     ▼
 [Dropout + Residual]── 本地
     │
     ▼
 [LayerNorm]──── 本地
     │
     ▼
 [all-gather]─── token 维度拼回完整 s
     │            通信: sh
     ▼
 [MLP]───────── 本地计算（或用 TP 进一步切分）
     │
     ▼
 [reduce-scatter]── token 维度结果分回各 GPU
     │            通信: sh
     ▼
 [Dropout + Residual]── 本地
     │
     ▼
 X_i' (s/P, h)
```

## 3. 与 Tensor Parallelism 混合

Megatron-SP 通常和 TP 组合使用（Megatron-LM 的标准做法）。两者切分维度正交：

```text
         TP: 按列/行切权重矩阵 W
         SP: 按 token 切输入 X

         ┌─────── h ───────┐
    ┌    ┌─────────────────┐
    │    │   GPU(0,0)      │  ← TP rank 0, SP rank 0
    │ s  ├─────────────────┤     (持有前 s/P 的 token)
    │    │   GPU(0,1)      │  ← TP rank 1, SP rank 0
    │    ├─────────────────┤
    │    │   GPU(1,0)      │  ← TP rank 0, SP rank 1
    │    ├─────────────────┤     (持有后 s/P 的 token)
    │    │   GPU(1,1)      │  ← TP rank 1, SP rank 1
    └    └─────────────────┘
```

TP 和 SP 的通信各自独立执行。SP 的核心收益是：将 LayerNorm 和 Dropout 的中间激活分散到各 GPU，**显著降低显存占用**——原本每卡要存完整 $s$ 个 token 的中间结果，现在只需 $s/P$。

对应 cp 总览中的对比参见 [CP 并行总览](../README.md)。
